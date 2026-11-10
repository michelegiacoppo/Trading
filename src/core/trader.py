"""Live / dry-run orchestration loop.

Start-up (cold start)::

    connect MT5 -> select symbols -> detect server timezone (auto)
    -> WARM-UP: last 1000 closed bars of M1/M5/M15/H1 per symbol (refreshes the Parquet cache)
    -> key-level map (PDH/PDL, Asian/London range, daily VWAP +-sigma) -> start listening

Main loop (single thread, the dashboard renders from its own thread)::

    every poll_interval_ms
      ├─ ensure MT5 connection (auto-reconnect) + AutoTrading status
      ├─ account equity (closed + floating) -> daily kill-switch -> flatten & LOCKED if hit
      ├─ ticks -> spread filter, tick-jump detector, stale-feed watchdog
      ├─ manage trades (limit expiry, TP1 65% + BE, micro-trailing, stale killer)
      └─ on each new CLOSED M1 bar per symbol
           ├─ pull M1/M5/M15 (+levels TF), features + key levels, gap detector
           ├─ strategy signal -> guardrails -> position sizer (key-level priority)
           └─ execution (pending LIMIT on pullback / MARKET with deviation cap)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import Settings, resolve_path
from src.core.clock import ServerClock
from src.core.data_feed import DataFeed, HistoricalStore, TickSnapshot
from src.core.execution import ExecutionEngine, MarketContext
from src.core.levels import KeyLevelMap, compute_levels
from src.core.mt5_connector import MT5Connector
from src.core.state import BotState, PendingRow, PositionRow, SymbolState
from src.risk.guardrails import EntryContext, Guardrails
from src.risk.position_sizer import PositionSizer
from src.strategies import indicators as ind
from src.strategies.base_strategy import BaseStrategy

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LiveTrader:
    def __init__(self, settings: Settings, connector: MT5Connector, strategy: BaseStrategy,
                 state: BotState, dry_run: bool = True, symbols: list[str] | None = None,
                 clock: ServerClock | None = None) -> None:  # fmt: skip
        self.s = settings
        self.conn = connector
        self.strategy = strategy
        self.state = state
        self.dry_run = dry_run
        self.symbols = symbols or settings.enabled_symbols
        self.clock = clock or strategy.clock
        self.feed = DataFeed(connector)
        self.store = HistoricalStore(resolve_path(settings.data.historical_dir), connector,
                                     settings.data.download_chunk_days)  # fmt: skip
        self.guards = Guardrails(settings.symbols, settings.guardrails, settings.risk,
                                 settings.sessions, self.clock)  # fmt: skip
        self.sizer = PositionSizer(settings.risk)
        self.exec = ExecutionEngine(connector, self.feed, settings, self.guards, dry_run=dry_run)
        self.market: dict[str, MarketContext] = {}
        self.levels: dict[str, KeyLevelMap] = {}
        self._last_tick_msc: dict[str, int] = {}
        self._last_tick_change: dict[str, float] = {}
        self._server_offset = pd.Timedelta(0)
        self._autotrading_ok = False
        self._offset_warned = False
        self._running = False

    # ------------------------------------------------------------- setup
    def setup(self) -> None:
        if not self.conn.connect():
            raise RuntimeError("Cannot connect to MetaTrader 5 (see logs)")
        for sym in self.symbols:
            if not self.conn.select_symbol(sym):
                raise RuntimeError(f"Symbol {sym} unavailable on this broker")
            spec = self.feed.symbol_spec(sym)
            self.state.symbols[sym] = SymbolState(
                sym, digits=spec.digits, max_spread=self.s.symbols[sym].max_spread_points
            )
            log.info("%s spec: %s", sym, spec)
        if self.clock.is_auto:
            tick = self.feed.last_tick(self.symbols[0])
            if tick:
                self._detect_offset(pd.Timestamp(tick.time_msc, unit="ms").to_pydatetime())
        self.state.strategy = self.strategy.name
        self.state.symbol = ", ".join(self.symbols)
        self.state.mode = "dryrun" if self.dry_run else "live"
        self.warmup()
        self.state.event(f"Started {self.state.mode.upper()} on {self.state.symbol}")

    def warmup(self) -> None:
        """Cold start: download recent history and build the key-level map before trading."""
        d = self.s.data
        for sym in self.symbols:
            t0 = time.perf_counter()
            frames = self.store.warmup(sym, d.warmup_timeframes, d.warmup_bars)
            lv_tf = self.s.levels.timeframe
            if self.s.levels.enabled and lv_tf in frames and not frames[lv_tf].empty:
                self.levels[sym] = KeyLevelMap.from_frame(
                    compute_levels(frames[lv_tf], self.s.levels, self.clock)
                )
            if "M1" in frames and not frames["M1"].empty:
                self._update_market(sym, frames)
            sizes = ", ".join(f"{tf}:{len(df)}" for tf, df in frames.items())
            msg = f"Warm-up {sym} [{sizes}] in {time.perf_counter() - t0:.1f}s"
            self.state.warmup = sizes
            self.state.event(msg)
            lm = self.levels.get(sym)
            if lm and lm.levels:
                txt = " | ".join(
                    f"{k}={v:.2f}" for k, v in lm.levels.items() if not k.startswith("vwap_")
                )
                self.state.event(f"{sym} key levels: {txt}")

    def _detect_offset(self, server_now: datetime) -> None:
        if self.clock.set_offset_from_tick(server_now, _utcnow()):
            log.info(
                "Server timezone auto-detected: UTC%+.1fh", self.clock.offset.total_seconds() / 3600
            )
        elif not self._offset_warned:
            self._offset_warned = True
            self.state.event("Stale tick (market closed?): server timezone will be detected on the "
                             "next live tick", "warn")  # fmt: skip

    def now(self) -> datetime:
        """Broker server time, extrapolated with the local clock between ticks."""
        return (pd.Timestamp(_utcnow()) + self._server_offset).to_pydatetime()

    # -------------------------------------------------------------- loop
    def run(self) -> None:
        self._running = True
        period = self.s.general.poll_interval_ms / 1000
        while self._running:
            t0 = time.perf_counter()
            try:
                self.step()
            except Exception as exc:  # keep the loop alive; never die with open positions
                log.exception("step failed")
                self.state.event(f"loop error: {exc}", "error")
            elapsed = time.perf_counter() - t0
            self.state.loop_ms = elapsed * 1000
            time.sleep(max(0.0, period - elapsed))

    def stop(self) -> None:
        self._running = False

    def step(self) -> None:
        st = self.state
        if not self.conn.ensure_connected():
            st.connected = False
            return
        st.connected = True
        st.reconnects = self.conn.reconnects

        ok, why = self.conn.autotrading_status()
        self._autotrading_ok = ok
        st.autotrading = "[green]ON[/]" if ok else f"[red]OFF ({why})[/]"

        ticks = self._poll_ticks()
        now = self.now()
        st.server_time = now
        st.local_time = self.guards.session.local(now)
        st.session = self.guards.session.active_session(now)

        ks = self.guards.kill_switch
        acc = self.conn.account_info()
        if acc is not None:
            equity = acc.equity
            if self.dry_run:  # paper P/L on top of the real account equity
                equity += sum(c.pnl for c in self.exec.paper.closed)
                equity += sum(p.profit for p in self.exec.paper.positions.values())
            ks.update_equity(equity, now)
            st.account = f"{acc.login} @ {acc.server}"
            st.currency, st.balance, st.equity = acc.currency, acc.balance, equity
            st.free_margin, st.margin_level = acc.margin_free, acc.margin_level or 0.0
            st.leverage = acc.leverage

        if ks.needs_flatten:
            st.event(
                f"KILL-SWITCH LOCKED: {ks.locked_reason} -> closing all & cancelling pendings",
                "error",
            )
            for trade in self.exec.flatten_all(now, ticks, "kill-switch"):
                ks.record_close(trade.pnl, now, final=trade.final)
            ks.flatten_done()

        for trade in self.exec.manage(now, ticks, self.market):
            ks.record_close(trade.pnl, now, final=trade.final)
            kind = "CLOSED" if trade.final else "PARTIAL"
            st.event(f"{kind} #{trade.ticket} {trade.symbol} {trade.reason} P/L {trade.pnl:+.2f}",
                     "trade" if trade.pnl >= 0 else "warn")  # fmt: skip

        for sym in self.symbols:
            if sym in ticks and self.feed.has_new_bar(sym, self.strategy.entry_tf):
                self._on_new_bar(sym, ticks[sym], now)

        self._refresh_state(ticks, now)

    # ------------------------------------------------------------- ticks
    def _poll_ticks(self) -> dict[str, TickSnapshot]:
        ticks: dict[str, TickSnapshot] = {}
        mono = time.monotonic()
        for sym in self.symbols:
            tick = self.feed.last_tick(sym)
            if tick is None:
                continue
            ticks[sym] = tick
            if self._last_tick_msc.get(sym) != tick.time_msc:
                self._last_tick_msc[sym] = tick.time_msc
                self._last_tick_change[sym] = mono
                server = pd.Timestamp(tick.time_msc, unit="ms")
                self._server_offset = server - pd.Timestamp(_utcnow())
                self.guards.spread[sym].update(tick.spread_points)
                if self.clock.is_auto and not self.clock.detected:
                    self._detect_offset(server.to_pydatetime())
                atr = self.market.get(sym, MarketContext()).atr
                if self.guards.gaps[sym].on_tick(tick.mid, atr, server.to_pydatetime()):
                    self.state.event(
                        f"{sym}: {self.guards.gaps[sym].last_event} -> entries blocked", "warn"
                    )
        return ticks

    def _tick_age(self, sym: str) -> float:
        return time.monotonic() - self._last_tick_change.get(sym, time.monotonic())

    # -------------------------------------------------------------- bars
    def _update_market(self, sym: str, frames: dict[str, pd.DataFrame]) -> None:
        m1 = frames["M1"]
        c = self.s.strategy
        atr1 = ind.atr(m1, c.atr_period)
        k = max(1, self.s.execution.trailing_candles)
        a = float(atr1.iloc[-1]) if np.isfinite(atr1.iloc[-1]) else 0.0
        mom = float((m1["close"].iloc[-1] - m1["close"].iloc[-4]) / a) if a and len(m1) > 4 else 0.0
        self.market[sym] = MarketContext(
            a, float(m1["high"].tail(k).max()), float(m1["low"].tail(k).min()), mom
        )
        ss = self.state.symbols[sym]
        ss.atr_m1, ss.momentum = a, mom
        if "M5" in frames and len(frames["M5"]) > c.atr_period:
            ss.atr_m5 = float(ind.atr(frames["M5"], c.atr_period).iloc[-1])
        med = float(atr1.tail(120).median()) if len(atr1) > 20 else 0.0
        ratio = a / med if med else 1.0
        ss.vol_state = (
            "[red]HIGH VOL[/]"
            if ratio > 1.6
            else "[cyan]LOW VOL[/]"
            if ratio < 0.6
            else "[green]NORMAL[/]"
        )

    def _on_new_bar(self, sym: str, tick: TickSnapshot, now: datetime) -> None:
        st = self.state
        sym_cfg = self.s.symbols[sym]
        spec = self.feed.symbol_spec(sym)
        frames = self.feed.get_frames(
            sym, self.strategy.required_timeframes, self.s.strategy.history_bars
        )
        if any(df.empty for df in frames.values()):
            return
        self._update_market(sym, frames)
        spread_price = tick.ask - tick.bid
        signal, row = self.strategy.latest_signal(frames, sym_cfg, spec, spread_price)
        if row is None:
            return

        atr = self.market[sym].atr
        m1 = frames[self.strategy.entry_tf]
        if len(m1) >= 2 and self.guards.gaps[sym].on_bar(float(m1["open"].iloc[-1]),
                                                         float(m1["close"].iloc[-2]), atr, now):  # fmt: skip
            st.event(f"{sym}: {self.guards.gaps[sym].last_event} -> entries blocked", "warn")
        lv = {k[3:]: float(row[k]) for k in row.index if k.startswith("lv_") and pd.notna(row[k])}
        if lv:
            self.levels[sym] = KeyLevelMap(pd.Timestamp(row["time"]).to_pydatetime(), lv)

        ss = st.symbols[sym]
        ss.bias = int(row.get("m15_bias", 0) or 0)
        ss.confirm = int(row.get("m5_confirm", 0) or 0)
        if signal is None:
            return

        open_all = self.exec.open_positions()
        pend_all = self.exec.pending_orders()
        ctx = EntryContext(
            symbol=sym, now=now, spread_points=tick.spread_points,
            atr_points=atr / spec.point if atr else None,
            last_tick_age_s=self._tick_age(sym),
            open_positions_total=len(open_all) + len(pend_all),
            open_positions_symbol=sum(p.symbol == sym for p in open_all) + sum(o.symbol == sym for o in pend_all),
        )  # fmt: skip
        decision = self.guards.check_entry(ctx)
        if not self.dry_run and not self._autotrading_ok:
            decision.block("AutoTrading disabled")
        tag = f"{'BUY' if signal.is_buy else 'SELL'} {signal.order_type.upper()} {signal.entry}"
        if signal.key_level:
            tag += f" [{signal.key_level}]"
        if not decision.allowed:
            ss.last_signal = f"[dim]{tag} blocked[/]"
            st.event(f"{sym} signal {tag} BLOCKED: {decision}", "warn")
            return

        acc = self.conn.account_info()
        mt5 = self.conn.mt5
        margin_1lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY if signal.is_buy else mt5.ORDER_TYPE_SELL,
                                            sym, 1.0, signal.entry)  # fmt: skip
        risk_pct = self.s.risk.risk_per_trade_pct * signal.risk_mult
        size = self.sizer.size(st.equity, signal.entry, signal.stop_loss, spec, risk_pct=risk_pct,
                               spread_price=spread_price, free_margin=getattr(acc, "margin_free", None),
                               margin_per_lot=margin_1lot)  # fmt: skip
        if not size.ok:
            st.event(f"{sym} signal skipped: {size.reason}", "warn")
            return

        res = self.exec.execute_signal(signal, size.lots, tick, now)
        if res.ok:
            self.guards.kill_switch.record_entry()
            ss.last_signal = f"{tag} @ {now:%H:%M}"
            st.event(f"{sym} {tag} {size.lots:.2f} lots SL {signal.stop_loss} TP {signal.take_profit or '-'} "
                     f"risk {size.risk_money:.2f} ({size.risk_pct_effective:.2f}%) | {signal.reason}",
                     "trade")  # fmt: skip
        else:
            st.event(f"{sym} order not sent: {res.comment or res.retcode}", "error")

    # ------------------------------------------------------------- state
    def _refresh_state(self, ticks: dict[str, TickSnapshot], now: datetime) -> None:
        st = self.state
        ks = self.guards.kill_switch
        st.kill_status, st.kill_reason = ks.status, ks.locked_reason
        st.day_pnl, st.day_pnl_pct = ks.day_pnl, ks.day_pnl_pct
        st.trades_today = ks.trades
        st.consecutive_losses = ks.consecutive_losses
        st.avg_slippage = self.guards.slippage.average
        for sym, tick in ticks.items():
            ss = st.symbols[sym]
            ss.bid, ss.ask, ss.spread = tick.bid, tick.ask, tick.spread_points
            ss.spread_median = self.guards.spread[sym].median
            gap_ok, gap_msg = self.guards.gaps[sym].check(now)
            ss.gap_state = "OK" if gap_ok else gap_msg
            atr = self.market.get(sym, MarketContext()).atr
            spec = self.feed.symbol_spec(sym)
            ss.guard = str(self.guards.check_entry(EntryContext(
                sym, now, tick.spread_points, atr / spec.point if atr else None, self._tick_age(sym))))  # fmt: skip
            if sym in self.levels:
                ss.key_levels = self.levels[sym].nearest(tick.bid, 4)
        rows = []
        for p in self.exec.open_positions():
            tick = ticks.get(p.symbol)
            px = (tick.bid if p.side > 0 else tick.ask) if tick else p.entry
            rows.append(PositionRow(p.ticket, p.symbol, p.side, p.volume, p.entry, p.sl, p.tp,
                                    p.r_multiple(px), p.profit, p.be_done, p.tp1_done))  # fmt: skip
        st.positions = rows
        st.pending = [PendingRow(o.ticket, o.symbol, o.side, o.volume, o.price, o.sl, o.tp, o.expires)
                      for o in self.exec.pending_orders()]  # fmt: skip

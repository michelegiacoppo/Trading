"""Live / dry-run orchestration loop.

One single-threaded loop (the dashboard renders from its own thread):

    every poll_interval_ms
      ├─ ensure MT5 connection (auto-reconnect) + AutoTrading status
      ├─ account equity -> daily kill-switch
      ├─ ticks -> spread filter, tick-jump detector, stale-feed watchdog
      ├─ manage open trades (partial / BE / trailing / time-stop / limit expiry)
      └─ on each new CLOSED entry-TF bar per symbol
           ├─ pull M1/M5/M15, compute features, gap detector
           ├─ strategy signal -> guardrails -> position sizer
           └─ execution (market or pending limit)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from src.config import Settings
from src.core.data_feed import DataFeed, TickSnapshot
from src.core.execution import ExecutionEngine
from src.core.mt5_connector import MT5Connector
from src.core.state import BotState, PositionRow, SymbolState
from src.risk.guardrails import EntryContext, Guardrails
from src.risk.position_sizer import PositionSizer
from src.strategies.base_strategy import BaseStrategy

log = logging.getLogger(__name__)


class LiveTrader:
    def __init__(
        self,
        settings: Settings,
        connector: MT5Connector,
        strategy: BaseStrategy,
        state: BotState,
        dry_run: bool = True,
        symbols: list[str] | None = None,
    ) -> None:
        self.s = settings
        self.conn = connector
        self.strategy = strategy
        self.state = state
        self.dry_run = dry_run
        self.symbols = symbols or settings.enabled_symbols
        self.feed = DataFeed(connector)
        self.guards = Guardrails(
            settings.symbols, settings.guardrails, settings.risk, settings.sessions
        )
        self.sizer = PositionSizer(settings.risk)
        self.exec = ExecutionEngine(connector, self.feed, settings, self.guards, dry_run=dry_run)
        self.atrs: dict[str, float] = {}
        self._last_tick_msc: dict[str, int] = {}
        self._last_tick_change: dict[str, float] = {}
        self._server_offset = timedelta(0)
        self._autotrading_ok = False
        self._running = False

    # ------------------------------------------------------------- setup
    def setup(self) -> None:
        if not self.conn.connect():
            raise RuntimeError("Cannot connect to MetaTrader 5 (see logs)")
        for sym in self.symbols:
            if not self.conn.select_symbol(sym):
                raise RuntimeError(f"Symbol {sym} unavailable on this broker")
            spec = self.feed.symbol_spec(sym)
            self.state.symbols[sym] = SymbolState(sym)
            log.info("%s spec: %s", sym, spec)
        self.state.strategy = self.strategy.name
        self.state.mode = "dryrun" if self.dry_run else "live"
        self.state.event(f"Started {self.state.mode.upper()} on {', '.join(self.symbols)}")

    def now(self) -> datetime:
        """Broker server time, extrapolated with the local clock between ticks."""
        return datetime.now(timezone.utc).replace(tzinfo=None) + self._server_offset

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

        acc = self.conn.account_info()
        if acc is not None:
            equity = acc.equity
            if self.dry_run:  # paper P/L on top of real account equity
                equity += sum(c.pnl for c in self.exec.paper.closed)
                equity += sum(p.profit for p in self.exec.paper.positions.values())
            self.guards.kill_switch.update_equity(equity, now)
            st.account = f"{acc.login} @ {acc.server}"
            st.currency, st.balance, st.equity = acc.currency, acc.balance, equity
            st.margin_level = acc.margin_level or 0.0

        for trade in self.exec.manage(now, ticks, self.atrs):
            self.guards.kill_switch.record_close(trade.pnl, now)
            st.event(
                f"CLOSED #{trade.ticket} {trade.symbol} {trade.reason} P/L {trade.pnl:+.2f}",
                "trade" if trade.pnl >= 0 else "warn",
            )

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
                server = datetime.fromtimestamp(tick.time_msc / 1000, tz=timezone.utc).replace(
                    tzinfo=None
                )
                self._server_offset = server - datetime.now(timezone.utc).replace(tzinfo=None)
                self.guards.spread[sym].update(tick.spread_points)
                if self.guards.gaps[sym].on_tick(tick.mid, self.atrs.get(sym, 0.0), server):
                    self.state.event(f"{sym}: {self.guards.gaps[sym].last_event}", "warn")
        return ticks

    def _tick_age(self, sym: str) -> float:
        return time.monotonic() - self._last_tick_change.get(sym, time.monotonic())

    # -------------------------------------------------------------- bars
    def _on_new_bar(self, sym: str, tick: TickSnapshot, now: datetime) -> None:
        st = self.state
        sym_cfg = self.s.symbols[sym]
        spec = self.feed.symbol_spec(sym)
        frames = self.feed.get_frames(sym, self.strategy.timeframes, self.s.strategy.history_bars)
        spread_price = tick.ask - tick.bid
        signal, row = self.strategy.latest_signal(frames, sym_cfg, spec, spread_price)
        if row is None:
            return

        atr = float(row["atr"]) if np.isfinite(row["atr"]) else 0.0
        self.atrs[sym] = atr
        m1 = frames[self.strategy.entry_tf]
        if len(m1) >= 2 and self.guards.gaps[sym].on_bar(
            float(m1["open"].iloc[-1]), float(m1["close"].iloc[-2]), atr
        ):
            st.event(f"{sym}: {self.guards.gaps[sym].last_event}", "warn")

        ss = st.symbols[sym]
        ss.atr = atr
        ss.bias = int(row.get("m15_bias", 0) or 0)
        ss.confirm = int(row.get("m5_confirm", 0) or 0)
        ss.velocity_x_atr = float(row["velocity"]) / atr if atr else 0.0
        if signal is None:
            return

        open_all = self.exec.open_positions()
        ctx = EntryContext(
            symbol=sym,
            now=now,
            spread_points=tick.spread_points,
            atr_points=atr / spec.point if atr else None,
            last_tick_age_s=self._tick_age(sym),
            open_positions_total=len(open_all) + self.exec.pending_count(),
            open_positions_symbol=sum(p.symbol == sym for p in open_all)
            + self.exec.pending_count(sym),
        )
        decision = self.guards.check_entry(ctx)
        if not self.dry_run and not self._autotrading_ok:
            decision.block("AutoTrading disabled")
        tag = f"{'BUY' if signal.is_buy else 'SELL'} {signal.order_type.upper()} {signal.entry}"
        if not decision.allowed:
            ss.last_signal = f"[dim]{tag} blocked[/]"
            st.event(f"{sym} signal {tag} BLOCKED: {decision}", "warn")
            return

        acc = self.conn.account_info()
        margin_1lot = self.conn.mt5.order_calc_margin(
            self.conn.mt5.ORDER_TYPE_BUY if signal.is_buy else self.conn.mt5.ORDER_TYPE_SELL,
            sym,
            1.0,
            signal.entry,
        )
        size = self.sizer.size(
            st.equity,
            signal.entry,
            signal.stop_loss,
            spec,
            spread_price=spread_price,
            free_margin=getattr(acc, "margin_free", None),
            margin_per_lot=margin_1lot,
        )
        if not size.ok:
            st.event(f"{sym} signal skipped: {size.reason}", "warn")
            return

        res = self.exec.execute_signal(signal, size.lots, tick, now)
        if res.ok:
            self.guards.kill_switch.record_entry()
            ss.last_signal = f"{tag} @ {now:%H:%M}"
            st.event(
                f"{sym} {tag} {size.lots:.2f} lots SL {signal.stop_loss} TP {signal.take_profit} "
                f"risk {size.risk_money:.2f} ({size.risk_pct_effective:.2f}%) | {signal.reason}",
                "trade",
            )
        else:
            st.event(f"{sym} order rejected retcode={res.retcode} {res.comment}", "error")

    # ------------------------------------------------------------- state
    def _refresh_state(self, ticks: dict[str, TickSnapshot], now: datetime) -> None:
        st = self.state
        ks = self.guards.kill_switch
        st.kill_switch = ks.tripped_reason
        st.day_start_equity = ks.start_equity
        st.day_realized = ks.realized_pnl
        st.trades_today = ks.trades
        st.consecutive_losses = ks.consecutive_losses
        st.avg_slippage = self.guards.slippage.average
        for sym, tick in ticks.items():
            ss = st.symbols[sym]
            ss.bid, ss.ask, ss.spread = tick.bid, tick.ask, tick.spread_points
            ss.spread_median = self.guards.spread[sym].median
            spec = self.feed.symbol_spec(sym)
            ss.guard = str(
                self.guards.check_entry(
                    EntryContext(
                        sym,
                        now,
                        tick.spread_points,
                        self.atrs.get(sym, 0) / spec.point or None,
                        self._tick_age(sym),
                    )
                )
            )
        rows = []
        for p in self.exec.open_positions():
            tick = ticks.get(p.symbol)
            px = (tick.bid if p.side > 0 else tick.ask) if tick else p.entry
            flags = ("P" if p.partial_done else "") + ("BE" if p.be_done else "")
            rows.append(
                PositionRow(
                    p.ticket,
                    p.symbol,
                    p.side,
                    p.volume,
                    p.entry,
                    p.sl,
                    p.tp,
                    p.r_multiple(px),
                    p.profit,
                    flags,
                )
            )
        st.positions = rows
        st.pending_orders = self.exec.pending_count()

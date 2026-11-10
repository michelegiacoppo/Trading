"""Realistic "anti-illusion" backtester on M1 bars.

Classic M1 backtests lie on XAUUSD because they ignore spread, slippage and the
order in which High and Low are touched inside a candle. This engine:

* **Real spread** - uses the broker's historical ``spread`` column of MT5 bars
  (points, per bar) or, when real ticks are supplied, the mean Bid/Ask spread of
  each minute. A synthetic spread (default + jitter) is used *only* when neither
  exists, and the report says so.
* **Slippage & commissions** - ``simulated_slippage_points`` is charged on every
  Market entry, Stop-Loss fill and market exit; round-turn ``commission_per_lot``
  on every closed lot. Stops that gap through are filled at the worse of SL/open.
* **Conservative intra-bar execution** - if SL and TP (or SL and TP1) are both inside
  the same candle, the **SL is assumed first** - unless tick data is available for
  that minute, in which case the real price path decides.
* **Causality** - signals are computed (vectorised) on the close of bar *i* and
  executed from bar *i+1*. Limit orders fill only if price trades *through* the
  level within ``order_expiration_candles``.
* **Same rules as live** - strategy, :class:`TradeManager`, position sizer, sessions
  (CET), spread filter, gap detector and the daily kill-switch (with flatten) are
  the production classes.

Simplification: one position and one pending order per backtest at a time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, SymbolSpec, resolve_path
from src.core.clock import ServerClock
from src.core.data_feed import TF_MINUTES
from src.core.execution import MarketContext, PositionView, TradeManager
from src.risk.guardrails import DailyKillSwitch, GapDetector, SessionFilter, SpreadFilter
from src.risk.position_sizer import PositionSizer, money_per_lot
from src.strategies import indicators as ind
from src.strategies.base_strategy import BaseStrategy

log = logging.getLogger(__name__)


@dataclass
class BTTrade:
    symbol: str
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    volume: float
    pnl: float
    r_multiple: float
    reason: str
    key_level: str = ""
    risk_money: float = 0.0


@dataclass
class BacktestResult:
    trades: list[BTTrade]
    equity_curve: pd.Series
    initial_equity: float
    stats: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        return self.stats

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


# ------------------------------------------------------------------ data io
def load_m1_csv(path: str | Path) -> pd.DataFrame:
    """Load M1 bars: MT5 export (``<DATE> <TIME> ... <SPREAD>``), plain CSV or Parquet."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
    else:
        df["time"] = pd.to_datetime(df["time"])
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df["tickvol"] if "tickvol" in df.columns else df.get("volume", 1)
    cols = ["time", "open", "high", "low", "close", "tick_volume"]
    if "spread" in df.columns:
        cols.append("spread")
    return df[cols].sort_values("time").reset_index(drop=True)


def spread_from_ticks(m1: pd.DataFrame, ticks: pd.DataFrame) -> pd.Series:
    """Mean real Bid/Ask spread (points) per M1 bar from tick data (NaN where no ticks)."""
    if ticks is None or ticks.empty:
        return pd.Series(np.nan, index=m1.index)
    per_min = ticks.set_index("time")["spread_points"].resample("1min").mean()
    return m1["time"].map(per_min).astype(float)


def synthetic_m1(n_bars: int = 20_000, start_price: float = 2650.0, seed: int = 7,
                 start: str = "2026-01-05 00:00") -> pd.DataFrame:  # fmt: skip
    """Regime-switching random walk with momentum bursts and occasional gaps (smoke tests)."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n_bars // 240 + 1, p=[0.3, 0.4, 0.3]), 240)
    drift = regime[:n_bars] * 0.12
    vol = 1.0 * (1 + 0.5 * np.abs(np.sin(np.arange(n_bars) / 300)))
    shocks = rng.standard_t(df=4, size=n_bars) * vol
    burst = (rng.random(n_bars) < 0.01) * rng.normal(0, 5.0, n_bars)
    rets = drift + shocks + burst
    close = start_price + np.cumsum(rets)
    open_ = np.r_[start_price, close[:-1]] + (rng.random(n_bars) < 0.002) * rng.normal(0, 8, n_bars)
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n_bars))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n_bars))
    volume = (100 + 400 * np.abs(rets) + rng.integers(0, 50, n_bars)).astype(int)
    times = pd.date_range(start, periods=n_bars, freq="1min")
    df = pd.DataFrame({"time": times, "open": open_, "high": hi, "low": lo, "close": close,
                       "tick_volume": volume, "spread": rng.integers(10, 26, n_bars)})  # fmt: skip
    return df[df["time"].dt.weekday < 5].reset_index(drop=True)


# ------------------------------------------------------------- tick path
class TickPath:
    """Resolves intra-bar ordering of two price levels from real ticks."""

    def __init__(self, ticks: pd.DataFrame | None) -> None:
        self.available = ticks is not None and not ticks.empty
        if self.available:
            assert ticks is not None
            t = ticks.sort_values("time")
            self.t = t["time"].to_numpy(dtype="datetime64[ns]")
            self.bid = t["bid"].to_numpy(dtype=float)
            self.ask = t["ask"].to_numpy(dtype=float)
        self.resolved = 0
        self.pessimistic = 0

    def first_hit(self, bar_time: pd.Timestamp, minutes: int, side: int, stop: float,
                  target: float) -> str | None:  # fmt: skip
        """'stop' | 'target' | None (no ticks for that bar)."""
        if not self.available:
            return None
        a = np.searchsorted(self.t, np.datetime64(bar_time, "ns"))
        b = np.searchsorted(self.t, np.datetime64(bar_time + pd.Timedelta(minutes=minutes), "ns"))
        if b <= a:
            return None
        px = self.bid[a:b] if side > 0 else self.ask[a:b]
        stop_hits = np.flatnonzero(px <= stop if side > 0 else px >= stop)
        tgt_hits = np.flatnonzero(px >= target if side > 0 else px <= target)
        if not len(stop_hits) and not len(tgt_hits):
            return None
        if not len(tgt_hits) or (len(stop_hits) and stop_hits[0] <= tgt_hits[0]):
            return "stop"
        return "target"


# ----------------------------------------------------------------- engine
class BacktestEngine:
    def __init__(self, settings: Settings, strategy: BaseStrategy, symbol: str,
                 spec: SymbolSpec | None = None, clock: ServerClock | None = None) -> None:  # fmt: skip
        self.s = settings
        self.strategy = strategy
        self.symbol = symbol
        self.sym_cfg = settings.symbols[symbol]
        self.spec = spec or self.sym_cfg.backtest_spec()
        self.clock = clock or ServerClock(settings.sessions.backtest_server_timezone)
        self.sizer = PositionSizer(settings.risk)
        self.manager = TradeManager(settings.execution, settings.risk.commission_per_lot)

    def build_frames(self, m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {tf: m1 if tf == "M1" else ind.resample_ohlc(m1, tf)
                for tf in self.strategy.required_timeframes}  # fmt: skip

    def run(self, m1: pd.DataFrame, ticks: pd.DataFrame | None = None) -> BacktestResult:
        bt, ex = self.s.backtest, self.s.execution
        spec, pt = self.spec, self.spec.point
        rng = np.random.default_rng(bt.seed)
        tf_min = TF_MINUTES[self.strategy.entry_tf]

        # ----- spread source: real bars column > real ticks > synthetic (flagged)
        m1 = m1.copy()
        tick_spread = spread_from_ticks(m1, ticks) if ticks is not None else None
        if "spread" in m1.columns and m1["spread"].notna().any() and (m1["spread"] > 0).any():
            spread_source = "MT5 bar spread (real)"
            if tick_spread is not None:
                m1["spread"] = tick_spread.fillna(m1["spread"])
                spread_source = "tick Bid/Ask (real) + bar spread"
        elif tick_spread is not None and tick_spread.notna().any():
            spread_source = "tick Bid/Ask (real)"
            m1["spread"] = tick_spread.fillna(tick_spread.median())
        else:
            spread_source = "SYNTHETIC (no real spread in data!)"
            m1["spread"] = bt.default_spread_points + rng.uniform(
                0, bt.spread_jitter_points, len(m1)
            )
        m1["spread"] = m1["spread"].ffill().fillna(bt.default_spread_points)

        feats = self.strategy.compute_features(self.build_frames(m1))
        n = len(feats)
        t = pd.to_datetime(feats["time"]).reset_index(drop=True)
        o, h, lo_, c = (feats[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr = feats["atr"].to_numpy(dtype=float)
        mom = feats["momentum"].to_numpy(dtype=float) if "momentum" in feats else np.zeros(n)
        sig = feats["signal"].to_numpy(dtype=int)
        spread_pts = feats["spread"].to_numpy(dtype=float)
        k = max(1, ex.trailing_candles)
        rec_hi = feats["high"].rolling(k).max().to_numpy(dtype=float)
        rec_lo = feats["low"].rolling(k).min().to_numpy(dtype=float)
        slip = bt.simulated_slippage_points * pt
        path = TickPath(ticks)

        session = SessionFilter(self.s.sessions, self.clock)
        spread_f = SpreadFilter(self.sym_cfg, self.s.guardrails)
        gaps = GapDetector(self.s.guardrails)
        ks = DailyKillSwitch(self.s.risk)
        equity = bt.initial_equity
        eq_vals = np.empty(n)
        trades: list[BTTrade] = []
        pos: PositionView | None = None
        pos_meta: dict[str, object] = {}
        pending: dict | None = None
        realized_in_pos = 0.0
        counters = {"limits_placed": 0, "limits_filled": 0, "limits_expired": 0, "signals": 0,
                    "blocked": 0, "kill_switch_locks": 0}  # fmt: skip
        block_reasons: dict[str, int] = {}

        def close(px: float, vol: float, reason: str, i: int) -> None:
            nonlocal equity, pos, realized_in_pos
            assert pos is not None
            pnl = pos.side * (px - pos.entry) / spec.tick_size * spec.tick_value * vol
            pnl -= self.s.risk.commission_per_lot * vol
            equity += pnl
            realized_in_pos += pnl
            pos.volume = round(pos.volume - vol, 8)
            final = pos.volume <= 1e-9
            ks.record_close(pnl, t[i].to_pydatetime(), final=final)
            if final:
                risk_money = float(pos_meta.get("risk_money", 0.0)) or 1e-9
                trades.append(BTTrade(self.symbol, pos.side, pd.Timestamp(pos.open_time), t[i],
                                      pos.entry, px, pos.initial_volume, realized_in_pos,
                                      realized_in_pos / risk_money, reason,
                                      str(pos_meta.get("key_level") or ""), risk_money))  # fmt: skip
                pos = None
                realized_in_pos = 0.0

        def open_position(order: dict, fill: float, when: pd.Timestamp) -> None:
            nonlocal pos, pos_meta
            pos = PositionView(0, self.symbol, order["side"], order["volume"], fill, order["sl"],
                               order["tp"], when.to_pydatetime(), order["volume"],
                               abs(fill - order["sl"]))  # fmt: skip
            pos_meta = {"key_level": order.get("key_level"),
                        "risk_money": money_per_lot(abs(fill - order["sl"]), spec) * order["volume"]
                        + self.s.risk.commission_per_lot * order["volume"]}  # fmt: skip

        for i in range(1, n):
            bar_t = t[i]
            now = (bar_t + timedelta(minutes=tf_min)).to_pydatetime()  # decisions at bar close
            sp = spread_pts[i] * pt
            spread_f.update(spread_pts[i])
            gaps.on_bar(o[i], c[i - 1], atr[i - 1] if np.isfinite(atr[i - 1]) else 0.0, now)

            # ---- kill switch on equity incl. floating P/L
            floating = 0.0
            if pos is not None:
                px = c[i] if pos.side > 0 else c[i] + sp
                floating = (
                    pos.side * (px - pos.entry) / spec.tick_size * spec.tick_value * pos.volume
                )
            was_locked = ks.tripped
            ks.update_equity(equity + floating, now)
            if ks.tripped and not was_locked:
                counters["kill_switch_locks"] += 1
            if ks.needs_flatten:
                pending = None
                if pos is not None:
                    exit_px = (c[i] - slip) if pos.side > 0 else (c[i] + sp + slip)
                    close(exit_px, pos.volume, f"kill-switch: {ks.locked_reason}", i)
                ks.flatten_done()

            # ---- pending limit: expiry / fill (price must trade THROUGH the level)
            if pending is not None and pos is None:
                lim, side = pending["price"], pending["side"]
                if i > pending["expires_idx"]:
                    pending = None
                    counters["limits_expired"] += 1
                elif (side > 0 and lo_[i] + sp < lim) or (side < 0 and h[i] > lim):
                    fill = min(lim, o[i] + sp) if side > 0 else max(lim, o[i])
                    open_position(pending, fill, bar_t)
                    counters["limits_filled"] += 1
                    pending = None

            # ---- open position: intrabar SL / TP / TP1 (conservative ordering)
            if pos is not None:
                side = pos.side
                tp1_px = pos.entry + side * ex.tp1_r * pos.initial_risk
                if side > 0:
                    sl_hit = pos.sl > 0 and lo_[i] <= pos.sl
                    tp_hit = pos.tp > 0 and h[i] >= pos.tp
                    tp1_hit = not pos.tp1_done and h[i] >= tp1_px
                    sl_fill = min(pos.sl, o[i]) - slip
                    tp_fill = max(pos.tp, o[i]) if pos.tp else 0.0
                else:
                    sl_hit = pos.sl > 0 and h[i] + sp >= pos.sl
                    tp_hit = pos.tp > 0 and lo_[i] + sp <= pos.tp
                    tp1_hit = not pos.tp1_done and lo_[i] + sp <= tp1_px
                    sl_fill = max(pos.sl, o[i] + sp) + slip
                    tp_fill = min(pos.tp, o[i] + sp) if pos.tp else 0.0

                target_hit = tp_hit or tp1_hit
                target_px = pos.tp if tp_hit else tp1_px
                stop_first = sl_hit
                if sl_hit and target_hit:
                    who = path.first_hit(bar_t, tf_min, side, pos.sl, target_px)
                    if who is None:
                        path.pessimistic += 1  # worst case: stop first
                    else:
                        path.resolved += 1
                        stop_first = who == "stop"
                if sl_hit and stop_first:
                    close(sl_fill, pos.volume, "SL" if not pos.be_done else "SL (BE/trail)", i)
                elif target_hit:
                    if tp1_hit:  # partial at TP1 always happens before the runner TP
                        self._apply_tp1(pos, tp1_px, i, close, sp)
                    if pos is not None and tp_hit:
                        close(tp_fill, pos.volume, "TP runner", i)

            # ---- management on bar close (trailing / stale killer / late TP1)
            if pos is not None:
                mkt = MarketContext(atr[i], rec_hi[i], rec_lo[i], mom[i])
                bid, ask = c[i], c[i] + sp
                for act in self.manager.evaluate(pos, bid, ask, mkt, now, spec):
                    exit_px = bid if pos.side > 0 else ask
                    if act.kind == "close":
                        close(exit_px - pos.side * slip, pos.volume, act.reason, i)
                        break
                    if act.kind == "tp1":
                        if act.volume > 0:
                            close(exit_px, act.volume, act.reason, i)
                        if pos is None:
                            break
                        pos.tp1_done = True
                    elif act.kind == "modify_sl":
                        pos.sl = act.new_sl
                        pos.be_done = pos.be_done or "break-even" in act.reason

            # ---- new signal on bar close -> order for the next bar
            if sig[i] != 0 and i + 1 < n:
                counters["signals"] += 1
                atr_pts = atr[i] / pt if np.isfinite(atr[i]) else None
                reason = None
                if pos is not None or pending is not None:
                    reason = "trade/order already open"
                else:
                    for label, check in (
                        ("session", session.check(now)),
                        ("kill-switch/cooldown", ks.check(now)),
                        ("spread", spread_f.check(spread_pts[i], atr_pts)),
                        ("gap block", gaps.check(now)),
                    ):
                        if not check[0]:
                            reason = (
                                f"{label}: {check[1].split(' ')[0]}"
                                if label == "session"
                                else label
                            )
                            break
                ok = reason is None
                s_obj = (
                    self.strategy.build_signal(feats.iloc[i], self.sym_cfg, spec, sp)
                    if ok
                    else None
                )
                size = None
                if s_obj is not None:
                    risk_pct = self.s.risk.risk_per_trade_pct * s_obj.risk_mult
                    size = self.sizer.size(equity, s_obj.entry, s_obj.stop_loss, spec,
                                           risk_pct=risk_pct, spread_price=sp)  # fmt: skip
                if s_obj is None or size is None or not size.ok:
                    counters["blocked"] += 1
                    if reason is None:
                        reason = "SL out of range" if s_obj is None else "size < min lot"
                    block_reasons[reason] = block_reasons.get(reason, 0) + 1
                else:
                    ks.record_entry()
                    order = {"side": s_obj.side, "price": s_obj.entry, "sl": s_obj.stop_loss,
                             "tp": s_obj.take_profit, "volume": size.lots, "key_level": s_obj.key_level,
                             "expires_idx": i + max(1, self.s.strategy.order_expiration_candles)}  # fmt: skip
                    if s_obj.order_type == "market":
                        j = i + 1
                        fill = (o[j] + spread_pts[j] * pt + slip) if s_obj.is_buy else (o[j] - slip)
                        shift = fill - s_obj.entry  # keep SL/TP distances anchored to the fill
                        order.update(sl=s_obj.stop_loss + shift,
                                     tp=s_obj.take_profit + shift if s_obj.take_profit else 0.0)  # fmt: skip
                        open_position(order, fill, t[j])
                    else:
                        pending = order
                        counters["limits_placed"] += 1

            eq_vals[i] = equity
        eq_vals[0] = bt.initial_equity

        if pos is not None:
            last_sp = spread_pts[-1] * pt
            close(c[-1] if pos.side > 0 else c[-1] + last_sp, pos.volume, "end of data", n - 1)
            eq_vals[-1] = equity

        curve = pd.Series(eq_vals, index=pd.DatetimeIndex(t), name="equity")
        res = BacktestResult(trades, curve, bt.initial_equity)
        funnel = getattr(self.strategy, "signal_funnel", None)
        res.meta = {
            **counters,
            "block_reasons": dict(sorted(block_reasons.items(), key=lambda kv: -kv[1])),
            "funnel": funnel(feats) if callable(funnel) else [],
            "spread_source": spread_source,
            "avg_spread_points": float(np.nanmean(spread_pts)),
            "intrabar_resolved_by_ticks": path.resolved,
            "intrabar_pessimistic": path.pessimistic,
            "ticks_available": path.available,
            "bars": n,
            "from": str(t.iloc[0]),
            "to": str(t.iloc[-1]),
        }
        res.stats = compute_stats(res)
        return res

    def _apply_tp1(self, pos: PositionView, tp1_px: float, i: int, close, sp: float) -> None:  # noqa: ANN001
        """TP1 touched intrabar (resting take-profit): partial close at TP1 + BE stop."""
        ex = self.s.execution
        spec = self.spec
        vol = round(np.floor(pos.initial_volume * ex.tp1_close_fraction / spec.volume_step + 1e-9)
                    * spec.volume_step, 8)  # fmt: skip
        if vol >= spec.volume_min and pos.volume - vol < spec.volume_min - 1e-9:
            close(tp1_px, pos.volume, "TP1 (no runner)", i)
            return
        if vol >= spec.volume_min:
            close(tp1_px, vol, "TP1", i)
        pos.tp1_done = True
        be = pos.entry + pos.side * self.manager.break_even_buffer(spec, sp)
        if (pos.side > 0 and be > pos.sl) or (pos.side < 0 and be < pos.sl):
            pos.sl = round(be, spec.digits)
            pos.be_done = True


# ------------------------------------------------------------------ stats
def compute_stats(res: BacktestResult) -> dict[str, float]:
    tr, eq = res.trades, res.equity_curve
    if eq.empty or not tr:
        return {"trades": float(len(tr)), "net_profit": float(eq.iloc[-1] - res.initial_equity)
                if not eq.empty else 0.0}  # fmt: skip
    pnl = np.array([x.pnl for x in tr])
    r = np.array([x.r_multiple for x in tr])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    peak = eq.cummax()
    dd_abs = (peak - eq).max()
    dd_pct = ((eq / peak) - 1.0).min() * 100
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(252))
        if len(daily) > 2 and daily.std() > 0
        else 0.0
    )
    downside = daily[daily < 0].std()
    sortino = (
        float(daily.mean() / downside * np.sqrt(252)) if len(daily) > 2 and downside > 0 else 0.0
    )
    net = float(eq.iloc[-1] - res.initial_equity)
    streak = best = 0
    for p in pnl:
        streak = streak + 1 if p <= 0 else 0
        best = max(best, streak)
    hold = np.mean([(x.exit_time - x.entry_time).total_seconds() / 60 for x in tr])
    gross_loss = -losses.sum()
    return {
        "trades": float(len(tr)),
        "net_profit": net,
        "return_pct": net / res.initial_equity * 100,
        "win_rate_pct": len(wins) / len(tr) * 100,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_usd": float(pnl.mean()),
        "expectancy_r": float(r.mean()),
        "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown_pct": float(dd_pct),
        "max_drawdown_usd": float(dd_abs),
        "sharpe": sharpe,
        "sortino": sortino,
        "recovery_factor": float(net / dd_abs) if dd_abs > 0 else float("inf"),
        "max_consecutive_losses": float(best),
        "avg_hold_minutes": float(hold),
    }


def hourly_analysis(res: BacktestResult, clock: ServerClock, tz: str,
                    bucket_minutes: int = 30) -> pd.DataFrame:  # fmt: skip
    """Performance by time-of-day slot of the entry, in the session timezone (CET)."""
    df = res.trades_frame()
    if df.empty:
        return df
    local = clock.series_to_zone(pd.to_datetime(df["entry_time"]), tz)
    slot_min = (local.dt.hour * 60 + local.dt.minute) // bucket_minutes * bucket_minutes
    df["slot"] = [f"{m // 60:02d}:{m % 60:02d}" for m in slot_min]
    g = df.groupby("slot")
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate_pct": g["pnl"].apply(lambda s: (s > 0).mean() * 100),
        "net_usd": g["pnl"].sum(),
        "avg_r": g["r_multiple"].mean(),
        "profit_factor": g["pnl"].apply(
            lambda s: s[s > 0].sum() / -s[s <= 0].sum() if (s <= 0).any() and s[s <= 0].sum() < 0 else np.inf
        ),
    })  # fmt: skip
    return out.sort_index()


def save_results(res: BacktestResult, settings: Settings) -> tuple[Path, Path]:
    bt = settings.backtest
    curve_path, trades_path = resolve_path(bt.equity_curve_file), resolve_path(bt.trades_file)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    res.equity_curve.to_csv(curve_path, header=True)
    res.trades_frame().to_csv(trades_path, index=False)
    return curve_path, trades_path

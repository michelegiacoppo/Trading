"""Fast event-driven backtester on M1 bars with spread, commission and slippage.

Execution model (conservative by construction):

* Bars are BID OHLC. ASK = BID + spread (per-bar spread from the CSV ``spread``
  column if present, otherwise ``default_spread_points`` + random jitter).
* Signals are computed on the close of bar *i* (vectorised, no look-ahead) and
  acted upon from bar *i+1*.
* Market entries fill at the next bar open +/- spread + adverse slippage.
* Limit entries fill only if price trades *through* the limit inside the expiry.
* Intrabar ambiguity: if both SL and TP are touched in the same bar, **SL wins**.
  Stops are filled at the worse of SL and the bar open (gap-through) + slippage.
* Management (partial close, break-even, ATR trailing, time-stop) re-uses
  :class:`src.core.execution.TradeManager`, evaluated on bar close.
* Guardrails (session, spread, daily kill-switch) are the live classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Settings, SymbolSpec
from src.core.execution import PositionView, TradeManager
from src.risk.guardrails import DailyKillSwitch, SessionFilter, SpreadFilter
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


@dataclass
class BacktestResult:
    trades: list[BTTrade]
    equity_curve: pd.Series
    initial_equity: float
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        return self.stats


def load_m1_csv(path: str | Path) -> pd.DataFrame:
    """Load M1 bars. Accepts MT5 exports (``<DATE> <TIME> ...``) or ``time,open,high,low,close``."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
    else:
        df["time"] = pd.to_datetime(df["time"])
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df.get("tickvol", df.get("volume", 1))
    cols = ["time", "open", "high", "low", "close", "tick_volume"] + (
        ["spread"] if "spread" in df.columns else []
    )
    return df[cols].sort_values("time").reset_index(drop=True)


def synthetic_m1(
    n_bars: int = 20_000,
    start_price: float = 2650.0,
    seed: int = 7,
    start: str = "2026-01-05 00:00",
) -> pd.DataFrame:
    """Regime-switching random walk with momentum bursts and occasional gaps (for smoke tests)."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n_bars // 240 + 1, p=[0.3, 0.4, 0.3]), 240)[
        :n_bars
    ]
    drift = regime * 0.12
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
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": hi,
            "low": lo,
            "close": close,
            "tick_volume": volume,
            "spread": rng.integers(10, 26, n_bars),
        }
    )
    return df[df["time"].dt.weekday < 5].reset_index(drop=True)


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        strategy: BaseStrategy,
        symbol: str,
        spec: SymbolSpec | None = None,
    ) -> None:
        self.s = settings
        self.strategy = strategy
        self.symbol = symbol
        self.sym_cfg = settings.symbols[symbol]
        self.spec = spec or self.sym_cfg.backtest_spec()
        self.sizer = PositionSizer(settings.risk)
        self.manager = TradeManager(settings.execution, settings.strategy.max_hold_minutes)

    def run(self, m1: pd.DataFrame) -> BacktestResult:
        bt = self.s.backtest
        spec = self.spec
        pt = spec.point
        rng = np.random.default_rng(bt.seed)
        tf_e, tf_c, tf_t = self.strategy.timeframes
        frames = {tf_e: m1, tf_c: ind.resample_ohlc(m1, tf_c), tf_t: ind.resample_ohlc(m1, tf_t)}
        feats = self.strategy.compute_features(frames)

        n = len(feats)
        t = feats["time"].to_numpy()
        o, h, lo_, c = (feats[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr = feats["atr"].to_numpy(dtype=float)
        sig = feats["signal"].to_numpy(dtype=int)
        if "spread" in feats.columns:
            spread_pts = feats["spread"].to_numpy(dtype=float)
        else:
            spread_pts = np.full(n, bt.default_spread_points, dtype=float)
        spread_pts = spread_pts + rng.uniform(0, bt.spread_jitter_points, n)
        slip = bt.slippage_points * pt

        session = SessionFilter(self.s.sessions)
        spread_f = SpreadFilter(self.sym_cfg, self.s.guardrails)
        ks = DailyKillSwitch(self.s.risk)
        equity = bt.initial_equity
        eq_times: list[pd.Timestamp] = []
        eq_vals: list[float] = []
        trades: list[BTTrade] = []
        pos: PositionView | None = None
        pending: dict | None = None
        realized_in_pos = 0.0

        def close(px: float, vol: float, reason: str, i: int) -> None:
            nonlocal equity, pos, realized_in_pos
            assert pos is not None
            pnl = pos.side * (px - pos.entry) / spec.tick_size * spec.tick_value * vol
            pnl -= self.s.risk.commission_per_lot * vol
            equity += pnl
            realized_in_pos += pnl
            now = pd.Timestamp(t[i]).to_pydatetime()
            ks.record_close(pnl, now)
            pos.volume = round(pos.volume - vol, 8)
            if pos.volume <= 1e-9:
                risk_money = money_per_lot(pos.initial_risk, spec) * pos.initial_volume
                trades.append(
                    BTTrade(
                        self.symbol,
                        pos.side,
                        pd.Timestamp(pos.open_time),
                        pd.Timestamp(t[i]),
                        pos.entry,
                        px,
                        pos.initial_volume,
                        realized_in_pos,
                        realized_in_pos / risk_money if risk_money else 0.0,
                        reason,
                    )
                )
                pos = None
                realized_in_pos = 0.0

        for i in range(1, n):
            now = pd.Timestamp(t[i]).to_pydatetime()
            sp = spread_pts[i] * pt
            spread_f.update(spread_pts[i])
            ks.update_equity(equity + (self._floating(pos, c[i], sp) if pos else 0.0), now)

            # ---- pending limit fill / expiry
            if pending is not None and pos is None:
                lim, side = pending["price"], pending["side"]
                if i > pending["expires"]:
                    pending = None
                elif (side > 0 and lo_[i] + sp < lim) or (side < 0 and h[i] > lim):
                    fill = min(lim, o[i] + sp) if side > 0 else max(lim, o[i])
                    pos = self._open(pending, fill, now)
                    pending = None

            # ---- open position: stops/targets intrabar (SL first), then management
            if pos is not None:
                if pos.side > 0:
                    sl_hit = pos.sl and lo_[i] <= pos.sl
                    tp_hit = pos.tp and h[i] >= pos.tp
                    if sl_hit:
                        close(min(pos.sl, o[i]) - slip, pos.volume, "SL", i)
                    elif tp_hit:
                        close(max(pos.tp, o[i]), pos.volume, "TP", i)
                else:
                    sl_hit = pos.sl and h[i] + sp >= pos.sl
                    tp_hit = pos.tp and lo_[i] + sp <= pos.tp
                    if sl_hit:
                        close(max(pos.sl, o[i] + sp) + slip, pos.volume, "SL", i)
                    elif tp_hit:
                        close(min(pos.tp, o[i] + sp), pos.volume, "TP", i)
                if pos is not None:
                    bid, ask = c[i], c[i] + sp
                    for act in self.manager.evaluate(pos, bid, ask, atr[i], now, spec):
                        exit_px = bid if pos.side > 0 else ask
                        if act.kind == "close":
                            close(exit_px - pos.side * slip, pos.volume, act.reason, i)
                            break
                        if act.kind == "partial":
                            close(exit_px, act.volume, act.reason, i)
                            if pos is None:
                                break
                            pos.partial_done = True
                        elif act.kind == "modify_sl":
                            pos.sl = act.new_sl

            # ---- new signal on bar close -> order for the next bar
            if pos is None and pending is None and sig[i] != 0 and i + 1 < n:
                ok = (
                    session.check(now)[0]
                    and ks.check(now)[0]
                    and spread_f.check(spread_pts[i], atr[i] / pt)[0]
                )
                if ok:
                    s_obj = self.strategy.build_signal(feats.iloc[i], self.sym_cfg, spec, sp)
                    if s_obj is not None:
                        size = self.sizer.size(
                            equity, s_obj.entry, s_obj.stop_loss, spec, spread_price=sp
                        )
                        if size.ok:
                            ks.record_entry()
                            order = {
                                "side": s_obj.side,
                                "price": s_obj.entry,
                                "sl": s_obj.stop_loss,
                                "tp": s_obj.take_profit,
                                "volume": size.lots,
                                "expires": i + max(1, s_obj.expiry_s // 60),
                            }
                            if s_obj.order_type == "market":
                                j = i + 1
                                fill = (
                                    (o[j] + spread_pts[j] * pt + slip)
                                    if s_obj.is_buy
                                    else (o[j] - slip)
                                )
                                shift = (
                                    fill - s_obj.entry
                                )  # keep SL/TP distances anchored to the fill
                                order.update(
                                    sl=s_obj.stop_loss + shift, tp=s_obj.take_profit + shift
                                )
                                pos = self._open(order, fill, pd.Timestamp(t[j]).to_pydatetime())
                            else:
                                pending = order

            eq_times.append(pd.Timestamp(t[i]))
            eq_vals.append(equity)

        if pos is not None:
            close(c[-1] if pos.side > 0 else c[-1] + spread_pts[-1] * pt, pos.volume, "end", n - 1)
            eq_vals[-1] = equity

        curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_times), name="equity")
        result = BacktestResult(trades, curve, bt.initial_equity)
        result.stats = compute_stats(result)
        return result

    def _open(self, order: dict, fill: float, now) -> PositionView:
        return PositionView(
            0,
            self.symbol,
            order["side"],
            order["volume"],
            fill,
            order["sl"],
            order["tp"],
            now,
            order["volume"],
            abs(fill - order["sl"]),
        )

    def _floating(self, pos: PositionView, bid: float, sp: float) -> float:
        px = bid if pos.side > 0 else bid + sp
        return pos.side * (px - pos.entry) / self.spec.tick_size * self.spec.tick_value * pos.volume


def compute_stats(res: BacktestResult) -> dict[str, float]:
    tr = res.trades
    eq = res.equity_curve
    if eq.empty:
        return {"trades": 0}
    pnl = np.array([x.pnl for x in tr]) if tr else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    dd = (eq / eq.cummax() - 1.0) * 100
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(252))
        if len(daily) > 2 and daily.std() > 0
        else 0.0
    )
    return {
        "trades": float(len(tr)),
        "net_profit": float(eq.iloc[-1] - res.initial_equity),
        "return_pct": float((eq.iloc[-1] / res.initial_equity - 1) * 100),
        "win_rate_pct": float(len(wins) / len(tr) * 100) if tr else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum())
        if len(losses) and losses.sum() < 0
        else float("inf")
        if len(wins)
        else 0.0,
        "avg_r": float(np.mean([x.r_multiple for x in tr])) if tr else 0.0,
        "expectancy": float(pnl.mean()) if tr else 0.0,
        "max_drawdown_pct": float(dd.min()),
        "sharpe_daily": sharpe,
    }


def save_results(res: BacktestResult, settings: Settings) -> None:
    bt = settings.backtest
    curve_path = _resolve(bt.equity_curve_file)
    trades_path = _resolve(bt.trades_file)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    res.equity_curve.to_csv(curve_path, header=True)
    pd.DataFrame([t.__dict__ for t in res.trades]).to_csv(trades_path, index=False)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

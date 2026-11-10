"""Shared-equity portfolio simulator (one account, global risk rules).

Per-symbol / per-strategy backtests produce *candidate* trades, each with its result in
R (``r_multiple`` = P/L / money at risk, costs included). This module replays all of them
on ONE account, in time order, exactly like the live bot would:

* position size = ``risk_per_trade_pct`` of the **current** equity (compounding);
* ``max_open_positions`` and ``max_positions_per_symbol`` are global;
* daily kill-switch (realised loss %, consecutive losses, profit target, max trades/day)
  locks the whole account until the next day.

This fixes the over-statement of summing independent per-symbol equity curves
(N symbols each sized on the full account = N times the risk).
Approximation: the kill-switch uses realised P/L (floating P/L is not replayed), and lot
rounding is ignored (R scales linearly with size).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import pandas as pd

from src.backtest.engine import BacktestResult, BTTrade, compute_stats
from src.config import RiskConfig, Settings


@dataclass
class PortfolioRules:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_daily_loss_pct: float = 2.5
    max_consecutive_losses: int = 4
    daily_profit_target_pct: float = 0.0
    max_trades_per_day: int = 30

    @classmethod
    def from_risk(cls, risk: RiskConfig, initial_equity: float) -> PortfolioRules:
        return cls(
            initial_equity=initial_equity,
            risk_per_trade_pct=risk.risk_per_trade_pct,
            max_open_positions=risk.max_open_positions,
            max_positions_per_symbol=risk.max_positions_per_symbol,
            max_daily_loss_pct=risk.max_daily_loss_pct,
            max_consecutive_losses=risk.max_consecutive_losses,
            daily_profit_target_pct=risk.daily_profit_target_pct,
            max_trades_per_day=risk.max_trades_per_day,
        )


@dataclass
class PortfolioResult(BacktestResult):
    rejected: dict[str, int] = field(default_factory=dict)
    locks: int = 0


def simulate_portfolio(trades: list[BTTrade], rules: PortfolioRules,
                       ref_risk_pct: float | None = None) -> PortfolioResult:  # fmt: skip
    """Replay candidate trades on a single account.

    ``ref_risk_pct``: the risk % the candidates were generated with. A trade generated at
    ``ref * k`` (e.g. key-level priority) is replayed at ``rules.risk_per_trade_pct * k``.
    """
    events: list[tuple[pd.Timestamp, int, int]] = []  # (time, 0=exit/1=entry, trade idx)
    for i, t in enumerate(trades):
        entry = pd.Timestamp(t.entry_time)
        exit_ = pd.Timestamp(t.exit_time)
        events.append((entry, 1, i))
        # exits of OTHER trades go before entries at the same instant (frees the slot), but a
        # trade opened and closed within the same bar must be opened before it is closed
        events.append((max(exit_, entry), 0 if exit_ > entry else 2, i))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    equity = rules.initial_equity
    open_: dict[int, float] = {}  # trade idx -> money at risk
    per_symbol: dict[str, int] = {}
    day = None
    day_start = equity
    realised_day = 0.0
    trades_today = 0
    consec = 0
    locked = False
    locks = 0
    rejected: dict[str, int] = {}
    taken: list[BTTrade] = []
    curve_t: list[pd.Timestamp] = []
    curve_v: list[float] = []

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for ts, kind, i in events:
        if ts.date() != day:  # new trading day: reset the daily breaker
            day, day_start, realised_day, trades_today, consec, locked = (
                ts.date(),
                equity,
                0.0,
                0,
                0,
                False,
            )
        t = trades[i]
        if kind == 1:
            if locked:
                reject("kill-switch locked")
            elif len(open_) >= rules.max_open_positions:
                reject("max open positions")
            elif per_symbol.get(t.symbol, 0) >= rules.max_positions_per_symbol:
                reject("max positions on symbol")
            elif trades_today >= rules.max_trades_per_day:
                reject("max trades per day")
            else:
                scale = (t.risk_pct / ref_risk_pct) if ref_risk_pct and t.risk_pct else 1.0
                open_[i] = equity * rules.risk_per_trade_pct * scale / 100.0
                per_symbol[t.symbol] = per_symbol.get(t.symbol, 0) + 1
                trades_today += 1
            continue
        if i not in open_:
            continue
        risk_money = open_.pop(i)
        per_symbol[t.symbol] -= 1
        pnl = t.r_multiple * risk_money
        equity += pnl
        realised_day += pnl
        consec = consec + 1 if pnl < 0 else 0
        taken.append(replace(t, pnl=pnl, risk_money=risk_money))
        curve_t.append(ts)
        curve_v.append(equity)
        if not locked and day_start > 0:
            day_pct = realised_day / day_start * 100
            if (day_pct <= -rules.max_daily_loss_pct or consec >= rules.max_consecutive_losses
                    or (rules.daily_profit_target_pct > 0 and day_pct >= rules.daily_profit_target_pct)):  # fmt: skip
                locked = True
                locks += 1

    if curve_t:
        start = min(pd.Timestamp(t.entry_time) for t in trades)
        idx = pd.DatetimeIndex([start, *curve_t])
        curve = pd.Series([rules.initial_equity, *curve_v], index=idx, name="equity")
        curve = curve[~curve.index.duplicated(keep="last")]
    else:
        curve = pd.Series(dtype=float, name="equity")
    res = PortfolioResult(taken, curve, rules.initial_equity, rejected=rejected, locks=locks)
    res.stats = compute_stats(res) if len(curve) else {"trades": 0.0}
    res.meta = {"rejected": rejected, "kill_switch_locks": locks, "candidates": len(trades)}
    return res


def candidate_settings(settings: Settings) -> Settings:
    """Copy of the settings for candidate generation: per-symbol breakers off (the account-level
    breaker is applied afterwards by :func:`simulate_portfolio`)."""
    s = copy.deepcopy(settings)
    s.risk.max_daily_loss_pct = 1e9
    s.risk.daily_trailing_dd_pct = 0.0
    s.risk.daily_profit_target_pct = 0.0
    s.risk.max_consecutive_losses = 10**9
    s.risk.max_trades_per_day = 10**9
    s.risk.cooldown_after_loss_s = 0
    return s

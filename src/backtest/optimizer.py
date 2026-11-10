"""Walk-forward style auto-optimizer (in-sample selection, out-of-sample validation).

Every parameter combination is backtested once on the full history (the strategy
is causal, so this is equivalent to running segments with a warm-up). Trades are
then split by entry time: the **in-sample** part ranks the combinations, the
**out-of-sample** part shows whether the edge survives. Only OOS numbers should
drive changes to ``settings.yaml``.
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BTTrade
from src.config import Settings, SymbolSpec
from src.core.clock import ServerClock
from src.strategies.xau_velocity_scalp import create_strategy

DEFAULT_GRID: dict[str, list[Any]] = {
    "strategy.displacement_body_atr": [1.2, 1.5, 1.8],
    "strategy.pullback_pct": [0.35, 0.50],
    "strategy.trigger_mode": ["sweep", "any"],
    "execution.tp1_r": [0.8, 1.0],
}


def _set(settings: Settings, dotted: str, value: Any) -> None:
    section, attr = dotted.split(".", 1)
    setattr(getattr(settings, section), attr, value)


def trade_metrics(trades: list[BTTrade]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "net": 0.0,
        }
    pnl = np.array([t.pnl for t in trades])
    r = np.array([t.r_multiple for t in trades])
    loss = -pnl[pnl <= 0].sum()
    return {
        "trades": len(trades),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": float(pnl[pnl > 0].sum() / loss) if loss > 0 else math.inf,
        "expectancy_r": float(r.mean()),
        "net": float(pnl.sum()),
    }


def score(m: dict[str, float], min_trades: int = 15) -> float:
    """Expectancy in R scaled by sqrt(N): rewards edge *and* statistical weight."""
    if m["trades"] < min_trades:
        return -math.inf
    return m["expectancy_r"] * math.sqrt(m["trades"])


def optimize(settings: Settings, symbol: str, m1: pd.DataFrame, clock: ServerClock,
             ticks: pd.DataFrame | None = None, grid: dict[str, list[Any]] | None = None,
             top: int = 5, spec: SymbolSpec | None = None) -> pd.DataFrame:  # fmt: skip
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    split_time = m1["time"].iloc[int(len(m1) * settings.backtest.optimize_train_pct)]
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        s = copy.deepcopy(settings)
        for k, v in zip(keys, combo, strict=True):
            _set(s, k, v)
        strat = create_strategy(s.strategy, s.levels, clock)
        res = BacktestEngine(s, strat, symbol, spec=spec, clock=clock).run(m1, ticks)
        ins = [t for t in res.trades if t.entry_time < split_time]
        oos = [t for t in res.trades if t.entry_time >= split_time]
        mi, mo = trade_metrics(ins), trade_metrics(oos)
        rows.append({**dict(zip(keys, combo, strict=True)),
                     **{f"IS_{k}": v for k, v in mi.items()},
                     **{f"OOS_{k}": v for k, v in mo.items()},
                     "IS_score": score(mi)})  # fmt: skip
    df = pd.DataFrame(rows).sort_values("IS_score", ascending=False).reset_index(drop=True)
    df.attrs["split_time"] = str(split_time)
    return df.head(top)

"""Strategy contract shared by live trading and the backtester.

A strategy exposes ONE vectorised method, :meth:`BaseStrategy.compute_features`,
which returns a DataFrame aligned on the entry timeframe with at least a
``signal`` column (+1 long, -1 short, 0 flat). The backtester calls it once on
the full history; the live loop calls it on the latest window and reads the
last row. Same code path => no live/backtest divergence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.config import LevelsConfig, StrategyConfig, SymbolConfig, SymbolSpec
from src.core.clock import ServerClock

Side = Literal[1, -1]  # convenience alias for strategy authors


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: int  # +1 buy / -1 sell
    order_type: str  # "market" | "limit"
    entry: float  # execution-side price (ask for buys, bid for sells)
    stop_loss: float
    take_profit: float
    atr: float
    time: pd.Timestamp
    reason: str = ""
    expiry_s: int = 0  # pending-order lifetime (0 = market)
    key_level: str | None = None  # institutional level swept by the setup, if any
    risk_mult: float = 1.0  # sizing multiplier (key-level priority)

    @property
    def is_buy(self) -> bool:
        return self.side > 0

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop_loss)


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(
        self,
        cfg: StrategyConfig,
        levels_cfg: LevelsConfig | None = None,
        clock: ServerClock | None = None,
    ) -> None:
        self.cfg = cfg
        self.levels_cfg = levels_cfg or LevelsConfig(enabled=False)
        self.clock = clock or ServerClock()

    @property
    def timeframes(self) -> list[str]:
        tfs = self.cfg.timeframes
        return [tfs["entry"], tfs["confirm"], tfs["trend"]]

    @property
    def required_timeframes(self) -> list[str]:
        """Timeframes the strategy needs in ``frames`` (may include the levels TF)."""
        return self.timeframes

    @property
    def entry_tf(self) -> str:
        return self.cfg.timeframes["entry"]

    @abstractmethod
    def compute_features(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Vectorised features on the entry TF; must contain ``signal`` and ``atr``."""

    @abstractmethod
    def build_signal(
        self,
        row: pd.Series,
        symbol_cfg: SymbolConfig,
        spec: SymbolSpec,
        spread_price: float,
    ) -> Signal | None:
        """Turn a feature row with ``signal != 0`` into a concrete order proposal."""

    def latest_signal(
        self,
        frames: dict[str, pd.DataFrame],
        symbol_cfg: SymbolConfig,
        spec: SymbolSpec,
        spread_price: float,
    ) -> tuple[Signal | None, pd.Series | None]:
        if any(df is None or df.empty for df in frames.values()):
            return None, None
        feats = self.compute_features(frames)
        if feats.empty:
            return None, None
        row = feats.iloc[-1]
        if int(row.get("signal", 0)) == 0:
            return None, row
        return self.build_signal(row, symbol_cfg, spec, spread_price), row

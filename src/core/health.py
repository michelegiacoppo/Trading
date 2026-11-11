"""Live strategy health monitor: switches a strategy off when reality diverges from the backtest.

After ``min_trades`` closed trades, if the live average R is more than ``z_disable`` standard
errors below the expectancy measured by the lab, new entries are blocked (open trades keep
being managed). It is the "learn from live trading" safety valve: the strategy proved
itself in research; if the market stops agreeing, the bot stops trading it.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import HealthConfig

log = logging.getLogger(__name__)


@dataclass
class StrategyHealth:
    cfg: HealthConfig
    r_history: list[float] = field(default_factory=list)
    disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def n(self) -> int:
        return len(self.r_history)

    @property
    def live_mean_r(self) -> float:
        return sum(self.r_history) / self.n if self.n else 0.0

    @property
    def z_score(self) -> float:
        """How many standard errors the live mean is above (+) / below (-) the expected mean."""
        if not self.n:
            return 0.0
        se = max(self.cfg.expected_sd_r, 1e-6) / math.sqrt(self.n)
        return (self.live_mean_r - self.cfg.expected_mean_r) / se

    def record(self, r: float) -> None:
        self.r_history.append(float(r))
        if not self.enabled or self.disabled_reason or self.n < self.cfg.min_trades:
            return
        if self.z_score < -self.cfg.z_disable:
            self.disabled_reason = (f"live {self.live_mean_r:+.2f}R/trade over {self.n} trades vs "
                                    f"expected {self.cfg.expected_mean_r:+.2f}R (z={self.z_score:.1f})")  # fmt: skip
            log.critical("STRATEGY DISABLED by health monitor: %s", self.disabled_reason)

    @property
    def allows_entries(self) -> bool:
        return self.disabled_reason is None

    def status(self) -> str:
        if not self.enabled:
            return "-"
        if self.disabled_reason:
            return "DISABLED"
        return f"{self.n} tr, {self.live_mean_r:+.2f}R (z {self.z_score:+.1f})"


class LiveTradeLog:
    """Append-only CSV of closed live/dry-run trades (input for the next lab run / analysis)."""

    FIELDS = ["time", "profile", "symbol", "ticket", "pnl", "risk_money", "r", "reason"]

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, profile: str, symbol: str, ticket: int, pnl: float, risk_money: float,
               r: float, reason: str, when: datetime) -> None:  # fmt: skip
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.FIELDS)
            w.writerow([f"{when:%Y-%m-%d %H:%M:%S}", profile, symbol, ticket, f"{pnl:.2f}",
                        f"{risk_money:.2f}", f"{r:.4f}", reason])  # fmt: skip

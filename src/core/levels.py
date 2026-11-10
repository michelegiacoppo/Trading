"""Institutional key levels for XAUUSD (PDH/PDL, Asian & London ranges, daily VWAP + bands).

All levels are computed **causally** on closed bars of ``levels.timeframe``
(default M5) and attached to the entry timeframe with ``align_htf`` - the same
code runs in the backtester (full history) and live (warm-up window).
Day boundaries and session ranges are in **UTC**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import LevelsConfig
from src.core.clock import ServerClock

SUPPORT_LEVELS = ["pdl", "asia_low", "london_low", "vwap_dn2", "vwap_dn1"]
RESISTANCE_LEVELS = ["pdh", "asia_high", "london_high", "vwap_up2", "vwap_up1"]
LEVEL_COLUMNS = [
    "pdh", "pdl", "asia_high", "asia_low", "london_high", "london_low",
    "vwap", "vwap_up1", "vwap_dn1", "vwap_up2", "vwap_dn2",
]  # fmt: skip
LEVEL_LABELS = {
    "pdh": "PDH", "pdl": "PDL", "asia_high": "Asia High", "asia_low": "Asia Low",
    "london_high": "London High", "london_low": "London Low", "vwap": "VWAP",
    "vwap_up1": "VWAP +1σ", "vwap_dn1": "VWAP -1σ", "vwap_up2": "VWAP +2σ", "vwap_dn2": "VWAP -2σ",
}  # fmt: skip


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def compute_levels(bars: pd.DataFrame, cfg: LevelsConfig, clock: ServerClock) -> pd.DataFrame:
    """Return ``bars[time] + LEVEL_COLUMNS``; every value uses only bars up to that row."""
    df = bars[["time", "high", "low", "close", "tick_volume"]].copy()
    utc = clock.series_to_utc(df["time"])
    day = utc.dt.floor("D")
    mod = utc.dt.hour * 60 + utc.dt.minute

    # --- Previous day high/low (UTC day)
    daily = df.groupby(day).agg(h=("high", "max"), l=("low", "min"))
    prev = daily.shift(1)
    df["pdh"] = day.map(prev["h"]).to_numpy()
    df["pdl"] = day.map(prev["l"]).to_numpy()

    # --- Session ranges of *today* (running high/low, frozen after the session ends)
    for name, (a, b) in (("asia", cfg.asian_session_utc), ("london", cfg.london_session_utc)):
        in_sess = (mod >= _minutes(a)) & (mod < _minutes(b))
        hi = df["high"].where(in_sess)
        lo = df["low"].where(in_sess)
        df[f"{name}_high"] = hi.groupby(day).cummax().groupby(day).ffill()
        df[f"{name}_low"] = lo.groupby(day).cummin().groupby(day).ffill()

    # --- Daily rolling VWAP and volume-weighted std bands
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].astype(float).clip(lower=1.0)
    cum_v = vol.groupby(day).cumsum()
    vwap = (tp * vol).groupby(day).cumsum() / cum_v
    var = (vol * tp**2).groupby(day).cumsum() / cum_v - vwap**2
    sd = np.sqrt(var.clip(lower=0.0))
    df["vwap"] = vwap
    bands = sorted(cfg.vwap_band_std) + [cfg.vwap_band_std[-1]] * 2
    for i, k in enumerate(bands[:2], start=1):
        df[f"vwap_up{i}"] = vwap + k * sd
        df[f"vwap_dn{i}"] = vwap - k * sd
    return df[["time", *LEVEL_COLUMNS]]


@dataclass
class KeyLevelMap:
    """Snapshot of the current levels (shown on the dashboard, used for logging)."""

    as_of: datetime | None = None
    levels: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, levels: pd.DataFrame) -> KeyLevelMap:
        if levels.empty:
            return cls()
        row = levels.iloc[-1]
        vals = {k: float(row[k]) for k in LEVEL_COLUMNS if k in row and pd.notna(row[k])}
        return cls(pd.Timestamp(row["time"]).to_pydatetime(), vals)

    def nearest(self, price: float, n: int = 3) -> list[tuple[str, float, float]]:
        """(label, level, distance) of the n closest levels to ``price``."""
        rows = [(LEVEL_LABELS[k], v, v - price) for k, v in self.levels.items()]
        return sorted(rows, key=lambda r: abs(r[2]))[:n]

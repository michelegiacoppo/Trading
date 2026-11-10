"""Generic, strategy-agnostic market features for meta-labeling (causal: bar-close values).

The same function is used by the lab (to label historical signals) and by the live bot (to
score a new signal), so the model always sees identically computed inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.clock import ServerClock
from src.strategies import indicators as ind

FEATURES = [
    "side", "mom_1", "mom_3", "mom_12", "atr_ratio", "dist_ema50", "dist_ema200", "rsi14",
    "range_pos20", "vol_ratio", "spread_atr", "dist_vwap", "hour_sin", "hour_cos", "dow",
]  # fmt: skip


def market_features(
    bars: pd.DataFrame, point: float, clock: ServerClock | None = None
) -> pd.DataFrame:
    """Feature table aligned 1:1 with ``bars`` (without the trade ``side`` column)."""
    clock = clock or ServerClock()
    df = bars.reset_index(drop=True)
    c = df["close"]
    atr = ind.atr(df, 14)
    f = pd.DataFrame(index=df.index)
    f["time"] = df["time"]
    for k in (1, 3, 12):
        f[f"mom_{k}"] = (c - c.shift(k)) / atr
    f["atr_ratio"] = atr / atr.rolling(100, min_periods=20).mean()
    f["dist_ema50"] = (c - ind.ema(c, 50)) / atr
    f["dist_ema200"] = (c - ind.ema(c, 200)) / atr
    f["rsi14"] = ind.rsi(c, 14)
    hi, lo = df["high"].rolling(20).max(), df["low"].rolling(20).min()
    f["range_pos20"] = (c - lo) / (hi - lo).replace(0, np.nan)
    vol = df["tick_volume"].astype(float)
    f["vol_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    spread = (
        df["spread"].astype(float) if "spread" in df.columns else pd.Series(0.0, index=df.index)
    )
    f["spread_atr"] = spread * point / atr
    utc = clock.series_to_utc(df["time"])
    day = utc.dt.floor("D")
    tp = (df["high"] + df["low"] + c) / 3
    v = vol.clip(lower=1)
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    f["dist_vwap"] = (c - vwap) / atr
    hour = utc.dt.hour + utc.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = utc.dt.dayofweek.astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def features_at(feat: pd.DataFrame, signal_times: pd.Series, tf_minutes: int) -> pd.DataFrame:
    """Rows of ``feat`` for bars that CLOSED at ``signal_times`` (bar open = signal - tf)."""
    key = pd.to_datetime(signal_times) - pd.Timedelta(minutes=tf_minutes)
    idx = feat.set_index("time")
    idx = idx[~idx.index.duplicated(keep="last")]
    return idx.reindex(key.to_numpy()).reset_index(drop=True)

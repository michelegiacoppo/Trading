"""Vectorised indicators and multi-timeframe alignment (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.data_feed import TF_MINUTES


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR."""
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(loss.notna())


def align_htf(
    base: pd.DataFrame, base_tf: str, htf: pd.DataFrame, htf_tf: str, cols: list[str], prefix: str
) -> pd.DataFrame:
    """Attach higher-timeframe columns to ``base`` without look-ahead.

    A bar stamped ``t`` (open time) is only *known* at ``t + tf``. Each base bar
    is matched (backward as-of join on known-times) to the latest HTF bar that
    had already closed when the base bar closed.
    """
    left = base.copy()
    left["_known"] = left["time"] + pd.Timedelta(minutes=TF_MINUTES[base_tf])
    right = htf[["time", *cols]].copy()
    right["_known"] = right["time"] + pd.Timedelta(minutes=TF_MINUTES[htf_tf])
    right = right.drop(columns="time").rename(columns={c: f"{prefix}{c}" for c in cols})
    merged = pd.merge_asof(
        left.sort_values("_known"), right.sort_values("_known"), on="_known", direction="backward"
    )
    return merged.drop(columns="_known")


def resample_ohlc(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Build higher-timeframe bars from M1 (bars labelled by open time)."""
    rule = f"{TF_MINUTES[tf]}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    if "spread" in m1.columns:
        agg["spread"] = "mean"
    out = m1.set_index("time").resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"]).reset_index()

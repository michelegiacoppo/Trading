"""Support / resistance zones from swing pivots (causal).

A swing high at bar ``i`` (highest high of ``i-k .. i+k``) is only *known* at bar ``i+k``:
levels enter the map ``k`` bars late, never earlier. Pivots closer than ``tol_atr`` x ATR are
merged into a zone whose **strength** counts how many times price turned there (highs and
lows both count: a broken resistance becomes support - role reversal). Old pivots expire after
``lookback`` bars.

For every bar the map returns the nearest zone below (support) and above (resistance) the
close with at least ``min_touches`` touches, its strength and the distance in ATR.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def swing_pivots(df: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    """(pivot_high_value, pivot_low_value) placed on the bar where the pivot becomes known."""
    w = 2 * k + 1
    hi_c = df["high"].shift(k)
    lo_c = df["low"].shift(k)
    is_ph = hi_c == df["high"].rolling(w).max()
    is_pl = lo_c == df["low"].rolling(w).min()
    return hi_c.where(is_ph), lo_c.where(is_pl)


def sr_zones(df: pd.DataFrame, atr: pd.Series, k: int = 3, lookback: int = 300, tol_atr: float = 0.35,
             min_touches: int = 2) -> pd.DataFrame:  # fmt: skip
    """Columns: support, support_touches, resistance, resistance_touches, dist_sup_atr, dist_res_atr."""
    ph, pl = swing_pivots(df, k)
    close = df["close"].to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)
    ph_v, pl_v = ph.to_numpy(dtype=float), pl.to_numpy(dtype=float)
    n = len(df)
    sup = np.full(n, np.nan)
    sup_t = np.zeros(n)
    res = np.full(n, np.nan)
    res_t = np.zeros(n)
    levels: deque[tuple[int, float]] = deque()  # (bar index known, price)
    for i in range(n):
        for v in (ph_v[i], pl_v[i]):
            if not np.isnan(v):
                levels.append((i, v))
        while levels and levels[0][0] < i - lookback:
            levels.popleft()
        a = atr_v[i]
        if not levels or not np.isfinite(a) or a <= 0:
            continue
        vals = np.fromiter((v for _, v in levels), dtype=float, count=len(levels))
        tol = tol_atr * a
        touches = (np.abs(vals[:, None] - vals[None, :]) <= tol).sum(axis=1)
        strong = touches >= min_touches
        below = strong & (vals < close[i])
        above = strong & (vals > close[i])
        if below.any():
            j = np.argmax(np.where(below, vals, -np.inf))
            sup[i], sup_t[i] = vals[j], touches[j]
        if above.any():
            j = np.argmin(np.where(above, vals, np.inf))
            res[i], res_t[i] = vals[j], touches[j]
    out = pd.DataFrame(index=df.index)
    out["support"], out["support_touches"] = sup, sup_t
    out["resistance"], out["resistance_touches"] = res, res_t
    out["dist_sup_atr"] = (df["close"] - out["support"]) / atr
    out["dist_res_atr"] = (out["resistance"] - df["close"]) / atr
    return out


def market_structure(df: pd.DataFrame, k: int = 3) -> pd.Series:
    """+1 higher highs & higher lows, -1 lower highs & lower lows, 0 otherwise (pivot based)."""
    ph, pl = swing_pivots(df, k)
    last_h = ph.ffill()
    last_l = pl.ffill()
    prev_h = ph.dropna().shift(1).reindex(df.index).ffill()
    prev_l = pl.dropna().shift(1).reindex(df.index).ffill()
    up = (last_h > prev_h) & (last_l > prev_l)
    dn = (last_h < prev_h) & (last_l < prev_l)
    return pd.Series(np.select([up, dn], [1, -1], 0), index=df.index)

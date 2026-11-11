"""Japanese candlestick patterns (vectorised, causal: each flag uses the current and past bars).

Every function returns a boolean Series aligned with ``df``. Bullish / bearish variants are
returned as a pair. :data:`PATTERNS` lists them with a quality weight used by the
price-action confluence score.
"""

from __future__ import annotations

import pandas as pd


def _parts(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    body = c - o
    rng = (h - lo).where(h > lo)
    return {
        "o": o, "h": h, "l": lo, "c": c, "body": body, "abody": body.abs(), "rng": rng,
        "upper": h - pd.concat([o, c], axis=1).max(axis=1),
        "lower": pd.concat([o, c], axis=1).min(axis=1) - lo,
    }  # fmt: skip


def pin_bar(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Hammer / shooting star: long rejection wick (>= 60% of range, >= 2x body)."""
    p = _parts(df)
    bull = (
        (p["lower"] >= 0.6 * p["rng"])
        & (p["lower"] >= 2 * p["abody"])
        & (p["upper"] <= 0.25 * p["rng"])
    )
    bear = (
        (p["upper"] >= 0.6 * p["rng"])
        & (p["upper"] >= 2 * p["abody"])
        & (p["lower"] <= 0.25 * p["rng"])
    )
    return bull.fillna(False), bear.fillna(False)


def engulfing(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    p = _parts(df)
    pb, po, pc = p["body"].shift(1), p["o"].shift(1), p["c"].shift(1)
    bull = (pb < 0) & (p["body"] > 0) & (p["o"] <= pc) & (p["c"] >= po) & (p["abody"] > pb.abs())
    bear = (pb > 0) & (p["body"] < 0) & (p["o"] >= pc) & (p["c"] <= po) & (p["abody"] > pb.abs())
    return bull.fillna(False), bear.fillna(False)


def inside_bar_breakout(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Bar t-1 inside bar t-2 (compression), bar t closes outside bar t-1."""
    h, lo, c = df["high"], df["low"], df["close"]
    inside = (h.shift(1) < h.shift(2)) & (lo.shift(1) > lo.shift(2))
    return (inside & (c > h.shift(1))).fillna(False), (inside & (c < lo.shift(1))).fillna(False)


def star(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Morning / evening star: strong bar, small indecision bar, strong reversal past the midpoint."""
    p = _parts(df)
    b2, r2 = p["body"].shift(2), p["rng"].shift(2)
    small1 = p["abody"].shift(1) <= 0.3 * p["rng"].shift(1)
    mid2 = (p["o"].shift(2) + p["c"].shift(2)) / 2
    bull = (b2 < 0) & (b2.abs() >= 0.6 * r2) & small1 & (p["body"] > 0) & (p["c"] > mid2)
    bear = (b2 > 0) & (b2.abs() >= 0.6 * r2) & small1 & (p["body"] < 0) & (p["c"] < mid2)
    return bull.fillna(False), bear.fillna(False)


def tweezer(df: pd.DataFrame, atr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Two bars with (almost) equal lows / highs and opposite colours."""
    p = _parts(df)
    tol = 0.1 * atr
    bull = ((p["l"] - p["l"].shift(1)).abs() <= tol) & (p["body"].shift(1) < 0) & (p["body"] > 0)
    bear = ((p["h"] - p["h"].shift(1)).abs() <= tol) & (p["body"].shift(1) > 0) & (p["body"] < 0)
    return bull.fillna(False), bear.fillna(False)


def three_soldiers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Three white soldiers / three black crows: three strong bars closing progressively."""
    p = _parts(df)
    strong = p["abody"] >= 0.5 * p["rng"]
    up = (p["body"] > 0) & strong & (p["c"] > p["c"].shift(1))
    dn = (p["body"] < 0) & strong & (p["c"] < p["c"].shift(1))
    bull = up & up.shift(1, fill_value=False) & up.shift(2, fill_value=False)
    bear = dn & dn.shift(1, fill_value=False) & dn.shift(2, fill_value=False)
    return bull.fillna(False), bear.fillna(False)


def marubozu(df: pd.DataFrame, atr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Full-body momentum candle (>= 85% body, >= 1 ATR)."""
    p = _parts(df)
    full = (p["abody"] >= 0.85 * p["rng"]) & (p["rng"] >= atr)
    return (full & (p["body"] > 0)).fillna(False), (full & (p["body"] < 0)).fillna(False)


# name -> (function needing atr?, quality weight in the confluence score, reversal pattern?)
PATTERNS: dict[str, tuple[object, bool, float, bool]] = {
    "pin_bar": (pin_bar, False, 1.5, True),
    "engulfing": (engulfing, False, 1.5, True),
    "star": (star, False, 2.0, True),
    "tweezer": (tweezer, True, 1.0, True),
    "inside_breakout": (inside_bar_breakout, False, 1.0, False),
    "three_soldiers": (three_soldiers, False, 1.0, False),
    "marubozu": (marubozu, True, 0.5, False),
}

LABELS_IT = {
    "pin_bar": ("Hammer/Pin bar", "Shooting star/Pin bar"),
    "engulfing": ("Engulfing rialzista", "Engulfing ribassista"),
    "star": ("Morning star", "Evening star"),
    "tweezer": ("Tweezer bottom", "Tweezer top"),
    "inside_breakout": ("Inside bar breakout rialzista", "Inside bar breakout ribassista"),
    "three_soldiers": ("Tre soldati bianchi", "Tre corvi neri"),
    "marubozu": ("Marubozu rialzista", "Marubozu ribassista"),
}


def detect_all(df: pd.DataFrame, atr: pd.Series) -> dict[str, tuple[pd.Series, pd.Series]]:
    out = {}
    for name, (fn, needs_atr, _, _) in PATTERNS.items():
        out[name] = fn(df, atr) if needs_atr else fn(df)  # type: ignore[operator]
    return out

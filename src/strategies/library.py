"""Strategy library for the research lab: diverse, well-known strategy families.

Each family works on ONE timeframe (``strategy.timeframes.entry``), reads its parameters
from ``strategy.params`` and declares the trade management that suits it (``EXEC`` /
``SESSIONS`` overrides applied by the lab and written into generated profiles).

Families (all causal, vectorised, market entry at the next bar open):

* ``trend_donchian``  - Donchian channel breakout + optional EMA filter, ATR chandelier exit
* ``ema_cross``       - fast/slow EMA crossover, ATR trailing exit
* ``tsmom``           - time-series momentum (sign change of N-bar return), ATR trailing exit
* ``orb``             - opening-range breakout of the London / New York open, fixed R target,
                        flat after a few hours
* ``vwap_reversion``  - stretch beyond the daily VWAP sigma band and re-entry, target = VWAP
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.config import LevelsConfig, StrategyConfig, SymbolConfig, SymbolSpec
from src.core.clock import ServerClock
from src.core.levels import compute_levels
from src.strategies import indicators as ind
from src.strategies.base_strategy import BaseStrategy, Signal

TREND_EXEC: dict[str, Any] = {
    "tp1_r": 99.0, "tp1_close_fraction": 0.0, "trailing_mode": "atr", "trailing_only_after_tp1": False,
    "max_trade_duration_minutes": 0, "stale_require_fade": False,
}  # fmt: skip
INTRADAY_EXEC: dict[str, Any] = {
    "tp1_r": 99.0, "tp1_close_fraction": 0.0, "trailing_only_after_tp1": True,
    "max_trade_duration_minutes": 240, "stale_require_fade": False,
}  # fmt: skip
ALL_DAY: dict[str, Any] = {"windows": {}}  # trade around the clock (rollover/weekend still blocked)


class SingleTFStrategy(BaseStrategy):
    """Base for library families: one timeframe, params dict, generic order construction."""

    name = "single_tf"
    DEFAULTS: ClassVar[dict[str, Any]] = {"atr_period": 14, "sl_atr": 2.0, "tp_r": 0.0}
    GRID: ClassVar[dict[str, list[Any]]] = {}
    TIMEFRAMES: ClassVar[list[str]] = ["H1"]
    EXEC: ClassVar[dict[str, Any]] = TREND_EXEC
    SESSIONS: ClassVar[dict[str, Any]] = ALL_DAY

    def __init__(self, cfg: StrategyConfig, levels_cfg: LevelsConfig | None = None,
                 clock: ServerClock | None = None) -> None:  # fmt: skip
        super().__init__(cfg, levels_cfg, clock)
        self.p: dict[str, Any] = {
            **SingleTFStrategy.DEFAULTS,
            **self.DEFAULTS,
            **(cfg.params or {}),
        }

    @property
    def tf(self) -> str:
        return self.cfg.timeframes.get("entry", self.TIMEFRAMES[0])

    @property
    def timeframes(self) -> list[str]:
        return [self.tf, self.tf, self.tf]

    @property
    def required_timeframes(self) -> list[str]:
        return [self.tf]

    @classmethod
    def exec_overrides(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Execution settings for this family (may depend on params)."""
        out = dict(cls.EXEC)
        if out.get("trailing_mode") == "atr":
            out["trailing_atr_mult"] = float(params.get("trail_atr", params.get("sl_atr", 2.0)))
        return out

    # -------------------------------------------------------------- hooks
    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        raise NotImplementedError

    def compute_features(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = frames[self.tf].copy().reset_index(drop=True)
        df["atr"] = ind.atr(df, int(self.p["atr_period"]))
        df["momentum"] = (df["close"] - df["close"].shift(3)) / df["atr"]
        df["sl_dist"] = float(self.p["sl_atr"]) * df["atr"]
        df["tp_dist"] = np.nan
        long_, short = self.signals(df)
        ok = df["atr"].notna() & (df["atr"] > 0)
        df["signal"] = np.where(long_.fillna(False) & ok, 1,
                                np.where(short.fillna(False) & ok, -1, 0)).astype(int)  # fmt: skip
        df["trigger"] = self.name
        return df

    def build_signal(self, row: pd.Series, symbol_cfg: SymbolConfig, spec: SymbolSpec,
                     spread_price: float) -> Signal | None:  # fmt: skip
        side = int(row["signal"])
        atr = float(row["atr"])
        sl_dist = float(row["sl_dist"])
        if side == 0 or not np.isfinite(atr) or not np.isfinite(sl_dist) or sl_dist <= 0:
            return None
        if sl_dist < max(3 * spread_price, 5 * spec.point):  # stop inside the noise / spread
            return None
        close = float(row["close"])
        entry = close + spread_price if side > 0 else close
        sl = entry - side * sl_dist
        tp_dist = row.get("tp_dist")
        if tp_dist is not None and np.isfinite(tp_dist) and tp_dist > 0:
            tp = entry + side * float(tp_dist)
        elif float(self.p.get("tp_r", 0)) > 0:
            tp = entry + side * float(self.p["tp_r"]) * sl_dist
        else:
            tp = 0.0
        d = spec.digits
        return Signal(symbol=symbol_cfg.name, side=side, order_type="market", entry=round(entry, d),
                      stop_loss=round(sl, d), take_profit=round(tp, d), atr=atr,
                      time=pd.Timestamp(row["time"]), reason=f"{self.name} {self.tf}")  # fmt: skip


def _first(cond: pd.Series) -> pd.Series:
    """True only on the first bar of a run of True values (new event, not continuation)."""
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


class TrendDonchian(SingleTFStrategy):
    name = "trend_donchian"
    DEFAULTS: ClassVar[dict[str, Any]] = {"lookback": 20, "sl_atr": 2.0, "ema_filter": 200}
    GRID: ClassVar[dict[str, list[Any]]] = {
        "lookback": [20, 55],
        "sl_atr": [2.0, 3.0],
        "ema_filter": [0, 200],
    }
    TIMEFRAMES: ClassVar[list[str]] = ["H1", "H4"]

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        n = int(self.p["lookback"])
        hi = df["high"].shift(1).rolling(n).max()
        lo = df["low"].shift(1).rolling(n).min()
        long_, short = df["close"] > hi, df["close"] < lo
        if int(self.p["ema_filter"]) > 0:
            ema = ind.ema(df["close"], int(self.p["ema_filter"]))
            long_ &= df["close"] > ema
            short &= df["close"] < ema
        return _first(long_), _first(short)


class EmaCross(SingleTFStrategy):
    name = "ema_cross"
    DEFAULTS: ClassVar[dict[str, Any]] = {"fast": 20, "slow": 100, "sl_atr": 2.0}
    GRID: ClassVar[dict[str, list[Any]]] = {
        "fast": [10, 20],
        "slow": [50, 100],
        "sl_atr": [2.0, 3.0],
    }
    TIMEFRAMES: ClassVar[list[str]] = ["H1", "H4"]

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        f = ind.ema(df["close"], int(self.p["fast"]))
        s = ind.ema(df["close"], int(self.p["slow"]))
        return _first(f > s), _first(f < s)


class TSMomentum(SingleTFStrategy):
    name = "tsmom"
    DEFAULTS: ClassVar[dict[str, Any]] = {"lookback": 72, "sl_atr": 2.5}
    GRID: ClassVar[dict[str, list[Any]]] = {"lookback": [24, 72, 168], "sl_atr": [2.5]}
    TIMEFRAMES: ClassVar[list[str]] = ["H1", "H4"]

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        roc = df["close"] / df["close"].shift(int(self.p["lookback"])) - 1.0
        return _first(roc > 0), _first(roc < 0)


SESSION_OPENS = {"london": ("Europe/London", 8, 0), "ny": ("America/New_York", 9, 30)}


class OpeningRangeBreakout(SingleTFStrategy):
    name = "orb"
    DEFAULTS: ClassVar[dict[str, Any]] = {"session": "london", "range_minutes": 30, "tp_r": 2.0,
                                          "window_minutes": 180, "sl_atr": 0.0}  # fmt: skip
    GRID: ClassVar[dict[str, list[Any]]] = {"session": ["london", "ny"], "range_minutes": [30, 60],
                                            "tp_r": [1.5, 2.5]}  # fmt: skip
    TIMEFRAMES: ClassVar[list[str]] = ["M5", "M15"]
    EXEC: ClassVar[dict[str, Any]] = INTRADAY_EXEC

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        tz, h, m = SESSION_OPENS[str(self.p["session"])]
        local = self.clock.series_to_zone(df["time"], tz)
        day = local.dt.floor("D")
        mins = (local - day).dt.total_seconds() / 60 - (h * 60 + m)
        rng_min = float(self.p["range_minutes"])
        in_range = (mins >= 0) & (mins < rng_min)
        r_hi = df["high"].where(in_range).groupby(day).transform("max")
        r_lo = df["low"].where(in_range).groupby(day).transform("min")
        # the range is only known once it is complete: use it strictly after rng_min
        after = (mins >= rng_min) & (mins < rng_min + float(self.p["window_minutes"]))
        brk_l = after & (df["close"] > r_hi)
        brk_s = after & (df["close"] < r_lo)
        first_l = brk_l & (brk_l.astype(int).groupby(day).cumsum() == 1)
        first_s = brk_s & (brk_s.astype(int).groupby(day).cumsum() == 1)
        df["sl_dist"] = np.where(
            first_l, df["close"] - r_lo, np.where(first_s, r_hi - df["close"], np.nan)
        )
        return first_l, first_s


class VwapReversion(SingleTFStrategy):
    name = "vwap_reversion"
    DEFAULTS: ClassVar[dict[str, Any]] = {"band_k": 2.0, "sl_atr": 1.5}
    GRID: ClassVar[dict[str, list[Any]]] = {"band_k": [2.0, 2.5], "sl_atr": [1.5, 2.5]}
    TIMEFRAMES: ClassVar[list[str]] = ["M5", "M15"]
    EXEC: ClassVar[dict[str, Any]] = INTRADAY_EXEC

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        k = float(self.p["band_k"])
        lv = compute_levels(df, LevelsConfig(vwap_band_std=[k, k]), self.clock)
        vwap, up, dn = lv["vwap"].to_numpy(), lv["vwap_up1"].to_numpy(), lv["vwap_dn1"].to_numpy()
        close, prev = df["close"], df["close"].shift(1)
        # re-entry inside the band after a stretch beyond it (the band is known at bar close)
        long_ = (prev < pd.Series(dn).shift(1).to_numpy()) & (close > dn)
        short = (prev > pd.Series(up).shift(1).to_numpy()) & (close < up)
        df["tp_dist"] = np.abs(vwap - close.to_numpy())
        return pd.Series(long_, index=df.index), pd.Series(short, index=df.index)


LIBRARY: dict[str, type[SingleTFStrategy]] = {
    cls.name: cls
    for cls in (TrendDonchian, EmaCross, TSMomentum, OpeningRangeBreakout, VwapReversion)
}

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
from src.strategies.candles import LABELS_IT, PATTERNS, detect_all
from src.strategies.sr_levels import market_structure, sr_zones

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
    NEWS: ClassVar[dict[str, Any]] = {}  # news-blackout overrides for this family
    REQUIRES_CALENDAR: ClassVar[bool] = False

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
                      time=pd.Timestamp(row["time"]),
                      reason=str(row.get("reason_text") or f"{self.name} {self.tf}"))  # fmt: skip


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


class NewsDrift(SingleTFStrategy):
    """Post-release continuation: after a high-impact release of one of the symbol's currencies,
    wait ``delay_minutes`` (spread normalises, first spike fades) and follow the move if it is
    large enough - optionally only when it agrees with the data surprise (actual vs forecast).
    Needs the economic calendar (mql5/XQS_ExportCalendar.mq5)."""

    name = "news_drift"
    DEFAULTS: ClassVar[dict[str, Any]] = {"delay_minutes": 10, "move_atr": 1.0, "use_surprise": True,
                                          "min_surprise": 0.5, "sl_atr": 1.5, "tp_r": 2.0}  # fmt: skip
    GRID: ClassVar[dict[str, list[Any]]] = {"delay_minutes": [5, 15], "move_atr": [0.5, 1.0],
                                            "use_surprise": [True, False]}  # fmt: skip
    TIMEFRAMES: ClassVar[list[str]] = ["M5"]
    EXEC: ClassVar[dict[str, Any]] = {**INTRADAY_EXEC, "max_trade_duration_minutes": 120}
    NEWS: ClassVar[dict[str, Any]] = {"blackout_minutes_before": 10, "blackout_minutes_after": 0}
    REQUIRES_CALENDAR: ClassVar[bool] = True

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        none = pd.Series(False, index=df.index)
        if self.calendar is None or not self.symbol:
            return none, none
        ev = self.calendar.events_for(self.symbol, 3)
        if ev.empty or len(df) < 2:
            return none, none
        tf = pd.Timedelta(df["time"].diff().median())
        close_t = (df["time"] + tf).to_numpy(dtype="datetime64[ns]")
        et = ev["time"].to_numpy(dtype="datetime64[ns]")
        k = np.searchsorted(et, close_t, side="right") - 1  # last release before the bar close
        has = k >= 0
        kk = np.clip(k, 0, None)
        since = np.full(len(df), np.inf)
        since[has] = (close_t[has] - et[k[has]]) / np.timedelta64(1, "m")
        delay = float(self.p["delay_minutes"])
        first = (since >= delay) & (since < delay + tf / pd.Timedelta(minutes=1))
        # open of the bar in which the release happened = price just before the news
        bar_open_t = df["time"].to_numpy(dtype="datetime64[ns]")
        rel_bar = np.clip(np.searchsorted(bar_open_t, et[kk], side="right") - 1, 0, None)
        move = df["close"].to_numpy() - df["open"].to_numpy()[rel_bar]
        big = np.abs(move) >= float(self.p["move_atr"]) * df["atr"].to_numpy()
        long_, short = first & big & (move > 0), first & big & (move < 0)
        if self.p["use_surprise"]:
            sz = np.nan_to_num((ev["surprise_z"] * ev["symbol_sign"]).to_numpy()[kk])
            strong = np.abs(sz) >= float(self.p["min_surprise"])
            long_ &= strong & (sz > 0)
            short &= strong & (sz < 0)
        return pd.Series(long_, index=df.index), pd.Series(short, index=df.index)


_ZONE_CACHE: dict[tuple, pd.DataFrame] = {}


class PriceAction(SingleTFStrategy):
    """A rule-based "professional trader": candlestick pattern + location + context + confirmation.

    For every candle it asks the questions a discretionary trader asks:

    1. **Pattern** - is there a reversal (pin bar, engulfing, star, tweezer) or continuation
       (inside-bar breakout, three soldiers, marubozu) candle?
    2. **Location** - did the reversal form ON a support / resistance zone built from past
       swing points (more touches = stronger zone)? Continuation patterns need the trend.
    3. **Context** - trend (price vs EMA200, EMA50 slope), market structure (HH/HL vs LH/LL),
       volume expansion.
    4. **Confirmation** - optionally wait for the next candle to close beyond the pattern.
    5. **Room** - the target is the next opposite zone; the trade is taken only if the
       reward/risk to that zone is at least ``min_rr``.

    Each factor adds to a confluence score; a signal needs ``score >= min_score``. The
    reasoning is written in ``reason_text`` (logged with every trade).
    """

    name = "price_action"
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "min_score": 3.5, "min_rr": 2.0, "confirm": 0, "trend_filter": True, "zone_tol_atr": 0.5,
        "pivot_k": 3, "zone_lookback": 300, "sl_buffer_atr": 0.2, "max_sl_atr": 3.0, "sl_atr": 0.0,
    }  # fmt: skip
    GRID: ClassVar[dict[str, list[Any]]] = {"min_score": [3.0, 4.0], "min_rr": [1.5, 2.5],
                                            "confirm": [0, 1], "trend_filter": [True, False]}  # fmt: skip
    TIMEFRAMES: ClassVar[list[str]] = ["M15", "H1", "H4"]
    EXEC: ClassVar[dict[str, Any]] = {
        "tp1_r": 99.0, "tp1_close_fraction": 0.0, "trailing_only_after_tp1": True,
        "max_trade_duration_minutes": 0, "stale_require_fade": False,
    }  # fmt: skip

    def _zones(self, df: pd.DataFrame) -> pd.DataFrame:
        k, lb = int(self.p["pivot_k"]), int(self.p["zone_lookback"])
        key = (
            self.symbol,
            self.tf,
            len(df),
            str(df["time"].iloc[0]),
            str(df["time"].iloc[-1]),
            k,
            lb,
        )
        if key not in _ZONE_CACHE:
            if len(_ZONE_CACHE) > 64:
                _ZONE_CACHE.clear()
            _ZONE_CACHE[key] = sr_zones(df, df["atr"], k=k, lookback=lb)
        return _ZONE_CACHE[key]

    def signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        p = self.p
        atr = df["atr"]
        c, h, lo = df["close"], df["high"], df["low"]
        z = self._zones(df)
        pats = detect_all(df, atr)
        ema50, ema200 = ind.ema(c, 50), ind.ema(c, 200)
        trend_up = (c > ema200) & (ema50 > ema50.shift(5))
        trend_dn = (c < ema200) & (ema50 < ema50.shift(5))
        struct = market_structure(df, int(p["pivot_k"]))
        vol = df["tick_volume"].astype(float)
        vol_up = vol > 1.3 * vol.rolling(20).mean().shift(1)
        tol = float(p["zone_tol_atr"]) * atr
        low3, high3 = lo.rolling(3).min(), h.rolling(3).max()
        at_sup = (low3 <= z["support"] + tol) & (c > z["support"])
        at_res = (high3 >= z["resistance"] - tol) & (c < z["resistance"])

        def side_score(bull: bool) -> tuple[pd.Series, pd.Series]:
            idx = 0 if bull else 1
            rev_w = pd.Series(0.0, index=df.index)
            cont_w = pd.Series(0.0, index=df.index)
            name_s = pd.Series("", index=df.index, dtype=object)
            for name, (_, _, w, reversal) in PATTERNS.items():
                flag = pats[name][idx]
                target = rev_w if reversal else cont_w
                better = flag & (w > target)
                target[better] = w
                name_s[better & (name_s == "")] = LABELS_IT[name][idx]
            at = at_sup if bull else at_res
            touches = z["support_touches"] if bull else z["resistance_touches"]
            trend = trend_up if bull else trend_dn
            counter = trend_dn if bull else trend_up
            st = struct == (1 if bull else -1)
            valid = ((rev_w > 0) & at) | ((cont_w > 0) & trend)
            score = (np.maximum(rev_w.where(at, 0.0), cont_w.where(trend, 0.0))
                     + 1.5 * at.astype(float) + (0.5 * (touches - 1).clip(0, 3)).where(at, 0.0)
                     + trend.astype(float) + st.astype(float) + 0.5 * vol_up.astype(float))  # fmt: skip
            ok = valid & (score >= float(p["min_score"]))
            if p["trend_filter"]:
                ok &= ~counter
            return ok.fillna(False), score.where(ok)

        bull_ok, bull_sc = side_score(True)
        bear_ok, bear_sc = side_score(False)
        if int(p["confirm"]):  # wait for the next candle to close beyond the pattern extreme
            bull_ok = bull_ok.shift(1, fill_value=False) & (c > h.shift(1))
            bear_ok = bear_ok.shift(1, fill_value=False) & (c < lo.shift(1))
            bull_sc, bear_sc = bull_sc.shift(1), bear_sc.shift(1)
        n_bars = 3 + int(p["confirm"])
        buf = float(p["sl_buffer_atr"]) * atr
        sl_long = c - (lo.rolling(n_bars).min() - buf)
        sl_short = (h.rolling(n_bars).max() + buf) - c
        tp_long = (z["resistance"] - 0.1 * atr - c).where(z["resistance"].notna(), 2.5 * sl_long)
        tp_short = (c - z["support"] - 0.1 * atr).where(z["support"].notna(), 2.5 * sl_short)
        max_sl = float(p["max_sl_atr"]) * atr
        rr_l, rr_s = tp_long / sl_long, tp_short / sl_short
        long_ = bull_ok & (sl_long > 0) & (sl_long <= max_sl) & (rr_l >= float(p["min_rr"]))
        short = bear_ok & (sl_short > 0) & (sl_short <= max_sl) & (rr_s >= float(p["min_rr"]))
        both = long_ & short
        long_, short = long_ & ~both, short & ~both
        df["sl_dist"] = np.where(long_, sl_long, np.where(short, sl_short, np.nan))
        df["tp_dist"] = np.where(long_, tp_long, np.where(short, tp_short, np.nan))
        df["reason_text"] = ""
        for i in np.flatnonzero((long_ | short).to_numpy()):
            bull = bool(long_.iloc[i])
            j = i - int(p["confirm"])
            pat = next((LABELS_IT[nm][0 if bull else 1] for nm in PATTERNS if pats[nm][0 if bull else 1].iloc[j]),
                       "pattern")  # fmt: skip
            lvl = z["support"].iloc[i] if bull else z["resistance"].iloc[i]
            tch = z["support_touches"].iloc[i] if bull else z["resistance_touches"].iloc[i]
            where = (f"su {'supporto' if bull else 'resistenza'} {lvl:.5g} ({int(tch)} tocchi)"
                     if np.isfinite(lvl) else "in trend")  # fmt: skip
            tr = "trend a favore" if (trend_up if bull else trend_dn).iloc[i] else "trend neutro"
            rr = (rr_l if bull else rr_s).iloc[i]
            sc = (bull_sc if bull else bear_sc).iloc[i]
            conf = ", confermato" if int(p["confirm"]) else ""
            df.loc[df.index[i], "reason_text"] = (
                f"{pat} {where}, {tr}{conf}, R:R {rr:.1f}, score {sc:.1f}"
            )
        return long_, short


LIBRARY: dict[str, type[SingleTFStrategy]] = {
    cls.name: cls
    for cls in (
        TrendDonchian,
        EmaCross,
        TSMomentum,
        OpeningRangeBreakout,
        VwapReversion,
        NewsDrift,
        PriceAction,
    )
}

"""XAU Velocity Scalp - momentum-burst "hit & run" scalper with anti-gap logic.

Pipeline (all on CLOSED bars):

1. **M15 bias**   close vs EMA(50) *and* EMA slope over N bars agree.
2. **M5 confirm** EMA(9) vs EMA(21) aligned with the bias, price on the right side.
3. **M1 trigger** a velocity burst: ``close - close[-N] >= k*ATR`` in bias
   direction, closed on a strong body near the extreme, with a tick-volume spike
   and RSI not yet exhausted.
4. **Anti-gap**   skip trigger bars whose range is > X*ATR (news spike / gap)
   and bars that opened with a gap vs the previous close.
5. **Entry**      market, or a limit order on a shallow pullback into the burst
   (better fill, filters out one-tick fakeouts, protects from slippage).
6. **Stop**       beyond the burst's swing extreme or k*ATR (whichever is wider),
   clamped to [min, max] points, then pushed away from round numbers
   (``.00``/``.50``) where stop-hunting liquidity clusters sit.
7. **Target**     fixed R-multiple; management (partial/BE/trailing/time-stop)
   is done by :mod:`src.core.execution`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import StrategyConfig, SymbolConfig, SymbolSpec
from src.risk.position_sizer import atr_stop_distance
from src.strategies import indicators as ind
from src.strategies.base_strategy import BaseStrategy, Signal


def avoid_round_levels(sl: float, side: int, step: float, buffer: float) -> float:
    """Push a stop beyond the nearest round level if it sits within ``buffer`` of it."""
    if step <= 0 or buffer <= 0:
        return sl
    level = round(sl / step) * step
    if abs(sl - level) < buffer:
        return level - buffer if side > 0 else level + buffer
    return sl


class XauVelocityScalp(BaseStrategy):
    name = "xau_velocity_scalp"

    def __init__(self, cfg: StrategyConfig) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------ features
    def compute_features(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        c = self.cfg
        tf_e, tf_c, tf_t = self.timeframes
        m1 = frames[tf_e].copy()
        m5 = frames[tf_c].copy()
        m15 = frames[tf_t].copy()

        # --- M15 directional bias
        m15["ema_t"] = ind.ema(m15["close"], c.trend_ema)
        slope = m15["ema_t"] - m15["ema_t"].shift(c.trend_slope_bars)
        m15["bias"] = np.select(
            [
                (m15["close"] > m15["ema_t"]) & (slope > 0),
                (m15["close"] < m15["ema_t"]) & (slope < 0),
            ],
            [1, -1],
            0,
        )

        # --- M5 confirmation
        m5["ema_f"] = ind.ema(m5["close"], c.confirm_ema_fast)
        m5["ema_s"] = ind.ema(m5["close"], c.confirm_ema_slow)
        m5["confirm"] = np.select(
            [
                (m5["ema_f"] > m5["ema_s"]) & (m5["close"] > m5["ema_f"]),
                (m5["ema_f"] < m5["ema_s"]) & (m5["close"] < m5["ema_f"]),
            ],
            [1, -1],
            0,
        )

        # --- M1 trigger features
        m1["atr"] = ind.atr(m1, c.atr_period)
        rng = (m1["high"] - m1["low"]).replace(0.0, np.nan)
        m1["velocity"] = m1["close"] - m1["close"].shift(c.velocity_lookback)
        m1["body_ratio"] = (m1["close"] - m1["open"]).abs() / rng
        m1["close_loc"] = (m1["close"] - m1["low"]) / rng  # 1 = closed at high
        m1["vol_ma"] = m1["tick_volume"].rolling(c.volume_ma_period).mean().shift(1)
        m1["rsi"] = ind.rsi(m1["close"], c.rsi_period)
        m1["swing_low"] = m1["low"].rolling(c.velocity_lookback + 1).min()
        m1["swing_high"] = m1["high"].rolling(c.velocity_lookback + 1).max()
        m1["gap"] = (m1["open"] - m1["close"].shift(1)).abs()

        df = ind.align_htf(m1, tf_e, m15, tf_t, ["bias"], "m15_")
        df = ind.align_htf(df, tf_e, m5, tf_c, ["confirm"], "m5_")
        df[["m15_bias", "m5_confirm"]] = df[["m15_bias", "m5_confirm"]].fillna(0).astype(int)

        atr_ = df["atr"]
        burst = df["velocity"].abs() >= c.velocity_atr_mult * atr_
        strong = df["body_ratio"] >= c.body_ratio_min
        vol_ok = df["tick_volume"] >= c.volume_spike_mult * df["vol_ma"]
        sane_bar = (df["high"] - df["low"]) <= c.max_bar_atr_mult * atr_
        no_gap = df["gap"] <= 0.5 * atr_
        common = burst & strong & vol_ok & sane_bar & no_gap & atr_.notna()

        long_ = (
            common
            & (df["m15_bias"] == 1)
            & (df["m5_confirm"] == 1)
            & (df["velocity"] > 0)
            & (df["close"] > df["open"])
            & (df["close_loc"] >= c.close_location_min)
            & (df["rsi"] <= c.rsi_long_max)
        )
        short = (
            common
            & (df["m15_bias"] == -1)
            & (df["m5_confirm"] == -1)
            & (df["velocity"] < 0)
            & (df["close"] < df["open"])
            & (df["close_loc"] <= 1 - c.close_location_min)
            & (df["rsi"] >= c.rsi_short_min)
        )
        df["signal"] = np.where(long_, 1, np.where(short, -1, 0)).astype(int)
        return df.reset_index(drop=True)

    # --------------------------------------------------------------- order
    def build_signal(
        self,
        row: pd.Series,
        symbol_cfg: SymbolConfig,
        spec: SymbolSpec,
        spread_price: float,
    ) -> Signal | None:
        c = self.cfg
        side = int(row["signal"])
        atr_ = float(row["atr"])
        if side == 0 or not np.isfinite(atr_) or atr_ <= 0:
            return None
        close = float(row["close"])  # bid
        buffer = symbol_cfg.stop_hunt_buffer_points * spec.point

        if side > 0:
            ref = close + spread_price  # current ask
            entry = ref - c.limit_pullback_atr * atr_ if c.entry_mode == "limit" else ref
            structure = entry - float(row["swing_low"]) + buffer
        else:
            ref = close  # current bid
            entry = ref + c.limit_pullback_atr * atr_ if c.entry_mode == "limit" else ref
            # shorts are stopped on the ASK -> add the spread on top of the bid swing
            structure = float(row["swing_high"]) + spread_price - entry + buffer

        dist = atr_stop_distance(
            atr_, c.sl_atr_mult, spec.point, c.sl_min_points, c.sl_max_points, max(structure, 0.0)
        )
        if dist is None:
            return None
        sl = entry - side * dist
        sl = avoid_round_levels(sl, side, symbol_cfg.stop_hunt_round_step, buffer)
        risk = abs(entry - sl)
        tp = entry + side * c.tp_r_multiple * risk

        digits = spec.digits
        return Signal(
            symbol=symbol_cfg.name,
            side=side,
            order_type=c.entry_mode,
            entry=round(entry, digits),
            stop_loss=round(sl, digits),
            take_profit=round(tp, digits),
            atr=atr_,
            time=pd.Timestamp(row["time"]),
            reason=f"velocity {row['velocity']:.2f} ({row['velocity'] / atr_:.1f}xATR) rsi {row['rsi']:.0f}",
            expiry_s=c.limit_expiry_s if c.entry_mode == "limit" else 0,
        )


STRATEGIES: dict[str, type[BaseStrategy]] = {XauVelocityScalp.name: XauVelocityScalp}


def create_strategy(cfg: StrategyConfig) -> BaseStrategy:
    try:
        return STRATEGIES[cfg.name](cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown strategy '{cfg.name}'. Available: {list(STRATEGIES)}") from exc

"""XAU Velocity & Micro-Pullback Scalper.

Pipeline (closed bars only, vectorised -> identical in live and backtest):

1. **Directional bias (M15)** - EMA 9 > EMA 21 with price above the slow EMA, confirmed by
   the slope of the daily VWAP (``ema_vwap``) or by higher-highs/higher-lows
   (``ema_structure``). Optional M5 EMA confirmation.
2. **Trigger (M1)** - a *Displacement* candle (body >= 1.5x ATR(14) and tick volume
   >= 1.5x its moving average) that either
   * follows a **liquidity sweep**: within the last K candles price pierced the
     high/low of the previous N candles (or an institutional key level) and closed
     back inside, or
   * **breaks a volatility squeeze** (Bollinger bands inside Keltner channels).
3. **Institutional levels** - PDH/PDL, Asian & London high/low, daily VWAP +-1/2 sigma
   (:mod:`src.core.levels`). Sweeps of these levels get absolute priority: full risk
   (``priority``) or exclusive trading (``required``); other setups trade at reduced risk.
4. **Smart entry anti-gap** - a LIMIT order at a 35-50% pullback of the impulse body,
   or at the near edge of the micro Fair Value Gap, valid ``order_expiration_candles``.
   Market mode is available (slippage capped by ``max_deviation_points``).
5. **Structural stop** - beyond the sweep extreme + ATR buffer, at least ``sl_min_points``,
   pushed away from round numbers (stop-hunt magnets). Shorts add the spread (stopped on ASK).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LevelsConfig, StrategyConfig, SymbolConfig, SymbolSpec
from src.core.clock import ServerClock
from src.core.data_feed import TF_MINUTES
from src.core.levels import LEVEL_LABELS, RESISTANCE_LEVELS, SUPPORT_LEVELS, compute_levels
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


def _daily_vwap(df: pd.DataFrame, day: pd.Series) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].astype(float).clip(lower=1.0)
    return (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()


def _first_level(flags: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """Name of the first (highest-priority) level whose flag is True, else ''."""
    names = list(flags)
    if not names:
        return pd.Series("", index=index)
    conds = [flags[n].fillna(False).to_numpy(dtype=bool) for n in names]
    return pd.Series(np.select(conds, [LEVEL_LABELS[n] for n in names], default=""), index=index)


class XauVelocityScalp(BaseStrategy):
    name = "xau_velocity_scalp"

    @property
    def required_timeframes(self) -> list[str]:
        tfs = list(self.timeframes)
        if self.levels_cfg.enabled and self.levels_cfg.timeframe not in tfs:
            tfs.append(self.levels_cfg.timeframe)
        return tfs

    # ------------------------------------------------------------ features
    def compute_features(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        c = self.cfg
        tf_e, tf_c, tf_t = self.timeframes
        m1, m5, m15 = frames[tf_e].copy(), frames[tf_c].copy(), frames[tf_t].copy()

        # --- 1. M15 bias: EMA 9/21 + VWAP slope or market structure
        m15["ema_f"] = ind.ema(m15["close"], c.bias_ema_fast)
        m15["ema_s"] = ind.ema(m15["close"], c.bias_ema_slow)
        bull = (m15["ema_f"] > m15["ema_s"]) & (m15["close"] > m15["ema_s"])
        bear = (m15["ema_f"] < m15["ema_s"]) & (m15["close"] < m15["ema_s"])
        if c.bias_mode == "ema_vwap":
            day = self.clock.series_to_utc(m15["time"]).dt.floor("D")
            slope = _daily_vwap(m15, day).groupby(day).diff(c.vwap_slope_bars)
            bull &= slope > 0
            bear &= slope < 0
        elif c.bias_mode == "ema_structure":
            n = c.structure_bars
            hi_now, hi_prev = m15["high"].rolling(n).max(), m15["high"].shift(n).rolling(n).max()
            lo_now, lo_prev = m15["low"].rolling(n).min(), m15["low"].shift(n).rolling(n).min()
            bull &= (hi_now > hi_prev) & (lo_now > lo_prev)
            bear &= (hi_now < hi_prev) & (lo_now < lo_prev)
        m15["bias"] = np.select([bull, bear], [1, -1], 0)

        # --- optional M5 confirmation
        m5["ema_f"] = ind.ema(m5["close"], c.confirm_ema_fast)
        m5["ema_s"] = ind.ema(m5["close"], c.confirm_ema_slow)
        m5["confirm"] = np.select(
            [m5["ema_f"] > m5["ema_s"], m5["ema_f"] < m5["ema_s"]], [1, -1], 0
        )

        # --- 2. M1 displacement / sweep / squeeze
        atr = ind.atr(m1, c.atr_period)
        m1["atr"] = atr
        body = m1["close"] - m1["open"]
        m1["body_atr"] = body / atr
        m1["vol_ma"] = m1["tick_volume"].rolling(c.volume_ma_period).mean().shift(1)
        m1["vol_ratio"] = m1["tick_volume"] / m1["vol_ma"]
        vol_ok = m1["vol_ratio"] >= c.displacement_volume_mult
        m1["disp_bull"] = (body >= c.displacement_body_atr * atr) & vol_ok
        m1["disp_bear"] = (-body >= c.displacement_body_atr * atr) & vol_ok

        n, k = c.sweep_lookback, c.sweep_window
        prior_low = m1["low"].shift(1).rolling(n).min()
        prior_high = m1["high"].shift(1).rolling(n).max()
        sweep_low_bar = (m1["low"] < prior_low) & (m1["close"] > prior_low)
        sweep_high_bar = (m1["high"] > prior_high) & (m1["close"] < prior_high)
        m1["sweep_bull"] = sweep_low_bar.astype(float).rolling(k, min_periods=1).max() > 0
        m1["sweep_bear"] = sweep_high_bar.astype(float).rolling(k, min_periods=1).max() > 0

        p = c.squeeze_period
        mid = m1["close"].rolling(p).mean()
        sd = m1["close"].rolling(p).std(ddof=0)
        bb_up, bb_dn = mid + c.squeeze_bb_std * sd, mid - c.squeeze_bb_std * sd
        kc_mid = ind.ema(m1["close"], p)
        kc_up, kc_dn = kc_mid + c.squeeze_kc_atr_mult * atr, kc_mid - c.squeeze_kc_atr_mult * atr
        sq_on = ((bb_up < kc_up) & (bb_dn > kc_dn)).astype(float)
        sq_recent = sq_on.shift(1).rolling(c.squeeze_window, min_periods=1).max() > 0
        m1["squeeze_bull"] = sq_recent & (m1["close"] > bb_up)
        m1["squeeze_bear"] = sq_recent & (m1["close"] < bb_dn)

        m1["struct_low"] = m1["low"].rolling(k, min_periods=1).min()
        m1["struct_high"] = m1["high"].rolling(k, min_periods=1).max()
        m1["fvg_bull_top"] = m1["low"].where(m1["low"] > m1["high"].shift(2))
        m1["fvg_bear_bottom"] = m1["high"].where(m1["high"] < m1["low"].shift(2))
        m1["velocity"] = m1["close"] - m1["close"].shift(3)
        m1["momentum"] = m1["velocity"] / atr
        # trend-continuation "pullback" trigger: touch of the M1 EMA, then resumption candle
        ema_pb = ind.ema(m1["close"], c.pullback_ema)
        tol = c.pullback_touch_atr * atr
        w = c.pullback_window
        touch_bull = (m1["low"] <= ema_pb + tol).astype(float).rolling(w, min_periods=1).max() > 0
        touch_bear = (m1["high"] >= ema_pb - tol).astype(float).rolling(w, min_periods=1).max() > 0
        ema_up = ema_pb > ema_pb.shift(3)
        ema_dn = ema_pb < ema_pb.shift(3)
        m1["pb_bull"] = (
            touch_bull & ema_up & (m1["close"] > ema_pb) & (body >= c.pullback_body_atr * atr)
            & (m1["close"] > m1["high"].shift(1))
        )  # fmt: skip
        m1["pb_bear"] = (
            touch_bear & ema_dn & (m1["close"] < ema_pb) & (-body >= c.pullback_body_atr * atr)
            & (m1["close"] < m1["low"].shift(1))
        )  # fmt: skip
        m1["sane"] = (m1["high"] - m1["low"]) <= c.max_bar_atr_mult * atr
        m1["no_gap"] = (m1["open"] - m1["close"].shift(1)).abs() <= c.max_open_gap_atr * atr

        df = ind.align_htf(m1, tf_e, m15, tf_t, ["bias"], "m15_")
        df = ind.align_htf(df, tf_e, m5, tf_c, ["confirm"], "m5_")
        df[["m15_bias", "m5_confirm"]] = df[["m15_bias", "m5_confirm"]].fillna(0).astype(int)

        # --- 3. institutional key levels (causal, from closed levels-TF bars)
        lv = self.levels_cfg
        df["key_level_bull"] = ""
        df["key_level_bear"] = ""
        if lv.enabled and lv.timeframe in frames:
            levels = compute_levels(frames[lv.timeframe], lv, self.clock)
            cols = [col for col in levels.columns if col != "time"]
            df = ind.align_htf(df, tf_e, levels, lv.timeframe, cols, "lv_")
            tol = lv.sweep_tolerance_atr * df["atr"]
            bull_flags: dict[str, pd.Series] = {}
            bear_flags: dict[str, pd.Series] = {}
            for name in SUPPORT_LEVELS:
                lvl = df[f"lv_{name}"]
                hit = (df["low"] <= lvl + tol) & (df["close"] > lvl)
                bull_flags[name] = hit.astype(float).rolling(k, min_periods=1).max() > 0
            for name in RESISTANCE_LEVELS:
                lvl = df[f"lv_{name}"]
                hit = (df["high"] >= lvl - tol) & (df["close"] < lvl)
                bear_flags[name] = hit.astype(float).rolling(k, min_periods=1).max() > 0
            df["key_level_bull"] = _first_level(bull_flags, df.index)
            df["key_level_bear"] = _first_level(bear_flags, df.index)

        key_bull = df["key_level_bull"] != ""
        key_bear = df["key_level_bear"] != ""
        sweep_ctx_bull = df["sweep_bull"] | key_bull
        sweep_ctx_bear = df["sweep_bear"] | key_bear
        mode = c.trigger_mode
        if mode in ("sweep", "pullback"):
            ctx_bull, ctx_bear = sweep_ctx_bull, sweep_ctx_bear
        elif mode == "squeeze":
            ctx_bull, ctx_bear = df["squeeze_bull"], df["squeeze_bear"]
        else:  # any | all
            ctx_bull = sweep_ctx_bull | df["squeeze_bull"]
            ctx_bear = sweep_ctx_bear | df["squeeze_bear"]
        breakout_bull = df["disp_bull"] & ctx_bull
        breakout_bear = df["disp_bear"] & ctx_bear
        pb_bull, pb_bear = df["pb_bull"].astype(bool), df["pb_bear"].astype(bool)
        if mode == "pullback":
            trig_bull, trig_bear = pb_bull, pb_bear
        elif mode == "all":
            trig_bull, trig_bear = breakout_bull | pb_bull, breakout_bear | pb_bear
        else:
            trig_bull, trig_bear = breakout_bull, breakout_bear
        if lv.enabled and lv.filter_mode == "required":
            trig_bull = trig_bull & key_bull
            trig_bear = trig_bear & key_bear
        df["breakout_bull"], df["breakout_bear"] = breakout_bull, breakout_bear
        df["trig_bull"], df["trig_bear"] = trig_bull, trig_bear

        ones = pd.Series(True, index=df.index)
        confirm_bull = (df["m5_confirm"] == 1) if c.confirm_enabled else ones
        confirm_bear = (df["m5_confirm"] == -1) if c.confirm_enabled else ones
        base = df["atr"].notna() & df["sane"].astype(bool) & df["no_gap"].astype(bool)
        long_ = base & trig_bull & (df["m15_bias"] == 1) & confirm_bull
        short = base & trig_bear & (df["m15_bias"] == -1) & confirm_bear
        df["signal"] = np.where(long_, 1, np.where(short, -1, 0)).astype(int)
        is_long = df["signal"] > 0
        brk = np.where(is_long, breakout_bull, breakout_bear)
        swp = np.where(is_long, sweep_ctx_bull, sweep_ctx_bear)
        df["trigger"] = np.where(~brk, "pullback", np.where(swp, "sweep", "squeeze"))
        return df.reset_index(drop=True)

    # -------------------------------------------------------------- funnel
    def signal_funnel(self, df: pd.DataFrame) -> list[tuple[str, int, int]]:
        """How many bars survive each filter, cumulatively (long, short).

        Shows which condition is starving the strategy of signals.
        """
        c = self.cfg
        atr = df["atr"]
        body = df["close"] - df["open"]
        valid = atr.notna() & df["vol_ma"].notna()
        rows: list[tuple[str, int, int]] = [
            ("M1 bars (valid ATR)", int(valid.sum()), int(valid.sum()))
        ]

        def cnt(a: pd.Series, b: pd.Series) -> tuple[int, int]:
            return int((a.fillna(False).astype(bool) & valid).sum()), int(
                (b.fillna(False).astype(bool) & valid).sum()
            )

        if c.trigger_mode != "pullback":
            d_l, d_s = body >= c.displacement_body_atr * atr, -body >= c.displacement_body_atr * atr
            rows.append((f"  breakout: body >= {c.displacement_body_atr}x ATR", *cnt(d_l, d_s)))
            v = df["vol_ratio"] >= c.displacement_volume_mult
            rows.append((f"  breakout: + tick volume >= {c.displacement_volume_mult}x MA",
                         *cnt(d_l & v, d_s & v)))  # fmt: skip
            rows.append(("  breakout: + sweep/squeeze/key-level context",
                         *cnt(df["breakout_bull"], df["breakout_bear"])))  # fmt: skip
        if c.trigger_mode in ("pullback", "all"):
            rows.append((f"  pullback: EMA{c.pullback_ema} touch + resumption",
                         *cnt(df["pb_bull"], df["pb_bear"])))  # fmt: skip
        acc_l, acc_s = df["trig_bull"].astype(bool) & valid, df["trig_bear"].astype(bool) & valid
        rows.append((f"TRIGGER ({c.trigger_mode})", int(acc_l.sum()), int(acc_s.sum())))
        clean = df["sane"].astype(bool) & df["no_gap"].astype(bool)
        steps = [
            ("+ no news bar / no open gap", clean, clean),
            (f"+ M15 bias ({c.bias_mode})", df["m15_bias"] == 1, df["m15_bias"] == -1),
            ("+ M5 confirmation" + ("" if c.confirm_enabled else " (disabled)"),
             (df["m5_confirm"] == 1) | (not c.confirm_enabled),
             (df["m5_confirm"] == -1) | (not c.confirm_enabled)),
        ]  # fmt: skip
        for label, cond_l, cond_s in steps:
            acc_l &= cond_l.fillna(False).astype(bool)
            acc_s &= cond_s.fillna(False).astype(bool)
            rows.append((label, int(acc_l.sum()), int(acc_s.sum())))
        rows.append(("= SIGNALS", int((df["signal"] == 1).sum()), int((df["signal"] == -1).sum())))
        return rows

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
        atr = float(row["atr"])
        if side == 0 or not np.isfinite(atr) or atr <= 0:
            return None
        o, close = float(row["open"]), float(row["close"])  # BID prices
        pt = spec.point
        round_buffer = symbol_cfg.stop_hunt_buffer_points * pt

        if side > 0:
            if c.entry_mode == "market":
                entry = close + spread_price  # current ASK
            else:
                level = close - c.pullback_pct * (close - o)  # 35-50% of the impulse body
                fvg = row.get("fvg_bull_top")
                if c.limit_anchor == "fvg" and pd.notna(fvg) and float(fvg) < close:
                    level = float(fvg)  # near edge of the bullish FVG
                entry = level + spread_price  # BUY LIMIT triggers on ASK
            sl = float(row["struct_low"]) - c.sl_atr_buffer * atr
            key = row.get("key_level_bull") or None
        else:
            if c.entry_mode == "market":
                entry = close
            else:
                level = close + c.pullback_pct * (o - close)
                fvg = row.get("fvg_bear_bottom")
                if c.limit_anchor == "fvg" and pd.notna(fvg) and float(fvg) > close:
                    level = float(fvg)
                entry = level  # SELL LIMIT triggers on BID
            sl = float(row["struct_high"]) + spread_price + c.sl_atr_buffer * atr
            key = row.get("key_level_bear") or None

        sl_min = (
            symbol_cfg.sl_min_points if symbol_cfg.sl_min_points is not None else c.sl_min_points
        )
        sl_max = (
            symbol_cfg.sl_max_points if symbol_cfg.sl_max_points is not None else c.sl_max_points
        )
        dist = side * (entry - sl)
        if dist < sl_min * pt:  # never inside tick noise
            sl = entry - side * sl_min * pt
        elif dist > sl_max * pt:
            return None
        sl = avoid_round_levels(sl, side, symbol_cfg.stop_hunt_round_step, round_buffer)
        risk = abs(entry - sl)
        tp = entry + side * c.runner_tp_r * risk if c.runner_tp_r > 0 else 0.0

        risk_mult = 1.0
        if self.levels_cfg.enabled and self.levels_cfg.filter_mode == "priority" and not key:
            risk_mult = self.levels_cfg.non_key_risk_mult

        expiry = c.order_expiration_candles * TF_MINUTES[self.entry_tf] * 60
        reason = (
            f"{row.get('trigger', '?')} disp {abs(float(row['body_atr'])):.1f}xATR "
            f"vol {float(row['vol_ratio']):.1f}x" + (f" @ {key}" if key else "")
        )
        d = spec.digits
        return Signal(
            symbol=symbol_cfg.name,
            side=side,
            order_type="market" if c.entry_mode == "market" else "limit",
            entry=round(entry, d),
            stop_loss=round(sl, d),
            take_profit=round(tp, d),
            atr=atr,
            time=pd.Timestamp(row["time"]),
            reason=reason,
            expiry_s=0 if c.entry_mode == "market" else expiry,
            key_level=key,
            risk_mult=risk_mult,
        )


STRATEGIES: dict[str, type[BaseStrategy]] = {XauVelocityScalp.name: XauVelocityScalp}


def create_strategy(
    cfg: StrategyConfig,
    levels_cfg: LevelsConfig | None = None,
    clock: ServerClock | None = None,
) -> BaseStrategy:
    try:
        return STRATEGIES[cfg.name](cfg, levels_cfg, clock)
    except KeyError as exc:
        raise ValueError(f"Unknown strategy '{cfg.name}'. Available: {list(STRATEGIES)}") from exc

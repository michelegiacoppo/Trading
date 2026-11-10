"""Dynamic lot sizing: equity x risk% / (SL distance in money per lot).

Design rule: **never round up into more risk**. Volumes are floored to the
broker's ``volume_step``; if the result is below ``volume_min`` the trade is
refused (0 lots) instead of being forced to the minimum lot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import RiskConfig, SymbolSpec


@dataclass(frozen=True)
class SizingResult:
    lots: float
    risk_money: float  # money at risk with the returned lots (incl. commission)
    risk_pct_effective: float  # actual % of equity risked
    loss_per_lot: float  # money lost per 1.0 lot if SL is hit
    reason: str = "ok"

    @property
    def ok(self) -> bool:
        return self.lots > 0


def floor_to_step(volume: float, step: float) -> float:
    if step <= 0:
        return volume
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(volume / step + 1e-9) * step, decimals)


def money_per_lot(distance_price: float, spec: SymbolSpec) -> float:
    """Money P/L of 1.0 lot for a price move of ``distance_price``."""
    if spec.tick_size <= 0:
        raise ValueError(f"{spec.name}: invalid tick_size {spec.tick_size}")
    return abs(distance_price) / spec.tick_size * spec.tick_value


class PositionSizer:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def size(
        self,
        equity: float,
        entry: float,
        stop_loss: float,
        spec: SymbolSpec,
        risk_pct: float | None = None,
        spread_price: float = 0.0,
        free_margin: float | None = None,
        margin_per_lot: float | None = None,
    ) -> SizingResult:
        """Compute lots so that a stop-out costs ``risk_pct`` of equity.

        ``spread_price`` is added to the SL distance (you pay it on exit), and the
        round-turn commission is added to the per-lot loss.
        """
        risk_pct = self.cfg.risk_per_trade_pct if risk_pct is None else risk_pct
        if equity <= 0:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "equity <= 0")
        distance = abs(entry - stop_loss) + max(spread_price, 0.0)
        if distance <= 0:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "zero stop distance")

        loss_per_lot = money_per_lot(distance, spec) + self.cfg.commission_per_lot
        risk_budget = equity * risk_pct / 100.0
        raw_lots = risk_budget / loss_per_lot

        cap = min(spec.volume_max, self.cfg.max_lot)
        if free_margin is not None and margin_per_lot and margin_per_lot > 0:
            usable = free_margin * self.cfg.max_margin_usage_pct / 100.0
            cap = min(cap, usable / margin_per_lot)

        lots = floor_to_step(min(raw_lots, cap), spec.volume_step)
        if lots < spec.volume_min:
            return SizingResult(
                0.0,
                0.0,
                0.0,
                loss_per_lot,
                f"size {raw_lots:.4f} < volume_min {spec.volume_min} (risk too small / SL too wide)",
            )
        risk_money = lots * loss_per_lot
        return SizingResult(lots, risk_money, risk_money / equity * 100.0, loss_per_lot)


def atr_stop_distance(
    atr: float,
    atr_mult: float,
    point: float,
    min_points: float,
    max_points: float,
    structure_distance: float = 0.0,
) -> float | None:
    """SL distance in price = max(ATR*k, structural swing, min). ``None`` if > max."""
    dist = max(atr * atr_mult, structure_distance, min_points * point)
    if dist > max_points * point:
        return None
    return dist

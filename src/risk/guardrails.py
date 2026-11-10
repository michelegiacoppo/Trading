"""Protective layer that sits between signals and execution.

Every component is pure Python with an injectable clock so it can be unit
tested and reused unchanged by the backtester:

* :class:`SpreadFilter`     - absolute cap + rolling-median + ATR-relative spread check
* :class:`GapDetector`      - bar gaps and tick-to-tick liquidity holes
* :class:`SlippageMonitor`  - realised slippage tracking with auto-pause
* :class:`SessionFilter`    - trading windows, rollover, weekend, news blackouts
* :class:`DailyKillSwitch`  - daily loss / trailing DD / consecutive losses / trade count
* :class:`Guardrails`       - facade returning a single :class:`GuardDecision`
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from src.config import GuardrailConfig, RiskConfig, SessionConfig, SymbolConfig

log = logging.getLogger(__name__)


@dataclass
class GuardDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.allowed = False
        self.reasons.append(reason)

    def __str__(self) -> str:
        return "OK" if self.allowed else "; ".join(self.reasons)


# ============================================================== spread
class SpreadFilter:
    def __init__(self, symbol_cfg: SymbolConfig, cfg: GuardrailConfig) -> None:
        self.sym = symbol_cfg
        self.cfg = cfg
        self._samples: deque[float] = deque(maxlen=cfg.spread_window)

    def update(self, spread_points: float) -> None:
        self._samples.append(float(spread_points))

    @property
    def median(self) -> float | None:
        if len(self._samples) < self.cfg.spread_min_samples:
            return None
        return statistics.median(self._samples)

    def check(self, spread_points: float, atr_points: float | None = None) -> tuple[bool, str]:
        if spread_points > self.sym.max_spread_points:
            return False, f"spread {spread_points:.0f}p > cap {self.sym.max_spread_points:.0f}p"
        med = self.median
        if med is not None and med > 0 and spread_points > med * self.sym.spread_median_mult:
            return (
                False,
                f"spread spike {spread_points:.0f}p > {self.sym.spread_median_mult}x median {med:.0f}p",
            )
        if atr_points and atr_points > 0:
            ratio = spread_points / atr_points
            if ratio > self.sym.spread_atr_ratio_max:
                return False, f"spread/ATR {ratio:.2f} > {self.sym.spread_atr_ratio_max:.2f}"
        return True, "ok"


# ================================================================= gap
class GapDetector:
    """Blocks entries after a bar gap or an abnormal tick jump (thin liquidity)."""

    def __init__(self, cfg: GuardrailConfig) -> None:
        self.cfg = cfg
        self._bars_left = 0
        self._tick_block_until: datetime | None = None
        self._last_mid: float | None = None
        self.last_event: str = ""

    def on_bar(self, open_: float, prev_close: float, atr: float) -> bool:
        """Call on every new closed bar. Returns True if a gap was detected."""
        if self._bars_left > 0:
            self._bars_left -= 1
        if atr <= 0:
            return False
        gap = abs(open_ - prev_close)
        if gap > self.cfg.gap_atr_mult * atr:
            self._bars_left = self.cfg.gap_cooldown_bars
            self.last_event = f"bar gap {gap:.2f} > {self.cfg.gap_atr_mult}xATR"
            log.warning(self.last_event)
            return True
        return False

    def on_tick(self, mid: float, atr: float, now: datetime) -> bool:
        jumped = False
        if self._last_mid is not None and atr > 0:
            jump = abs(mid - self._last_mid)
            if jump > self.cfg.tick_jump_atr_mult * atr:
                self._tick_block_until = now + timedelta(seconds=self.cfg.tick_jump_cooldown_s)
                self.last_event = f"tick jump {jump:.2f} > {self.cfg.tick_jump_atr_mult}xATR"
                log.warning(self.last_event)
                jumped = True
        self._last_mid = mid
        return jumped

    def check(self, now: datetime) -> tuple[bool, str]:
        if self._bars_left > 0:
            return False, f"gap cooldown ({self._bars_left} bars)"
        if self._tick_block_until and now < self._tick_block_until:
            return False, "tick-jump cooldown"
        return True, "ok"


# ============================================================ slippage
class SlippageMonitor:
    """Tracks adverse slippage (points). Pauses trading when the rolling mean is too high."""

    def __init__(self, cfg: GuardrailConfig) -> None:
        self.cfg = cfg
        self._samples: deque[float] = deque(maxlen=cfg.slippage_window)
        self._paused_until: datetime | None = None

    def record(
        self, requested: float, filled: float, is_buy: bool, point: float, now: datetime
    ) -> float:
        """Positive value = adverse slippage in points."""
        slip = ((filled - requested) if is_buy else (requested - filled)) / point
        self._samples.append(slip)
        avg = self.average
        if len(self._samples) >= 3 and avg > self.cfg.max_avg_slippage_points:
            self._paused_until = now + timedelta(seconds=self.cfg.slippage_pause_s)
            log.warning(
                "Avg slippage %.1fp > %.1fp -> pause", avg, self.cfg.max_avg_slippage_points
            )
        return slip

    @property
    def average(self) -> float:
        return statistics.fmean(self._samples) if self._samples else 0.0

    def check(self, now: datetime) -> tuple[bool, str]:
        if self._paused_until and now < self._paused_until:
            return False, f"slippage pause until {self._paused_until:%H:%M:%S}"
        return True, "ok"


# ============================================================ sessions
def _t(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _in_window(t: time, start: time, end: time) -> bool:
    """[start, end) with support for windows crossing midnight."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end


class SessionFilter:
    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        self.windows = [(_t(a), _t(b)) for a, b in cfg.windows]
        self.rollover = (
            tuple(_t(x) for x in cfg.rollover_blackout) if cfg.rollover_blackout else None
        )
        self.friday_cutoff = _t(cfg.friday_cutoff) if cfg.friday_cutoff else None
        self.monday_start = _t(cfg.monday_start) if cfg.monday_start else None
        self.news: list[datetime] = [datetime.fromisoformat(x) for x in cfg.news_blackouts]

    def check(self, now: datetime) -> tuple[bool, str]:
        t = now.time()
        wd = now.weekday()
        if wd >= 5:
            return False, "weekend"
        if wd == 4 and self.friday_cutoff and t >= self.friday_cutoff:
            return False, "friday cutoff"
        if wd == 0 and self.monday_start and t < self.monday_start:
            return False, "monday open (gap risk)"
        if self.rollover and _in_window(t, self.rollover[0], self.rollover[1]):
            return False, "rollover blackout"
        before = timedelta(minutes=self.cfg.news_blackout_minutes_before)
        after = timedelta(minutes=self.cfg.news_blackout_minutes_after)
        for ev in self.news:
            if ev - before <= now <= ev + after:
                return False, f"news blackout {ev:%H:%M}"
        if self.windows and not any(_in_window(t, a, b) for a, b in self.windows):
            return False, "outside session"
        return True, "ok"


# ========================================================= kill switch
class DailyKillSwitch:
    """Daily circuit breaker. Once tripped it stays tripped until the next server day."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.day: date | None = None
        self.start_equity = 0.0
        self.peak_equity = 0.0
        self.realized_pnl = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.last_loss_time: datetime | None = None
        self.tripped_reason: str | None = None

    def new_day(self, day: date, equity: float) -> None:
        self.day = day
        self.start_equity = self.peak_equity = equity
        self.realized_pnl = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.tripped_reason = None
        log.debug("Kill-switch reset for %s, start equity %.2f", day, equity)

    def update_equity(self, equity: float, now: datetime) -> None:
        if self.day != now.date():
            self.new_day(now.date(), equity)
        self.peak_equity = max(self.peak_equity, equity)
        if self.tripped_reason or self.start_equity <= 0:
            return
        day_pnl_pct = (equity - self.start_equity) / self.start_equity * 100.0
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100.0
        if day_pnl_pct <= -self.cfg.daily_loss_limit_pct:
            self.trip(f"daily loss {day_pnl_pct:.2f}% <= -{self.cfg.daily_loss_limit_pct}%")
        elif dd_pct >= self.cfg.daily_trailing_dd_pct:
            self.trip(f"intraday DD {dd_pct:.2f}% >= {self.cfg.daily_trailing_dd_pct}%")
        elif self.cfg.daily_profit_lock_pct > 0 and day_pnl_pct >= self.cfg.daily_profit_lock_pct:
            self.trip(f"profit lock reached +{day_pnl_pct:.2f}%")

    def record_entry(self) -> None:
        """Count a new trade (entries, not partial exits, drive ``max_trades_per_day``)."""
        self.trades += 1

    def record_close(self, pnl: float, now: datetime) -> None:
        """Register a realised exit (full or partial)."""
        self.realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = now
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.trip(f"{self.consecutive_losses} consecutive losses")

    def trip(self, reason: str) -> None:
        if not self.tripped_reason:
            self.tripped_reason = reason
            log.critical("KILL-SWITCH TRIPPED: %s", reason)

    @property
    def tripped(self) -> bool:
        return self.tripped_reason is not None

    @property
    def day_pnl_pct(self) -> float:
        return 0.0 if self.start_equity <= 0 else self.realized_pnl / self.start_equity * 100

    def check(self, now: datetime) -> tuple[bool, str]:
        if self.tripped_reason:
            return False, f"KILL-SWITCH: {self.tripped_reason}"
        if self.trades >= self.cfg.max_trades_per_day:
            return False, f"max trades/day {self.cfg.max_trades_per_day}"
        if self.last_loss_time and now < self.last_loss_time + timedelta(
            seconds=self.cfg.cooldown_after_loss_s
        ):
            return False, "post-loss cooldown"
        return True, "ok"


# ============================================================== facade
@dataclass
class EntryContext:
    symbol: str
    now: datetime
    spread_points: float
    atr_points: float | None = None
    last_tick_age_s: float = 0.0
    open_positions_total: int = 0
    open_positions_symbol: int = 0


class Guardrails:
    def __init__(
        self,
        symbols: dict[str, SymbolConfig],
        guard_cfg: GuardrailConfig,
        risk_cfg: RiskConfig,
        session_cfg: SessionConfig,
    ) -> None:
        self.guard_cfg = guard_cfg
        self.risk_cfg = risk_cfg
        self.spread = {s: SpreadFilter(c, guard_cfg) for s, c in symbols.items()}
        self.gaps = {s: GapDetector(guard_cfg) for s in symbols}
        self.slippage = SlippageMonitor(guard_cfg)
        self.session = SessionFilter(session_cfg)
        self.kill_switch = DailyKillSwitch(risk_cfg)

    def check_entry(self, ctx: EntryContext) -> GuardDecision:
        d = GuardDecision(True)
        checks = (
            self.kill_switch.check(ctx.now),
            self.session.check(ctx.now),
            self.spread[ctx.symbol].check(ctx.spread_points, ctx.atr_points),
            self.gaps[ctx.symbol].check(ctx.now),
            self.slippage.check(ctx.now),
        )
        for ok, reason in checks:
            if not ok:
                d.block(reason)
        if ctx.last_tick_age_s > self.guard_cfg.stale_tick_s:
            d.block(f"stale feed ({ctx.last_tick_age_s:.0f}s)")
        if ctx.open_positions_total >= self.risk_cfg.max_open_positions:
            d.block("max open positions")
        if ctx.open_positions_symbol >= self.risk_cfg.max_positions_per_symbol:
            d.block("max positions on symbol")
        return d

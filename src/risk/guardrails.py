"""Protective layer that sits between signals and execution.

Every component is pure Python with an injectable clock (``now`` is always a
broker-server-time argument) so it is unit tested and reused unchanged by the
backtester:

* :class:`SpreadFilter`     - absolute cap + rolling-median spike + spread/ATR ratio
* :class:`GapDetector`      - bar gaps (open vs previous close) and tick jumps -> N-minute block
* :class:`SlippageMonitor`  - realised slippage tracking with auto-pause
* :class:`SessionFilter`    - London/NY windows in CET, rollover, weekend, news, blocked slots
* :class:`DailyKillSwitch`  - daily loss (closed + floating), profit target, trailing DD,
  consecutive losses -> ``LOCKED`` until the next day (flatten requested once)
* :class:`Guardrails`       - facade returning a single :class:`GuardDecision`
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from src.config import GuardrailConfig, RiskConfig, SessionConfig, SymbolConfig
from src.core.clock import ServerClock

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
            return False, f"spread {spread_points:.0f}p > max {self.sym.max_spread_points:.0f}p"
        med = self.median
        if med is not None and med > 0 and spread_points > med * self.sym.spread_median_mult:
            return False, (
                f"spread spike {spread_points:.0f}p > {self.sym.spread_median_mult}x median {med:.0f}p"
            )
        if atr_points and atr_points > 0:
            ratio = spread_points / atr_points
            if ratio > self.sym.spread_atr_ratio_max:
                return False, f"spread/ATR {ratio:.2f} > {self.sym.spread_atr_ratio_max:.2f}"
        return True, "ok"


# ================================================================= gap
class GapDetector:
    """Blocks new entries for ``gap_block_minutes`` after a gap or a liquidity hole."""

    def __init__(self, cfg: GuardrailConfig) -> None:
        self.cfg = cfg
        self.blocked_until: datetime | None = None
        self._last_mid: float | None = None
        self.last_event: str = ""

    def _block(self, now: datetime, msg: str) -> None:
        self.blocked_until = now + timedelta(minutes=self.cfg.gap_block_minutes)
        self.last_event = msg
        log.info("%s -> entries blocked until %s", msg, f"{self.blocked_until:%H:%M}")

    def on_bar(self, open_: float, prev_close: float, atr: float, now: datetime) -> bool:
        """New closed bar: anomalous gap between previous Close and this Open?"""
        if atr <= 0:
            return False
        gap = abs(open_ - prev_close)
        if gap > self.cfg.gap_atr_mult * atr:
            self._block(now, f"bar gap {gap:.2f} > {self.cfg.gap_atr_mult}xATR")
            return True
        return False

    def on_tick(self, mid: float, atr: float, now: datetime) -> bool:
        """Every tick: jump from the previous tick (news / illiquidity)."""
        jumped = False
        if self._last_mid is not None and atr > 0:
            jump = abs(mid - self._last_mid)
            if jump > self.cfg.tick_jump_atr_mult * atr:
                self._block(now, f"tick jump {jump:.2f} > {self.cfg.tick_jump_atr_mult}xATR")
                jumped = True
        self._last_mid = mid
        return jumped

    def check(self, now: datetime) -> tuple[bool, str]:
        if self.blocked_until and now < self.blocked_until:
            return False, f"gap block until {self.blocked_until:%H:%M}"
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
    """All rules are evaluated in ``cfg.timezone`` (CET by default)."""

    def __init__(self, cfg: SessionConfig, clock: ServerClock | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or ServerClock.from_config(cfg.server_timezone)
        self.windows = {name: (_t(a), _t(b)) for name, (a, b) in (cfg.windows or {}).items()}
        self.rollover = (
            tuple(_t(x) for x in cfg.rollover_blackout) if cfg.rollover_blackout else None
        )
        self.friday_cutoff = _t(cfg.friday_cutoff) if cfg.friday_cutoff else None
        self.monday_start = _t(cfg.monday_start) if cfg.monday_start else None
        self.blocked_slots = {str(x) for x in (cfg.blocked_slots or [])}
        self.news: list[datetime] = [datetime.fromisoformat(x) for x in cfg.news_blackouts]

    def local(self, now_server: datetime) -> datetime:
        return self.clock.to_zone(now_server, self.cfg.timezone)

    def active_session(self, now_server: datetime) -> str:
        t = self.local(now_server).time()
        for name, (a, b) in self.windows.items():
            if _in_window(t, a, b):
                return name
        return "Closed"

    def check(self, now_server: datetime) -> tuple[bool, str]:
        now = self.local(now_server)
        t, wd = now.time(), now.weekday()
        if wd >= 5:
            return False, "weekend"
        if wd == 4 and self.friday_cutoff and t >= self.friday_cutoff:
            return False, "friday cutoff"
        if wd == 0 and self.monday_start and t < self.monday_start:
            return False, "monday open (gap risk)"
        if self.rollover and _in_window(t, self.rollover[0], self.rollover[1]):
            return False, "rollover blackout"
        slot = f"{now.hour:02d}:{now.minute // 30 * 30:02d}"
        if slot in self.blocked_slots:
            return False, f"blocked slot {slot}"
        before = timedelta(minutes=self.cfg.news_blackout_minutes_before)
        after = timedelta(minutes=self.cfg.news_blackout_minutes_after)
        for ev in self.news:
            if ev - before <= now <= ev + after:
                return False, f"news blackout {ev:%H:%M}"
        if self.windows and not any(_in_window(t, a, b) for a, b in self.windows.values()):
            return False, "outside session"
        return True, "ok"


# ========================================================= kill switch
class DailyKillSwitch:
    """Daily circuit breaker. ``ACTIVE`` -> ``LOCKED`` until the next server day.

    When it locks, ``needs_flatten`` becomes True once: the caller must close
    all positions and cancel pending orders, then call :meth:`flatten_done`.
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.day: date | None = None
        self.start_equity = 0.0
        self.peak_equity = 0.0
        self.last_equity = 0.0
        self.realized_pnl = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.last_loss_time: datetime | None = None
        self.locked_reason: str | None = None
        self.needs_flatten = False

    @property
    def status(self) -> str:
        return "LOCKED" if self.locked_reason else "ACTIVE"

    @property
    def tripped(self) -> bool:
        return self.locked_reason is not None

    @property
    def tripped_reason(self) -> str | None:  # backward-compatible alias
        return self.locked_reason

    def new_day(self, day: date, equity: float) -> None:
        self.day = day
        self.start_equity = self.peak_equity = self.last_equity = equity
        self.realized_pnl = 0.0
        self.trades = 0
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.locked_reason = None
        self.needs_flatten = False
        log.debug("Kill-switch reset for %s, start equity %.2f", day, equity)

    def update_equity(self, equity: float, now: datetime) -> None:
        """``equity`` must include floating P/L (closed + open)."""
        if self.day != now.date():
            self.new_day(now.date(), equity)
        self.last_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.locked_reason or self.start_equity <= 0:
            return
        pnl_pct = self.day_pnl_pct
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100.0
        if pnl_pct <= -self.cfg.max_daily_loss_pct:
            self.lock(f"daily loss {pnl_pct:.2f}% <= -{self.cfg.max_daily_loss_pct}%")
        elif self.cfg.daily_trailing_dd_pct > 0 and dd_pct >= self.cfg.daily_trailing_dd_pct:
            self.lock(f"intraday DD {dd_pct:.2f}% >= {self.cfg.daily_trailing_dd_pct}%")
        elif self.cfg.daily_profit_target_pct > 0 and pnl_pct >= self.cfg.daily_profit_target_pct:
            self.lock(f"daily profit target +{pnl_pct:.2f}% reached")

    def record_entry(self) -> None:
        """Count a new trade (entries, not partial exits, drive ``max_trades_per_day``)."""
        self.trades += 1

    def record_close(self, pnl: float, now: datetime, final: bool = True) -> None:
        """Register a realised exit. Partial exits (``final=False``) don't touch the loss streak."""
        self.realized_pnl += pnl
        if not final:
            return
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = now
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.lock(f"{self.consecutive_losses} consecutive losses")

    def lock(self, reason: str) -> None:
        if not self.locked_reason:
            self.locked_reason = reason
            self.needs_flatten = self.cfg.flatten_on_lock
            log.critical("KILL-SWITCH LOCKED: %s", reason)

    def flatten_done(self) -> None:
        self.needs_flatten = False

    @property
    def day_pnl(self) -> float:
        return self.last_equity - self.start_equity

    @property
    def day_pnl_pct(self) -> float:
        return 0.0 if self.start_equity <= 0 else self.day_pnl / self.start_equity * 100.0

    def check(self, now: datetime) -> tuple[bool, str]:
        if self.locked_reason:
            return False, f"LOCKED: {self.locked_reason}"
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
        clock: ServerClock | None = None,
        calendar: Any = None,
        news_cfg: Any = None,
    ) -> None:
        self.guard_cfg = guard_cfg
        self.calendar = calendar  # src.core.news.EconomicCalendar | None
        self.news_cfg = news_cfg
        self.risk_cfg = risk_cfg
        self.spread = {s: SpreadFilter(c, guard_cfg) for s, c in symbols.items()}
        self.gaps = {s: GapDetector(guard_cfg) for s in symbols}
        self.slippage = SlippageMonitor(guard_cfg)
        self.session = SessionFilter(session_cfg, clock)
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
        if self.calendar is not None and self.news_cfg is not None:
            news = self.calendar.blackout(ctx.symbol, ctx.now, self.news_cfg)
            if news.blocked:
                d.block(news.reason)
        if ctx.last_tick_age_s > self.guard_cfg.stale_tick_s:
            d.block(f"stale feed ({ctx.last_tick_age_s:.0f}s)")
        if ctx.open_positions_total >= self.risk_cfg.max_open_positions:
            d.block("max concurrent trades")
        if ctx.open_positions_symbol >= self.risk_cfg.max_positions_per_symbol:
            d.block("max trades on symbol")
        return d

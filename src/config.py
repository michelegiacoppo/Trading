"""Typed configuration loader (``config/settings.yaml`` + ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def _build(cls: type[T], data: dict[str, Any] | None) -> T:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    data = data or {}
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(frozen=True)
class SymbolSpec:
    """Contract specification. Live: from ``mt5.symbol_info``. Backtest: from YAML."""

    name: str
    point: float
    digits: int
    tick_size: float
    tick_value: float
    contract_size: float = 1.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 0
    freeze_level: int = 0
    filling_mode: int = 0


@dataclass
class SymbolConfig:
    name: str = ""
    enabled: bool = True
    max_spread_points: float = 35
    spread_median_mult: float = 1.8
    spread_atr_ratio_max: float = 0.15
    stop_hunt_round_step: float = 0.0
    stop_hunt_buffer_points: float = 0
    spec: dict[str, Any] = field(default_factory=dict)

    def backtest_spec(self) -> SymbolSpec:
        return SymbolSpec(name=self.name, **self.spec)


@dataclass
class GeneralConfig:
    magic_number: int = 20261110
    comment_prefix: str = "XQS"
    poll_interval_ms: int = 250
    log_level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class MT5Config:
    timeout_ms: int = 10000
    max_reconnect_attempts: int = 5
    reconnect_backoff_s: float = 2.0


@dataclass
class StrategyConfig:
    name: str = "xau_velocity_scalp"
    timeframes: dict[str, str] = field(
        default_factory=lambda: {"entry": "M1", "confirm": "M5", "trend": "M15"}
    )
    history_bars: int = 400
    trend_ema: int = 50
    trend_slope_bars: int = 3
    confirm_ema_fast: int = 9
    confirm_ema_slow: int = 21
    atr_period: int = 14
    velocity_lookback: int = 3
    velocity_atr_mult: float = 1.2
    body_ratio_min: float = 0.55
    close_location_min: float = 0.70
    volume_ma_period: int = 20
    volume_spike_mult: float = 1.4
    rsi_period: int = 7
    rsi_long_max: float = 80
    rsi_short_min: float = 20
    max_bar_atr_mult: float = 3.0
    entry_mode: str = "limit"
    limit_pullback_atr: float = 0.25
    limit_expiry_s: int = 90
    sl_atr_mult: float = 1.3
    sl_min_points: float = 80
    sl_max_points: float = 600
    tp_r_multiple: float = 1.6
    max_hold_minutes: int = 25


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    commission_per_lot: float = 7.0
    max_lot: float = 5.0
    max_open_positions: int = 2
    max_positions_per_symbol: int = 1
    max_margin_usage_pct: float = 30.0
    daily_loss_limit_pct: float = 3.0
    daily_trailing_dd_pct: float = 4.0
    daily_profit_lock_pct: float = 0.0
    max_consecutive_losses: int = 4
    max_trades_per_day: int = 30
    cooldown_after_loss_s: int = 120


@dataclass
class GuardrailConfig:
    spread_window: int = 300
    spread_min_samples: int = 30
    gap_atr_mult: float = 1.5
    gap_cooldown_bars: int = 3
    tick_jump_atr_mult: float = 1.0
    tick_jump_cooldown_s: int = 30
    stale_tick_s: int = 10
    slippage_window: int = 10
    max_avg_slippage_points: float = 25
    slippage_pause_s: int = 600
    max_deviation_points: int = 20


@dataclass
class SessionConfig:
    windows: list[list[str]] = field(default_factory=list)
    rollover_blackout: list[str] = field(default_factory=lambda: ["23:50", "01:10"])
    friday_cutoff: str = "21:00"
    monday_start: str = "01:30"
    news_blackouts: list[str] = field(default_factory=list)
    news_blackout_minutes_before: int = 5
    news_blackout_minutes_after: int = 10


@dataclass
class ExecutionConfig:
    partial_close_r: float = 1.0
    partial_close_fraction: float = 0.5
    break_even_r: float = 1.0
    break_even_buffer_points: float = 10
    trailing_start_r: float = 1.2
    trailing_atr_mult: float = 1.0
    order_retries: int = 3
    retry_delay_ms: int = 150


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    default_spread_points: float = 20
    spread_jitter_points: float = 8
    slippage_points: float = 5
    seed: int = 42
    equity_curve_file: str = "logs/equity_curve.csv"
    trades_file: str = "logs/backtest_trades.csv"


@dataclass
class DashboardConfig:
    refresh_per_second: int = 4
    max_events: int = 14


@dataclass
class Credentials:
    login: int | None
    password: str | None
    server: str | None
    path: str | None

    @property
    def complete(self) -> bool:
        return bool(self.login and self.password and self.server)


@dataclass
class Settings:
    general: GeneralConfig
    mt5: MT5Config
    symbols: dict[str, SymbolConfig]
    strategy: StrategyConfig
    risk: RiskConfig
    guardrails: GuardrailConfig
    sessions: SessionConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    dashboard: DashboardConfig

    @property
    def enabled_symbols(self) -> list[str]:
        return [s for s, c in self.symbols.items() if c.enabled]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Settings:
        symbols = {
            name: _build(SymbolConfig, {**(cfg or {}), "name": name})
            for name, cfg in (raw.get("symbols") or {}).items()
        }
        return cls(
            general=_build(GeneralConfig, raw.get("general")),
            mt5=_build(MT5Config, raw.get("mt5")),
            symbols=symbols,
            strategy=_build(StrategyConfig, raw.get("strategy")),
            risk=_build(RiskConfig, raw.get("risk")),
            guardrails=_build(GuardrailConfig, raw.get("guardrails")),
            sessions=_build(SessionConfig, raw.get("sessions")),
            execution=_build(ExecutionConfig, raw.get("execution")),
            backtest=_build(BacktestConfig, raw.get("backtest")),
            dashboard=_build(DashboardConfig, raw.get("dashboard")),
        )


def load_settings(path: str | Path | None = None) -> Settings:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Settings.from_dict(raw)


def load_credentials() -> Credentials:
    """Read MT5 credentials from ``config/.env`` or ``.env`` (never from YAML)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / "config" / ".env")
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is a hard requirement
        pass
    login = os.getenv("MT5_LOGIN")
    return Credentials(
        login=int(login) if login and login.isdigit() else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH") or None,
    )

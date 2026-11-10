"""Typed configuration loader (``config/settings.yaml`` + ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
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
    currency_profit: str = "USD"
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
    timeout_ms: int = 60000
    max_reconnect_attempts: int = 5
    reconnect_backoff_s: float = 2.0


@dataclass
class StrategyConfig:
    name: str = "xau_velocity_scalp"
    timeframes: dict[str, str] = field(
        default_factory=lambda: {"entry": "M1", "confirm": "M5", "trend": "M15"}
    )
    history_bars: int = 1000
    # bias M15
    bias_ema_fast: int = 9
    bias_ema_slow: int = 21
    bias_mode: str = "ema_vwap"  # ema | ema_vwap | ema_structure
    vwap_slope_bars: int = 4
    structure_bars: int = 8
    confirm_enabled: bool = True
    confirm_ema_fast: int = 9
    confirm_ema_slow: int = 21
    # trigger
    trigger_mode: str = "any"  # sweep | squeeze | any
    atr_period: int = 14
    displacement_body_atr: float = 1.5
    displacement_volume_mult: float = 1.5
    volume_ma_period: int = 20
    sweep_lookback: int = 10
    sweep_window: int = 3
    squeeze_period: int = 20
    squeeze_bb_std: float = 2.0
    squeeze_kc_atr_mult: float = 1.5
    squeeze_window: int = 3
    max_bar_atr_mult: float = 4.0
    max_open_gap_atr: float = 0.5
    # entry
    entry_mode: str = "limit"  # limit | market
    limit_anchor: str = "retracement"  # retracement | fvg
    pullback_pct: float = 0.40
    order_expiration_candles: int = 3
    # stop / target
    sl_atr_buffer: float = 0.30
    sl_min_points: float = 100
    sl_max_points: float = 800
    runner_tp_r: float = 3.0


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    commission_per_lot: float = 7.0
    max_lot: float = 5.0
    max_open_positions: int = 2
    max_positions_per_symbol: int = 2
    max_margin_usage_pct: float = 30.0
    max_daily_loss_pct: float = 2.5
    daily_trailing_dd_pct: float = 0.0
    daily_profit_target_pct: float = 0.0
    flatten_on_lock: bool = True
    max_consecutive_losses: int = 4
    max_trades_per_day: int = 30
    cooldown_after_loss_s: int = 120


@dataclass
class GuardrailConfig:
    spread_window: int = 300
    spread_min_samples: int = 30
    gap_atr_mult: float = 1.5
    tick_jump_atr_mult: float = 1.0
    gap_block_minutes: int = 10
    stale_tick_s: int = 10
    slippage_window: int = 10
    max_avg_slippage_points: float = 25
    slippage_pause_s: int = 600
    max_deviation_points: int = 20


@dataclass
class SessionConfig:
    timezone: str = "Europe/Rome"
    server_timezone: str = "auto"
    backtest_server_timezone: str = "Europe/Athens"
    windows: dict[str, list[str]] = field(default_factory=dict)
    rollover_blackout: list[str] = field(default_factory=lambda: ["22:45", "01:15"])
    friday_cutoff: str = "21:30"
    monday_start: str = "02:00"
    blocked_slots: list[str] = field(default_factory=list)  # 30-min slots, e.g. '12:30'
    news_blackouts: list[str] = field(default_factory=list)
    news_blackout_minutes_before: int = 5
    news_blackout_minutes_after: int = 10


@dataclass
class ExecutionConfig:
    tp1_r: float = 1.0
    tp1_close_fraction: float = 0.65
    break_even_extra_points: float = 5
    trailing_mode: str = "candles"  # candles | atr
    trailing_candles: int = 2
    trailing_atr_mult: float = 1.0
    trailing_only_after_tp1: bool = True
    max_trade_duration_minutes: int = 25
    stale_require_fade: bool = True
    stale_momentum_threshold: float = 0.3
    order_retries: int = 3
    retry_delay_ms: int = 150


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    default_spread_points: float = 20
    spread_jitter_points: float = 8
    simulated_slippage_points: float = 5
    seed: int = 42
    hourly_min_trades: int = 5
    optimize_train_pct: float = 0.70
    equity_curve_file: str = "logs/equity_curve.csv"
    trades_file: str = "logs/backtest_trades.csv"


@dataclass
class DataConfig:
    historical_dir: str = "data/historical"
    warmup_bars: int = 1000
    warmup_timeframes: list[str] = field(default_factory=lambda: ["M1", "M5", "M15", "H1"])
    download_chunk_days: int = 15


@dataclass
class LevelsConfig:
    enabled: bool = True
    timeframe: str = "M5"
    asian_session_utc: list[str] = field(default_factory=lambda: ["00:00", "07:00"])
    london_session_utc: list[str] = field(default_factory=lambda: ["07:00", "11:00"])
    vwap_band_std: list[float] = field(default_factory=lambda: [1.0, 2.0])
    filter_mode: str = "priority"  # priority | required | off
    non_key_risk_mult: float = 0.5
    sweep_tolerance_atr: float = 0.10


@dataclass
class DashboardConfig:
    refresh_per_second: int = 4
    max_events: int = 10


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
    data: DataConfig = field(default_factory=DataConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)

    @property
    def enabled_symbols(self) -> list[str]:
        return [s for s, c in self.symbols.items() if c.enabled]

    def ensure_symbols(self, names: list[str] | None) -> None:
        """Map broker-specific names (``XAUUSD.m``, ``XAUUSDm``, ``GOLD``...) onto a configured symbol.

        Unknown names inherit the parameters of the configured symbol they start with
        (``GOLD*`` -> ``XAUUSD``) and become the enabled symbols for this run.
        """
        if not names:
            return
        aliases = {"GOLD": "XAUUSD", "XAU": "XAUUSD", "NAS100": "US100", "USTEC": "US100"}
        for name in names:
            if name in self.symbols:
                continue
            upper = name.upper()
            base = next((k for k in self.symbols if upper.startswith(k.upper())), None)
            base = base or next((v for a, v in aliases.items() if upper.startswith(a)), None)
            if base is None or base not in self.symbols:
                raise ValueError(
                    f"Symbol '{name}' not in settings.yaml and no similar entry found: "
                    f"add it under `symbols:`"
                )
            self.symbols[name] = replace(self.symbols[base], name=name)
        for key, cfg in self.symbols.items():
            cfg.enabled = key in names

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
            data=_build(DataConfig, raw.get("data")),
            levels=_build(LevelsConfig, raw.get("levels")),
        )


def resolve_path(path: str | Path) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


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

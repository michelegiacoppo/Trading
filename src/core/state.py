"""Shared, thread-safe snapshot of the bot state consumed by the TUI dashboard."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SymbolState:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    digits: int = 2
    spread: float = 0.0
    max_spread: float = 0.0
    spread_median: float | None = None
    atr_m1: float = 0.0
    atr_m5: float = 0.0
    bias: int = 0
    confirm: int = 0
    momentum: float = 0.0
    vol_state: str = "-"
    gap_state: str = "OK"
    guard: str = "-"
    last_signal: str = "-"
    key_levels: list[tuple[str, float, float]] = field(default_factory=list)  # label, level, dist


@dataclass
class PositionRow:
    ticket: int
    symbol: str
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    r_now: float
    profit: float
    be: bool = False
    tp1: bool = False


@dataclass
class PendingRow:
    ticket: int
    symbol: str
    side: int
    volume: float
    price: float
    sl: float
    tp: float
    expires: datetime


@dataclass
class BotState:
    mode: str = "dryrun"
    strategy: str = ""
    symbol: str = ""
    session: str = "-"
    connected: bool = False
    autotrading: str = "-"
    server_time: datetime | None = None
    local_time: datetime | None = None
    account: str = "-"
    currency: str = ""
    balance: float = 0.0
    equity: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    leverage: int = 0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    kill_status: str = "ACTIVE"
    kill_reason: str | None = None
    avg_slippage: float = 0.0
    reconnects: int = 0
    loop_ms: float = 0.0
    warmup: str = "-"
    symbols: dict[str, SymbolState] = field(default_factory=dict)
    positions: list[PositionRow] = field(default_factory=list)
    pending: list[PendingRow] = field(default_factory=list)
    events: deque[tuple[datetime, str, str]] = field(default_factory=lambda: deque(maxlen=200))
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def event(self, msg: str, level: str = "info", when: datetime | None = None) -> None:
        with self.lock:
            self.events.appendleft((when or datetime.now(), level, msg))

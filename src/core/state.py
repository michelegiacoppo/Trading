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
    spread: float = 0.0
    spread_median: float | None = None
    atr: float = 0.0
    bias: int = 0
    confirm: int = 0
    velocity_x_atr: float = 0.0
    guard: str = "-"
    last_signal: str = "-"


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
    flags: str = ""


@dataclass
class BotState:
    mode: str = "dryrun"
    strategy: str = ""
    connected: bool = False
    autotrading: str = "-"
    server_time: datetime | None = None
    account: str = "-"
    currency: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin_level: float = 0.0
    day_start_equity: float = 0.0
    day_realized: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    kill_switch: str | None = None
    avg_slippage: float = 0.0
    reconnects: int = 0
    loop_ms: float = 0.0
    symbols: dict[str, SymbolState] = field(default_factory=dict)
    positions: list[PositionRow] = field(default_factory=list)
    pending_orders: int = 0
    events: deque[tuple[datetime, str, str]] = field(default_factory=lambda: deque(maxlen=200))
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def event(self, msg: str, level: str = "info", when: datetime | None = None) -> None:
        with self.lock:
            self.events.appendleft((when or datetime.now(), level, msg))

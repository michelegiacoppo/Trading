"""MetaTrader 5 connection layer: initialize/login, auto-reconnect, AutoTrading checks.

The ``MetaTrader5`` package is Windows-only, so it is imported lazily. Every
other module receives the raw ``mt5`` module through :pyattr:`MT5Connector.mt5`,
which keeps the rest of the code base importable (and testable) on Linux.
"""

from __future__ import annotations

import logging
import time
from types import ModuleType
from typing import Any

from src.config import Credentials, MT5Config

log = logging.getLogger(__name__)


def import_mt5() -> ModuleType | None:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError:
        return None
    return mt5


class MT5NotAvailableError(RuntimeError):
    """Raised when the MetaTrader5 package/terminal cannot be used."""


class MT5Connector:
    """Owns the terminal session. Use as a context manager or call connect()/shutdown()."""

    def __init__(
        self,
        credentials: Credentials,
        cfg: MT5Config | None = None,
        mt5_module: ModuleType | None = None,
    ) -> None:
        self.credentials = credentials
        self.cfg = cfg or MT5Config()
        self._mt5 = mt5_module or import_mt5()
        self._connected = False
        self.reconnects = 0

    # ------------------------------------------------------------------ props
    @property
    def mt5(self) -> ModuleType:
        if self._mt5 is None:
            raise MT5NotAvailableError(
                "MetaTrader5 package not installed (Windows only). "
                "Install with `pip install MetaTrader5` on the trading host."
            )
        return self._mt5

    @property
    def available(self) -> bool:
        return self._mt5 is not None

    # ------------------------------------------------------------- lifecycle
    def connect(self) -> bool:
        mt5 = self.mt5
        kwargs: dict[str, Any] = {"timeout": self.cfg.timeout_ms}
        if self.credentials.path:
            kwargs["path"] = self.credentials.path
        if self.credentials.complete:
            kwargs.update(
                login=self.credentials.login,
                password=self.credentials.password,
                server=self.credentials.server,
            )
        if not mt5.initialize(**kwargs):
            log.error("mt5.initialize failed: %s", mt5.last_error())
            self._connected = False
            return False

        info = mt5.account_info()
        if info is None:
            log.error("account_info() is None after initialize: %s", mt5.last_error())
            self._connected = False
            return False
        self._connected = True
        log.info(
            "Connected to MT5 | account=%s server=%s balance=%.2f %s leverage=1:%s",
            info.login,
            info.server,
            info.balance,
            info.currency,
            info.leverage,
        )
        return True

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    def __enter__(self) -> MT5Connector:
        if not self.connect():
            raise MT5NotAvailableError(f"Unable to connect to MT5: {self.mt5.last_error()}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------ health
    def is_connected(self) -> bool:
        if not self._connected or self._mt5 is None:
            return False
        term = self._mt5.terminal_info()
        return bool(term and term.connected)

    def ensure_connected(self) -> bool:
        """Check the link and reconnect with exponential back-off if it dropped."""
        if self.is_connected():
            return True
        log.warning("MT5 link down - attempting reconnect")
        delay = self.cfg.reconnect_backoff_s
        for attempt in range(1, self.cfg.max_reconnect_attempts + 1):
            self.shutdown()
            if self.connect() and self.is_connected():
                self.reconnects += 1
                log.info("Reconnected on attempt %d", attempt)
                return True
            log.warning("Reconnect attempt %d failed, sleeping %.1fs", attempt, delay)
            time.sleep(delay)
            delay *= 2
        log.critical("MT5 reconnect failed after %d attempts", self.cfg.max_reconnect_attempts)
        return False

    def autotrading_status(self) -> tuple[bool, str]:
        """Return (allowed, reason). Checks terminal button, account and EA permissions."""
        mt5 = self.mt5
        term = mt5.terminal_info()
        acc = mt5.account_info()
        if term is None or acc is None:
            return False, "terminal/account info unavailable"
        if not term.trade_allowed:
            return False, "AutoTrading button is OFF in the terminal"
        if not acc.trade_allowed:
            return False, "trading disabled for this account"
        if not getattr(acc, "trade_expert", True):
            return False, "expert trading disabled by broker"
        if getattr(term, "tradeapi_disabled", False):
            return False, "Python trade API disabled (Tools > Options > Expert Advisors)"
        return True, "ok"

    def select_symbol(self, symbol: str) -> bool:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            log.error("Symbol %s not found on this broker", symbol)
            return False
        if not info.visible and not self.mt5.symbol_select(symbol, True):
            log.error("symbol_select(%s) failed: %s", symbol, self.mt5.last_error())
            return False
        return True

    def account_info(self) -> Any:
        return self.mt5.account_info()

    def server_time(self, symbol: str) -> int | None:
        """Broker server time (epoch seconds, server TZ) from the last tick."""
        tick = self.mt5.symbol_info_tick(symbol)
        return int(tick.time) if tick else None

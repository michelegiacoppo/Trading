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

# Trade server return codes (MQL5 docs: TRADE_RETCODE_*)
RETCODES: dict[int, str] = {
    0: "OK (order_check)", 10004: "REQUOTE", 10006: "REJECT", 10007: "CANCEL", 10008: "PLACED",
    10009: "DONE", 10010: "DONE_PARTIAL", 10011: "ERROR", 10012: "TIMEOUT",
    10013: "INVALID request", 10014: "INVALID_VOLUME", 10015: "INVALID_PRICE",
    10016: "INVALID_STOPS (SL/TP too close?)", 10017: "TRADE_DISABLED", 10018: "MARKET_CLOSED",
    10019: "NO_MONEY", 10020: "PRICE_CHANGED", 10021: "PRICE_OFF (no quotes)",
    10022: "INVALID_EXPIRATION", 10023: "ORDER_CHANGED", 10024: "TOO_MANY_REQUESTS",
    10025: "NO_CHANGES", 10026: "SERVER_DISABLES_AT", 10027: "CLIENT_DISABLES_AT (AutoTrading OFF)",
    10028: "LOCKED", 10029: "FROZEN (freeze level)", 10030: "INVALID_FILL (filling mode)",
    10031: "CONNECTION", 10032: "ONLY_REAL", 10033: "LIMIT_ORDERS", 10034: "LIMIT_VOLUME",
    10035: "INVALID_ORDER", 10036: "POSITION_CLOSED", 10038: "INVALID_CLOSE_VOLUME",
    10039: "CLOSE_ORDER_EXIST", 10040: "LIMIT_POSITIONS", 10044: "CLOSE_ONLY", 10045: "FIFO_CLOSE",
}  # fmt: skip
RETCODE_DONE = {10008, 10009, 10010}

# mt5.initialize() error codes -> actionable hints for the user
INIT_HINTS: dict[int, str] = {
    -10005: "IPC timeout: MT5 is not answering. Log in to an account (File > Login to Trade Account), "
    "do not run MT5 as administrator unless Python is too, close extra MT5 terminals or set MT5_PATH "
    "in config/.env, then retry.",
    -10004: "No IPC connection: open the MetaTrader 5 terminal and log in to an account.",
    -10003: "Terminal launch failed: check MT5_PATH in config/.env (path to terminal64.exe).",
    -6: "Authorization failed: check MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in config/.env.",
    -2: "Invalid parameters: check MT5_LOGIN (numbers only) and MT5_SERVER in config/.env.",
}
RETCODE_RETRY = {10004, 10020, 10021, 10024, 10031}  # requote, price changed/off, busy, link


def describe_retcode(code: int | None) -> str:
    return f"{code} {RETCODES.get(code or -1, 'UNKNOWN')}"


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
            err = mt5.last_error()
            code = err[0] if isinstance(err, tuple) and err else None
            log.error("mt5.initialize failed: %s", err)
            hint = INIT_HINTS.get(code)
            if hint:
                log.error("HINT: %s", hint)
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

    # ------------------------------------------------------------ broker
    def filling_mode(self, symbol: str, pending: bool = False) -> int:
        """Auto-detect the filling policy accepted by the broker for ``symbol``.

        ``symbol_info.filling_mode`` is a bitmask: 1 = FOK, 2 = IOC. Pending
        orders always use RETURN. If neither bit is set the broker only
        supports RETURN (exchange / market execution).
        """
        mt5 = self.mt5
        if pending:
            return mt5.ORDER_FILLING_RETURN
        info = mt5.symbol_info(symbol)
        mask = int(getattr(info, "filling_mode", 0) or 0)
        if mask & 1:
            return mt5.ORDER_FILLING_FOK
        if mask & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def send_request(
        self,
        request: dict[str, Any],
        retries: int = 3,
        retry_delay_ms: int = 150,
        refresh_price: bool = False,
    ) -> Any:
        """``order_check`` -> ``order_send`` with retry on requote/price changes.

        Every rejection is logged with its retcode and description. Returns the
        last ``OrderSendResult`` (or ``OrderCheckResult`` if the pre-check failed).
        """
        mt5 = self.mt5
        result: Any = None
        for attempt in range(1, max(1, retries) + 1):
            if refresh_price and request.get("action") == mt5.TRADE_ACTION_DEAL:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick is None:
                    log.error("No tick for %s, request aborted", request["symbol"])
                    return None
                request["price"] = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
            check = mt5.order_check(request)
            if check is None:
                log.error("order_check returned None: %s", mt5.last_error())
                return None
            if check.retcode not in (0, *RETCODE_DONE):
                log.error(
                    "order_check REJECTED %s | %s | %s",
                    describe_retcode(check.retcode),
                    check.comment,
                    _short(request),
                )
                return check
            result = mt5.order_send(request)
            if result is None:
                log.error("order_send returned None: %s", mt5.last_error())
            elif result.retcode in RETCODE_DONE:
                log.debug("order_send %s %s", describe_retcode(result.retcode), _short(request))
                return result
            elif result.retcode not in RETCODE_RETRY:
                log.error(
                    "order_send FAILED %s | %s | %s",
                    describe_retcode(result.retcode),
                    result.comment,
                    _short(request),
                )
                return result
            log.warning(
                "order_send retry %d/%d: %s",
                attempt,
                retries,
                describe_retcode(getattr(result, "retcode", None)),
            )
            time.sleep(retry_delay_ms / 1000)
        return result

    def bot_positions(self, magic: int, symbol: str | None = None) -> list[Any]:
        raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [p for p in raw or () if p.magic == magic]

    def bot_orders(self, magic: int, symbol: str | None = None) -> list[Any]:
        raw = self.mt5.orders_get(symbol=symbol) if symbol else self.mt5.orders_get()
        return [o for o in raw or () if o.magic == magic]

    def account_info(self) -> Any:
        return self.mt5.account_info()

    def server_time(self, symbol: str) -> int | None:
        """Broker server time (epoch seconds, server TZ) from the last tick."""
        tick = self.mt5.symbol_info_tick(symbol)
        return int(tick.time) if tick else None


def _short(req: dict[str, Any]) -> str:
    keys = ("action", "symbol", "type", "volume", "price", "sl", "tp", "position", "order")
    return " ".join(f"{k}={req[k]}" for k in keys if k in req)

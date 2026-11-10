"""Order execution and in-trade management.

* :class:`TradeManager`    - **pure** management logic (partial close, break-even,
  ATR trailing, time-stop). Shared with the backtester.
* :class:`ExecutionEngine` - MT5 order routing: filling-mode auto-detection,
  ``order_check`` pre-flight, requote retries, slippage measurement, pending
  limit orders with expiry, partial closes and SL modifications. In ``dry_run``
  mode orders are routed to an in-memory paper book fed by live ticks.
"""

from __future__ import annotations

import itertools
import logging
import time as _time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import ExecutionConfig, Settings, SymbolSpec
from src.core.data_feed import DataFeed, TickSnapshot
from src.core.mt5_connector import MT5Connector
from src.risk.guardrails import Guardrails
from src.risk.position_sizer import floor_to_step, money_per_lot
from src.strategies.base_strategy import Signal

log = logging.getLogger(__name__)

# MT5 retcodes worth retrying with a fresh price
RETRY_RETCODES = {
    10004,
    10020,
    10021,
    10024,
    10031,
}  # requote, price changed, off quotes, too many, no connection
RETCODE_DONE = {10008, 10009, 10010}  # placed, done, done partial


def server_dt(epoch_s: float) -> datetime:
    """MT5 epochs are broker-server wall time encoded as UTC: decode tz-naive."""
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).replace(tzinfo=None)


# =================================================================== models
@dataclass
class PositionView:
    ticket: int
    symbol: str
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    open_time: datetime
    initial_volume: float
    initial_risk: float  # price distance entry -> initial SL
    partial_done: bool = False
    be_done: bool = False
    profit: float = 0.0

    def r_multiple(self, exit_price: float) -> float:
        if self.initial_risk <= 0:
            return 0.0
        return self.side * (exit_price - self.entry) / self.initial_risk


@dataclass(frozen=True)
class ManagementAction:
    kind: str  # "partial" | "modify_sl" | "close"
    volume: float = 0.0
    new_sl: float = 0.0
    reason: str = ""


@dataclass
class ExecutionResult:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    comment: str = ""
    slippage_points: float = 0.0


@dataclass
class ClosedTrade:
    ticket: int
    symbol: str
    pnl: float
    time: datetime
    reason: str = ""


# ============================================================ trade manager
class TradeManager:
    """Stateless rules -> actions. Uses the exit side price (bid for longs, ask for shorts)."""

    def __init__(self, cfg: ExecutionConfig, max_hold_minutes: int) -> None:
        self.cfg = cfg
        self.max_hold = timedelta(minutes=max_hold_minutes)

    def evaluate(
        self,
        pos: PositionView,
        bid: float,
        ask: float,
        atr: float,
        now: datetime,
        spec: SymbolSpec,
    ) -> list[ManagementAction]:
        price = bid if pos.side > 0 else ask
        r = pos.r_multiple(price)
        actions: list[ManagementAction] = []

        if (
            self.max_hold.total_seconds() > 0
            and now - pos.open_time >= self.max_hold
            and r < self.cfg.partial_close_r
        ):
            return [ManagementAction("close", pos.volume, reason=f"time-stop ({r:+.2f}R)")]

        if (
            not pos.partial_done
            and self.cfg.partial_close_fraction > 0
            and r >= self.cfg.partial_close_r
        ):
            vol = floor_to_step(
                pos.initial_volume * self.cfg.partial_close_fraction, spec.volume_step
            )
            if vol >= spec.volume_min and pos.volume - vol >= spec.volume_min - 1e-9:
                actions.append(ManagementAction("partial", vol, reason=f"partial @ {r:+.2f}R"))

        best_sl = pos.sl
        reason = ""
        if r >= self.cfg.break_even_r:
            be = pos.entry + pos.side * self.cfg.break_even_buffer_points * spec.point
            if self._improves(pos.side, be, best_sl):
                best_sl, reason = be, f"break-even @ {r:+.2f}R"
        if r >= self.cfg.trailing_start_r and atr > 0:
            trail = price - pos.side * self.cfg.trailing_atr_mult * atr
            if self._improves(pos.side, trail, best_sl):
                best_sl, reason = trail, f"trail {self.cfg.trailing_atr_mult}xATR @ {r:+.2f}R"

        if best_sl != pos.sl:
            min_dist = max(spec.stops_level, spec.freeze_level) * spec.point
            if abs(price - best_sl) >= min_dist and self._improves(pos.side, best_sl, pos.sl):
                actions.append(
                    ManagementAction("modify_sl", new_sl=round(best_sl, spec.digits), reason=reason)
                )
        return actions

    @staticmethod
    def _improves(side: int, new_sl: float, old_sl: float) -> bool:
        if old_sl == 0:
            return True
        return new_sl > old_sl if side > 0 else new_sl < old_sl


# ======================================================== execution engine
@dataclass
class _PendingOrder:
    ticket: int
    signal: Signal
    volume: float
    expires: datetime


@dataclass
class PaperBook:
    """Minimal broker simulator for --mode dryrun."""

    positions: dict[int, PositionView] = field(default_factory=dict)
    pending: dict[int, _PendingOrder] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def next_id(self) -> int:
        return next(self._ids)


class ExecutionEngine:
    def __init__(
        self,
        connector: MT5Connector,
        feed: DataFeed,
        settings: Settings,
        guardrails: Guardrails,
        dry_run: bool = True,
    ) -> None:
        self.conn = connector
        self.feed = feed
        self.s = settings
        self.guards = guardrails
        self.dry_run = dry_run
        self.manager = TradeManager(settings.execution, settings.strategy.max_hold_minutes)
        self.magic = settings.general.magic_number
        self.paper = PaperBook()
        self._meta: dict[int, PositionView] = {}  # live: ticket -> initial risk/state
        self._pending_expiry: dict[int, datetime] = {}
        self._seen_deals: set[int] = set()

    # ------------------------------------------------------------ helpers
    @property
    def mt5(self) -> Any:
        return self.conn.mt5

    def _filling(self, spec: SymbolSpec, pending: bool = False) -> int:
        mt5 = self.mt5
        if pending:
            return mt5.ORDER_FILLING_RETURN
        if spec.filling_mode & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if spec.filling_mode & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _comment(self, tag: str) -> str:
        return f"{self.s.general.comment_prefix}-{tag}"[:31]

    @staticmethod
    def _enforce_stops(signal: Signal, spec: SymbolSpec, price: float) -> tuple[float, float]:
        """Widen SL/TP if they violate the broker's minimum stop distance."""
        min_dist = (spec.stops_level + 2) * spec.point if spec.stops_level else 0.0
        sl, tp = signal.stop_loss, signal.take_profit
        if min_dist and abs(price - sl) < min_dist:
            sl = price - signal.side * min_dist
        if min_dist and abs(tp - price) < min_dist:
            tp = price + signal.side * min_dist
        return round(sl, spec.digits), round(tp, spec.digits)

    def _send(self, request: dict[str, Any], refresh_price: bool) -> Any:
        mt5 = self.mt5
        result = None
        for attempt in range(1, self.s.execution.order_retries + 1):
            if refresh_price:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick is None:
                    return None
                request["price"] = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
            check = mt5.order_check(request)
            if check is None or check.retcode not in (0, *RETCODE_DONE):
                log.error(
                    "order_check rejected: %s | %s",
                    getattr(check, "comment", None),
                    mt5.last_error(),
                )
                return check
            result = mt5.order_send(request)
            if result is None:
                log.error("order_send returned None: %s", mt5.last_error())
            elif result.retcode in RETCODE_DONE:
                return result
            elif result.retcode not in RETRY_RETCODES:
                log.error("order_send failed retcode=%s %s", result.retcode, result.comment)
                return result
            log.warning(
                "retryable retcode=%s, attempt %d", getattr(result, "retcode", None), attempt
            )
            _time.sleep(self.s.execution.retry_delay_ms / 1000)
        return result

    # ------------------------------------------------------------ entries
    def execute_signal(
        self, signal: Signal, volume: float, tick: TickSnapshot, now: datetime
    ) -> ExecutionResult:
        if signal.order_type == "limit":
            return self.place_limit(signal, volume, tick, now)
        return self.market_order(signal, volume, tick, now)

    def market_order(
        self, signal: Signal, volume: float, tick: TickSnapshot, now: datetime
    ) -> ExecutionResult:
        spec = self.feed.symbol_spec(signal.symbol)
        requested = tick.ask if signal.is_buy else tick.bid
        sl, tp = self._enforce_stops(signal, spec, requested)

        if self.dry_run:
            ticket = self.paper.next_id()
            self.paper.positions[ticket] = PositionView(
                ticket,
                signal.symbol,
                signal.side,
                volume,
                requested,
                sl,
                tp,
                now,
                volume,
                abs(requested - sl),
            )
            log.info(
                "[DRY] MARKET %s %s %.2f @ %s SL %s TP %s",
                _side(signal.side),
                signal.symbol,
                volume,
                requested,
                sl,
                tp,
            )
            return ExecutionResult(True, ticket, requested, volume, 10009, "paper")

        mt5 = self.mt5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal.is_buy else mt5.ORDER_TYPE_SELL,
            "price": requested,
            "sl": sl,
            "tp": tp,
            "deviation": self.s.guardrails.max_deviation_points,
            "magic": self.magic,
            "comment": self._comment("MKT"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(spec),
        }
        res = self._send(request, refresh_price=True)
        if res is None or res.retcode not in RETCODE_DONE:
            return ExecutionResult(
                False, retcode=getattr(res, "retcode", -1), comment=getattr(res, "comment", "")
            )
        slip = self.guards.slippage.record(requested, res.price, signal.is_buy, spec.point, now)
        log.info(
            "MARKET %s %s %.2f filled @ %s (slip %.1fp)",
            _side(signal.side),
            signal.symbol,
            res.volume,
            res.price,
            slip,
        )
        self._meta[res.order] = PositionView(
            res.order,
            signal.symbol,
            signal.side,
            res.volume,
            res.price,
            sl,
            tp,
            now,
            res.volume,
            abs(res.price - sl),
        )
        return ExecutionResult(
            True, res.order, res.price, res.volume, res.retcode, res.comment, slip
        )

    def place_limit(
        self, signal: Signal, volume: float, tick: TickSnapshot, now: datetime
    ) -> ExecutionResult:
        spec = self.feed.symbol_spec(signal.symbol)
        price = signal.entry
        # a limit already through the market would fill instantly at a worse price -> go market
        if (signal.is_buy and price >= tick.ask) or (not signal.is_buy and price <= tick.bid):
            return self.market_order(signal, volume, tick, now)
        sl, tp = self._enforce_stops(signal, spec, price)
        expires = now + timedelta(seconds=signal.expiry_s or 60)

        if self.dry_run:
            ticket = self.paper.next_id()
            sig = replace(signal, stop_loss=sl, take_profit=tp)
            self.paper.pending[ticket] = _PendingOrder(ticket, sig, volume, expires)
            log.info(
                "[DRY] LIMIT %s %s %.2f @ %s SL %s TP %s exp %s",
                _side(signal.side),
                signal.symbol,
                volume,
                price,
                sl,
                tp,
                f"{expires:%H:%M:%S}",
            )
            return ExecutionResult(True, ticket, price, volume, 10008, "paper pending")

        mt5 = self.mt5
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": signal.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY_LIMIT if signal.is_buy else mt5.ORDER_TYPE_SELL_LIMIT,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": self.magic,
            "comment": self._comment("LMT"),
            "type_time": mt5.ORDER_TIME_GTC,  # expiry handled locally (broker-agnostic)
            "type_filling": self._filling(spec, pending=True),
        }
        res = self._send(request, refresh_price=False)
        if res is None or res.retcode not in RETCODE_DONE:
            return ExecutionResult(
                False, retcode=getattr(res, "retcode", -1), comment=getattr(res, "comment", "")
            )
        self._pending_expiry[res.order] = expires
        log.info(
            "LIMIT %s %s %.2f @ %s placed (ticket %s)",
            _side(signal.side),
            signal.symbol,
            volume,
            price,
            res.order,
        )
        return ExecutionResult(True, res.order, price, volume, res.retcode, res.comment)

    # ----------------------------------------------------------- queries
    def open_positions(self, symbol: str | None = None) -> list[PositionView]:
        if self.dry_run:
            return [
                p for p in self.paper.positions.values() if symbol is None or p.symbol == symbol
            ]
        raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        out: list[PositionView] = []
        for p in raw or ():
            if p.magic != self.magic:
                continue
            side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            meta = self._meta.get(p.ticket)
            if meta is None:  # restart / unknown: infer initial risk from current SL
                meta = PositionView(
                    p.ticket,
                    p.symbol,
                    side,
                    p.volume,
                    p.price_open,
                    p.sl,
                    p.tp,
                    server_dt(p.time),
                    p.volume,
                    abs(p.price_open - p.sl) if p.sl else 0.0,
                )
                self._meta[p.ticket] = meta
            meta.volume, meta.sl, meta.tp, meta.profit = p.volume, p.sl, p.tp, p.profit
            out.append(meta)
        return out

    def pending_count(self, symbol: str | None = None) -> int:
        if self.dry_run:
            return sum(
                1
                for o in self.paper.pending.values()
                if symbol is None or o.signal.symbol == symbol
            )
        orders = self.mt5.orders_get(symbol=symbol) if symbol else self.mt5.orders_get()
        return sum(1 for o in orders or () if o.magic == self.magic)

    # -------------------------------------------------------- management
    def close_volume(self, pos: PositionView, volume: float, reason: str) -> bool:
        if self.dry_run:
            return True  # handled by paper book
        mt5 = self.mt5
        spec = self.feed.symbol_spec(pos.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if pos.side > 0 else mt5.ORDER_TYPE_BUY,
            "price": 0.0,
            "deviation": self.s.guardrails.max_deviation_points,
            "magic": self.magic,
            "comment": self._comment(reason[:10]),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(spec),
        }
        res = self._send(request, refresh_price=True)
        ok = res is not None and res.retcode in RETCODE_DONE
        log.info(
            "CLOSE %.2f of #%s (%s): %s",
            volume,
            pos.ticket,
            reason,
            "ok" if ok else getattr(res, "comment", "fail"),
        )
        return ok

    def modify_sl(self, pos: PositionView, new_sl: float, reason: str) -> bool:
        if self.dry_run:
            return True
        mt5 = self.mt5
        res = mt5.order_send(
            {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
                "magic": self.magic,
            }
        )
        ok = res is not None and res.retcode in RETCODE_DONE
        log.info(
            "SL #%s -> %s (%s): %s",
            pos.ticket,
            new_sl,
            reason,
            "ok" if ok else getattr(res, "comment", "fail"),
        )
        return ok

    def manage(
        self, now: datetime, ticks: dict[str, TickSnapshot], atrs: dict[str, float]
    ) -> list[ClosedTrade]:
        """Run every loop: expire limits, manage open trades, return newly closed trades."""
        if self.dry_run:
            return self._manage_paper(now, ticks, atrs)
        self._cancel_expired(now)
        for pos in self.open_positions():
            tick = ticks.get(pos.symbol)
            if tick is None:
                continue
            spec = self.feed.symbol_spec(pos.symbol)
            for act in self.manager.evaluate(
                pos, tick.bid, tick.ask, atrs.get(pos.symbol, 0.0), now, spec
            ):
                if act.kind == "close":
                    self.close_volume(pos, pos.volume, act.reason)
                    break
                if act.kind == "partial" and self.close_volume(pos, act.volume, act.reason):
                    pos.partial_done = True
                elif act.kind == "modify_sl" and self.modify_sl(pos, act.new_sl, act.reason):
                    pos.be_done = pos.be_done or "break-even" in act.reason
        return self._poll_closed_deals(now)

    def _cancel_expired(self, now: datetime) -> None:
        mt5 = self.mt5
        live = {o.ticket for o in mt5.orders_get() or () if o.magic == self.magic}
        for ticket, exp in list(self._pending_expiry.items()):
            if ticket not in live:
                self._pending_expiry.pop(ticket, None)  # filled or cancelled
            elif now >= exp:
                res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
                log.info("Limit #%s expired -> cancel %s", ticket, getattr(res, "retcode", None))
                self._pending_expiry.pop(ticket, None)

    def _poll_closed_deals(self, now: datetime) -> list[ClosedTrade]:
        mt5 = self.mt5
        start = datetime(now.year, now.month, now.day)
        deals = mt5.history_deals_get(start, now + timedelta(days=1)) or ()
        closed: list[ClosedTrade] = []
        for d in deals:
            if d.magic != self.magic or d.ticket in self._seen_deals:
                continue
            self._seen_deals.add(d.ticket)
            if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
                pnl = d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
                closed.append(
                    ClosedTrade(d.position_id, d.symbol, pnl, server_dt(d.time), d.comment)
                )
        return closed

    # -------------------------------------------------------- paper book
    def _manage_paper(
        self, now: datetime, ticks: dict[str, TickSnapshot], atrs: dict[str, float]
    ) -> list[ClosedTrade]:
        book = self.paper
        closed: list[ClosedTrade] = []
        for ticket, order in list(book.pending.items()):
            tick = ticks.get(order.signal.symbol)
            if now >= order.expires:
                book.pending.pop(ticket)
                log.info("[DRY] limit #%s expired", ticket)
                continue
            if tick is None:
                continue
            sig = order.signal
            if (sig.is_buy and tick.ask <= sig.entry) or (not sig.is_buy and tick.bid >= sig.entry):
                book.pending.pop(ticket)
                book.positions[ticket] = PositionView(
                    ticket,
                    sig.symbol,
                    sig.side,
                    order.volume,
                    sig.entry,
                    sig.stop_loss,
                    sig.take_profit,
                    now,
                    order.volume,
                    abs(sig.entry - sig.stop_loss),
                )
                log.info("[DRY] limit #%s filled @ %s", ticket, sig.entry)

        for pos in list(book.positions.values()):
            tick = ticks.get(pos.symbol)
            if tick is None:
                continue
            spec = self.feed.symbol_spec(pos.symbol)
            px = tick.bid if pos.side > 0 else tick.ask
            sign = 1 if pos.side * (px - pos.entry) >= 0 else -1
            pos.profit = sign * money_per_lot(px - pos.entry, spec) * pos.volume
            hit_sl = pos.sl and (px <= pos.sl if pos.side > 0 else px >= pos.sl)
            hit_tp = pos.tp and (px >= pos.tp if pos.side > 0 else px <= pos.tp)
            if hit_sl or hit_tp:
                closed.append(
                    self._paper_close(pos, px, pos.volume, "SL" if hit_sl else "TP", now, spec)
                )
                continue
            for act in self.manager.evaluate(
                pos, tick.bid, tick.ask, atrs.get(pos.symbol, 0.0), now, spec
            ):
                if act.kind == "close":
                    closed.append(self._paper_close(pos, px, pos.volume, act.reason, now, spec))
                    break
                if act.kind == "partial":
                    closed.append(self._paper_close(pos, px, act.volume, act.reason, now, spec))
                    pos.partial_done = True
                elif act.kind == "modify_sl":
                    pos.sl = act.new_sl
                    pos.be_done = pos.be_done or "break-even" in act.reason
                    log.info("[DRY] SL #%s -> %s (%s)", pos.ticket, act.new_sl, act.reason)
        book.closed.extend(closed)
        return closed

    def _paper_close(
        self,
        pos: PositionView,
        px: float,
        volume: float,
        reason: str,
        now: datetime,
        spec: SymbolSpec,
    ) -> ClosedTrade:
        sign = 1 if pos.side * (px - pos.entry) >= 0 else -1
        pnl = sign * money_per_lot(px - pos.entry, spec) * volume
        pnl -= self.s.risk.commission_per_lot * volume
        pos.volume = round(pos.volume - volume, 8)
        if pos.volume <= 1e-9:
            self.paper.positions.pop(pos.ticket, None)
        log.info("[DRY] close %.2f #%s @ %s (%s) pnl %.2f", volume, pos.ticket, px, reason, pnl)
        return ClosedTrade(pos.ticket, pos.symbol, pnl, now, reason)


def _side(side: int) -> str:
    return "BUY" if side > 0 else "SELL"

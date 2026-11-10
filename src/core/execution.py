"""Order execution and "Hit & Run" in-trade management.

* :class:`TradeManager`    - **pure** management rules, shared by live, dry-run and backtest:
    - **TP1 / fast break-even**: at ``+tp1_r`` close ``tp1_close_fraction`` (65%) and move the
      stop of the runner to break-even + spread + commissions + extra points (zero risk);
    - **micro-trailing** of the runner on the low/high of the last N M1 candles (or ATR);
    - **stale trade killer**: after ``max_trade_duration_minutes`` without TP1, if momentum
      faded, close at market near break-even.
* :class:`ExecutionEngine` - MT5 routing through :class:`MT5Connector` (filling-mode
  detection, ``order_check`` before every ``order_send``, retcode logging, magic number),
  LIMIT pullback orders with candle-based expiry, MARKET orders with hard deviation and
  anti-chase checks, partial closes, SL modifications, flatten-all. In ``dry_run`` mode
  orders go to an in-memory :class:`PaperBook` fed by live ticks.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import ExecutionConfig, Settings, SymbolSpec
from src.core.data_feed import DataFeed, TickSnapshot
from src.core.mt5_connector import RETCODE_DONE, MT5Connector, describe_retcode
from src.risk.guardrails import Guardrails
from src.risk.position_sizer import floor_to_step, money_per_lot
from src.strategies.base_strategy import Signal

log = logging.getLogger(__name__)


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
    tp1_done: bool = False
    be_done: bool = False
    profit: float = 0.0

    def r_multiple(self, exit_price: float) -> float:
        if self.initial_risk <= 0:
            return 0.0
        return self.side * (exit_price - self.entry) / self.initial_risk


@dataclass
class PendingView:
    ticket: int
    symbol: str
    side: int
    volume: float
    price: float
    sl: float
    tp: float
    expires: datetime


@dataclass(frozen=True)
class MarketContext:
    """Per-symbol inputs for trade management (refreshed on each closed M1 bar)."""

    atr: float = 0.0
    recent_high: float = 0.0  # max high of the last N closed candles (BID)
    recent_low: float = 0.0  # min low of the last N closed candles (BID)
    momentum: float = 0.0  # (close - close[-3]) / ATR, signed


@dataclass(frozen=True)
class ManagementAction:
    kind: str  # "tp1" | "modify_sl" | "close"
    volume: float = 0.0  # tp1: volume to close (0 = flag only, lot too small)
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
    final: bool = True  # False for partial exits (TP1)


# ============================================================ trade manager
class TradeManager:
    """Stateless rules -> actions. Uses the exit-side price (BID for longs, ASK for shorts)."""

    def __init__(self, cfg: ExecutionConfig, commission_per_lot: float = 0.0) -> None:
        self.cfg = cfg
        self.commission_per_lot = commission_per_lot
        self.max_hold = timedelta(minutes=cfg.max_trade_duration_minutes)

    def break_even_buffer(self, spec: SymbolSpec, spread_price: float) -> float:
        """Price distance covering spread + round-turn commission + extra points."""
        per_price_unit = spec.tick_value / spec.tick_size if spec.tick_size else 0.0
        commission_price = self.commission_per_lot / per_price_unit if per_price_unit else 0.0
        return spread_price + commission_price + self.cfg.break_even_extra_points * spec.point

    def evaluate(
        self,
        pos: PositionView,
        bid: float,
        ask: float,
        mkt: MarketContext,
        now: datetime,
        spec: SymbolSpec,
    ) -> list[ManagementAction]:
        c = self.cfg
        side = pos.side
        price = bid if side > 0 else ask
        spread = max(ask - bid, 0.0)
        r = pos.r_multiple(price)

        # --- stale trade killer
        if (
            not pos.tp1_done
            and c.max_trade_duration_minutes > 0
            and now - pos.open_time >= self.max_hold
        ):
            faded = (
                side * mkt.momentum < c.stale_momentum_threshold if c.stale_require_fade else True
            )
            if faded:
                return [ManagementAction("close", pos.volume, reason=f"stale trade {r:+.2f}R")]

        actions: list[ManagementAction] = []
        best_sl, reason = pos.sl, ""
        tp1_now = False

        # --- TP1: partial close + break-even (+costs)
        if not pos.tp1_done and r >= c.tp1_r:
            vol = floor_to_step(pos.initial_volume * c.tp1_close_fraction, spec.volume_step)
            remaining = round(pos.volume - vol, 8)
            if vol >= spec.volume_min and remaining < spec.volume_min - 1e-9:
                return [ManagementAction("close", pos.volume, reason=f"TP1 {r:+.2f}R (no runner)")]
            vol = vol if vol >= spec.volume_min else 0.0  # lot too small: BE only
            actions.append(
                ManagementAction("tp1", vol, reason=f"TP1 {c.tp1_close_fraction:.0%} @ {r:+.2f}R")
            )
            be = pos.entry + side * self.break_even_buffer(spec, spread)
            if self._improves(side, be, best_sl) and side * (price - be) > 0:
                best_sl, reason = be, "break-even + costs"
            tp1_now = True

        # --- micro-trailing on the runner
        if pos.tp1_done or tp1_now or not c.trailing_only_after_tp1:
            trail = 0.0
            if c.trailing_mode == "candles" and mkt.recent_low > 0 and mkt.recent_high > 0:
                trail = mkt.recent_low if side > 0 else mkt.recent_high + spread
            elif mkt.atr > 0:
                trail = price - side * c.trailing_atr_mult * mkt.atr
            if trail and self._improves(side, trail, best_sl) and side * (price - trail) > 0:
                best_sl, reason = trail, f"trail ({c.trailing_mode})"

        if best_sl != pos.sl and self._improves(side, best_sl, pos.sl):
            min_dist = max(spec.stops_level, spec.freeze_level) * spec.point
            if abs(price - best_sl) >= min_dist:
                actions.append(
                    ManagementAction("modify_sl", new_sl=round(best_sl, spec.digits), reason=reason)
                )
        return actions

    @staticmethod
    def _improves(side: int, new_sl: float, old_sl: float) -> bool:
        if old_sl == 0:
            return True
        return new_sl > old_sl if side > 0 else new_sl < old_sl


# ============================================================== paper book
@dataclass
class _PaperPending:
    view: PendingView
    signal: Signal


@dataclass
class PaperBook:
    """Minimal broker simulator for ``--mode dryrun``."""

    positions: dict[int, PositionView] = field(default_factory=dict)
    pending: dict[int, _PaperPending] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def next_id(self) -> int:
        return next(self._ids)


# ======================================================== execution engine
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
        self.manager = TradeManager(settings.execution, settings.risk.commission_per_lot)
        self.magic = settings.general.magic_number
        self.paper = PaperBook()
        self._meta: dict[int, PositionView] = {}
        self._pending_meta: dict[int, PendingView] = {}
        self._seen_deals: set[int] = set()

    @property
    def mt5(self) -> Any:
        return self.conn.mt5

    def _comment(self, tag: str) -> str:
        return f"{self.s.general.comment_prefix}-{tag}"[:31]

    def _send(self, request: dict[str, Any], refresh_price: bool) -> Any:
        ex = self.s.execution
        return self.conn.send_request(request, ex.order_retries, ex.retry_delay_ms, refresh_price)

    @staticmethod
    def _enforce_stops(signal: Signal, spec: SymbolSpec, price: float) -> tuple[float, float]:
        """Widen SL/TP if they violate the broker's minimum stop distance."""
        min_dist = (spec.stops_level + 2) * spec.point if spec.stops_level else 0.0
        sl, tp = signal.stop_loss, signal.take_profit
        if min_dist and abs(price - sl) < min_dist:
            sl = price - signal.side * min_dist
        if tp and min_dist and abs(tp - price) < min_dist:
            tp = price + signal.side * min_dist
        return round(sl, spec.digits), round(tp, spec.digits)

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
        max_dev = self.s.guardrails.max_deviation_points
        chase = signal.side * (requested - signal.entry) / spec.point
        if chase > max_dev:  # price already ran away from the signal: don't chase the spike
            msg = f"price ran {chase:.0f}p beyond signal (max {max_dev}p)"
            log.warning("%s %s skipped: %s", signal.symbol, _side(signal.side), msg)
            return ExecutionResult(False, comment=msg)
        sl, tp = self._enforce_stops(signal, spec, requested)

        if self.dry_run:
            ticket = self.paper.next_id()
            self.paper.positions[ticket] = PositionView(
                ticket, signal.symbol, signal.side, volume, requested, sl, tp, now, volume,
                abs(requested - sl),
            )  # fmt: skip
            log.info("[DRY] MARKET %s %s %.2f @ %s SL %s TP %s", _side(signal.side),
                     signal.symbol, volume, requested, sl, tp)  # fmt: skip
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
            "deviation": max_dev,  # hard slippage cap enforced by the trade server
            "magic": self.magic,
            "comment": self._comment("MKT"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.conn.filling_mode(signal.symbol),
        }
        res = self._send(request, refresh_price=True)
        if res is None or res.retcode not in RETCODE_DONE:
            return ExecutionResult(False, retcode=getattr(res, "retcode", -1),
                                   comment=getattr(res, "comment", ""))  # fmt: skip
        slip = self.guards.slippage.record(requested, res.price, signal.is_buy, spec.point, now)
        log.info("MARKET %s %s %.2f filled @ %s (slip %.1fp)", _side(signal.side), signal.symbol,
                 res.volume, res.price, slip)  # fmt: skip
        ticket = getattr(res, "order", 0)
        self._meta[ticket] = PositionView(
            ticket, signal.symbol, signal.side, res.volume, res.price, sl, tp, now, res.volume,
            abs(res.price - sl),
        )  # fmt: skip
        return ExecutionResult(True, ticket, res.price, res.volume, res.retcode, res.comment, slip)

    def place_limit(
        self, signal: Signal, volume: float, tick: TickSnapshot, now: datetime
    ) -> ExecutionResult:
        spec = self.feed.symbol_spec(signal.symbol)
        price = signal.entry
        if (signal.is_buy and price >= tick.ask) or (not signal.is_buy and price <= tick.bid):
            # already retraced through the level: take it at market (anti-chase still applies)
            return self.market_order(signal, volume, tick, now)
        sl, tp = self._enforce_stops(signal, spec, price)
        expires = now + timedelta(seconds=signal.expiry_s or 180)
        view = PendingView(0, signal.symbol, signal.side, volume, price, sl, tp, expires)

        if self.dry_run:
            view.ticket = self.paper.next_id()
            sig = replace(signal, stop_loss=sl, take_profit=tp)
            self.paper.pending[view.ticket] = _PaperPending(view, sig)
            log.info("[DRY] %s LIMIT %s %.2f @ %s SL %s TP %s exp %s", _side(signal.side),
                     signal.symbol, volume, price, sl, tp, f"{expires:%H:%M:%S}")  # fmt: skip
            return ExecutionResult(True, view.ticket, price, volume, 10008, "paper pending")

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
            "type_time": mt5.ORDER_TIME_GTC,  # expiry enforced locally (broker-agnostic)
            "type_filling": self.conn.filling_mode(signal.symbol, pending=True),
        }
        res = self._send(request, refresh_price=False)
        if res is None or res.retcode not in RETCODE_DONE:
            return ExecutionResult(False, retcode=getattr(res, "retcode", -1),
                                   comment=getattr(res, "comment", ""))  # fmt: skip
        view.ticket = res.order
        self._pending_meta[res.order] = view
        log.info("%s LIMIT %s %.2f @ %s placed #%s (expires %s)", _side(signal.side), signal.symbol,
                 volume, price, res.order, f"{expires:%H:%M:%S}")  # fmt: skip
        return ExecutionResult(True, res.order, price, volume, res.retcode, res.comment)

    # ----------------------------------------------------------- queries
    def open_positions(self, symbol: str | None = None) -> list[PositionView]:
        if self.dry_run:
            return [p for p in self.paper.positions.values() if symbol in (None, p.symbol)]
        out: list[PositionView] = []
        for p in self.conn.bot_positions(self.magic, symbol):
            side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            meta = self._meta.get(p.ticket)
            if meta is None:  # filled limit / restart: build from the broker position
                meta = PositionView(p.ticket, p.symbol, side, p.volume, p.price_open, p.sl, p.tp,
                                    server_dt(p.time), p.volume,
                                    abs(p.price_open - p.sl) if p.sl else 0.0)  # fmt: skip
                self._meta[p.ticket] = meta
            meta.volume, meta.sl, meta.tp, meta.profit = p.volume, p.sl, p.tp, p.profit
            out.append(meta)
        return out

    def pending_orders(self, symbol: str | None = None) -> list[PendingView]:
        if self.dry_run:
            return [o.view for o in self.paper.pending.values() if symbol in (None, o.view.symbol)]
        out: list[PendingView] = []
        for o in self.conn.bot_orders(self.magic, symbol):
            view = self._pending_meta.get(o.ticket)
            if view is None:
                side = (
                    1
                    if o.type in (self.mt5.ORDER_TYPE_BUY_LIMIT, self.mt5.ORDER_TYPE_BUY_STOP)
                    else -1
                )
                view = PendingView(o.ticket, o.symbol, side, o.volume_current, o.price_open, o.sl, o.tp,
                                   server_dt(o.time_setup) + timedelta(minutes=3))  # fmt: skip
                self._pending_meta[o.ticket] = view
            out.append(view)
        return out

    def pending_count(self, symbol: str | None = None) -> int:
        return len(self.pending_orders(symbol))

    # -------------------------------------------------------- management
    def close_volume(self, pos: PositionView, volume: float, reason: str) -> bool:
        mt5 = self.mt5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if pos.side > 0 else mt5.ORDER_TYPE_BUY,
            "price": 0.0,
            "deviation": self.s.guardrails.max_deviation_points,
            "magic": self.magic,
            "comment": self._comment(reason[:12]),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.conn.filling_mode(pos.symbol),
        }
        res = self._send(request, refresh_price=True)
        ok = res is not None and res.retcode in RETCODE_DONE
        log.info("CLOSE %.2f of #%s (%s): %s", volume, pos.ticket, reason,
                 describe_retcode(getattr(res, "retcode", None)))  # fmt: skip
        return ok

    def modify_sl(self, pos: PositionView, new_sl: float, reason: str) -> bool:
        mt5 = self.mt5
        res = self._send(
            {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": pos.ticket,
             "sl": new_sl, "tp": pos.tp, "magic": self.magic},
            refresh_price=False,
        )  # fmt: skip
        ok = res is not None and res.retcode in RETCODE_DONE
        log.info("SL #%s -> %s (%s): %s", pos.ticket, new_sl, reason,
                 describe_retcode(getattr(res, "retcode", None)))  # fmt: skip
        return ok

    def cancel_order(self, ticket: int, reason: str) -> bool:
        if self.dry_run:
            return self.paper.pending.pop(ticket, None) is not None
        mt5 = self.mt5
        res = self._send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}, refresh_price=False)
        self._pending_meta.pop(ticket, None)
        log.info(
            "CANCEL #%s (%s): %s", ticket, reason, describe_retcode(getattr(res, "retcode", None))
        )
        return res is not None and res.retcode in RETCODE_DONE

    def flatten_all(
        self, now: datetime, ticks: dict[str, TickSnapshot], reason: str
    ) -> list[ClosedTrade]:
        """Kill-switch: cancel every pending order and close every bot position."""
        for o in self.pending_orders():
            self.cancel_order(o.ticket, reason)
        if self.dry_run:
            closed = []
            for pos in list(self.paper.positions.values()):
                tick = ticks.get(pos.symbol)
                if tick:
                    px = tick.bid if pos.side > 0 else tick.ask
                    closed.append(self._paper_close(pos, px, pos.volume, reason, now))
            self.paper.closed.extend(closed)
            return closed
        for pos in self.open_positions():
            self.close_volume(pos, pos.volume, reason)
        return self._poll_closed_deals(now)

    def manage(
        self, now: datetime, ticks: dict[str, TickSnapshot], ctx: dict[str, MarketContext]
    ) -> list[ClosedTrade]:
        """Run every loop: expire limits, manage open trades, return newly closed trades."""
        if self.dry_run:
            return self._manage_paper(now, ticks, ctx)
        for o in self.pending_orders():
            if now >= o.expires:
                self.cancel_order(o.ticket, "limit expired (price ran away)")
        live_orders = {o.ticket for o in self.conn.bot_orders(self.magic)}
        for t in [t for t in self._pending_meta if t not in live_orders]:
            self._pending_meta.pop(t, None)  # filled or cancelled
        for pos in self.open_positions():
            tick = ticks.get(pos.symbol)
            if tick is None:
                continue
            spec = self.feed.symbol_spec(pos.symbol)
            for act in self.manager.evaluate(
                pos, tick.bid, tick.ask, ctx.get(pos.symbol, MarketContext()), now, spec
            ):
                if act.kind == "close":
                    self.close_volume(pos, pos.volume, act.reason)
                    break
                if act.kind == "tp1":
                    if act.volume == 0 or self.close_volume(pos, act.volume, act.reason):
                        pos.tp1_done = True
                elif act.kind == "modify_sl" and self.modify_sl(pos, act.new_sl, act.reason):
                    pos.be_done = pos.be_done or "break-even" in act.reason
        return self._poll_closed_deals(now)

    def _poll_closed_deals(self, now: datetime) -> list[ClosedTrade]:
        mt5 = self.mt5
        start = datetime(now.year, now.month, now.day)
        deals = mt5.history_deals_get(start, now + timedelta(days=1)) or ()
        open_tickets = {p.ticket for p in self.conn.bot_positions(self.magic)}
        closed: list[ClosedTrade] = []
        for d in deals:
            if d.magic != self.magic or d.ticket in self._seen_deals:
                continue
            self._seen_deals.add(d.ticket)
            if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
                pnl = d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
                final = d.position_id not in open_tickets
                closed.append(
                    ClosedTrade(d.position_id, d.symbol, pnl, server_dt(d.time), d.comment, final)
                )
        return closed

    # -------------------------------------------------------- paper book
    def _manage_paper(
        self, now: datetime, ticks: dict[str, TickSnapshot], ctx: dict[str, MarketContext]
    ) -> list[ClosedTrade]:
        book = self.paper
        closed: list[ClosedTrade] = []
        for ticket, order in list(book.pending.items()):
            v, sig = order.view, order.signal
            if now >= v.expires:
                book.pending.pop(ticket)
                log.info("[DRY] limit #%s expired (price ran away) -> cancelled", ticket)
                continue
            tick = ticks.get(v.symbol)
            if tick and (
                (sig.is_buy and tick.ask <= v.price) or (not sig.is_buy and tick.bid >= v.price)
            ):
                book.pending.pop(ticket)
                book.positions[ticket] = PositionView(ticket, v.symbol, v.side, v.volume, v.price, v.sl,
                                                      v.tp, now, v.volume, abs(v.price - v.sl))  # fmt: skip
                log.info("[DRY] limit #%s filled @ %s", ticket, v.price)

        for pos in list(book.positions.values()):
            tick = ticks.get(pos.symbol)
            if tick is None:
                continue
            spec = self.feed.symbol_spec(pos.symbol)
            px = tick.bid if pos.side > 0 else tick.ask
            pos.profit = pos.side * (px - pos.entry) / spec.tick_size * spec.tick_value * pos.volume
            hit_sl = pos.sl and (px <= pos.sl if pos.side > 0 else px >= pos.sl)
            hit_tp = pos.tp and (px >= pos.tp if pos.side > 0 else px <= pos.tp)
            if hit_sl or hit_tp:
                closed.append(self._paper_close(pos, px, pos.volume, "SL" if hit_sl else "TP", now))
                continue
            for act in self.manager.evaluate(
                pos, tick.bid, tick.ask, ctx.get(pos.symbol, MarketContext()), now, spec
            ):
                if act.kind == "close":
                    closed.append(self._paper_close(pos, px, pos.volume, act.reason, now))
                    break
                if act.kind == "tp1":
                    if act.volume > 0:
                        closed.append(self._paper_close(pos, px, act.volume, act.reason, now))
                    pos.tp1_done = True
                elif act.kind == "modify_sl":
                    pos.sl = act.new_sl
                    pos.be_done = pos.be_done or "break-even" in act.reason
                    log.info("[DRY] SL #%s -> %s (%s)", pos.ticket, act.new_sl, act.reason)
        book.closed.extend(closed)
        return closed

    def _paper_close(self, pos: PositionView, px: float, volume: float, reason: str,
                     now: datetime) -> ClosedTrade:  # fmt: skip
        spec = self.feed.symbol_spec(pos.symbol)
        pnl = pos.side * (px - pos.entry) / spec.tick_size * spec.tick_value * volume
        pnl -= self.s.risk.commission_per_lot * volume
        pos.volume = round(pos.volume - volume, 8)
        final = pos.volume <= 1e-9
        if final:
            self.paper.positions.pop(pos.ticket, None)
        log.info("[DRY] close %.2f #%s @ %s (%s) pnl %.2f", volume, pos.ticket, px, reason, pnl)
        return ClosedTrade(pos.ticket, pos.symbol, pnl, now, reason, final)


def _side(side: int) -> str:
    return "BUY" if side > 0 else "SELL"


__all__ = [
    "ClosedTrade", "ExecutionEngine", "ExecutionResult", "ManagementAction", "MarketContext",
    "PaperBook", "PendingView", "PositionView", "TradeManager", "money_per_lot", "server_dt",
]  # fmt: skip

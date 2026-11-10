"""Live "Quant Desk" terminal dashboard built with ``rich``.

Rendering is decoupled from trading: the trading loop mutates a
:class:`src.core.state.BotState`, and :class:`Dashboard` renders snapshots of it
at ``refresh_per_second`` from its own thread.
"""

from __future__ import annotations

import threading
from datetime import datetime

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.state import BotState

_ARROW = {1: "[green]▲ BULL[/]", -1: "[red]▼ BEAR[/]", 0: "[dim]● FLAT[/]"}


def _pnl(v: float, fmt: str = "{:+,.2f}") -> str:
    color = "green" if v > 0 else "red" if v < 0 else "white"
    return f"[{color}]{fmt.format(v)}[/]"


def render_header(st: BotState) -> Panel:
    status = "[bold green]● ONLINE[/]" if st.connected else "[bold red]● OFFLINE[/]"
    mode_color = {"live": "bold red", "dryrun": "bold yellow"}.get(st.mode, "bold cyan")
    ks = (
        f"[bold white on red] KILL-SWITCH: {st.kill_switch} [/]"
        if st.kill_switch
        else "[green]armed[/]"
    )
    t = Text.from_markup(
        f"[bold gold1]XAU QUANT SCALPER[/]  │  mode [{mode_color}]{st.mode.upper()}[/]  │  {status}  │  "
        f"AutoTrading: {st.autotrading}  │  server {st.server_time:%Y-%m-%d %H:%M:%S}  │  "
        f"loop {st.loop_ms:.0f} ms  │  risk guard: {ks}"
        if st.server_time
        else f"[bold gold1]XAU QUANT SCALPER[/]  │  mode [{mode_color}]{st.mode.upper()}[/]  │  {status}"
    )
    return Panel(t, box=box.HEAVY, style="on grey11")


def render_account(st: BotState) -> Panel:
    day_pnl = st.equity - st.day_start_equity if st.day_start_equity else 0.0
    day_pct = day_pnl / st.day_start_equity * 100 if st.day_start_equity else 0.0
    tb = Table.grid(padding=(0, 2))
    tb.add_column(style="dim")
    tb.add_column(justify="right")
    tb.add_row("Account", st.account)
    tb.add_row("Balance", f"{st.balance:,.2f} {st.currency}")
    tb.add_row("Equity", f"[bold]{st.equity:,.2f}[/] {st.currency}")
    tb.add_row("Margin lvl", f"{st.margin_level:,.0f}%" if st.margin_level else "-")
    tb.add_row("Day P/L", f"{_pnl(day_pnl)} ({_pnl(day_pct, '{:+.2f}%')})")
    tb.add_row("Realized", _pnl(st.day_realized))
    tb.add_row("Trades today", str(st.trades_today))
    tb.add_row("Consec. losses", str(st.consecutive_losses))
    tb.add_row("Avg slippage", f"{st.avg_slippage:.1f} pt")
    tb.add_row("Reconnects", str(st.reconnects))
    return Panel(tb, title="[bold]Account & Risk", border_style="cyan")


def render_market(st: BotState) -> Panel:
    tb = Table(box=box.SIMPLE_HEAVY, expand=True)
    for col, just in [
        ("Symbol", "left"),
        ("Bid", "right"),
        ("Ask", "right"),
        ("Spread", "right"),
        ("Med", "right"),
        ("ATR", "right"),
        ("M15", "center"),
        ("M5", "center"),
        ("Vel/ATR", "right"),
        ("Guard", "left"),
        ("Last signal", "left"),
    ]:
        tb.add_column(col, justify=just, no_wrap=col != "Guard")
    for s in st.symbols.values():
        guard = "[green]OK[/]" if s.guard == "OK" else f"[yellow]{s.guard}[/]"
        vel_col = "bold magenta" if abs(s.velocity_x_atr) >= 1 else "white"
        tb.add_row(
            f"[bold]{s.symbol}[/]",
            f"{s.bid:.2f}",
            f"{s.ask:.2f}",
            f"{s.spread:.0f}",
            f"{s.spread_median:.0f}" if s.spread_median else "-",
            f"{s.atr:.2f}",
            _ARROW.get(s.bias, "-"),
            _ARROW.get(s.confirm, "-"),
            f"[{vel_col}]{s.velocity_x_atr:+.2f}[/]",
            guard,
            s.last_signal,
        )
    return Panel(tb, title="[bold]Market Scanner", border_style="gold1")


def render_positions(st: BotState) -> Panel:
    tb = Table(box=box.SIMPLE, expand=True)
    for col in ("Ticket", "Symbol", "Side", "Lots", "Entry", "SL", "TP", "R", "P/L", "Flags"):
        tb.add_column(col, justify="right" if col not in ("Symbol", "Side", "Flags") else "left")
    for p in st.positions:
        side = "[green]BUY[/]" if p.side > 0 else "[red]SELL[/]"
        tb.add_row(
            str(p.ticket),
            p.symbol,
            side,
            f"{p.volume:.2f}",
            f"{p.entry:.2f}",
            f"{p.sl:.2f}",
            f"{p.tp:.2f}",
            _pnl(p.r_now, "{:+.2f}"),
            _pnl(p.profit),
            p.flags,
        )
    if not st.positions:
        tb.add_row("-", "[dim]flat[/]", "", "", "", "", "", "", "", "")
    title = f"[bold]Open Positions[/] ({len(st.positions)})  pending: {st.pending_orders}"
    return Panel(tb, title=title, border_style="green")


def render_events(st: BotState, max_events: int) -> Panel:
    colors = {"info": "white", "trade": "bold green", "warn": "yellow", "error": "bold red"}
    with st.lock:
        events = list(st.events)[:max_events]
    lines = [
        Text.from_markup(f"[dim]{t:%H:%M:%S}[/] [{colors.get(lvl, 'white')}]{msg}[/]")
        for t, lvl, msg in events
    ]
    return Panel(
        Group(*lines) if lines else Text("waiting for events…", style="dim"),
        title="[bold]Event Log",
        border_style="magenta",
    )


def build_layout(st: BotState, max_events: int = 14) -> Layout:
    root = Layout()
    root.split_column(Layout(name="header", size=3), Layout(name="body"))
    root["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=1))
    root["left"].split_column(
        Layout(name="market", ratio=2),
        Layout(name="positions", ratio=2),
        Layout(name="events", ratio=3),
    )
    root["header"].update(render_header(st))
    root["right"].update(render_account(st))
    root["market"].update(render_market(st))
    root["positions"].update(render_positions(st))
    root["events"].update(render_events(st, max_events))
    return root


class Dashboard:
    """Background renderer. ``start()`` / ``stop()`` from the trading thread."""

    def __init__(self, state: BotState, refresh_per_second: int = 4, max_events: int = 14) -> None:
        self.state = state
        self.refresh = max(1, refresh_per_second)
        self.max_events = max_events
        self.console = Console()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        with Live(
            build_layout(self.state, self.max_events),
            console=self.console,
            refresh_per_second=self.refresh,
            screen=True,
        ) as live:
            while not self._stop.wait(1.0 / self.refresh):
                live.update(build_layout(self.state, self.max_events))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def demo_state() -> BotState:
    """Static sample state (used by tests and `python -m src.ui.dashboard`)."""
    from src.core.state import PositionRow, SymbolState

    st = BotState(
        mode="dryrun",
        strategy="xau_velocity_scalp",
        connected=True,
        autotrading="[green]ON[/]",
        server_time=datetime.now(),
        account="12345678 @ Demo",
        currency="USD",
        balance=10_000,
        equity=10_085.4,
        margin_level=2150,
        day_start_equity=10_000,
        day_realized=62.1,
        trades_today=4,
        avg_slippage=3.2,
        loop_ms=12,
    )
    st.symbols["XAUUSD"] = SymbolState(
        "XAUUSD", 2651.32, 2651.52, 20, 19, 1.85, 1, 1, 1.34, "OK", "BUY LMT 2650.98 @ 14:31"
    )
    st.positions.append(
        PositionRow(1, "XAUUSD", 1, 0.30, 2649.80, 2649.90, 2653.10, 0.92, 45.6, "BE")
    )
    st.event("Limit #1 filled @ 2649.80", "trade")
    st.event("Spread spike 48p > 1.8x median -> blocked", "warn")
    return st


if __name__ == "__main__":  # pragma: no cover - visual preview
    Console().print(build_layout(demo_state()))

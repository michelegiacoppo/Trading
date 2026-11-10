"""Live "Quant Desk" terminal dashboard built with ``rich``.

Panels: header (MT5 link, symbol, LIVE/DRY-RUN, active session), account & risk,
market microstructure (bid/ask, spread vs max with traffic light, ATR M1/M5,
volatility & gap state, nearest key levels), positions & pending orders, and a
live execution log with millisecond timestamps.

Rendering is decoupled from trading: the trading loop mutates a
:class:`src.core.state.BotState`; :class:`Dashboard` renders snapshots of it from
its own thread at ``refresh_per_second`` (2-4 Hz).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.state import BotState, PendingRow, PositionRow, SymbolState

_ARROW = {1: "[green]▲ BULL[/]", -1: "[red]▼ BEAR[/]", 0: "[dim]● FLAT[/]"}
_CCY = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥", "CHF": "CHF "}


def _pnl(v: float, fmt: str = "{:+,.2f}") -> str:
    color = "green" if v > 0 else "red" if v < 0 else "white"
    return f"[{color}]{fmt.format(v)}[/]"


def render_header(st: BotState) -> Panel:
    link = "[bold green]● MT5 ONLINE[/]" if st.connected else "[bold red]● MT5 OFFLINE[/]"
    mode = (
        "[bold white on red] LIVE [/]"
        if st.mode == "live"
        else "[bold black on yellow] DRY-RUN [/]"
    )
    sess_color = "green" if st.session not in ("Closed", "-") else "dim"
    clock = f"{st.local_time:%H:%M:%S} CET" if st.local_time else "--:--:--"
    t = Text.from_markup(
        f"[bold gold1]XAU QUANT DESK[/]  {mode}  {link}  │  [bold]{st.symbol or '-'}[/]  │  "
        f"session [{sess_color}]{st.session}[/]  │  {clock}  │  AutoTrading {st.autotrading}  │  "
        f"loop {st.loop_ms:.0f} ms"
    )
    return Panel(t, box=box.HEAVY, style="on grey11")


def render_account(st: BotState) -> Panel:
    ccy = _CCY.get(st.currency, f"{st.currency} ")
    tb = Table.grid(padding=(0, 2))
    tb.add_column(style="dim")
    tb.add_column(justify="right")
    tb.add_row("Account", st.account)
    tb.add_row("Balance", f"{ccy}{st.balance:,.2f}")
    tb.add_row("Equity", f"[bold]{ccy}{st.equity:,.2f}[/]")
    tb.add_row("Free margin", f"{ccy}{st.free_margin:,.2f}")
    lev = f"1:{st.leverage}" if st.leverage else "-"
    tb.add_row(
        "Leverage / ML", f"{lev} / {st.margin_level:,.0f}%" if st.margin_level else f"{lev} / -"
    )
    tb.add_row(
        "Daily P&L",
        f"{_pnl(st.day_pnl, '{:+,.2f}')} {ccy.strip()}  ({_pnl(st.day_pnl_pct, '{:+.2f}%')})",
    )
    tb.add_row("Trades today", str(st.trades_today))
    tb.add_row("Consec. losses", str(st.consecutive_losses))
    tb.add_row("Avg slippage", f"{st.avg_slippage:.1f} pt")
    if st.kill_status == "LOCKED":
        ks = f"[bold white on red] LOCKED [/]\n[red]{st.kill_reason}[/]"
    else:
        ks = "[bold black on green] ACTIVE [/]"
    tb.add_row("Kill-switch", ks)
    tb.add_row("Reconnects", str(st.reconnects))
    tb.add_row("Warm-up", st.warmup)
    return Panel(tb, title="[bold]Account & Risk", border_style="cyan")


def _micro_table(s: SymbolState) -> Table:
    d = s.digits
    tb = Table.grid(padding=(0, 2))
    tb.add_column(style="dim")
    tb.add_column(justify="right")
    tb.add_row("Bid / Ask", f"[bold]{s.bid:.{d}f}[/] / [bold]{s.ask:.{d}f}[/]")
    light = "[green]●[/]" if s.max_spread and s.spread <= s.max_spread else "[red]●[/]"
    med = f" (med {s.spread_median:.0f})" if s.spread_median else ""
    tb.add_row("Spread", f"{light} {s.spread:.0f} / max {s.max_spread:.0f} pt{med}")
    tb.add_row("ATR M1 / M5", f"{s.atr_m1:.{d}f} / {s.atr_m5:.{d}f}")
    tb.add_row("Volatility", s.vol_state)
    gap = "[green]OK[/]" if s.gap_state == "OK" else f"[bold red]{s.gap_state}[/]"
    tb.add_row("Gap / jump", gap)
    tb.add_row("Bias M15 / M5", f"{_ARROW.get(s.bias, '-')} / {_ARROW.get(s.confirm, '-')}")
    tb.add_row("Momentum", f"{s.momentum:+.2f} xATR")
    guard = "[green]READY[/]" if s.guard == "OK" else f"[yellow]{s.guard}[/]"
    tb.add_row("Entry guard", guard)
    tb.add_row("Last signal", s.last_signal)
    return tb


def _levels_table(s: SymbolState) -> Table:
    tb = Table(box=box.SIMPLE, show_header=True, header_style="dim", expand=True)
    tb.add_column("Key level")
    tb.add_column("Price", justify="right")
    tb.add_column("Dist", justify="right")
    for label, lvl, dist in s.key_levels:
        color = "red" if dist > 0 else "green"
        tb.add_row(label, f"{lvl:.{s.digits}f}", f"[{color}]{dist:+.{s.digits}f}[/]")
    if not s.key_levels:
        tb.add_row("[dim]n/a[/]", "", "")
    return tb


def render_market_compact(st: BotState) -> Panel:
    """One row per symbol - used when trading many symbols."""
    tb = Table(box=box.SIMPLE_HEAVY, expand=True)
    for col, just in (("Symbol", "left"), ("Bid", "right"), ("Ask", "right"), ("Spread", "right"),
                      ("ATR M1", "right"), ("M15", "center"), ("Mom", "right"), ("Vol", "left"),
                      ("Gap", "left"), ("Guard", "left"), ("Last signal", "left")):  # fmt: skip
        tb.add_column(col, justify=just, no_wrap=col not in ("Guard", "Last signal"))
    for s in st.symbols.values():
        d = s.digits
        light = "[green]●[/]" if s.max_spread and s.spread <= s.max_spread else "[red]●[/]"
        gap = "[green]OK[/]" if s.gap_state == "OK" else "[bold red]BLOCK[/]"
        guard = "[green]READY[/]" if s.guard == "OK" else f"[yellow]{s.guard[:38]}[/]"
        tb.add_row(f"[bold]{s.symbol}[/]", f"{s.bid:.{d}f}", f"{s.ask:.{d}f}",
                   f"{light} {s.spread:.0f}/{s.max_spread:.0f}", f"{s.atr_m1:.{d}f}",
                   _ARROW.get(s.bias, "-"), f"{s.momentum:+.1f}", s.vol_state, gap, guard,
                   s.last_signal)  # fmt: skip
    return Panel(
        tb, title=f"[bold]Market Microstructure ({len(st.symbols)} symbols)", border_style="gold1"
    )


def render_market(st: BotState) -> Panel:
    if len(st.symbols) > 2:
        return render_market_compact(st)
    blocks = []
    for s in st.symbols.values():
        grid = Table.grid(expand=True)
        grid.add_column(ratio=3)
        grid.add_column(ratio=2)
        grid.add_row(_micro_table(s), _levels_table(s))
        blocks.append(
            Panel(grid, title=f"[bold gold1]{s.symbol}", border_style="grey37", box=box.ROUNDED)
        )
    body = Group(*blocks) if blocks else Text("no symbols", style="dim")
    return Panel(body, title="[bold]Market Microstructure", border_style="gold1")


def render_positions(positions: list[PositionRow], pending: list[PendingRow]) -> Panel:
    tb = Table(box=box.SIMPLE, expand=True)
    for col in ("Ticket", "Type", "Symbol", "Lots", "Open", "SL", "TP", "R", "BE", "Profit"):
        tb.add_column(col, justify="left" if col in ("Type", "Symbol", "BE") else "right")
    for p in positions:
        side = "[green]BUY[/]" if p.side > 0 else "[red]SELL[/]"
        be = (
            "[green]✔ BE[/]" + (" [cyan]TP1[/]" if p.tp1 else "")
            if p.be
            else ("[cyan]TP1[/]" if p.tp1 else "[dim]–[/]")
        )
        tb.add_row(str(p.ticket), side, p.symbol, f"{p.volume:.2f}", f"{p.entry:.2f}", f"{p.sl:.2f}",
                   f"{p.tp:.2f}" if p.tp else "-", _pnl(p.r_now, "{:+.2f}"), be, _pnl(p.profit))  # fmt: skip
    for o in pending:
        side = "[green]BUY LMT[/]" if o.side > 0 else "[red]SELL LMT[/]"
        tb.add_row(f"[dim]{o.ticket}[/]", side, o.symbol, f"{o.volume:.2f}", f"{o.price:.2f}",
                   f"{o.sl:.2f}", f"{o.tp:.2f}" if o.tp else "-", "", f"[dim]exp {o.expires:%H:%M:%S}[/]",
                   "[dim]pending[/]")  # fmt: skip
    if not positions and not pending:
        tb.add_row("-", "[dim]flat[/]", "", "", "", "", "", "", "", "")
    title = f"[bold]Active Positions ({len(positions)}) & Pending Orders ({len(pending)})"
    return Panel(tb, title=title, border_style="green")


def render_events(st: BotState, max_events: int) -> Panel:
    colors = {"info": "white", "trade": "bold green", "warn": "yellow", "error": "bold red"}
    with st.lock:
        events = list(st.events)[:max_events]
    lines = [
        Text.from_markup(
            f"[dim]{t:%H:%M:%S}.{t.microsecond // 1000:03d}[/] [{colors.get(lvl, 'white')}]{msg}[/]"
        )  # fmt: skip
        for t, lvl, msg in events
    ]
    return Panel(Group(*lines) if lines else Text("waiting for events…", style="dim"),
                 title="[bold]Live Execution Log", border_style="magenta")  # fmt: skip


def build_layout(st: BotState, max_events: int = 10) -> Layout:
    root = Layout()
    root.split_column(Layout(name="header", size=3), Layout(name="body"))
    root["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=1))
    root["left"].split_column(Layout(name="market", ratio=5), Layout(name="positions", ratio=3),
                              Layout(name="events", ratio=4))  # fmt: skip
    root["header"].update(render_header(st))
    root["right"].update(render_account(st))
    root["market"].update(render_market(st))
    root["positions"].update(render_positions(st.positions, st.pending))
    root["events"].update(render_events(st, max_events))
    return root


class Dashboard:
    """Background renderer. ``start()`` / ``stop()`` from the trading thread."""

    def __init__(self, state: BotState, refresh_per_second: int = 4, max_events: int = 10) -> None:
        self.state = state
        self.refresh = min(max(1, refresh_per_second), 10)
        self.max_events = max_events
        self.console = Console()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        with Live(build_layout(self.state, self.max_events), console=self.console,
                  refresh_per_second=self.refresh, screen=True) as live:  # fmt: skip
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
    """Static sample state (used by tests and ``python -m src.ui.dashboard``)."""
    now = datetime.now()
    st = BotState(mode="dryrun", strategy="xau_velocity_scalp", symbol="XAUUSD", session="London",
                  connected=True, autotrading="[green]ON[/]", server_time=now, local_time=now,
                  account="12345678 @ Demo", currency="EUR", balance=10_000, equity=10_085.4,
                  free_margin=9_300, margin_level=2150, leverage=100, day_pnl=85.4, day_pnl_pct=0.85,
                  trades_today=4, avg_slippage=3.2, loop_ms=12, warmup="M1:1000, M5:1000")  # fmt: skip
    st.symbols["XAUUSD"] = SymbolState(
        "XAUUSD",
        2651.32,
        2651.52,
        2,
        20,
        35,
        19,
        1.85,
        3.9,
        1,
        1,
        1.4,
        "[green]NORMAL[/]",
        "OK",
        "OK",
        "BUY LIMIT 2650.98 [Asia Low] @ 09:31",
        [("Asia Low", 2649.80, -1.52), ("VWAP", 2652.40, 1.08), ("PDH", 2655.10, 3.78)],
    )
    st.positions.append(
        PositionRow(1, "XAUUSD", 1, 0.11, 2649.80, 2650.12, 2659.00, 1.12, 21.6, True, True)
    )
    st.pending.append(
        PendingRow(2, "XAUUSD", -1, 0.20, 2655.40, 2657.30, 2649.70, now + timedelta(minutes=3))
    )
    st.event("Limit #1 filled @ 2649.80", "trade")
    st.event("TP1 65% closed @ +1.00R, SL -> BE + costs", "trade")
    st.event("Spread spike 48p > 1.8x median -> blocked", "warn")
    return st


if __name__ == "__main__":  # pragma: no cover - visual preview
    Console(width=180, height=48).print(build_layout(demo_state()))

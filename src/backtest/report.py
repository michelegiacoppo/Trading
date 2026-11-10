"""Institutional-style backtest report rendered with ``rich``."""

from __future__ import annotations

import math

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.engine import BacktestResult, hourly_analysis
from src.config import Settings
from src.core.clock import ServerClock


def _fmt(v: float, fmt: str = "{:,.2f}") -> str:
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return "∞"
    return fmt.format(v)


def _col(v: float, good: float = 0.0) -> str:
    if isinstance(v, float) and math.isinf(v):
        return "green"
    return "green" if v > good else "red" if v < good else "white"


def losing_slots(hourly: pd.DataFrame, min_trades: int) -> list[str]:
    if hourly.empty:
        return []
    bad = hourly[(hourly["trades"] >= min_trades) & (hourly["net_usd"] < 0)]
    return list(bad.index)


def print_report(res: BacktestResult, settings: Settings, symbol: str, clock: ServerClock,
                 console: Console | None = None) -> pd.DataFrame:  # fmt: skip
    con = console or Console()
    st, meta = res.stats, res.meta
    tz = settings.sessions.timezone

    synthetic = str(meta.get("spread_source", "")).startswith("SYNTHETIC")
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim")
    head.add_column()
    head.add_row(
        "Strategy / symbol", f"[bold]{settings.strategy.name}[/] on [bold gold1]{symbol}[/]"
    )
    head.add_row(
        "Period (server)", f"{meta.get('from')} → {meta.get('to')}  ({meta.get('bars'):,} M1 bars)"
    )
    head.add_row("Spread source", f"[{'bold red' if synthetic else 'green'}]{meta.get('spread_source')}[/]"
                 f"  avg {meta.get('avg_spread_points', 0):.1f} pt")  # fmt: skip
    head.add_row("Costs", f"slippage {settings.backtest.simulated_slippage_points} pt on market/stop · "
                 f"commission {settings.risk.commission_per_lot} /lot RT")  # fmt: skip
    ticks = "real ticks" if meta.get("ticks_available") else "no ticks → worst case (SL first)"
    head.add_row("Intra-bar SL/TP", f"{ticks} · resolved by ticks {meta.get('intrabar_resolved_by_ticks', 0)}"
                 f" · pessimistic {meta.get('intrabar_pessimistic', 0)}")  # fmt: skip
    con.print(Panel(head, title="[bold]XAU Quant Scalper — Backtest Report", border_style="gold1"))

    if not res.trades:
        con.print("[yellow]No trades generated in this period.[/]")
        return pd.DataFrame()

    perf = Table(title="Performance", box=box.SIMPLE_HEAVY)
    perf.add_column("Metric")
    perf.add_column("Value", justify="right")
    rows = [
        ("Trades", _fmt(st["trades"], "{:.0f}"), "white"),
        ("Net profit", _fmt(st["net_profit"]) + f"  ({st['return_pct']:+.2f}%)", _col(st["net_profit"])),
        ("Win rate", _fmt(st["win_rate_pct"], "{:.1f}%"), _col(st["win_rate_pct"], 50)),
        ("Profit factor", _fmt(st["profit_factor"]), _col(st["profit_factor"], 1)),
        ("Expectancy / trade", f"{_fmt(st['expectancy_usd'])} $  |  {st['expectancy_r']:+.3f} R",
         _col(st["expectancy_usd"])),
        ("Avg win / avg loss", f"{_fmt(st['avg_win_usd'])} / {_fmt(st['avg_loss_usd'])}", "white"),
        ("Max drawdown", f"{st['max_drawdown_pct']:.2f}%  ({_fmt(st['max_drawdown_usd'])} $)", "red"),
        ("Sharpe (daily, ann.)", _fmt(st["sharpe"]), _col(st["sharpe"], 1)),
        ("Sortino", _fmt(st["sortino"]), _col(st["sortino"], 1)),
        ("Recovery factor", _fmt(st["recovery_factor"]), _col(st["recovery_factor"], 1)),
        ("Max consecutive losses", _fmt(st["max_consecutive_losses"], "{:.0f}"), "white"),
        ("Avg holding time", _fmt(st["avg_hold_minutes"], "{:.1f} min"), "white"),
    ]  # fmt: skip
    for name, val, color in rows:
        perf.add_row(name, f"[{color}]{val}[/]")

    exe = Table(title="Execution quality", box=box.SIMPLE_HEAVY)
    exe.add_column("Metric")
    exe.add_column("Value", justify="right")
    placed = int(meta.get("limits_placed", 0))
    filled = int(meta.get("limits_filled", 0))
    exe.add_row("Signals (raw)", str(meta.get("signals", 0)))
    exe.add_row("Blocked by guards/sizing", str(meta.get("blocked", 0)))
    exe.add_row("Limit orders placed", str(placed))
    exe.add_row("Limit fill rate", f"{filled / placed * 100:.0f}%" if placed else "-")
    exe.add_row("Limits expired (no chase)", str(meta.get("limits_expired", 0)))
    exe.add_row("Kill-switch locks", str(meta.get("kill_switch_locks", 0)))
    tdf = res.trades_frame()
    for label, sub in (("Key-level sweeps", tdf[tdf["key_level"] != ""]),
                       ("Other setups", tdf[tdf["key_level"] == ""])):  # fmt: skip
        if len(sub):
            exe.add_row(label, f"{len(sub)} tr · win {(sub['pnl'] > 0).mean() * 100:.0f}% · "
                        f"{sub['r_multiple'].mean():+.2f}R")  # fmt: skip

    con.print(perf)
    con.print(exe)

    # ------------------------------------------------------- hourly analysis
    hourly = hourly_analysis(res, clock, tz)
    ht = Table(title=f"Hourly analysis (entry time, {tz})", box=box.SIMPLE)
    for col, just in (("Slot", "left"), ("Trades", "right"), ("Win %", "right"),
                      ("Net $", "right"), ("Avg R", "right"), ("PF", "right"), ("", "left")):  # fmt: skip
        ht.add_column(col, justify=just)
    max_abs = max(1.0, hourly["net_usd"].abs().max())
    for slot, r in hourly.iterrows():
        bar_len = int(abs(r["net_usd"]) / max_abs * 20)
        color = _col(r["net_usd"])
        ht.add_row(str(slot), f"{int(r['trades'])}", f"{r['win_rate_pct']:.0f}%",
                   f"[{color}]{r['net_usd']:+,.2f}[/]", f"[{color}]{r['avg_r']:+.2f}[/]",
                   _fmt(r["profit_factor"]), f"[{color}]{'█' * bar_len}[/]")  # fmt: skip
    con.print(ht)
    bad = losing_slots(hourly, settings.backtest.hourly_min_trades)
    if bad:
        con.print(f"[yellow]Losing slots (≥{settings.backtest.hourly_min_trades} trades, net < 0):[/] "
                  f"{', '.join(bad)}\n  → to exclude them add to settings.yaml: "
                  f"[bold]sessions.blocked_slots: {bad}[/]")  # fmt: skip
    if synthetic:
        con.print("[bold red]⚠ Spread was synthetic: results are NOT reliable for XAUUSD.[/]")
    return hourly

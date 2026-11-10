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
    """Slots that are clearly bad: enough trades AND avg R <= -0.3 AND PF < 0.7."""
    if hourly.empty:
        return []
    bad = hourly[(hourly["trades"] >= min_trades) & (hourly["avg_r"] <= -0.3)
                 & (hourly["profit_factor"] < 0.7)]  # fmt: skip
    return list(bad.index)


def print_symbol_table(results: dict[str, BacktestResult], con: Console, title: str) -> None:
    """One row per symbol: where the strategy works and where it does not."""
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    for col in ("Symbol", "Trades", "Tr/week", "Win %", "PF", "Exp R", "Net $", "Max DD %",
                "Avg spread", "Signals", "Blocked"):  # fmt: skip
        t.add_column(col, justify="left" if col == "Symbol" else "right")
    ordered = sorted(results.items(), key=lambda kv: -kv[1].stats.get("net_profit", 0.0))
    for sym, r in ordered:
        st, m = r.stats, r.meta
        n = int(st.get("trades", 0))
        exp_r = st.get("expectancy_r", 0.0)
        net = st.get("net_profit", 0.0)
        pf = st.get("profit_factor", 0.0)
        t.add_row(
            sym, str(n), f"{st.get('trades_per_week', 0.0):.1f}",
            f"{st.get('win_rate_pct', 0.0):.0f}%" if n else "-",
            _fmt(pf) if n else "-",
            f"[{_col(exp_r)}]{exp_r:+.3f}[/]" if n else "-",
            f"[{_col(net)}]{net:+,.2f}[/]",
            f"{st.get('max_drawdown_pct', 0.0):.2f}" if n else "-",
            f"{m.get('avg_spread_points', 0.0):.1f} pt",
            str(m.get("signals", 0)), str(m.get("blocked", 0)),
        )  # fmt: skip
    con.print(t, width=max(con.width, 130))


def print_diagnostics(res: BacktestResult, con: Console) -> None:
    """Signal funnel + reasons why signals were not traded."""
    meta = res.meta
    funnel = meta.get("funnel") or []
    if funnel:
        ft = Table(title="Signal funnel (M1 bars passing each filter, cumulative)", box=box.SIMPLE)
        ft.add_column("Filter")
        ft.add_column("Long", justify="right")
        ft.add_column("Short", justify="right")
        ft.add_column("Kept", justify="right")
        total = (funnel[0][1] + funnel[0][2]) or 1
        prev = total
        for label, n_long, n_short in funnel:
            tot = n_long + n_short
            if label.startswith("M1 bars"):
                kept = ""
            elif label.startswith(("  ", "TRIGGER")):
                kept = f"{tot / total * 100:.1f}% of bars"  # branch rows: share of all bars
            else:
                kept = f"{tot / prev * 100:.0f}%" if prev else ""
            style = "bold" if label.startswith(("=", "TRIGGER")) else ""
            ft.add_row(
                f"[{style}]{label}[/]" if style else label, f"{n_long:,}", f"{n_short:,}", kept
            )
            if not label.startswith("  "):
                prev = tot
        con.print(ft)
    reasons = meta.get("block_reasons") or {}
    if reasons:
        bt = Table(title="Signals not traded — reason", box=box.SIMPLE)
        bt.add_column("Reason")
        bt.add_column("Count", justify="right")
        for r, n in reasons.items():
            bt.add_row(r, str(n))
        con.print(bt)
    if meta.get("signals", 0) and res.equity_curve.size:
        days = (res.equity_curve.index[-1] - res.equity_curve.index[0]).days or 1
        per_week = meta["signals"] / days * 7
        if per_week < 3:
            con.print(f"[yellow]Only {per_week:.1f} raw signals/week: filters are very selective, "
                      "results will not be statistically meaningful. See the funnel above.[/]")  # fmt: skip


def print_report(res: BacktestResult, settings: Settings, symbol: str, clock: ServerClock,
                 console: Console | None = None) -> pd.DataFrame:  # fmt: skip
    con = console or Console()
    st, meta = res.stats, res.meta
    tz = settings.sessions.timezone

    synthetic = str(meta.get("spread_source", "")).startswith("SYNTHETIC")
    if str(meta.get("spread_source", "")).startswith("ZERO"):
        con.print("[bold white on red] GROSS TEST: spread, slippage and commissions removed. "
                  "Diagnostic only - NOT a tradable result. [/]")  # fmt: skip
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim")
    head.add_column()
    head.add_row(
        "Strategy / symbol", f"[bold]{settings.strategy.name}[/] on [bold gold1]{symbol}[/]"
    )
    head.add_row(
        "Period (server)", f"{meta.get('from')} → {meta.get('to')}  ({meta.get('bars'):,} M1 bars)"
    )
    avg_sp = meta.get("avg_spread_points")
    avg_txt = f"  avg {avg_sp:.1f} pt" if isinstance(avg_sp, float) and avg_sp == avg_sp else ""
    head.add_row("Spread source", f"[{'bold red' if synthetic else 'green'}]{meta.get('spread_source')}[/]"
                 + avg_txt)  # fmt: skip
    head.add_row("Costs", f"slippage {settings.backtest.simulated_slippage_points} pt on market/stop · "
                 f"commission {settings.risk.commission_per_lot} /lot RT")  # fmt: skip
    ticks = "real ticks" if meta.get("ticks_available") else "no ticks → worst case (SL first)"
    head.add_row("Intra-bar SL/TP", f"{ticks} · resolved by ticks {meta.get('intrabar_resolved_by_ticks', 0)}"
                 f" · pessimistic {meta.get('intrabar_pessimistic', 0)}")  # fmt: skip
    con.print(Panel(head, title="[bold]XAU Quant Scalper — Backtest Report", border_style="gold1"))

    print_diagnostics(res, con)
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
        ("Expectancy 95% CI (R)",
         f"{st['expectancy_r'] - st.get('expectancy_r_ci95', 0):+.3f} … "
         f"{st['expectancy_r'] + st.get('expectancy_r_ci95', 0):+.3f}  (t = {st.get('expectancy_r_tstat', 0):+.1f})",
         "green" if st["expectancy_r"] - st.get("expectancy_r_ci95", 0) > 0
         else "red" if st["expectancy_r"] + st.get("expectancy_r_ci95", 0) < 0 else "yellow"),
        ("Avg win / avg loss", f"{_fmt(st['avg_win_usd'])} / {_fmt(st['avg_loss_usd'])}", "white"),
        ("Max drawdown", f"{st['max_drawdown_pct']:.2f}%  ({_fmt(st['max_drawdown_usd'])} $)", "red"),
        ("Sharpe (daily, ann.)", _fmt(st["sharpe"]), _col(st["sharpe"], 1)),
        ("Sortino", _fmt(st["sortino"]), _col(st["sortino"], 1)),
        ("Recovery factor", _fmt(st["recovery_factor"]), _col(st["recovery_factor"], 1)),
        ("Max consecutive losses", _fmt(st["max_consecutive_losses"], "{:.0f}"), "white"),
        ("Trades per week", _fmt(st.get("trades_per_week", 0.0), "{:.1f}"), "white"),
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
    if st.get("expectancy_r", 0) + st.get("expectancy_r_ci95", 0) < 0:
        con.print("[bold red]The strategy is losing with statistical confidence: filtering time slots "
                  "will not fix it (it only fits the past). Rework the signal first.[/]")  # fmt: skip
    elif bad and len(bad) <= max(2, len(hourly) // 4):
        con.print(f"[yellow]Clearly losing slots (≥{settings.backtest.hourly_min_trades} trades, avg ≤ -0.3R, "
                  f"PF < 0.7):[/] {', '.join(bad)}\n  → candidates for sessions.blocked_slots - "
                  "confirm on out-of-sample data before excluding them.")  # fmt: skip
    if synthetic:
        con.print("[bold red]⚠ Spread was synthetic: results are NOT reliable for XAUUSD.[/]")
    return hourly

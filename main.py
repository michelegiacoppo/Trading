"""CLI entry point.

python main.py --mode diag                          # broker / symbol diagnostics (no orders)
python main.py --mode dryrun                        # live data, paper execution, TUI
python main.py --mode live --confirm-live           # REAL orders
python main.py --mode download --days 180           # fill the Parquet cache (M1/M5/M15/H1)
python main.py --mode backtest --days 60            # download 60 days from MT5 + report
python main.py --mode backtest --days 30 --ticks    # + real ticks (spread & intra-bar path)
python main.py --mode backtest --days 60 --offline  # use the local cache only
python main.py --mode backtest --days 90 --optimize # in-sample / out-of-sample optimizer
python main.py --mode backtest --csv data/XAUUSD_M1.csv
python main.py --mode backtest --synthetic          # smoke test without data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT, Settings, load_credentials, load_settings, resolve_path
from src.core.clock import ServerClock
from src.core.state import BotState
from src.strategies.xau_velocity_scalp import create_strategy

console = Console(record=True)  # everything printed is also saved to logs/ultimo_report.txt


class StateLogHandler(logging.Handler):
    """Mirror WARNING+ log records into the dashboard event log."""

    def __init__(self, state: BotState) -> None:
        super().__init__(level=logging.WARNING)
        self.state = state

    def emit(self, record: logging.LogRecord) -> None:
        level = "error" if record.levelno >= logging.ERROR else "warn"
        self.state.event(record.getMessage(), level)


def setup_logging(settings: Settings, mode: str, to_console: bool) -> Path:
    log_dir = PROJECT_ROOT / settings.general.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_{datetime.now():%Y%m%d}.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if to_console:
        from rich.logging import RichHandler

        handlers.append(RichHandler(console=console, show_path=False, level=logging.WARNING))
    logging.basicConfig(
        level=settings.general.log_level.upper(),
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return log_file


def _mt5_store(settings: Settings, need_mt5: bool):  # noqa: ANN202
    """Return (connector|None, HistoricalStore, ServerClock)."""
    from src.core.data_feed import HistoricalStore
    from src.core.mt5_connector import MT5Connector

    conn = None
    if need_mt5:
        conn = MT5Connector(load_credentials(), settings.mt5)
        if not conn.available or not conn.connect():
            conn = None
    store = HistoricalStore(resolve_path(settings.data.historical_dir), conn,
                            settings.data.download_chunk_days)  # fmt: skip
    clock = ServerClock(settings.sessions.backtest_server_timezone)
    if conn is not None and settings.sessions.server_timezone == "auto":
        tick = (
            conn.mt5.symbol_info_tick(settings.enabled_symbols[0])
            if settings.enabled_symbols
            else None
        )
        auto = ServerClock()
        if tick and auto.set_offset_from_tick(
            pd.Timestamp(int(tick.time), unit="s").to_pydatetime()
        ):
            clock = auto  # else (market closed / stale tick): keep backtest_server_timezone
    elif settings.sessions.server_timezone != "auto":
        clock = ServerClock(settings.sessions.server_timezone)
    return conn, store, clock


# ------------------------------------------------------------------ modes
def run_download(settings: Settings, args: argparse.Namespace) -> int:
    conn, store, _ = _mt5_store(settings, need_mt5=True)
    if conn is None:
        console.print("[red]MT5 not available: download needs the terminal (Windows).[/]")
        return 1
    try:
        for sym in args.resolved_symbols:
            if not conn.select_symbol(sym):
                console.print(f"[yellow]{sym}: not available on this broker, skipped[/]")
                continue
            for tf in settings.data.warmup_timeframes:
                df = store.fetch_historical_data(sym, tf, args.days)
                console.print(f"{sym} {tf}: {len(df):,} bars cached")
            if args.ticks:
                end = datetime.now()
                ticks = store.fetch_historical_ticks(sym, end - timedelta(days=args.days), end)
                console.print(f"{sym} ticks: {len(ticks):,} cached")
    finally:
        conn.shutdown()
    return 0


def load_backtest_data(settings: Settings, args: argparse.Namespace, symbols: list[str]):  # noqa: ANN201
    """Return ({symbol: (m1, ticks|None, broker SymbolSpec|None)}, clock)."""
    from src.backtest.engine import load_m1_csv, synthetic_m1

    clock = ServerClock(settings.sessions.backtest_server_timezone)
    if args.csv:  # a CSV holds one instrument
        return {symbols[0]: (load_m1_csv(args.csv), None, None)}, clock
    if args.synthetic:  # synthetic prices are gold-like: one symbol only
        return {symbols[0]: (synthetic_m1(n_bars=args.bars), None, None)}, clock

    conn, store, clock = _mt5_store(settings, need_mt5=not args.offline)
    if conn is None and not args.offline:
        console.print("[yellow]MT5 not available → using the local Parquet cache.[/]")
    data: dict[str, tuple] = {}
    short_history: list[str] = []
    try:
        for symbol in symbols:
            spec = None
            if conn is not None:
                from src.core.data_feed import DataFeed

                if not conn.select_symbol(symbol):
                    console.print(f"[yellow]{symbol}: not available on this broker, skipped[/]")
                    continue
                spec = DataFeed(conn).symbol_spec(symbol)  # real tick_value in account currency
                console.print(
                    f"{symbol}: updating {args.days} days of M1 from MT5 (incremental cache)…"
                )
                m1 = store.fetch_historical_data(symbol, "M1", args.days)
            else:
                m1 = store.slice_days(store.load_bars(symbol, "M1"), args.days)
            if len(m1) < 2000:
                console.print(f"[yellow]{symbol}: no/too little M1 data, skipped[/]")
                continue
            m1 = m1.iloc[:-1]  # drop the possibly-forming last bar
            span_days = (m1["time"].iloc[-1] - m1["time"].iloc[0]).days
            if span_days < args.days * 0.8:
                short_history.append(f"{symbol} ({span_days}d)")
            ticks = None
            if args.ticks:
                start, end = m1["time"].iloc[0].to_pydatetime(), m1["time"].iloc[-1].to_pydatetime()
                if conn is not None:
                    console.print(f"{symbol}: downloading real ticks (COPY_TICKS_ALL)…")
                    ticks = store.fetch_historical_ticks(symbol, start, end)
                else:
                    ticks = store.load_ticks(symbol, start, end)
            data[symbol] = (m1, ticks, spec)
    finally:
        if conn is not None:
            conn.shutdown()
    if short_history:
        console.print(
            f"[bold yellow]⚠ Requested {args.days} days but MT5 returned less for: "
            f"{', '.join(short_history)}.[/]\n"
            "  Fix: MT5 > Tools > Options > Charts > 'Max bars in chart' = Unlimited, restart MT5,\n"
            "  open an M1 chart of the symbol and scroll back (Home key), then run again."
        )
    if not data:
        raise SystemExit(
            "No data for any symbol: run `--mode download` on the MT5 host or use --csv."
        )
    return data, clock


def apply_gross(settings: Settings) -> None:
    """Remove all trading costs (diagnostic: separates 'no edge' from 'edge eaten by costs')."""
    settings.backtest.gross = True
    settings.backtest.simulated_slippage_points = 0.0
    settings.risk.commission_per_lot = 0.0


def run_portfolio(settings: Settings, data: dict[str, tuple], clock: ServerClock):  # noqa: ANN201
    """Backtest every symbol, then replay all trades on ONE account with global risk rules.

    Returns (portfolio result, standalone per-symbol results).
    """
    from src.backtest.engine import BacktestEngine, combine_results
    from src.lab.portfolio import PortfolioRules, candidate_settings, simulate_portfolio

    cand = candidate_settings(settings)
    results = {}
    strategy = create_strategy(cand.strategy, cand.levels, clock)
    for sym, (m1, ticks, spec) in data.items():
        results[sym] = BacktestEngine(cand, strategy, sym, spec=spec, clock=clock).run(m1, ticks)
    merged = combine_results(results, settings.backtest.initial_equity)
    trades = [t for r in results.values() for t in r.trades]
    rules = PortfolioRules.from_risk(settings.risk, settings.backtest.initial_equity)
    port = simulate_portfolio(trades, rules, ref_risk_pct=settings.risk.risk_per_trade_pct)
    port.meta = {**merged.meta, **port.meta, "portfolio": True}
    if not len(port.equity_curve):
        port.equity_curve = merged.equity_curve
    return port, results


def run_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from src.backtest.engine import save_results
    from src.backtest.report import print_report, print_symbol_table

    if not (args.csv or args.synthetic or args.days):
        console.print("[red]Backtest needs --days N (MT5/cache), --csv <file> or --synthetic[/]")
        return 2
    data, clock = load_backtest_data(settings, args, args.resolved_symbols)

    if args.compare:
        return run_compare(settings, args, data, clock)
    if args.optimize:
        return run_optimize(settings, data, clock)

    label = next(iter(data)) if len(data) == 1 else f"PORTFOLIO ({len(data)} symbols)"
    console.print(f"Backtesting {', '.join(data)} …")
    res, per_symbol = run_portfolio(settings, data, clock)
    print_report(res, settings, label, clock, console)
    if len(per_symbol) > 1:
        print_symbol_table(per_symbol, console, "Results by symbol")
    curve, trades = save_results(res, settings)
    console.print(f"[dim]Equity curve → {curve} | trades → {trades}[/]")
    return 0


def run_optimize(settings: Settings, data: dict[str, tuple], clock: ServerClock) -> int:
    from src.backtest.optimizer import DEFAULT_GRID, optimize

    combos = 1
    for v in DEFAULT_GRID.values():
        combos *= len(v)
    console.print(f"[bold]Auto-optimizer[/]: {combos} combinations x {len(data)} symbols "
                  f"(~{combos * len(data) * 8 / 60:.0f} min). Ranking in-sample, validating out-of-sample…")  # fmt: skip
    top = optimize(settings, data, clock)
    table = Table(
        title=f"Top combinations — in-sample → out-of-sample (split {top.attrs.get('split_time')})"
    )
    params = [c for c in top.columns if "." in c]
    metrics = [("IS_trades", "IS n"), ("IS_expectancy_r", "IS exp R"), ("IS_profit_factor", "IS PF"),
               ("OOS_trades", "OOS n"), ("OOS_win_rate_pct", "OOS win%"),
               ("OOS_expectancy_r", "OOS exp R"), ("OOS_profit_factor", "OOS PF"), ("OOS_net", "OOS net $")]  # fmt: skip
    for col in params:
        table.add_column(col.split(".", 1)[1], justify="right")
    for _, head in metrics:
        table.add_column(head, justify="right")
    for _, row in top.iterrows():
        cells = [str(row[c]) for c in params]
        for key, _ in metrics:
            v = row[key]
            txt = f"{v:.0f}" if key.endswith("trades") else f"{v:.2f}"
            if key.startswith("OOS_exp"):
                txt = f"[{'green' if v > 0 else 'red'}]{txt}[/]"
            cells.append(txt)
        table.add_row(*cells)
    console.print(table, width=max(console.width, 150))
    console.print("[dim]Adopt a combination only if its OOS expectancy/PF stay positive.[/]")
    return 0


def run_compare(
    settings: Settings, args: argparse.Namespace, data: dict[str, tuple], clock: ServerClock
) -> int:
    """Run the base config and every profile on the same data and compare them side by side."""
    from src.backtest.report import print_diagnostics, print_symbol_table
    from src.config import available_profiles

    names = ["base", *available_profiles()]
    rows = []
    for name in names:
        s = load_settings(args.config, name)
        s.general.symbol_suffix = settings.general.symbol_suffix
        if settings.backtest.gross:
            apply_gross(s)
        s.resolve_symbols(list(data))
        res, per_symbol = run_portfolio(s, data, clock)
        console.rule(f"[bold]Profile: {name}")
        print_diagnostics(res, console)
        if len(per_symbol) > 1:
            print_symbol_table(per_symbol, console, f"Profile {name} — by symbol")
        rows.append((name, res.stats))
    n_bars = sum(len(m1) for m1, _, _ in data.values())
    table = Table(title=f"Profile comparison — {len(data)} symbol(s), {n_bars:,} M1 bars")
    cols = [("trades", "Trades", "{:.0f}"), ("trades_per_week", "Tr/week", "{:.1f}"),
            ("win_rate_pct", "Win %", "{:.1f}"), ("profit_factor", "PF", "{:.2f}"),
            ("expectancy_r", "Exp R", "{:+.3f}"), ("expectancy_usd", "Exp $", "{:+.2f}"),
            ("net_profit", "Net $", "{:+,.2f}"), ("max_drawdown_pct", "Max DD %", "{:.2f}"),
            ("sharpe", "Sharpe", "{:.2f}")]  # fmt: skip
    table.add_column("Profile")
    for _, head, _ in cols:
        table.add_column(head, justify="right")
    for name, st in rows:
        cells = []
        for key, _, fmt in cols:
            v = st.get(key)
            txt = "-" if v is None else ("∞" if v == float("inf") else fmt.format(v))
            if key in ("expectancy_r", "net_profit") and v is not None:
                txt = f"[{'green' if v > 0 else 'red'}]{txt}[/]"
            cells.append(txt)
        table.add_row(name, *cells)
    console.print(table, width=max(console.width, 130))
    console.print("[dim]Profiles in config/profiles/*.yaml. A profile is credible only with >= 200 trades "
                  "and positive expectancy; confirm with --optimize (out-of-sample).[/]")  # fmt: skip
    return 0


def run_lab(settings: Settings, args: argparse.Namespace) -> int:
    """Strategy research lab: search, walk-forward, anti-luck filters, ML filter, vault."""
    from src.lab.runner import Lab, load_lab_data, needed_timeframes

    families = args.families or settings.lab.families
    tfs = needed_timeframes(families)
    if args.synthetic:
        from src.backtest.engine import synthetic_m1
        from src.strategies.indicators import resample_ohlc

        data = {}
        for k, sym in enumerate(args.resolved_symbols[:3]):
            m1 = synthetic_m1(n_bars=args.bars, seed=11 + k)
            data[sym] = {tf: (resample_ohlc(m1, tf), None) for tf in tfs}
        clock = ServerClock(settings.sessions.backtest_server_timezone)
    else:
        conn, store, clock = _mt5_store(settings, need_mt5=not args.offline)
        if conn is None and not args.offline:
            console.print("[yellow]MT5 not available → using the local Parquet cache.[/]")
        try:
            console.print(
                f"Loading history for {len(args.resolved_symbols)} symbols, timeframes {', '.join(tfs)}…"
            )
            data = load_lab_data(settings, args.resolved_symbols, store, conn, console, tfs)
        finally:
            if conn is not None:
                conn.shutdown()
    if not data:
        console.print("[red]No data for the lab. Run it with MT5 open (or download first).[/]")
        return 1
    Lab(settings, data, clock, console, families=families, quick=args.quick).run()
    return 0


def expand_profiles(names: list[str] | None) -> list[str]:
    """``--profiles lab`` = every profile promoted by the strategy lab (config/profiles/lab_*)."""
    from src.config import available_profiles

    if not names:
        return []
    out: list[str] = []
    for n in names:
        if n == "lab":
            out += [p for p in available_profiles() if p.startswith("lab_")]
        else:
            out.append(n)
    if names and not out:
        console.print("[yellow]No lab_* profile found: run the lab first (6_LABORATORIO.bat).[/]")
    return list(dict.fromkeys(out))


def run_trading(settings: Settings, args: argparse.Namespace, dry_run: bool) -> int:
    from src.core.mt5_connector import MT5Connector
    from src.core.trader import LiveTrader
    from src.ui.dashboard import Dashboard

    if not dry_run and not args.confirm_live:
        console.print("[bold red]Refusing to trade LIVE without --confirm-live[/]")
        return 2
    state = BotState(mode="dryrun" if dry_run else "live")
    logging.getLogger().addHandler(StateLogHandler(state))
    connector = MT5Connector(load_credentials(), settings.mt5)
    if not connector.available:
        console.print("[red]MetaTrader5 package not available on this OS (Windows only).[/]")
        return 1
    clock = ServerClock.from_config(settings.sessions.server_timezone)
    profiles = expand_profiles(args.profiles)
    if profiles:  # several strategies on one account (e.g. every lab_* profile)
        from src.core.multi_trader import MultiTrader

        traders = []
        for name in profiles:
            s = load_settings(args.config, name)
            s.general.symbol_suffix = settings.general.symbol_suffix
            syms = s.resolve_symbols(args.symbols)
            strat = create_strategy(s.strategy, s.levels, clock)
            traders.append(LiveTrader(s, connector, strat, state, dry_run=dry_run, symbols=syms,
                                      clock=clock, tag=name.removeprefix("lab_")))  # fmt: skip
        trader = MultiTrader(traders)
    else:
        strategy = create_strategy(settings.strategy, settings.levels, clock)
        trader = LiveTrader(settings, connector, strategy, state, dry_run=dry_run,
                            symbols=args.resolved_symbols, clock=clock)  # fmt: skip
    trader.setup()
    dashboard = None if args.no_ui else Dashboard(state, settings.dashboard.refresh_per_second,
                                                  settings.dashboard.max_events)  # fmt: skip
    if dashboard:
        dashboard.start()
    try:
        trader.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")
    finally:
        trader.stop()
        if dashboard:
            dashboard.stop()
        connector.shutdown()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAU Quant Scalper - MT5 fast-intraday framework")
    p.add_argument(
        "--mode",
        choices=["live", "dryrun", "backtest", "diag", "download", "lab"],
        default="dryrun",
    )
    p.add_argument("--config", default=None, help="path to settings.yaml")
    p.add_argument("--symbols", nargs="+", default=None, help="override enabled symbols")
    p.add_argument("--days", type=int, default=None, help="history depth for backtest/download")
    p.add_argument(
        "--ticks", action="store_true", help="also use real ticks (spread + intra-bar path)"
    )
    p.add_argument(
        "--offline", action="store_true", help="backtest from the local Parquet cache only"
    )
    p.add_argument(
        "--profile", default=None, help="config/profiles/<name>.yaml overlay (e.g. attivo)"
    )
    p.add_argument(
        "--compare", action="store_true", help="backtest base + all profiles side by side"
    )
    p.add_argument(
        "--gross", action="store_true", help="diagnostic backtest with zero costs (not tradable)"
    )
    p.add_argument("--profiles", nargs="+", default=None,
                   help="dryrun/live: run several profiles together ('lab' = all lab_* profiles)")  # fmt: skip
    p.add_argument("--families", nargs="+", default=None, help="lab: strategy families to test")
    p.add_argument("--quick", action="store_true", help="lab: reduced parameter grid")
    p.add_argument("--optimize", action="store_true", help="run the IS/OOS auto-optimizer")
    p.add_argument(
        "--csv", default=None, help="M1 CSV/Parquet (MT5 export or time,open,high,low,close)"
    )
    p.add_argument("--synthetic", action="store_true", help="backtest on synthetic data")
    p.add_argument("--bars", type=int, default=60_000, help="synthetic bars")
    p.add_argument(
        "--no-ui", action="store_true", help="disable the TUI dashboard (log to console)"
    )
    p.add_argument("--confirm-live", action="store_true", help="required for --mode live")
    args = p.parse_args(argv)
    if args.mode == "download" and not args.days:
        args.days = 90
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config, args.profile)
    args.resolved_symbols = settings.resolve_symbols(args.symbols)
    if getattr(args, "gross", False):
        apply_gross(settings)
    ui_mode = args.mode in ("live", "dryrun") and not args.no_ui
    setup_logging(settings, args.mode, to_console=not ui_mode)

    if args.mode in ("backtest", "diag", "lab"):
        console.width = max(console.width, 150)  # wide tables (the saved report stays readable)
        try:
            if args.mode == "lab":
                logging.getLogger("src.risk").setLevel(logging.CRITICAL + 1)
                return run_lab(settings, args)
            if args.mode == "backtest":
                # simulated kill-switch / gap events are summarised in the report, not per bar
                logging.getLogger("src.risk").setLevel(logging.CRITICAL + 1)
                return run_backtest(settings, args)
            from src.core.diagnostics import run_diagnostics

            ok = run_diagnostics(settings, load_credentials(), args.resolved_symbols, console)
            return 0 if ok else 1
        finally:
            save_console_report(settings, args.mode)
    if args.mode == "download":
        return run_download(settings, args)
    if args.profiles and not expand_profiles(args.profiles):
        return 1
    return run_trading(settings, args, dry_run=args.mode == "dryrun")


def save_console_report(settings: Settings, mode: str) -> None:
    """Save the whole console output (plain text) so it can be shared without copy/paste losses."""
    name = {"backtest": "ultimo_report.txt", "lab": "ultimo_laboratorio.txt"}.get(
        mode, "ultima_diagnostica.txt"
    )
    out = PROJECT_ROOT / settings.general.log_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(console.export_text(clear=False), encoding="utf-8")
    console.print(f"[dim]Full output saved to {out}[/]")


if __name__ == "__main__":
    sys.exit(main())

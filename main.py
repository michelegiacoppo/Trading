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

console = Console()


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


def _symbol(settings: Settings, args: argparse.Namespace) -> str:
    return args.symbols[0] if args.symbols else (settings.enabled_symbols or ["XAUUSD"])[0]


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
        for sym in args.symbols or settings.enabled_symbols:
            conn.select_symbol(sym)
            for tf in settings.data.warmup_timeframes:
                df = store.fetch_historical_data(sym, tf, args.days)
                console.print(f"{sym} {tf}: {len(df):,} bars cached → {store.bars_path(sym, tf)}")
            if args.ticks:
                end = datetime.now()
                ticks = store.fetch_historical_ticks(sym, end - timedelta(days=args.days), end)
                console.print(f"{sym} ticks: {len(ticks):,} cached")
    finally:
        conn.shutdown()
    return 0


def load_backtest_data(settings: Settings, args: argparse.Namespace, symbol: str):  # noqa: ANN201
    """Return (m1, ticks|None, clock, broker SymbolSpec|None)."""
    from src.backtest.engine import load_m1_csv, synthetic_m1

    clock = ServerClock(settings.sessions.backtest_server_timezone)
    if args.csv:
        return load_m1_csv(args.csv), None, clock, None
    if args.synthetic:
        return synthetic_m1(n_bars=args.bars), None, clock, None

    conn, store, clock = _mt5_store(settings, need_mt5=not args.offline)
    spec = None
    try:
        if conn is not None:
            from src.core.data_feed import DataFeed

            conn.select_symbol(symbol)
            spec = DataFeed(conn).symbol_spec(symbol)  # real tick_value in account currency
            console.print(
                f"Downloading/updating {args.days} days of {symbol} M1 from MT5 (incremental cache)…"
            )
            m1 = store.fetch_historical_data(symbol, "M1", args.days)
        else:
            if not args.offline:
                console.print("[yellow]MT5 not available → using the local Parquet cache.[/]")
            m1 = store.slice_days(store.load_bars(symbol, "M1"), args.days)
        if m1.empty:
            raise SystemExit(
                f"No M1 data for {symbol}: run `--mode download` on the MT5 host or use --csv."
            )
        m1 = m1.iloc[:-1]  # drop the possibly-forming last bar
        ticks = None
        if args.ticks:
            start, end = m1["time"].iloc[0].to_pydatetime(), m1["time"].iloc[-1].to_pydatetime()
            if conn is not None:
                console.print("Downloading real ticks (COPY_TICKS_ALL) — this can take a while…")
                ticks = store.fetch_historical_ticks(symbol, start, end)
            else:
                ticks = store.load_ticks(symbol, start, end)
            console.print(f"Ticks loaded: {len(ticks):,}")
    finally:
        if conn is not None:
            conn.shutdown()
    return m1, ticks, clock, spec


def run_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from src.backtest.engine import BacktestEngine, save_results
    from src.backtest.report import print_report

    symbol = _symbol(settings, args)
    if not (args.csv or args.synthetic or args.days):
        console.print("[red]Backtest needs --days N (MT5/cache), --csv <file> or --synthetic[/]")
        return 2
    m1, ticks, clock, spec = load_backtest_data(settings, args, symbol)

    if args.optimize:
        from src.backtest.optimizer import optimize

        console.print(
            "[bold]Auto-optimizer[/]: grid search, ranking in-sample, validating out-of-sample…"
        )
        top = optimize(settings, symbol, m1, clock, ticks, spec=spec)
        table = Table(
            title=f"Top combinations — in-sample → out-of-sample (split {top.attrs.get('split_time')})"
        )
        params = [c for c in top.columns if "." in c]
        metrics = [
            ("IS_trades", "IS n"),
            ("IS_expectancy_r", "IS exp R"),
            ("IS_profit_factor", "IS PF"),
            ("OOS_trades", "OOS n"),
            ("OOS_win_rate_pct", "OOS win%"),
            ("OOS_expectancy_r", "OOS exp R"),
            ("OOS_profit_factor", "OOS PF"),
            ("OOS_net", "OOS net $"),
        ]
        for col in params:
            table.add_column(col.split(".", 1)[1], justify="right")
        for _, head in metrics:
            table.add_column(head, justify="right")
        for _, row in top.iterrows():
            cells = [str(row[c]) for c in params]
            for key, _ in metrics:
                v = row[key]
                color = (
                    "green"
                    if key.startswith("OOS_exp") and v > 0
                    else "red"
                    if key.startswith("OOS_exp")
                    else ""
                )
                txt = f"{v:.0f}" if key.endswith("trades") else f"{v:.2f}"
                cells.append(f"[{color}]{txt}[/]" if color else txt)
            table.add_row(*cells)
        Console(width=max(console.width, 150)).print(table)
        console.print("[dim]Adopt a combination only if its OOS expectancy/PF stay positive.[/]")
        return 0

    strategy = create_strategy(settings.strategy, settings.levels, clock)
    res = BacktestEngine(settings, strategy, symbol, spec=spec, clock=clock).run(m1, ticks)
    print_report(res, settings, symbol, clock, console)
    curve, trades = save_results(res, settings)
    console.print(f"[dim]Equity curve → {curve} | trades → {trades}[/]")
    return 0


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
    strategy = create_strategy(settings.strategy, settings.levels, clock)
    trader = LiveTrader(settings, connector, strategy, state, dry_run=dry_run,
                        symbols=args.symbols, clock=clock)  # fmt: skip
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
        "--mode", choices=["live", "dryrun", "backtest", "diag", "download"], default="dryrun"
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
    settings = load_settings(args.config)
    settings.ensure_symbols(args.symbols)
    ui_mode = args.mode in ("live", "dryrun") and not args.no_ui
    setup_logging(settings, args.mode, to_console=not ui_mode)

    if args.mode == "backtest":
        # simulated kill-switch / gap events are summarised in the report, not logged per bar
        logging.getLogger("src.risk").setLevel(logging.CRITICAL + 1)
        return run_backtest(settings, args)
    if args.mode == "download":
        return run_download(settings, args)
    if args.mode == "diag":
        from src.core.diagnostics import run_diagnostics

        return 0 if run_diagnostics(settings, load_credentials(), args.symbols, console) else 1
    return run_trading(settings, args, dry_run=args.mode == "dryrun")


if __name__ == "__main__":
    sys.exit(main())

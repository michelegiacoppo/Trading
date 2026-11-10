"""CLI entry point.

python main.py --mode diag                     # broker / symbol diagnostics
python main.py --mode dryrun                   # live data, paper execution, TUI
python main.py --mode live --confirm-live      # REAL orders
python main.py --mode backtest --csv data/XAUUSD_M1.csv
python main.py --mode backtest --synthetic     # smoke test without data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT, Settings, load_credentials, load_settings
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

        handlers.append(RichHandler(console=console, show_path=False))
    logging.basicConfig(
        level=settings.general.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file


def run_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from src.backtest.engine import BacktestEngine, load_m1_csv, save_results, synthetic_m1

    symbol = args.symbols[0] if args.symbols else (settings.enabled_symbols or ["XAUUSD"])[0]
    if args.csv:
        m1 = load_m1_csv(args.csv)
    elif args.synthetic:
        m1 = synthetic_m1(n_bars=args.bars)
    else:
        console.print("[red]Backtest needs --csv <file> or --synthetic[/]")
        return 2
    console.print(
        f"Backtesting [bold]{settings.strategy.name}[/] on {symbol}: {len(m1):,} M1 bars "
        f"({m1['time'].iloc[0]} → {m1['time'].iloc[-1]})"
    )
    engine = BacktestEngine(settings, create_strategy(settings.strategy), symbol)
    res = engine.run(m1)
    save_results(res, settings)
    table = Table(title="Backtest results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in res.stats.items():
        table.add_row(k, f"{v:,.2f}")
    console.print(table)
    console.print(
        f"Equity curve → {settings.backtest.equity_curve_file} | trades → {settings.backtest.trades_file}"
    )
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
    trader = LiveTrader(
        settings,
        connector,
        create_strategy(settings.strategy),
        state,
        dry_run=dry_run,
        symbols=args.symbols,
    )
    trader.setup()
    dashboard = (
        None
        if args.no_ui
        else Dashboard(state, settings.dashboard.refresh_per_second, settings.dashboard.max_events)
    )
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
    p.add_argument("--mode", choices=["live", "dryrun", "backtest", "diag"], default="dryrun")
    p.add_argument("--config", default=None, help="path to settings.yaml")
    p.add_argument("--symbols", nargs="+", default=None, help="override enabled symbols")
    p.add_argument(
        "--csv", default=None, help="M1 CSV for backtest (MT5 export or time,open,high,low,close)"
    )
    p.add_argument("--synthetic", action="store_true", help="backtest on synthetic data")
    p.add_argument("--bars", type=int, default=30_000, help="synthetic bars")
    p.add_argument(
        "--no-ui", action="store_true", help="disable the TUI dashboard (log to console)"
    )
    p.add_argument("--confirm-live", action="store_true", help="required for --mode live")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    ui_mode = args.mode in ("live", "dryrun") and not args.no_ui
    setup_logging(settings, args.mode, to_console=not ui_mode)
    if args.mode == "backtest":
        logging.getLogger("src").setLevel(logging.WARNING)

    if args.mode == "backtest":
        return run_backtest(settings, args)
    if args.mode == "diag":
        from src.core.diagnostics import run_diagnostics

        return 0 if run_diagnostics(settings, load_credentials(), args.symbols, console) else 1
    return run_trading(settings, args, dry_run=args.mode == "dryrun")


if __name__ == "__main__":
    sys.exit(main())

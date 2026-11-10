"""Live broker diagnostics: terminal, account, symbol specs, tick value, filling modes.

Run via ``python main.py --mode diag`` or ``python tests/test_mt5_diag.py``.
Nothing is ever sent to the market: ``order_check`` is used for a dry validation.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.config import Credentials, Settings
from src.core.data_feed import DataFeed
from src.core.mt5_connector import MT5Connector
from src.risk.position_sizer import PositionSizer, money_per_lot
from src.strategies import indicators as ind


def _filling_names(mask: int) -> str:
    names = [n for bit, n in ((1, "FOK"), (2, "IOC"), (4, "BOC")) if mask & bit]
    return ", ".join(names) or "RETURN only"


def run_diagnostics(
    settings: Settings,
    creds: Credentials,
    symbols: list[str] | None = None,
    console: Console | None = None,
) -> bool:
    con = console or Console()
    conn = MT5Connector(creds, settings.mt5)
    if not conn.available:
        con.print("[red]✗ MetaTrader5 package not installed (Windows only).[/]")
        return False
    if not conn.connect():
        con.print(f"[red]✗ initialize/login failed: {conn.mt5.last_error()}[/]")
        return False
    mt5 = conn.mt5
    all_ok = True
    try:
        term, acc = mt5.terminal_info(), mt5.account_info()
        t = Table(title="Terminal & Account", show_header=False)
        t.add_row("Terminal", f"{term.name} build {term.build} | connected={term.connected}")
        t.add_row("Company", f"{term.company}")
        t.add_row(
            "Account",
            f"{acc.login} @ {acc.server} ({'DEMO' if acc.trade_mode == 0 else 'REAL/CONTEST'})",
        )
        t.add_row("Balance / Equity", f"{acc.balance:.2f} / {acc.equity:.2f} {acc.currency}")
        t.add_row("Leverage", f"1:{acc.leverage}")
        t.add_row(
            "Margin mode",
            {0: "Netting (retail)", 1: "Exchange", 2: "Hedging"}.get(
                acc.margin_mode, str(acc.margin_mode)
            ),
        )
        at_ok, why = conn.autotrading_status()
        t.add_row("AutoTrading", "[green]ON[/]" if at_ok else f"[red]OFF - {why}[/]")
        con.print(t)
        all_ok &= at_ok

        feed = DataFeed(conn)
        sizer = PositionSizer(settings.risk)
        for sym in symbols or settings.enabled_symbols:
            if not conn.select_symbol(sym):
                con.print(f"[red]✗ {sym} not available[/]")
                all_ok = False
                continue
            info = mt5.symbol_info(sym)
            spec = feed.symbol_spec(sym, refresh=True)
            tick = feed.last_tick(sym)
            st = Table(title=f"{sym}", show_header=False)
            st.add_row("digits / point", f"{info.digits} / {info.point}")
            st.add_row("tick size", f"{info.trade_tick_size}")
            st.add_row(
                "tick value (profit/loss)",
                f"{info.trade_tick_value} ({info.trade_tick_value_profit} / {info.trade_tick_value_loss})",
            )
            st.add_row("contract size", f"{info.trade_contract_size}")
            st.add_row(
                "volume min/max/step", f"{info.volume_min} / {info.volume_max} / {info.volume_step}"
            )
            st.add_row(
                "stops / freeze level", f"{info.trade_stops_level} / {info.trade_freeze_level} pt"
            )
            st.add_row("filling modes", _filling_names(info.filling_mode))
            st.add_row(
                "execution mode",
                {0: "Request", 1: "Instant", 2: "Market", 3: "Exchange"}.get(
                    info.trade_exemode, str(info.trade_exemode)
                ),
            )
            st.add_row("swap long / short", f"{info.swap_long} / {info.swap_short}")
            st.add_row(
                "money / 1 point / 1 lot", f"{money_per_lot(info.point, spec):.4f} {acc.currency}"
            )
            if tick:
                st.add_row(
                    "bid / ask / spread", f"{tick.bid} / {tick.ask} / {tick.spread_points} pt"
                )
                margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, sym, 1.0, tick.ask)
                st.add_row("margin for 1 lot", f"{margin} {acc.currency}")
                m1 = feed.get_rates(sym, "M1", 200)
                if not m1.empty:
                    atr = float(ind.atr(m1, settings.strategy.atr_period).iloc[-1])
                    sl = tick.ask - atr * settings.strategy.sl_atr_mult
                    res = sizer.size(
                        acc.equity,
                        tick.ask,
                        sl,
                        spec,
                        spread_price=tick.ask - tick.bid,
                        free_margin=acc.margin_free,
                        margin_per_lot=margin,
                    )
                    st.add_row("ATR(M1)", f"{atr:.{info.digits}f} ({atr / info.point:.0f} pt)")
                    st.add_row(
                        "sample sizing",
                        f"{res.lots} lots, risk {res.risk_money:.2f} ({res.risk_pct_effective:.2f}%) - {res.reason}",
                    )
                fill = (
                    mt5.ORDER_FILLING_FOK
                    if info.filling_mode & 1
                    else mt5.ORDER_FILLING_IOC
                    if info.filling_mode & 2
                    else mt5.ORDER_FILLING_RETURN
                )
                chk = mt5.order_check(
                    {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": sym,
                        "volume": info.volume_min,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": tick.ask,
                        "deviation": 20,
                        "magic": settings.general.magic_number,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": fill,
                    }
                )
                ok = chk is not None and chk.retcode == 0
                st.add_row(
                    "order_check (min lot)",
                    "[green]OK[/]"
                    if ok
                    else f"[red]{getattr(chk, 'retcode', None)} {getattr(chk, 'comment', mt5.last_error())}[/]",
                )
                all_ok &= ok
            con.print(st)
    finally:
        conn.shutdown()
    con.print(
        "[bold green]✓ Diagnostics passed[/]"
        if all_ok
        else "[bold yellow]⚠ Diagnostics completed with warnings[/]"
    )
    return all_ok

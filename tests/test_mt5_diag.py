"""Live MT5 diagnostic script (broker, tick_value, filling mode, AutoTrading).

* As a script (on the Windows trading host)::

      python tests/test_mt5_diag.py            # all enabled symbols
      python tests/test_mt5_diag.py XAUUSD US100

* Under pytest the live checks are skipped unless the MetaTrader5 package is
  installed AND ``RUN_MT5_DIAG=1`` is set, so CI on Linux stays green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_credentials, load_settings  # noqa: E402
from src.core.mt5_connector import import_mt5  # noqa: E402

LIVE = import_mt5() is not None and os.getenv("RUN_MT5_DIAG") == "1"


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="needs MetaTrader5 terminal and RUN_MT5_DIAG=1")
def test_live_diagnostics() -> None:
    from src.core.diagnostics import run_diagnostics

    assert run_diagnostics(load_settings(), load_credentials())


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="needs MetaTrader5 terminal and RUN_MT5_DIAG=1")
def test_tick_value_consistency() -> None:
    """tick_value / tick_size must match contract_size for USD-quoted symbols (e.g. XAUUSD)."""
    mt5 = import_mt5()
    assert mt5 is not None
    from src.core.mt5_connector import MT5Connector

    with MT5Connector(load_credentials()) as conn:
        acc = conn.account_info()
        for sym in load_settings().enabled_symbols:
            info = mt5.symbol_info(sym)
            if info is None or not sym.endswith("USD") or acc.currency != "USD":
                continue
            per_unit = info.trade_tick_value / info.trade_tick_size
            assert per_unit == pytest.approx(info.trade_contract_size, rel=0.05), sym


def test_connector_degrades_gracefully_without_mt5() -> None:
    """Importing and instantiating the connector must never crash on non-Windows hosts."""
    from src.core.mt5_connector import MT5Connector, MT5NotAvailableError

    conn = MT5Connector(load_credentials(), mt5_module=None)
    if conn.available:
        pytest.skip("MetaTrader5 installed on this host")
    with pytest.raises(MT5NotAvailableError):
        _ = conn.mt5


if __name__ == "__main__":
    from src.core.diagnostics import run_diagnostics

    syms = sys.argv[1:] or None
    sys.exit(0 if run_diagnostics(load_settings(), load_credentials(), syms) else 1)

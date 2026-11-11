"""Run several strategies (profiles) on ONE account with shared risk rules.

Each profile gets its own :class:`LiveTrader` (own strategy, timeframe, magic number, trade
management, ML filter and health monitor). They share:

* the MT5 connection and the dashboard state;
* ONE daily kill-switch (account-level loss limit / consecutive losses / max trades);
* account-wide ``max_open_positions`` / ``max_positions_per_symbol`` (peers are counted);
* in dry-run, one equity = real account equity + every paper book's P/L.

The loop is single-threaded: traders are stepped one after the other (the MetaTrader5
Python API is not meant to be called from several threads).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.trader import LiveTrader

log = logging.getLogger(__name__)


class MultiTrader:
    def __init__(self, traders: list[LiveTrader]) -> None:
        if not traders:
            raise ValueError("MultiTrader needs at least one trader")
        self.traders = traders
        shared_ks = traders[0].guards.kill_switch
        for t in traders:
            t.guards.kill_switch = shared_ks
            t.peers = [o for o in traders if o is not t]
            t.equity_provider = self._equity
        self._running = False

    def _equity(self, acc: Any) -> float:
        eq = acc.equity
        for t in self.traders:
            if t.dry_run:
                eq += t.paper_pnl()
        return eq

    def setup(self) -> None:
        for t in self.traders:
            t.setup()
        state = self.traders[0].state
        state.strategy = ", ".join(t.tag or t.strategy.name for t in self.traders)
        state.event(
            f"MultiTrader: {len(self.traders)} strategies on one account, shared kill-switch"
        )

    def run(self) -> None:
        self._running = True
        period = min(t.s.general.poll_interval_ms for t in self.traders) / 1000
        while self._running:
            t0 = time.perf_counter()
            for t in self.traders:
                try:
                    t.step()
                except Exception as exc:  # one failing strategy must not stop the others
                    log.exception("step failed for %s", t.tag)
                    t.state.event(f"[{t.tag}] loop error: {exc}", "error")
            elapsed = time.perf_counter() - t0
            self.traders[0].state.loop_ms = elapsed * 1000
            time.sleep(max(0.0, period - elapsed))

    def stop(self) -> None:
        self._running = False

"""Fast OHLCV / tick extraction from MT5.

Conventions (shared with the backtester, see ``src/strategies/indicators.py``):

* Bars are returned **closed only** (the forming bar is dropped) so strategies
  never act on unfinished candles.
* ``time`` is a tz-naive ``datetime64`` in broker server time = bar *open* time.
* Prices are BID (MT5 standard). ``spread`` is in points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SymbolSpec
from src.core.mt5_connector import MT5Connector

log = logging.getLogger(__name__)

TF_MINUTES: dict[str, int] = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M15": 15, "M30": 30, "H1": 60}


@dataclass(frozen=True)
class TickSnapshot:
    symbol: str
    time_msc: int
    bid: float
    ask: float
    spread_points: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class DataFeed:
    def __init__(self, connector: MT5Connector) -> None:
        self.conn = connector
        self._spec_cache: dict[str, SymbolSpec] = {}
        self._last_bar_time: dict[tuple[str, str], pd.Timestamp] = {}

    # ----------------------------------------------------------- helpers
    def _tf(self, tf: str) -> int:
        try:
            return getattr(self.conn.mt5, f"TIMEFRAME_{tf}")
        except AttributeError as exc:
            raise ValueError(f"Unsupported timeframe {tf}") from exc

    def symbol_spec(self, symbol: str, refresh: bool = False) -> SymbolSpec:
        if symbol in self._spec_cache and not refresh:
            return self._spec_cache[symbol]
        info = self.conn.mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"symbol_info({symbol}) returned None")
        spec = SymbolSpec(
            name=symbol,
            point=info.point,
            digits=info.digits,
            tick_size=info.trade_tick_size or info.point,
            # trade_tick_value_loss is the conservative value for losing trades
            tick_value=getattr(info, "trade_tick_value_loss", 0) or info.trade_tick_value,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            stops_level=info.trade_stops_level,
            freeze_level=info.trade_freeze_level,
            filling_mode=info.filling_mode,
        )
        self._spec_cache[symbol] = spec
        return spec

    # -------------------------------------------------------------- bars
    def get_rates(self, symbol: str, tf: str, count: int) -> pd.DataFrame:
        """Return the last ``count`` CLOSED bars (start_pos=1 skips the forming bar)."""
        raw = self.conn.mt5.copy_rates_from_pos(symbol, self._tf(tf), 1, count)
        if raw is None or len(raw) == 0:
            log.warning("No rates for %s %s: %s", symbol, tf, self.conn.mt5.last_error())
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        return df[cols].reset_index(drop=True)

    def get_frames(self, symbol: str, tfs: list[str], count: int) -> dict[str, pd.DataFrame]:
        return {tf: self.get_rates(symbol, tf, count) for tf in tfs}

    def has_new_bar(self, symbol: str, tf: str) -> bool:
        """True once per newly closed bar (cheap: pulls a single bar)."""
        raw = self.conn.mt5.copy_rates_from_pos(symbol, self._tf(tf), 1, 1)
        if raw is None or len(raw) == 0:
            return False
        t = pd.Timestamp(int(raw[0]["time"]), unit="s")
        key = (symbol, tf)
        if self._last_bar_time.get(key) == t:
            return False
        self._last_bar_time[key] = t
        return True

    # ------------------------------------------------------------- ticks
    def last_tick(self, symbol: str) -> TickSnapshot | None:
        tick = self.conn.mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            return None
        spec = self.symbol_spec(symbol)
        return TickSnapshot(
            symbol=symbol,
            time_msc=int(tick.time_msc),
            bid=float(tick.bid),
            ask=float(tick.ask),
            spread_points=round((tick.ask - tick.bid) / spec.point, 1),
        )

    def get_ticks(self, symbol: str, from_epoch_s: int, count: int = 5000) -> pd.DataFrame:
        mt5 = self.conn.mt5
        raw = mt5.copy_ticks_from(symbol, from_epoch_s, count, mt5.COPY_TICKS_INFO)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time_msc"], unit="ms")
        spec = self.symbol_spec(symbol)
        df["spread_points"] = np.round((df["ask"] - df["bid"]) / spec.point, 1)
        return df[["time", "bid", "ask", "spread_points"]]

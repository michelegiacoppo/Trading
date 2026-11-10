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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SymbolSpec
from src.core.mt5_connector import MT5Connector

log = logging.getLogger(__name__)

TF_MINUTES: dict[str, int] = {
    "M1": 1, "M2": 2, "M3": 3, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}  # fmt: skip
BAR_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
TICK_COLUMNS = ["time", "bid", "ask", "last", "flags", "spread_points"]


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
            currency_profit=getattr(info, "currency_profit", "USD"),
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
        return rates_to_frame(raw)

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


# =========================================================================
#  Historical data: MT5 download + incremental local Parquet cache
# =========================================================================
def rates_to_frame(raw: object) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "spread" not in df.columns:
        df["spread"] = np.nan
    return df[BAR_COLUMNS].reset_index(drop=True)


def ticks_to_frame(raw: object, point: float) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=TICK_COLUMNS)
    df["time"] = pd.to_datetime(df["time_msc"], unit="ms")
    df["spread_points"] = np.round((df["ask"] - df["bid"]) / point, 1)
    for col in ("last", "flags"):
        if col not in df.columns:
            df[col] = 0
    return df[TICK_COLUMNS].reset_index(drop=True)


class HistoricalStore:
    """Download bars/ticks from MT5 and keep an incremental Parquet cache.

    Layout::

        data/historical/XAUUSD_M1.parquet
        data/historical/ticks/XAUUSD/2026-11-10.parquet   (one file per server day)

    Only the missing range is requested from the terminal: new bars after the
    last cached one, and older bars when ``days_back`` reaches further back than
    the cache. The last (possibly forming) bar is always re-downloaded.
    MT5 range functions take UTC datetimes but return broker *server* time
    stamps; ranges are padded by 1 day on each side and then trimmed.
    """

    def __init__(
        self, root: str | Path, connector: MT5Connector | None = None, chunk_days: int = 15
    ) -> None:
        self.root = Path(root)
        self.conn = connector
        self.chunk_days = max(1, chunk_days)

    # ------------------------------------------------------------- paths
    def bars_path(self, symbol: str, tf: str) -> Path:
        return self.root / f"{symbol}_{tf}.parquet"

    def ticks_path(self, symbol: str, day: datetime) -> Path:
        return self.root / "ticks" / symbol / f"{day:%Y-%m-%d}.parquet"

    def load_bars(self, symbol: str, tf: str) -> pd.DataFrame:
        path = self.bars_path(symbol, tf)
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS)
        return pd.read_parquet(path)

    def _save(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _mt5(self) -> object:
        if self.conn is None:
            raise RuntimeError("HistoricalStore needs an MT5 connector to download data")
        return self.conn.mt5

    # -------------------------------------------------------------- bars
    def _download_bars(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        mt5 = self._mt5()
        tf_const = getattr(mt5, f"TIMEFRAME_{tf}")
        parts: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=self.chunk_days), end)
            raw = mt5.copy_rates_range(
                symbol,
                tf_const,
                cursor.replace(tzinfo=timezone.utc),
                chunk_end.replace(tzinfo=timezone.utc),
            )
            if raw is None:
                log.warning(
                    "copy_rates_range %s %s %s->%s: %s",
                    symbol,
                    tf,
                    cursor,
                    chunk_end,
                    mt5.last_error(),
                )
            elif len(raw):
                parts.append(rates_to_frame(raw))
            cursor = chunk_end
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BAR_COLUMNS)

    def fetch_historical_data(
        self, symbol: str, timeframe: str, days_back: int, now: datetime | None = None
    ) -> pd.DataFrame:
        """Return ``days_back`` days of bars, downloading only what the cache lacks."""
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        want_from = now - timedelta(days=days_back)
        cached = self.load_bars(symbol, timeframe)
        new_parts = [cached] if not cached.empty else []

        if cached.empty:
            new_parts.append(
                self._download_bars(
                    symbol, timeframe, want_from - timedelta(days=1), now + timedelta(days=1)
                )
            )
        else:
            first, last = cached["time"].min(), cached["time"].max()
            if want_from < first:  # extend backwards
                new_parts.append(
                    self._download_bars(
                        symbol,
                        timeframe,
                        want_from - timedelta(days=1),
                        first.to_pydatetime() + timedelta(days=1),
                    )
                )
            # forward: from last cached bar (re-download it, it may have been forming)
            new_parts.append(
                self._download_bars(
                    symbol,
                    timeframe,
                    last.to_pydatetime() - timedelta(days=1),
                    now + timedelta(days=1),
                )
            )

        merged = (
            pd.concat([p for p in new_parts if not p.empty], ignore_index=True)
            if any(not p.empty for p in new_parts)
            else pd.DataFrame(columns=BAR_COLUMNS)
        )
        if not merged.empty:
            merged = (
                merged.drop_duplicates("time", keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )
            self._save(merged, self.bars_path(symbol, timeframe))
            log.info(
                "%s %s cache: %d bars (%s -> %s)",
                symbol,
                timeframe,
                len(merged),
                merged["time"].iloc[0],
                merged["time"].iloc[-1],
            )
        return self.slice_days(merged, days_back)

    @staticmethod
    def slice_days(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
        if df.empty:
            return df
        cutoff = df["time"].max() - pd.Timedelta(days=days_back)
        return df[df["time"] >= cutoff].reset_index(drop=True)

    # ------------------------------------------------------------- ticks
    def fetch_historical_ticks(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Real Bid/Ask ticks (``COPY_TICKS_ALL``) cached per day.

        Past days already on disk are never re-downloaded; the current day is
        always refreshed.
        """
        mt5 = self._mt5()
        point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.01
        today = datetime.now(timezone.utc).date()
        frames: list[pd.DataFrame] = []
        day = datetime(start_date.year, start_date.month, start_date.day)
        while day <= end_date:
            path = self.ticks_path(symbol, day)
            if path.exists() and day.date() < today:
                frames.append(pd.read_parquet(path))
            else:
                raw = mt5.copy_ticks_range(
                    symbol,
                    day.replace(tzinfo=timezone.utc),
                    (day + timedelta(days=1)).replace(tzinfo=timezone.utc),
                    mt5.COPY_TICKS_ALL,
                )
                if raw is None:
                    log.warning("copy_ticks_range %s %s: %s", symbol, day.date(), mt5.last_error())
                else:
                    df = ticks_to_frame(raw, point)
                    if not df.empty:
                        self._save(df, path)
                        frames.append(df)
            day += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=TICK_COLUMNS)
        out = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("time")
        return out[
            (out["time"] >= start_date) & (out["time"] < end_date + timedelta(days=1))
        ].reset_index(drop=True)

    def load_ticks(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Offline: read cached tick files only."""
        frames = []
        day = datetime(start_date.year, start_date.month, start_date.day)
        while day <= end_date:
            path = self.ticks_path(symbol, day)
            if path.exists():
                frames.append(pd.read_parquet(path))
            day += timedelta(days=1)
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TICK_COLUMNS)
        )

    # ------------------------------------------------------------ warmup
    def warmup(self, symbol: str, timeframes: list[str], bars: int) -> dict[str, pd.DataFrame]:
        """Cold start: last ``bars`` closed candles per timeframe (also refreshes the cache)."""
        out: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            days = int(bars * TF_MINUTES[tf] / 1440 * 7 / 5) + 3  # weekends + holidays margin
            df = self.fetch_historical_data(symbol, tf, days)
            out[tf] = df.iloc[:-1].tail(bars).reset_index(drop=True) if len(df) > 1 else df
            log.info("Warm-up %s %s: %d bars", symbol, tf, len(out[tf]))
        return out

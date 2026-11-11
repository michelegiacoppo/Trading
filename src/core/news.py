"""Economic calendar (MT5 built-in calendar exported by ``mql5/XQS_ExportCalendar.mq5``).

The MetaTrader5 Python package cannot read the terminal's economic calendar, so a small MQL5
script writes it (actual / forecast / previous, importance) to a tab-separated file in the
terminal's *Common\\Files* folder. Times are broker **server time**, like the bars.

Uses:

* **News blackout** - no new entries ``before``/``after`` minutes around high-impact events of
  the currencies a symbol depends on (XAUUSD -> USD, EURUSD -> EUR+USD, GER40 -> EUR, ...).
* **ML features** - minutes since / until the last / next event, standardised surprise of the
  last release signed for the symbol (positive = supportive of the symbol rising).
* **news_drift** strategy family in the lab (post-release continuation).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NewsConfig, resolve_path

log = logging.getLogger(__name__)

CALENDAR_FILE = "xqs_calendar.csv"
INDEX_CCY = {"US100": "USD", "US30": "USD", "US500": "USD", "SPX500": "USD", "NAS100": "USD",
             "USTEC": "USD", "GER40": "EUR", "DE40": "EUR", "UK100": "GBP", "JP225": "JPY",
             "XAU": "USD", "XAG": "USD", "GOLD": "USD", "SILVER": "USD"}  # fmt: skip


def symbol_currencies(symbol: str) -> tuple[str | None, str | None]:
    """(base, quote) currencies driving ``symbol``; indices/metals map to their quote currency."""
    s = "".join(ch for ch in symbol.upper() if ch.isalnum())
    for prefix, ccy in INDEX_CCY.items():
        if s.startswith(prefix):
            return None, ccy
    if len(s) >= 6 and s[:6].isalpha():
        return s[:3], s[3:6]
    return None, None


def default_calendar_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files" / CALENDAR_FILE
    return resolve_path("data") / CALENDAR_FILE


@dataclass
class NewsCheck:
    blocked: bool
    reason: str = ""


class EconomicCalendar:
    COLUMNS = ["time", "currency", "event", "importance", "actual", "forecast", "previous"]

    def __init__(self, events: pd.DataFrame) -> None:
        ev = events.copy()
        ev["time"] = pd.to_datetime(ev["time"])
        for col in ("actual", "forecast", "previous"):
            ev[col] = pd.to_numeric(ev.get(col), errors="coerce")
        ev["importance"] = pd.to_numeric(ev["importance"], errors="coerce").fillna(0).astype(int)
        ev = ev.sort_values("time").reset_index(drop=True)
        # standardised surprise per event type: (actual - forecast) / typical |miss| of that event
        diff = ev["actual"] - ev["forecast"]
        scale = diff.abs().groupby(ev["event"]).transform(lambda x: x.expanding().median().shift(1))
        ev["surprise_z"] = (diff / scale.replace(0, np.nan)).clip(-5, 5)
        self.events = ev
        self._by_ccy = {c: g.reset_index(drop=True) for c, g in ev.groupby("currency")}

    # ----------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path) -> EconomicCalendar:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", encoding_errors="replace")
        df.columns = [c.strip().lower() for c in df.columns]
        df["time"] = pd.to_datetime(
            df["time"].astype(str).str.replace(".", "-", n=2), errors="coerce"
        )
        return cls(df.dropna(subset=["time"]))

    @classmethod
    def from_config(cls, cfg: NewsConfig) -> EconomicCalendar | None:
        if not cfg.enabled:
            return None
        path = (
            default_calendar_path()
            if cfg.calendar_file in ("", "auto")
            else resolve_path(cfg.calendar_file)
        )
        if not path.exists():
            log.warning("Economic calendar not found at %s: run the MQL5 export script "
                        "(8_CALENDARIO_NOTIZIE.bat). News filter disabled.", path)  # fmt: skip
            return None
        cal = cls.load(path)
        log.info("Economic calendar: %d events (%s -> %s)", len(cal.events),
                 cal.events["time"].min(), cal.events["time"].max())  # fmt: skip
        return cal

    @property
    def last_time(self) -> pd.Timestamp | None:
        return self.events["time"].max() if len(self.events) else None

    # ----------------------------------------------------------- queries
    def events_for(self, symbol: str, min_importance: int = 3) -> pd.DataFrame:
        base, quote = symbol_currencies(symbol)
        parts = []
        for ccy, sign in ((base, 1.0), (quote, -1.0)):
            if ccy and ccy in self._by_ccy:
                g = self._by_ccy[ccy]
                g = g[g["importance"] >= min_importance].copy()
                g["symbol_sign"] = sign  # good news for the quote currency is bad for the symbol
                parts.append(g)
        if not parts:
            return pd.DataFrame(columns=[*self.COLUMNS, "surprise_z", "symbol_sign"])
        return pd.concat(parts).sort_values("time").reset_index(drop=True)

    def blackout(self, symbol: str, now: datetime, cfg: NewsConfig) -> NewsCheck:
        ev = self.events_for(symbol, cfg.blackout_importance)
        if ev.empty:
            return NewsCheck(False)
        lo = pd.Timestamp(now) - timedelta(minutes=cfg.blackout_minutes_after)
        hi = pd.Timestamp(now) + timedelta(minutes=cfg.blackout_minutes_before)
        hit = ev[(ev["time"] >= lo) & (ev["time"] <= hi)]
        if hit.empty:
            return NewsCheck(False)
        e = hit.iloc[0]
        return NewsCheck(True, f"news {e['currency']} {str(e['event'])[:28]} {e['time']:%H:%M}")

    def next_events(
        self, symbol: str, now: datetime, n: int = 3, min_importance: int = 2
    ) -> pd.DataFrame:
        ev = self.events_for(symbol, min_importance)
        return ev[ev["time"] >= pd.Timestamp(now)].head(n)

    def features_for(self, symbol: str, times: pd.Series, min_importance: int = 3) -> pd.DataFrame:
        """For each (bar-close) time: minutes since last / until next event, signed surprise."""
        ev = self.events_for(symbol, min_importance)
        t = pd.to_datetime(times).reset_index(drop=True)
        out = pd.DataFrame({"news_min_since": np.nan, "news_min_to": np.nan, "news_surprise": np.nan},
                           index=t.index)  # fmt: skip
        if ev.empty:
            return out
        et = ev["time"].to_numpy(dtype="datetime64[ns]")
        tv = t.to_numpy(dtype="datetime64[ns]")
        idx_prev = np.searchsorted(et, tv, side="right") - 1  # last event at or before t
        idx_next = np.searchsorted(et, tv, side="right")  # first event after t
        has_prev = idx_prev >= 0
        has_next = idx_next < len(et)
        minute = np.timedelta64(1, "m")
        out.loc[has_prev, "news_min_since"] = (tv[has_prev] - et[idx_prev[has_prev]]) / minute
        out.loc[has_next, "news_min_to"] = (et[idx_next[has_next]] - tv[has_next]) / minute
        sz = (ev["surprise_z"] * ev["symbol_sign"]).to_numpy()
        out.loc[has_prev, "news_surprise"] = sz[idx_prev[has_prev]]
        for col in ("news_min_since", "news_min_to"):
            out[col] = out[col].clip(upper=10_080)  # cap at one week
        return out


_CACHE: dict[str, EconomicCalendar | None] = {}


def get_calendar(cfg: NewsConfig) -> EconomicCalendar | None:
    """Process-wide cached calendar (reloaded when the file changes)."""
    path = (
        default_calendar_path()
        if cfg.calendar_file in ("", "auto")
        else resolve_path(cfg.calendar_file)
    )
    key = f"{path}:{path.stat().st_mtime if path.exists() else 0}:{cfg.enabled}"
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = EconomicCalendar.from_config(cfg)
    return _CACHE[key]

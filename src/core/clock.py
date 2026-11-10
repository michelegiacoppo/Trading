"""Timezone handling: broker server time <-> UTC <-> session timezone (CET).

MT5 timestamps are broker *server wall time* (most brokers: EET with DST,
i.e. UTC+2/UTC+3, so that the daily candle closes at NY 17:00). Sessions are
configured in ``sessions.timezone`` (default Europe/Rome) and key levels in UTC.

``ServerClock`` converts in both directions with either:
* an IANA zone for the server (e.g. ``Europe/Athens``) - DST aware, or
* a fixed offset detected live from the tick stream (``server_timezone: auto``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd


class ServerClock:
    def __init__(self, server_tz: str | None = None, offset: timedelta | None = None) -> None:
        self.zone = ZoneInfo(server_tz) if server_tz and server_tz != "auto" else None
        self.offset = offset or timedelta(0)
        self.detected = offset is not None

    @property
    def is_auto(self) -> bool:
        return self.zone is None

    @classmethod
    def from_config(cls, server_tz: str) -> ServerClock:
        return cls(server_tz=None if server_tz == "auto" else server_tz)

    def set_offset_from_tick(self, server_now: datetime, utc_now: datetime | None = None) -> bool:
        """Detect the server UTC offset (rounded to 15 min) - used with ``auto``.

        Returns False (and keeps the previous offset) when the tick is stale - e.g. the bot
        started during the weekend and the last tick is Friday's - so the implied offset is
        outside the real-world range UTC-12..UTC+14.
        """
        utc_now = utc_now or datetime.now(timezone.utc).replace(tzinfo=None)
        minutes = round((server_now - utc_now).total_seconds() / 900) * 15
        if not -12 * 60 <= minutes <= 14 * 60:
            return False
        self.offset = timedelta(minutes=minutes)
        self.detected = True
        return True

    # ----------------------------------------------------------- scalars
    def to_utc(self, server_dt: datetime) -> datetime:
        if self.zone is not None:
            return server_dt.replace(tzinfo=self.zone).astimezone(timezone.utc).replace(tzinfo=None)
        return server_dt - self.offset

    def to_zone(self, server_dt: datetime, tz: str) -> datetime:
        utc = self.to_utc(server_dt).replace(tzinfo=timezone.utc)
        return utc.astimezone(ZoneInfo(tz)).replace(tzinfo=None)

    # ------------------------------------------------------------ series
    def series_to_utc(self, times: pd.Series) -> pd.Series:
        if self.zone is not None:
            loc = times.dt.tz_localize(self.zone, ambiguous="NaT", nonexistent="shift_forward")
            out = loc.dt.tz_convert("UTC").dt.tz_localize(None)
            return out.fillna(times - pd.Timedelta(hours=2))  # DST-fold hour: best effort
        return times - self.offset

    def series_to_zone(self, times: pd.Series, tz: str) -> pd.Series:
        return (
            self.series_to_utc(times).dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
        )

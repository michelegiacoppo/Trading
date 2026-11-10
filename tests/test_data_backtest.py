"""Tests for historical data cache, key levels and the anti-illusion backtester (offline)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
    TickPath,
    compute_stats,
    hourly_analysis,
    synthetic_m1,
)
from src.config import LevelsConfig, load_settings
from src.core.clock import ServerClock
from src.core.data_feed import HistoricalStore
from src.core.levels import KeyLevelMap, compute_levels
from src.strategies.xau_velocity_scalp import create_strategy

pytest.importorskip("pyarrow")
UTC = ServerClock("UTC")


# ------------------------------------------------------------- fake MT5
class FakeMT5:
    """Serves M1 bars / ticks from a DataFrame and records requested ranges."""

    TIMEFRAME_M1 = "M1"
    COPY_TICKS_ALL = -1

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.calls: list[tuple[datetime, datetime]] = []

    def copy_rates_range(self, symbol, tf, start, end):  # noqa: ANN001, ANN201
        s, e = pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        self.calls.append((s, e))
        df = self.bars[(self.bars["time"] >= s) & (self.bars["time"] <= e)]
        arr = np.zeros(len(df), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                                       ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4")])  # fmt: skip
        arr["time"] = df["time"].dt.as_unit("s").astype("int64")
        for k in ("open", "high", "low", "close", "tick_volume", "spread"):
            arr[k] = df[k].to_numpy()
        return arr

    def copy_ticks_range(self, symbol, start, end, flags):  # noqa: ANN001, ANN201
        s = pd.Timestamp(start).tz_localize(None)
        t = pd.date_range(s, periods=10, freq="1s")
        arr = np.zeros(10, dtype=[("time", "i8"), ("bid", "f8"), ("ask", "f8"), ("last", "f8"),
                                  ("time_msc", "i8"), ("flags", "i4")])  # fmt: skip
        arr["time_msc"] = t.as_unit("ms").asi8
        arr["bid"], arr["ask"] = 2650.0, 2650.2
        return arr

    def symbol_info(self, symbol):  # noqa: ANN001, ANN201
        return SimpleNamespace(point=0.01)

    def last_error(self):  # noqa: ANN201
        return (0, "ok")


def test_parquet_cache_is_incremental(tmp_path: Path) -> None:
    bars = synthetic_m1(3 * 1440, start="2026-11-02 00:00")  # Mon-Wed
    fake = FakeMT5(bars)
    store = HistoricalStore(tmp_path, SimpleNamespace(mt5=fake), chunk_days=30)

    first = store.fetch_historical_data(
        "XAUUSD", "M1", days_back=2, now=datetime(2026, 11, 3, 12, 0)
    )
    assert store.bars_path("XAUUSD", "M1").exists()
    assert first["time"].max() <= pd.Timestamp("2026-11-03 12:00") + pd.Timedelta(days=1)
    n_calls = len(fake.calls)

    second = store.fetch_historical_data(
        "XAUUSD", "M1", days_back=2, now=datetime(2026, 11, 4, 23, 59)
    )
    new_calls = fake.calls[n_calls:]
    assert len(new_calls) == 1  # only the forward gap
    last_cached = first["time"].max()
    assert new_calls[0][0] >= last_cached - pd.Timedelta(
        days=1, minutes=1
    )  # starts near the cache end
    assert second["time"].is_monotonic_increasing and not second["time"].duplicated().any()
    assert second["time"].max() > last_cached


def test_tick_cache_per_day(tmp_path: Path) -> None:
    fake = FakeMT5(synthetic_m1(100))
    store = HistoricalStore(tmp_path, SimpleNamespace(mt5=fake))
    ticks = store.fetch_historical_ticks("XAUUSD", datetime(2026, 1, 5), datetime(2026, 1, 6))
    assert len(ticks) == 20
    assert ticks["spread_points"].iloc[0] == pytest.approx(20.0)
    assert store.ticks_path("XAUUSD", datetime(2026, 1, 5)).exists()
    assert len(store.load_ticks("XAUUSD", datetime(2026, 1, 5), datetime(2026, 1, 6))) == 20


# --------------------------------------------------------------- levels
def test_key_levels_pdh_asia_vwap() -> None:
    t = pd.date_range("2026-11-09 00:00", "2026-11-10 12:00", freq="5min")
    df = pd.DataFrame({"time": t, "high": 100.0, "low": 99.0, "close": 99.5, "tick_volume": 10})
    df.loc[df["time"] == "2026-11-09 15:00", "high"] = 105.0  # yesterday high
    df.loc[df["time"] == "2026-11-10 03:00", "low"] = 97.0  # today Asian low
    df.loc[df["time"] == "2026-11-10 08:00", "high"] = 103.0  # today London high
    lv = compute_levels(df, LevelsConfig(), UTC).set_index("time")
    today = lv.loc["2026-11-10 11:00"]
    assert today["pdh"] == 105.0 and today["pdl"] == 99.0
    assert today["asia_low"] == 97.0
    assert today["london_high"] == 103.0
    assert lv.loc["2026-11-10 02:00", "asia_low"] == 99.0  # causal: before the 03:00 low
    assert today["vwap_dn2"] < today["vwap"] < today["vwap_up2"]
    km = KeyLevelMap.from_frame(lv.reset_index())
    assert km.nearest(103.1, 1)[0][0] == "London High"


# ------------------------------------------------------------- backtest
def test_tickpath_resolves_order() -> None:
    t0 = pd.Timestamp("2026-11-10 10:00")
    ticks = pd.DataFrame({"time": [t0 + pd.Timedelta(seconds=s) for s in (1, 2, 3)],
                          "bid": [2650.0, 2652.1, 2647.9], "ask": [2650.2, 2652.3, 2648.1]})  # fmt: skip
    path = TickPath(ticks)
    assert path.first_hit(t0, 1, side=1, stop=2648.0, target=2652.0) == "target"
    assert path.first_hit(t0, 1, side=-1, stop=2652.0, target=2648.0) == "stop"
    assert TickPath(None).first_hit(t0, 1, 1, 2648.0, 2652.0) is None


def _engine():  # noqa: ANN202
    s = load_settings()
    s.sessions.windows = {}
    s.sessions.rollover_blackout = []
    return s, BacktestEngine(s, create_strategy(s.strategy, s.levels, UTC), "XAUUSD", clock=UTC)


def test_backtest_uses_real_spread_and_is_bounded() -> None:
    s, eng = _engine()
    res = eng.run(synthetic_m1(30_000))
    assert res.meta["spread_source"].startswith("MT5 bar spread")
    assert res.stats["trades"] > 0
    for k in ("win_rate_pct", "profit_factor", "expectancy_usd", "expectancy_r", "max_drawdown_pct",
              "sharpe", "recovery_factor"):  # fmt: skip
        assert k in res.stats
    worst = min(t.pnl for t in res.trades)
    assert worst > -s.backtest.initial_equity * s.risk.risk_per_trade_pct / 100 * 1.8
    hourly = hourly_analysis(res, UTC, "Europe/Rome")
    assert hourly["trades"].sum() == len(res.trades)


def test_backtest_synthetic_spread_is_flagged() -> None:
    _, eng = _engine()
    res = eng.run(synthetic_m1(5_000).drop(columns="spread"))
    assert res.meta["spread_source"].startswith("SYNTHETIC")


def test_conservative_intrabar_sl_first() -> None:
    """A bar that touches both SL and TP must be booked as a loss when no ticks exist."""
    s, eng = _engine()
    s.backtest.simulated_slippage_points = 0
    m1 = synthetic_m1(3_000)
    res = eng.run(m1)
    # every SL exit must be <= the configured risk (+costs) - no "lucky" intrabar ordering
    for t in res.trades:
        if t.reason == "SL":
            assert t.r_multiple <= -0.95


def test_compute_stats_recovery_factor() -> None:
    from src.backtest.engine import BacktestResult, BTTrade

    t0 = pd.Timestamp("2026-11-10 10:00")
    trades = [BTTrade("X", 1, t0, t0 + timedelta(minutes=5), 1, 1, 0.1, p, p / 50, "x")
              for p in (100.0, -50.0, 100.0)]  # fmt: skip
    eq = pd.Series([10_000, 10_100, 10_050, 10_150],
                   index=pd.date_range("2026-11-10", periods=4, freq="1D"))  # fmt: skip
    st = compute_stats(BacktestResult(trades, eq, 10_000))
    assert st["profit_factor"] == pytest.approx(4.0)
    assert st["recovery_factor"] == pytest.approx(150 / 50)
    assert st["expectancy_usd"] == pytest.approx(50.0)

"""Tests for the research lab: portfolio simulator, strategy library, stats, walk-forward, meta."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import BTTrade
from src.lab.portfolio import PortfolioRules, simulate_portfolio

T0 = pd.Timestamp("2026-11-10 10:00")


def _t(sym: str, start_min: int, dur_min: int, r: float, risk_pct: float = 0.5) -> BTTrade:
    e = T0 + pd.Timedelta(minutes=start_min)
    return BTTrade(sym, 1, e, e + pd.Timedelta(minutes=dur_min), 1.0, 1.0, 1.0, 0.0, r, "x",
                   risk_pct=risk_pct)  # fmt: skip


class TestPortfolio:
    def test_compounding_and_global_risk(self) -> None:
        rules = PortfolioRules(initial_equity=10_000, risk_per_trade_pct=1.0, max_open_positions=5,
                               max_positions_per_symbol=1, max_consecutive_losses=99)  # fmt: skip
        res = simulate_portfolio([_t("A", 0, 10, 2.0), _t("B", 20, 10, -1.0)], rules)
        assert res.trades[0].pnl == pytest.approx(200.0)  # 1% of 10,000 x 2R
        assert res.trades[1].pnl == pytest.approx(-102.0)  # 1% of 10,200 x -1R
        assert res.equity_curve.iloc[-1] == pytest.approx(10_098.0)

    def test_max_open_positions_is_global(self) -> None:
        rules = PortfolioRules(max_open_positions=2, max_positions_per_symbol=1)
        trades = [_t("A", 0, 60, 1), _t("B", 1, 60, 1), _t("C", 2, 60, 1), _t("D", 70, 5, 1)]
        res = simulate_portfolio(trades, rules)
        assert len(res.trades) == 3
        assert res.rejected == {"max open positions": 1}

    def test_account_kill_switch_locks_all_symbols(self) -> None:
        rules = PortfolioRules(risk_per_trade_pct=1.0, max_daily_loss_pct=2.5,
                               max_open_positions=9, max_consecutive_losses=99)  # fmt: skip
        trades = [_t("A", 0, 5, -1.5), _t("B", 10, 5, -1.5), _t("C", 20, 5, 3.0)]
        res = simulate_portfolio(trades, rules)
        assert len(res.trades) == 2 and res.locks == 1
        assert res.rejected == {"kill-switch locked": 1}
        nxt = _t("C", 24 * 60, 5, 1.0)  # next day: unlocked
        assert len(simulate_portfolio([*trades, nxt], rules).trades) == 3

    def test_same_bar_trade_releases_symbol(self) -> None:
        rules = PortfolioRules(max_positions_per_symbol=1)
        res = simulate_portfolio([_t("A", 0, 0, 1.0), _t("A", 5, 5, 1.0)], rules)
        assert len(res.trades) == 2

    def test_risk_multiplier_is_preserved(self) -> None:
        rules = PortfolioRules(risk_per_trade_pct=0.5)
        res = simulate_portfolio([_t("A", 0, 5, 1.0, risk_pct=0.25)], rules, ref_risk_pct=0.5)
        assert res.trades[0].pnl == pytest.approx(10_000 * 0.25 / 100)


# ----------------------------------------------------------- library
import numpy as np  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.core.clock import ServerClock  # noqa: E402
from src.strategies import indicators as ind  # noqa: E402
from src.strategies.library import LIBRARY  # noqa: E402
from src.strategies.xau_velocity_scalp import create_strategy  # noqa: E402

UTC = ServerClock("UTC")


@pytest.mark.parametrize("family", sorted(LIBRARY))
def test_library_families_are_causal(family: str) -> None:
    from src.backtest.engine import synthetic_m1

    cls = LIBRARY[family]
    tf = cls.TIMEFRAMES[0]
    s = load_settings()
    s.strategy.name = family
    s.strategy.timeframes = {"entry": tf, "confirm": tf, "trend": tf}
    s.strategy.params = dict(cls.DEFAULTS)
    strat = create_strategy(s.strategy, s.levels, UTC)
    bars = ind.resample_ohlc(synthetic_m1(120_000, seed=4), tf)
    strat.symbol = "XAUUSD"
    if cls.REQUIRES_CALENDAR:
        strat.calendar = synthetic_calendar(bars["time"].iloc[0], bars["time"].iloc[-1])
    full = strat.compute_features({tf: bars}).set_index("time")
    cut = int(len(bars) * 0.7)
    part = strat.compute_features({tf: bars.iloc[:cut]}).set_index("time")
    common = part.index[:-2]
    assert full.loc[common, "signal"].equals(part.loc[common, "signal"]), family
    assert (full["signal"] != 0).sum() > 0, f"{family} produced no signal"


# ------------------------------------------------------------- stats
def test_deflated_sharpe_raises_the_bar_with_more_trials() -> None:
    from src.lab.stats import deflated_sharpe, probabilistic_sharpe

    rng = np.random.default_rng(0)
    r = rng.normal(0.15, 1.0, 300)
    psr = probabilistic_sharpe(r)
    assert psr > 0.95
    assert deflated_sharpe(r, 2, 0.01) > deflated_sharpe(r, 500, 0.01)
    noise = rng.normal(0.0, 1.0, 300)
    assert deflated_sharpe(noise, 100, 0.01) < 0.5


# -------------------------------------------------------------- meta
def _meta_table(signal: bool, n: int = 900) -> pd.DataFrame:
    from src.lab.features import FEATURES

    rng = np.random.default_rng(1)
    t = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    t["side"] = rng.choice([-1, 1], n)
    edge = 0.8 * np.sign(t["rsi14"]) if signal else 0.0  # planted dependency on one feature
    t["r"] = (
        np.where(rng.random(n) < 0.5 + 0.3 * np.sign(edge) * (abs(edge) > 0), 1.0, -1.0)
        if signal
        else rng.choice([1.0, -1.0], n)
    )
    t["signal_time"] = pd.date_range("2025-01-01", periods=n, freq="6h")
    t["exit_time"] = t["signal_time"] + pd.Timedelta(hours=2)
    return t


def test_meta_labeling_finds_planted_edge_and_rejects_noise() -> None:
    pytest.importorskip("sklearn")
    from src.lab.meta import train_meta

    model, rep = train_meta(_meta_table(True))
    assert rep.adopted and model is not None
    assert rep.oos_mean_r_kept > rep.oos_mean_r_all
    assert "rsi14" in rep.feature_importance
    _, rep_noise = train_meta(_meta_table(False))
    assert not rep_noise.adopted


def test_meta_model_roundtrip(tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("sklearn")
    from src.lab.meta import MetaModel, train_meta

    model, _ = train_meta(_meta_table(True))
    assert model is not None
    model.save(tmp_path / "m.joblib")
    loaded = MetaModel.load(tmp_path / "m.joblib")
    x = _meta_table(True).head(20)
    assert (loaded.predict_proba(x) == model.predict_proba(x)).all()


# --------------------------------------------------------- lab smoke
def test_lab_end_to_end_smoke(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from rich.console import Console

    from src.backtest.engine import synthetic_m1
    from src.lab import runner

    monkeypatch.setattr(runner, "PROFILES_DIR", tmp_path)
    s = load_settings()
    s.general.log_dir = str(tmp_path / "logs")
    s.lab.models_dir = str(tmp_path / "models")
    s.lab.min_trades = 20
    s.lab.min_train_trades = 10
    m1 = synthetic_m1(150_000, seed=3)
    data = {"XAUUSD": {tf: (ind.resample_ohlc(m1, tf), None) for tf in ("M15", "H1")}}
    con = Console(record=True, width=160)
    lab = runner.Lab(s, data, UTC, con, families=["orb", "tsmom"], quick=True)
    res = lab.run()
    text = con.export_text()
    assert "Walk-forward" in text and "Outcome of every" in text
    assert len(res["cands"]) > 0
    assert (tmp_path / "logs" / "lab" / "journal.jsonl").exists()
    assert all(c.status for c in res["cands"])


# ------------------------------------------------------------ health
def test_health_monitor_disables_on_divergence(tmp_path) -> None:  # noqa: ANN001
    from datetime import datetime

    from src.config import HealthConfig
    from src.core.health import LiveTradeLog, StrategyHealth

    h = StrategyHealth(HealthConfig(enabled=True, expected_mean_r=0.2, expected_sd_r=1.0,
                                    min_trades=20, z_disable=2.5))  # fmt: skip
    for _ in range(19):
        h.record(-1.0)
    assert h.allows_entries  # not judged before min_trades
    h.record(-1.0)
    assert not h.allows_entries and "z=" in (h.disabled_reason or "")
    ok = StrategyHealth(HealthConfig(enabled=True, expected_mean_r=0.2, expected_sd_r=1.0))
    for i in range(40):
        ok.record(1.0 if i % 2 else -0.6)
    assert ok.allows_entries
    log = LiveTradeLog(tmp_path / "t.csv")
    log.append("p", "XAUUSD", 1, 10.0, 20.0, 0.5, "TP", datetime(2026, 11, 11, 10))
    assert (tmp_path / "t.csv").read_text().count("\n") == 2


# --------------------------------------------------------------- news
def synthetic_calendar(start: pd.Timestamp, end: pd.Timestamp):  # noqa: ANN201
    from src.core.news import EconomicCalendar

    rng = np.random.default_rng(5)
    days = pd.date_range(start.normalize(), end, freq="B")
    times = [d + pd.Timedelta(hours=14, minutes=30) for d in days]
    n = len(times)
    fc = rng.normal(0, 1, n)
    return EconomicCalendar(pd.DataFrame({
        "time": times, "currency": "USD", "event": "Synthetic CPI", "importance": 3,
        "actual": fc + rng.normal(0, 0.5, n), "forecast": fc, "previous": fc,
    }))  # fmt: skip


def test_symbol_currencies() -> None:
    from src.core.news import symbol_currencies

    assert symbol_currencies("EURUSD.m") == ("EUR", "USD")
    assert symbol_currencies("XAUUSD") == (None, "USD")
    assert symbol_currencies("GOLD") == (None, "USD")
    assert symbol_currencies("GER40") == (None, "EUR")
    assert symbol_currencies("USDJPY-ECN") == ("USD", "JPY")


def test_news_blackout_and_features(tmp_path) -> None:  # noqa: ANN001
    from datetime import datetime

    from src.config import NewsConfig
    from src.core.news import EconomicCalendar

    tsv = tmp_path / "cal.csv"
    tsv.write_text(
        "time\tcurrency\tevent\timportance\tactual\tforecast\tprevious\timpact\n"
        "2026.11.10 15:30\tUSD\tCPI m/m\t3\t0.5\t0.3\t0.2\t1\n"
        "2026.11.10 10:00\tEUR\tZEW, Sentiment\t2\t10\t12\t8\t2\n"
        "2026.11.11 15:30\tUSD\tCPI m/m\t3\t0.1\t0.3\t0.5\t2\n",
        encoding="utf-8",
    )
    cal = EconomicCalendar.load(tsv)
    cfg = NewsConfig(blackout_minutes_before=10, blackout_minutes_after=15)
    assert cal.blackout("XAUUSD", datetime(2026, 11, 10, 15, 25), cfg).blocked
    assert cal.blackout("EURUSD", datetime(2026, 11, 10, 15, 40), cfg).blocked
    assert not cal.blackout("XAUUSD", datetime(2026, 11, 10, 15, 50), cfg).blocked
    assert not cal.blackout("EURUSD", datetime(2026, 11, 10, 10, 0), cfg).blocked  # medium impact
    assert not cal.blackout("USDJPY", datetime(2026, 11, 10, 3, 0), cfg).blocked
    f = cal.features_for(
        "XAUUSD", pd.Series(pd.to_datetime(["2026-11-11 16:00", "2026-11-11 15:00"]))
    )
    assert f["news_min_since"].iloc[0] == pytest.approx(30)
    assert f["news_min_to"].iloc[1] == pytest.approx(30)
    # second CPI missed the forecast: bad for USD -> positive (bullish) for XAUUSD
    assert f["news_surprise"].iloc[0] > 0


def test_news_drift_trades_after_releases() -> None:
    from src.backtest.engine import BacktestEngine, synthetic_m1
    from src.lab.portfolio import candidate_settings

    cls = LIBRARY["news_drift"]
    s = candidate_settings(load_settings())
    s.strategy.name = "news_drift"
    s.strategy.timeframes = {"entry": "M5", "confirm": "M5", "trend": "M5"}
    s.strategy.params = {**cls.DEFAULTS, "use_surprise": False, "move_atr": 0.3}
    s.sessions.windows = {}
    s.news.enabled = False  # calendar injected directly below
    for k, v in cls.NEWS.items():
        setattr(s.news, k, v)
    for k, v in cls.exec_overrides(s.strategy.params).items():
        setattr(s.execution, k, v)
    bars = ind.resample_ohlc(synthetic_m1(60_000, seed=8), "M5")
    strat = create_strategy(s.strategy, s.levels, UTC)
    eng = BacktestEngine(s, strat, "XAUUSD", clock=UTC)
    cal = synthetic_calendar(bars["time"].iloc[0], bars["time"].iloc[-1])
    import src.backtest.engine as engine_mod

    orig = engine_mod.get_calendar
    engine_mod.get_calendar = lambda cfg: cal
    try:
        res = eng.run(bars, base_tf="M5")
    finally:
        engine_mod.get_calendar = orig
    assert res.stats["trades"] > 0
    for t in res.trades:  # every entry happens shortly after a 14:30 release
        mins = (
            pd.Timestamp(t.signal_time) - pd.Timestamp(t.signal_time).normalize()
        ).total_seconds() / 60
        assert 14 * 60 + 30 + 10 <= mins <= 14 * 60 + 30 + 15

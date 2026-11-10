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

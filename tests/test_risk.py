"""Unit tests for sizing, guardrails, trade management and no-look-ahead guarantees."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.config import (
    ExecutionConfig,
    GuardrailConfig,
    RiskConfig,
    SessionConfig,
    SymbolConfig,
    SymbolSpec,
    load_settings,
)
from src.core.execution import PositionView, TradeManager
from src.risk.guardrails import (
    DailyKillSwitch,
    EntryContext,
    GapDetector,
    Guardrails,
    SessionFilter,
    SlippageMonitor,
    SpreadFilter,
)
from src.risk.position_sizer import PositionSizer, atr_stop_distance, floor_to_step, money_per_lot
from src.strategies import indicators as ind
from src.strategies.xau_velocity_scalp import avoid_round_levels

XAU = SymbolSpec(
    name="XAUUSD",
    point=0.01,
    digits=2,
    tick_size=0.01,
    tick_value=1.0,
    contract_size=100,
    volume_min=0.01,
    volume_max=50,
    volume_step=0.01,
)
NOW = datetime(2026, 11, 10, 10, 0)  # Tuesday


# ------------------------------------------------------------------ sizing
class TestPositionSizer:
    def test_money_per_lot_xau(self) -> None:
        # 1.00 USD move on 1 lot (100 oz) = 100 USD
        assert money_per_lot(1.0, XAU) == pytest.approx(100.0)

    def test_basic_risk(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, commission_per_lot=0.0))
        res = sizer.size(10_000, 2650.0, 2648.0, XAU)  # 2.00 SL -> 200 USD/lot, budget 100
        assert res.lots == pytest.approx(0.50)
        assert res.risk_money == pytest.approx(100.0)
        assert res.risk_pct_effective == pytest.approx(1.0)

    def test_commission_and_spread_reduce_size(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, commission_per_lot=7.0))
        res = sizer.size(10_000, 2650.0, 2648.0, XAU, spread_price=0.20)
        # loss/lot = 2.20*100 + 7 = 227 -> 100/227 = 0.4405 -> floor 0.44
        assert res.lots == pytest.approx(0.44)
        assert res.risk_money <= 100.0

    def test_never_rounds_up(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=0.5, commission_per_lot=0.0))
        for sl_dist in (0.37, 1.13, 2.71, 4.99):
            res = sizer.size(7_777, 2000.0, 2000.0 - sl_dist, XAU)
            assert res.risk_money <= 7_777 * 0.005 + 1e-9

    def test_below_min_lot_is_refused(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=0.1, commission_per_lot=0.0))
        res = sizer.size(100, 2650.0, 2640.0, XAU)  # 0.10 USD budget vs 1000 USD/lot
        assert not res.ok and res.lots == 0.0
        assert "volume_min" in res.reason

    def test_caps(self) -> None:
        sizer = PositionSizer(
            RiskConfig(
                risk_per_trade_pct=5.0,
                commission_per_lot=0.0,
                max_lot=1.0,
                max_margin_usage_pct=50.0,
            )
        )
        assert sizer.size(1_000_000, 2650, 2649, XAU).lots == 1.0
        res = sizer.size(1_000_000, 2650, 2649, XAU, free_margin=1000, margin_per_lot=2650)
        assert res.lots == pytest.approx(0.18)  # 500 / 2650 = 0.1886 -> 0.18

    @pytest.mark.parametrize("bad", [(0, 2650, 2649), (10_000, 2650, 2650)])
    def test_invalid_inputs(self, bad: tuple[float, float, float]) -> None:
        assert not PositionSizer(RiskConfig()).size(*bad, XAU).ok

    def test_floor_to_step(self) -> None:
        assert floor_to_step(0.129999, 0.01) == 0.12
        assert floor_to_step(0.13, 0.01) == 0.13
        assert floor_to_step(1.27, 0.1) == 1.2
        assert floor_to_step(3.9, 1.0) == 3.0

    def test_atr_stop_distance(self) -> None:
        assert atr_stop_distance(2.0, 1.5, 0.01, 80, 600) == pytest.approx(3.0)
        assert atr_stop_distance(0.2, 1.0, 0.01, 80, 600) == pytest.approx(0.8)  # min clamp
        assert atr_stop_distance(2.0, 1.0, 0.01, 80, 600, structure_distance=2.5) == pytest.approx(
            2.5
        )
        assert atr_stop_distance(10.0, 1.0, 0.01, 80, 600) is None  # too wide -> no trade


# --------------------------------------------------------------- guardrails
def _sym() -> SymbolConfig:
    return SymbolConfig(
        name="XAUUSD", max_spread_points=35, spread_median_mult=1.8, spread_atr_ratio_max=0.15
    )


class TestSpreadFilter:
    def test_hard_cap(self) -> None:
        f = SpreadFilter(_sym(), GuardrailConfig())
        assert not f.check(40)[0]
        assert f.check(20)[0]

    def test_dynamic_median_spike(self) -> None:
        f = SpreadFilter(_sym(), GuardrailConfig(spread_min_samples=10))
        for _ in range(50):
            f.update(15)
        assert f.median == 15
        ok, reason = f.check(30)  # 2x median, under hard cap
        assert not ok and "spike" in reason
        assert f.check(25)[0]

    def test_spread_vs_atr(self) -> None:
        f = SpreadFilter(_sym(), GuardrailConfig())
        assert not f.check(20, atr_points=100)[0]  # 0.20 > 0.15
        assert f.check(20, atr_points=200)[0]


class TestGapDetector:
    def test_bar_gap_blocks_for_cooldown(self) -> None:
        g = GapDetector(GuardrailConfig(gap_atr_mult=1.5, gap_cooldown_bars=2))
        assert g.on_bar(open_=2655.0, prev_close=2650.0, atr=2.0)
        assert not g.check(NOW)[0]
        g.on_bar(2655.1, 2655.0, 2.0)
        assert not g.check(NOW)[0]
        g.on_bar(2655.2, 2655.1, 2.0)
        assert g.check(NOW)[0]

    def test_tick_jump(self) -> None:
        g = GapDetector(GuardrailConfig(tick_jump_atr_mult=1.0, tick_jump_cooldown_s=30))
        g.on_tick(2650.0, 1.0, NOW)
        assert g.on_tick(2651.5, 1.0, NOW + timedelta(seconds=1))
        assert not g.check(NOW + timedelta(seconds=10))[0]
        assert g.check(NOW + timedelta(seconds=40))[0]


class TestSlippage:
    def test_pause_on_high_average(self) -> None:
        m = SlippageMonitor(GuardrailConfig(max_avg_slippage_points=10, slippage_pause_s=60))
        for _ in range(3):
            slip = m.record(2650.00, 2650.30, is_buy=True, point=0.01, now=NOW)
        assert slip == pytest.approx(30)
        assert not m.check(NOW + timedelta(seconds=10))[0]
        assert m.check(NOW + timedelta(seconds=61))[0]

    def test_positive_slippage_is_negative_points(self) -> None:
        m = SlippageMonitor(GuardrailConfig())
        assert m.record(2650.00, 2650.05, is_buy=False, point=0.01, now=NOW) == pytest.approx(-5)


class TestSessionFilter:
    cfg = SessionConfig(
        windows=[["09:00", "12:30"], ["14:30", "18:30"]],
        rollover_blackout=["23:50", "01:10"],
        friday_cutoff="21:00",
        monday_start="01:30",
        news_blackouts=["2026-11-10 15:30"],
    )

    @pytest.mark.parametrize(
        ("when", "ok"),
        [
            (datetime(2026, 11, 10, 10, 0), True),
            (datetime(2026, 11, 10, 13, 0), False),  # between sessions
            (datetime(2026, 11, 10, 15, 32), False),  # news blackout
            (datetime(2026, 11, 10, 15, 45), True),  # after blackout window
            (datetime(2026, 11, 14, 10, 0), False),  # saturday
            (datetime(2026, 11, 13, 21, 5), False),  # friday cutoff
        ],
    )
    def test_windows(self, when: datetime, ok: bool) -> None:
        assert SessionFilter(self.cfg).check(when)[0] is ok

    def test_rollover_across_midnight(self) -> None:
        f = SessionFilter(SessionConfig(windows=[], rollover_blackout=["23:50", "01:10"]))
        assert not f.check(datetime(2026, 11, 10, 23, 55))[0]
        assert not f.check(datetime(2026, 11, 11, 0, 30))[0]
        assert f.check(datetime(2026, 11, 11, 2, 0))[0]


class TestKillSwitch:
    def test_daily_loss_trips_and_resets_next_day(self) -> None:
        ks = DailyKillSwitch(RiskConfig(daily_loss_limit_pct=3.0, daily_trailing_dd_pct=10))
        ks.update_equity(10_000, NOW)
        ks.update_equity(9_750, NOW + timedelta(minutes=5))
        assert not ks.tripped
        ks.update_equity(9_690, NOW + timedelta(minutes=10))
        assert ks.tripped and not ks.check(NOW)[0]
        ks.update_equity(9_800, NOW + timedelta(minutes=20))  # recovery does not re-arm
        assert ks.tripped
        ks.update_equity(9_800, NOW + timedelta(days=1))
        assert not ks.tripped

    def test_trailing_drawdown_from_peak(self) -> None:
        ks = DailyKillSwitch(RiskConfig(daily_loss_limit_pct=50, daily_trailing_dd_pct=2.0))
        ks.update_equity(10_000, NOW)
        ks.update_equity(10_500, NOW)
        ks.update_equity(10_280, NOW)
        assert ks.tripped and "DD" in (ks.tripped_reason or "")

    def test_consecutive_losses_and_cooldown(self) -> None:
        ks = DailyKillSwitch(RiskConfig(max_consecutive_losses=3, cooldown_after_loss_s=60))
        ks.update_equity(10_000, NOW)
        ks.record_close(-10, NOW)
        assert ks.check(NOW + timedelta(seconds=30))[1] == "post-loss cooldown"
        assert ks.check(NOW + timedelta(seconds=61))[0]
        ks.record_close(+5, NOW)
        ks.record_close(-10, NOW)
        ks.record_close(-10, NOW)
        assert not ks.tripped
        ks.record_close(-10, NOW)
        assert ks.tripped

    def test_max_trades(self) -> None:
        ks = DailyKillSwitch(RiskConfig(max_trades_per_day=2))
        ks.update_equity(10_000, NOW)
        ks.record_entry()
        ks.record_entry()
        assert "max trades" in ks.check(NOW)[1]


def test_guardrails_facade_collects_all_reasons() -> None:
    g = Guardrails(
        {"XAUUSD": _sym()},
        GuardrailConfig(stale_tick_s=5),
        RiskConfig(max_open_positions=1),
        SessionConfig(windows=[["09:00", "10:00"]]),
    )
    ctx = EntryContext(
        "XAUUSD",
        datetime(2026, 11, 10, 11, 0),
        spread_points=50,
        last_tick_age_s=20,
        open_positions_total=1,
        open_positions_symbol=1,
    )
    d = g.check_entry(ctx)
    assert not d.allowed
    text = str(d)
    for frag in ("outside session", "spread", "stale", "max open"):
        assert frag in text


# --------------------------------------------------------- trade manager
class TestTradeManager:
    cfg = ExecutionConfig(
        partial_close_r=1.0,
        partial_close_fraction=0.5,
        break_even_r=1.0,
        break_even_buffer_points=10,
        trailing_start_r=1.2,
        trailing_atr_mult=1.0,
    )

    def _pos(self, side: int = 1) -> PositionView:
        return PositionView(
            1,
            "XAUUSD",
            side,
            0.20,
            2650.0,
            2650.0 - side * 2.0,
            2650.0 + side * 3.2,
            NOW,
            0.20,
            2.0,
        )

    def test_nothing_below_1r(self) -> None:
        tm = TradeManager(self.cfg, 25)
        assert tm.evaluate(self._pos(), 2651.0, 2651.2, 1.5, NOW, XAU) == []

    def test_partial_and_break_even_at_1r(self) -> None:
        tm = TradeManager(self.cfg, 25)
        acts = tm.evaluate(self._pos(), 2652.0, 2652.2, 1.5, NOW, XAU)
        kinds = {a.kind: a for a in acts}
        assert kinds["partial"].volume == pytest.approx(0.10)
        assert kinds["modify_sl"].new_sl == pytest.approx(2650.10)

    def test_trailing_short(self) -> None:
        tm = TradeManager(self.cfg, 25)
        pos = self._pos(side=-1)
        pos.partial_done = True
        acts = tm.evaluate(pos, 2646.8, 2647.0, 1.0, NOW, XAU)  # ask 2647 -> +1.5R
        sl = next(a for a in acts if a.kind == "modify_sl").new_sl
        assert sl == pytest.approx(2648.0)  # ask + 1*ATR, better than BE 2649.90

    def test_time_stop(self) -> None:
        tm = TradeManager(self.cfg, 25)
        acts = tm.evaluate(self._pos(), 2650.5, 2650.7, 1.0, NOW + timedelta(minutes=26), XAU)
        assert acts and acts[0].kind == "close"

    def test_sl_never_loosens(self) -> None:
        tm = TradeManager(self.cfg, 25)
        pos = self._pos()
        pos.sl = 2651.5  # already trailed above BE
        acts = tm.evaluate(pos, 2652.1, 2652.3, 5.0, NOW, XAU)
        assert all(a.kind != "modify_sl" for a in acts)


# ------------------------------------------------------------ strategy bits
def test_avoid_round_levels() -> None:
    assert avoid_round_levels(2650.05, 1, 0.5, 0.15) == pytest.approx(2649.85)
    assert avoid_round_levels(2649.92, -1, 0.5, 0.15) == pytest.approx(2650.15)
    assert avoid_round_levels(2650.25, 1, 0.5, 0.15) == pytest.approx(2650.25)


def test_align_htf_has_no_lookahead() -> None:
    m1 = pd.DataFrame({"time": pd.date_range("2026-11-10 10:00", periods=30, freq="1min")})
    m15 = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-11-10 09:45", "2026-11-10 10:00", "2026-11-10 10:15"]),
            "v": [1, 2, 3],
        }
    )
    out = ind.align_htf(m1, "M1", m15, "M15", ["v"], "h_").set_index("time")["h_v"]
    # the 10:00 M15 bar closes at 10:15 => first usable by the M1 bar 10:14 (closes 10:15)
    assert out[pd.Timestamp("2026-11-10 10:13")] == 1
    assert out[pd.Timestamp("2026-11-10 10:14")] == 2
    assert out[pd.Timestamp("2026-11-10 10:28")] == 2


def test_strategy_features_are_causal() -> None:
    """Signals on a truncated history must equal signals on the full history."""
    from src.backtest.engine import synthetic_m1
    from src.strategies.xau_velocity_scalp import create_strategy

    s = load_settings()
    strat = create_strategy(s.strategy)
    m1 = synthetic_m1(6_000, seed=3)

    def feats(df: pd.DataFrame) -> pd.DataFrame:
        return strat.compute_features(
            {"M1": df, "M5": ind.resample_ohlc(df, "M5"), "M15": ind.resample_ohlc(df, "M15")}
        )

    full = feats(m1)
    cut = 4_000
    # drop a trailing partial M5/M15 bucket from the truncated input to mimic closed bars only
    part = feats(m1.iloc[:cut])
    common = part["time"].iloc[:-15]
    a = full.set_index("time").loc[common, "signal"]
    b = part.set_index("time").loc[common, "signal"]
    assert (a.values == b.values).all()


def test_backtest_smoke() -> None:
    from src.backtest.engine import BacktestEngine, synthetic_m1
    from src.strategies.xau_velocity_scalp import create_strategy

    s = load_settings()
    res = BacktestEngine(s, create_strategy(s.strategy), "XAUUSD").run(synthetic_m1(15_000))
    assert res.stats["trades"] > 0
    assert len(res.equity_curve) > 0
    # single-trade loss can never exceed ~risk% (+ slippage/gap tolerance)
    worst = min(t.pnl for t in res.trades)
    assert worst > -s.backtest.initial_equity * s.risk.risk_per_trade_pct / 100 * 1.6


def test_dashboard_renders() -> None:
    from rich.console import Console

    from src.ui.dashboard import build_layout, demo_state

    con = Console(record=True, width=160, height=50)
    con.print(build_layout(demo_state()))
    assert "XAU QUANT SCALPER" in con.export_text()

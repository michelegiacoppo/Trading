"""Unit tests: position sizing, guardrails, kill-switch, Hit & Run management, strategy causality."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.config import (
    ExecutionConfig,
    GuardrailConfig,
    LevelsConfig,
    RiskConfig,
    SessionConfig,
    SymbolConfig,
    SymbolSpec,
    load_settings,
)
from src.core.clock import ServerClock
from src.core.execution import MarketContext, PositionView, TradeManager
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
from src.strategies.xau_velocity_scalp import avoid_round_levels, create_strategy

XAU = SymbolSpec(name="XAUUSD", point=0.01, digits=2, tick_size=0.01, tick_value=1.0,
                 contract_size=100, volume_min=0.01, volume_max=50, volume_step=0.01)  # fmt: skip
NOW = datetime(2026, 11, 10, 11, 0)  # Tuesday, server time
UTC_CLOCK = ServerClock("UTC")


# ------------------------------------------------------------------ sizing
class TestPositionSizer:
    def test_money_per_lot_xau(self) -> None:
        assert money_per_lot(1.0, XAU) == pytest.approx(100.0)  # 1$ move x 100 oz

    def test_exact_risk(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=0.5, commission_per_lot=0.0))
        res = sizer.size(10_000, 2650.0, 2649.0, XAU)  # 1.00 SL -> 100$/lot, budget 50$
        assert res.lots == pytest.approx(0.50)
        assert res.risk_pct_effective == pytest.approx(0.5)

    def test_commission_and_spread_reduce_size(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, commission_per_lot=7.0))
        res = sizer.size(10_000, 2650.0, 2648.0, XAU, spread_price=0.20)
        assert res.lots == pytest.approx(0.44)  # 100 / (220 + 7) = 0.4405 -> floor
        assert res.risk_money <= 100.0

    def test_never_rounds_up(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=0.5, commission_per_lot=0.0))
        for sl_dist in (0.37, 1.13, 2.71, 4.99):
            assert (
                sizer.size(7_777, 2000.0, 2000.0 - sl_dist, XAU).risk_money <= 7_777 * 0.005 + 1e-9
            )

    def test_below_min_lot_is_refused(self) -> None:
        res = PositionSizer(RiskConfig(risk_per_trade_pct=0.1, commission_per_lot=0.0)).size(
            100, 2650, 2640, XAU
        )
        assert not res.ok and "volume_min" in res.reason

    def test_risk_multiplier_and_caps(self) -> None:
        sizer = PositionSizer(RiskConfig(risk_per_trade_pct=1.0, commission_per_lot=0.0, max_lot=1.0,
                                         max_margin_usage_pct=50.0))  # fmt: skip
        assert sizer.size(10_000, 2650, 2649, XAU, risk_pct=0.5).lots == pytest.approx(0.5)
        assert sizer.size(1_000_000, 2650, 2649, XAU).lots == 1.0
        assert (
            sizer.size(1_000_000, 2650, 2649, XAU, free_margin=1000, margin_per_lot=2650).lots
            == 0.18
        )

    def test_floor_to_step(self) -> None:
        assert floor_to_step(0.129999, 0.01) == 0.12
        assert floor_to_step(1.27, 0.1) == 1.2
        assert floor_to_step(3.9, 1.0) == 3.0

    def test_atr_stop_distance(self) -> None:
        assert atr_stop_distance(2.0, 1.5, 0.01, 80, 600) == pytest.approx(3.0)
        assert atr_stop_distance(0.2, 1.0, 0.01, 80, 600) == pytest.approx(0.8)
        assert atr_stop_distance(10.0, 1.0, 0.01, 80, 600) is None


# --------------------------------------------------------------- guardrails
def _sym() -> SymbolConfig:
    return SymbolConfig(
        name="XAUUSD", max_spread_points=35, spread_median_mult=1.8, spread_atr_ratio_max=0.15
    )


class TestSpreadFilter:
    def test_hard_cap_and_atr_ratio(self) -> None:
        f = SpreadFilter(_sym(), GuardrailConfig())
        assert not f.check(40)[0]
        assert not f.check(20, atr_points=100)[0]
        assert f.check(20, atr_points=200)[0]

    def test_dynamic_median_spike(self) -> None:
        f = SpreadFilter(_sym(), GuardrailConfig(spread_min_samples=10))
        for _ in range(50):
            f.update(15)
        ok, reason = f.check(30)
        assert not ok and "spike" in reason


class TestGapDetector:
    def test_bar_gap_blocks_for_n_minutes(self) -> None:
        g = GapDetector(GuardrailConfig(gap_atr_mult=1.5, gap_block_minutes=10))
        assert g.on_bar(open_=2655.0, prev_close=2650.0, atr=2.0, now=NOW)
        assert not g.check(NOW + timedelta(minutes=9))[0]
        assert g.check(NOW + timedelta(minutes=10))[0]

    def test_small_gap_ignored(self) -> None:
        g = GapDetector(GuardrailConfig(gap_atr_mult=1.5))
        assert not g.on_bar(2651.0, 2650.0, 2.0, NOW)
        assert g.check(NOW)[0]

    def test_tick_jump(self) -> None:
        g = GapDetector(GuardrailConfig(tick_jump_atr_mult=1.0, gap_block_minutes=5))
        g.on_tick(2650.0, 1.0, NOW)
        assert g.on_tick(2651.5, 1.0, NOW + timedelta(seconds=1))
        assert not g.check(NOW + timedelta(minutes=4))[0]
        assert g.check(NOW + timedelta(minutes=6))[0]


class TestSlippage:
    def test_pause_on_high_average(self) -> None:
        m = SlippageMonitor(GuardrailConfig(max_avg_slippage_points=10, slippage_pause_s=60))
        for _ in range(3):
            assert m.record(2650.00, 2650.30, is_buy=True, point=0.01, now=NOW) == pytest.approx(30)
        assert not m.check(NOW + timedelta(seconds=10))[0]
        assert m.check(NOW + timedelta(seconds=61))[0]


class TestSessionFilterCET:
    """Server on Europe/Athens (EET), sessions in Europe/Rome (CET): 1h difference."""

    cfg = SessionConfig(timezone="Europe/Rome", server_timezone="Europe/Athens",
                        windows={"London": ["08:00", "11:30"], "NY": ["14:00", "18:30"]},
                        rollover_blackout=["22:45", "01:15"], friday_cutoff="21:30",
                        monday_start="02:00", blocked_slots=["17:30"],
                        news_blackouts=["2026-11-10 14:30"])  # fmt: skip

    @pytest.mark.parametrize(
        ("server_time", "ok", "session"),
        [
            (datetime(2026, 11, 10, 9, 30), True, "London"),   # 08:30 CET
            (datetime(2026, 11, 10, 8, 30), False, "Closed"),  # 07:30 CET: before London
            (datetime(2026, 11, 10, 13, 0), False, "Closed"),  # 12:00 CET: lunch gap
            (datetime(2026, 11, 10, 15, 32), False, "NY"),     # 14:32 CET: news blackout
            (datetime(2026, 11, 10, 18, 40), False, "NY"),     # 17:40 CET: blocked slot
            (datetime(2026, 11, 10, 16, 0), True, "NY"),       # 15:00 CET
            (datetime(2026, 11, 10, 23, 50), False, "Closed"), # 22:50 CET: rollover
            (datetime(2026, 11, 14, 11, 0), False, "London"),  # saturday
        ],
    )  # fmt: skip
    def test_windows(self, server_time: datetime, ok: bool, session: str) -> None:
        f = SessionFilter(self.cfg)
        assert f.check(server_time)[0] is ok
        assert f.active_session(server_time) == session

    def test_auto_offset(self) -> None:
        clock = ServerClock()
        clock.set_offset_from_tick(
            datetime(2026, 11, 10, 12, 0, 3), utc_now=datetime(2026, 11, 10, 10, 0)
        )
        assert clock.offset == timedelta(hours=2)
        # a stale tick (weekend start) must not produce an absurd offset
        assert not clock.set_offset_from_tick(
            datetime(2026, 11, 6, 23, 59), utc_now=datetime(2026, 11, 8, 10, 0)
        )
        assert clock.offset == timedelta(hours=2)
        f = SessionFilter(SessionConfig(windows={"London": ["08:00", "11:30"]}), clock)
        assert f.active_session(datetime(2026, 11, 10, 11, 0)) == "London"  # 09:00 UTC = 10:00 CET


class TestKillSwitch:
    def test_daily_loss_closed_plus_floating_locks_and_flattens(self) -> None:
        ks = DailyKillSwitch(RiskConfig(max_daily_loss_pct=2.5, flatten_on_lock=True))
        ks.update_equity(10_000, NOW)
        ks.update_equity(9_760, NOW + timedelta(minutes=5))
        assert ks.status == "ACTIVE"
        ks.update_equity(9_740, NOW + timedelta(minutes=10))  # -2.6% incl. floating
        assert ks.status == "LOCKED" and ks.needs_flatten
        ks.flatten_done()
        assert not ks.needs_flatten and not ks.check(NOW)[0]
        ks.update_equity(9_900, NOW + timedelta(minutes=20))  # recovery does not unlock
        assert ks.tripped
        ks.update_equity(9_900, NOW + timedelta(days=1))  # next day
        assert ks.status == "ACTIVE"

    def test_profit_target_locks(self) -> None:
        ks = DailyKillSwitch(RiskConfig(daily_profit_target_pct=2.0))
        ks.update_equity(10_000, NOW)
        ks.update_equity(10_210, NOW)
        assert ks.tripped and "profit target" in (ks.locked_reason or "")

    def test_partial_exits_do_not_break_loss_streak_logic(self) -> None:
        ks = DailyKillSwitch(RiskConfig(max_consecutive_losses=2, cooldown_after_loss_s=0))
        ks.update_equity(10_000, NOW)
        ks.record_close(-10, NOW)
        ks.record_close(+30, NOW, final=False)  # TP1 partial of the next trade
        assert ks.consecutive_losses == 1
        ks.record_close(-5, NOW)  # runner stopped at a loss
        assert ks.tripped

    def test_max_trades(self) -> None:
        ks = DailyKillSwitch(RiskConfig(max_trades_per_day=2))
        ks.update_equity(10_000, NOW)
        ks.record_entry()
        ks.record_entry()
        assert "max trades" in ks.check(NOW)[1]


def test_guardrails_facade_collects_all_reasons() -> None:
    g = Guardrails({"XAUUSD": _sym()}, GuardrailConfig(stale_tick_s=5),
                   RiskConfig(max_open_positions=2, max_positions_per_symbol=1),
                   SessionConfig(server_timezone="UTC", timezone="UTC", windows={"A": ["09:00", "10:00"]}))  # fmt: skip
    ctx = EntryContext("XAUUSD", datetime(2026, 11, 10, 11, 0), spread_points=50,
                       last_tick_age_s=20, open_positions_total=2, open_positions_symbol=1)  # fmt: skip
    text = str(g.check_entry(ctx))
    for frag in ("outside session", "spread", "stale", "max concurrent", "max trades on symbol"):
        assert frag in text


# --------------------------------------------------- Hit & Run management
class TestTradeManager:
    cfg = ExecutionConfig(tp1_r=1.0, tp1_close_fraction=0.65, break_even_extra_points=5,
                          trailing_mode="candles", trailing_candles=2, trailing_only_after_tp1=True,
                          max_trade_duration_minutes=25, stale_require_fade=True,
                          stale_momentum_threshold=0.3)  # fmt: skip

    def _pos(self, side: int = 1, vol: float = 0.20) -> PositionView:
        return PositionView(1, "XAUUSD", side, vol, 2650.0, 2650.0 - side * 2.0, 2650.0 + side * 6.0,
                            NOW, vol, 2.0)  # fmt: skip

    def test_nothing_before_tp1(self) -> None:
        tm = TradeManager(self.cfg, commission_per_lot=7.0)
        mkt = MarketContext(1.5, 2651.2, 2649.5, 1.0)
        assert tm.evaluate(self._pos(), 2651.0, 2651.2, mkt, NOW, XAU) == []

    def test_tp1_closes_65pct_and_moves_to_be_plus_costs(self) -> None:
        tm = TradeManager(self.cfg, commission_per_lot=7.0)
        mkt = MarketContext(1.5, 2652.1, 2649.0, 1.0)  # recent low below BE -> trailing not better
        acts = {a.kind: a for a in tm.evaluate(self._pos(), 2652.0, 2652.2, mkt, NOW, XAU)}
        assert acts["tp1"].volume == pytest.approx(0.13)  # 65% of 0.20
        # BE = entry + spread 0.20 + commission 7/100 = 0.07 + 5 pts 0.05 = 2650.32
        assert acts["modify_sl"].new_sl == pytest.approx(2650.32)

    def test_micro_trailing_on_runner_uses_last_candles(self) -> None:
        tm = TradeManager(self.cfg, commission_per_lot=0.0)
        pos = self._pos()
        pos.tp1_done, pos.sl = True, 2650.30
        acts = tm.evaluate(pos, 2654.0, 2654.2, MarketContext(1.5, 2654.5, 2652.8, 1.0), NOW, XAU)
        assert next(a for a in acts if a.kind == "modify_sl").new_sl == pytest.approx(2652.8)

    def test_short_trailing_adds_spread(self) -> None:
        tm = TradeManager(self.cfg, commission_per_lot=0.0)
        pos = self._pos(side=-1)
        pos.tp1_done, pos.sl = True, 2649.70
        acts = tm.evaluate(pos, 2645.8, 2646.0, MarketContext(1.5, 2647.0, 2645.5, -1.0), NOW, XAU)
        assert next(a for a in acts if a.kind == "modify_sl").new_sl == pytest.approx(2647.2)

    def test_stale_trade_killer_only_when_momentum_faded(self) -> None:
        tm = TradeManager(self.cfg)
        late = NOW + timedelta(minutes=26)
        strong = tm.evaluate(self._pos(), 2650.5, 2650.7, MarketContext(1.0, 0, 0, 0.8), late, XAU)
        assert all(a.kind != "close" for a in strong)
        faded = tm.evaluate(self._pos(), 2650.5, 2650.7, MarketContext(1.0, 0, 0, 0.1), late, XAU)
        assert faded and faded[0].kind == "close" and "stale" in faded[0].reason

    def test_min_lot_position_gets_break_even_only(self) -> None:
        """0.01 lots cannot be split: TP1 is flagged with volume 0 and the SL goes to BE."""
        tm = TradeManager(self.cfg)
        acts = {a.kind: a for a in tm.evaluate(self._pos(vol=0.01), 2652.0, 2652.2,
                                               MarketContext(1.0, 0, 0, 1), NOW, XAU)}  # fmt: skip
        assert acts["tp1"].volume == 0.0
        assert acts["modify_sl"].new_sl > 2650.0

    def test_sl_never_loosens(self) -> None:
        tm = TradeManager(self.cfg)
        pos = self._pos()
        pos.tp1_done, pos.sl = True, 2653.0
        acts = tm.evaluate(pos, 2654.0, 2654.2, MarketContext(1.0, 2654.5, 2652.0, 1), NOW, XAU)
        assert all(a.kind != "modify_sl" for a in acts)


# ------------------------------------------------------------ strategy bits
def test_avoid_round_levels() -> None:
    assert avoid_round_levels(2650.05, 1, 0.5, 0.15) == pytest.approx(2649.85)
    assert avoid_round_levels(2649.92, -1, 0.5, 0.15) == pytest.approx(2650.15)
    assert avoid_round_levels(2650.25, 1, 0.5, 0.15) == pytest.approx(2650.25)


def test_align_htf_has_no_lookahead() -> None:
    m1 = pd.DataFrame({"time": pd.date_range("2026-11-10 10:00", periods=30, freq="1min")})
    m15 = pd.DataFrame({"time": pd.to_datetime(["2026-11-10 09:45", "2026-11-10 10:00", "2026-11-10 10:15"]),
                        "v": [1, 2, 3]})  # fmt: skip
    out = ind.align_htf(m1, "M1", m15, "M15", ["v"], "h_").set_index("time")["h_v"]
    assert out[pd.Timestamp("2026-11-10 10:13")] == 1
    assert out[pd.Timestamp("2026-11-10 10:14")] == 2


def _frames(strat, df: pd.DataFrame) -> dict[str, pd.DataFrame]:  # noqa: ANN001
    return {tf: df if tf == "M1" else ind.resample_ohlc(df, tf) for tf in strat.required_timeframes}


def test_strategy_features_are_causal() -> None:
    """Signals (and key levels) on a truncated history must equal those on the full history."""
    from src.backtest.engine import synthetic_m1

    s = load_settings()
    strat = create_strategy(s.strategy, s.levels, UTC_CLOCK)
    m1 = synthetic_m1(8_000, seed=3)
    full = strat.compute_features(_frames(strat, m1)).set_index("time")
    part = strat.compute_features(_frames(strat, m1.iloc[:5_000])).set_index("time")
    common = part.index[:-20]  # drop the trailing partial HTF buckets
    for col in ("signal", "key_level_bull", "key_level_bear", "lv_pdh", "lv_asia_low"):
        a, b = full.loc[common, col], part.loc[common, col]
        assert a.fillna(-1).equals(b.fillna(-1)), col


def test_build_signal_limit_pullback_and_structural_sl() -> None:
    s = load_settings()
    s.strategy.entry_mode, s.strategy.pullback_pct, s.strategy.limit_anchor = (
        "limit",
        0.4,
        "retracement",
    )
    strat = create_strategy(s.strategy, LevelsConfig(enabled=True, filter_mode="priority",
                                                     non_key_risk_mult=0.5), UTC_CLOCK)  # fmt: skip
    row = pd.Series({"signal": 1, "atr": 1.0, "open": 2650.0, "close": 2652.0, "struct_low": 2649.0,
                     "struct_high": 2652.5, "fvg_bull_top": float("nan"), "key_level_bull": "",
                     "time": pd.Timestamp("2026-11-10 10:00"), "body_atr": 2.0, "vol_ratio": 2.0,
                     "trigger": "sweep"})  # fmt: skip
    sig = strat.build_signal(row, s.symbols["XAUUSD"], XAU, spread_price=0.20)
    assert sig is not None and sig.order_type == "limit"
    assert sig.entry == pytest.approx(2652.0 - 0.4 * 2.0 + 0.20)  # 40% of body + spread (ASK)
    assert sig.stop_loss < 2649.0 - 0.3 * 1.0 + 1e-9  # beyond sweep low + ATR buffer
    assert sig.expiry_s == 3 * 60
    assert sig.risk_mult == 0.5  # not on a key level -> reduced risk
    row["key_level_bull"] = "PDL"
    assert strat.build_signal(row, s.symbols["XAUUSD"], XAU, 0.20).risk_mult == 1.0


def test_dashboard_renders() -> None:
    from rich.console import Console

    from src.ui.dashboard import build_layout, demo_state

    con = Console(record=True, width=180, height=50)
    con.print(build_layout(demo_state()))
    text = con.export_text()
    for frag in (
        "XAU QUANT DESK",
        "Market Microstructure",
        "Live Execution Log",
        "Kill-switch",
        "London",
    ):
        assert frag in text

"""Strategy research lab: search -> walk-forward -> anti-luck filters -> meta-labeling -> holdout.

Pipeline
--------
1. **Search** every family x timeframe x parameter grid on every symbol (pooled across
   symbols: a strategy must work as a rule, not as a per-symbol curve fit).
2. **Holdout ("vault")**: the last ``holdout_months`` are never used to choose anything.
3. **Walk-forward**: at each step the best ``wf_top_k`` configurations are picked using only
   past data and traded on the next window. The concatenated result is the honest estimate
   of what *the selection process itself* earns.
4. **Anti-luck filters** on pre-holdout data: minimum trades, positive expectancy, Deflated
   Sharpe (accounts for how many configurations were tried), parameter-neighbourhood
   robustness, cost stress test (spread x1.5, slippage x2).
5. **Meta-labeling** (ML signal filter) for each survivor, validated out-of-sample.
6. **Holdout** evaluation of the survivors' portfolio - once. The journal records it so a
   reused vault is flagged.
7. Survivors that also pass the holdout become ``config/profiles/lab_*.yaml`` profiles.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from src.backtest.engine import BacktestEngine, BTTrade
from src.config import PROFILES_DIR, Settings, SymbolSpec, resolve_path
from src.core.clock import ServerClock
from src.core.data_feed import TF_MINUTES
from src.core.news import get_calendar
from src.lab.features import FEATURES, features_at, market_features
from src.lab.meta import MetaReport, train_meta
from src.lab.portfolio import (
    PortfolioResult,
    PortfolioRules,
    candidate_settings,
    simulate_portfolio,
)
from src.lab.stats import deflated_sharpe, r_metrics
from src.strategies.library import LIBRARY
from src.strategies.xau_velocity_scalp import create_strategy

log = logging.getLogger(__name__)
REF_RISK = 0.5  # risk % used for candidate generation (R-multiples do not depend on it)

# data: {symbol: {timeframe: (bars, broker spec | None)}}
LabData = dict[str, dict[str, tuple[pd.DataFrame, SymbolSpec | None]]]


@dataclass
class Candidate:
    family: str
    tf: str
    params: dict[str, Any]
    trades: list[BTTrade] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    by_symbol_pos: int = 0
    n_symbols: int = 0
    dsr: float = 0.0
    neighbors_pos: float = float("nan")
    stress_mean_r: float = float("nan")
    status: str = ""
    meta: MetaReport | None = None
    meta_path: str = ""

    @property
    def cid(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}|{self.tf}|{p}"

    @property
    def short_id(self) -> str:
        return hashlib.sha1(self.cid.encode()).hexdigest()[:6]

    def label(self) -> str:
        grid = LIBRARY[self.family].GRID
        p = " ".join(f"{k}={self.params[k]}" for k in grid if k in self.params)
        return f"{self.family} {self.tf} {p}"


class Lab:
    def __init__(self, settings: Settings, data: LabData, clock: ServerClock,
                 console: Console | None = None, families: list[str] | None = None,
                 quick: bool = False) -> None:  # fmt: skip
        self.s = settings
        self.cfg = settings.lab
        self.data = data
        self.clock = clock
        self.con = console or Console()
        self.calendar = get_calendar(settings.news)
        wanted = [f for f in (families or self.cfg.families) if f in LIBRARY]
        self.skipped = [f for f in wanted if LIBRARY[f].REQUIRES_CALENDAR and self.calendar is None]
        self.families = [f for f in wanted if f not in self.skipped]
        self.quick = quick
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = resolve_path(settings.general.log_dir) / "lab" / self.run_id
        self._feat_cache: dict[tuple[str, str], pd.DataFrame] = {}
        last = max(bars["time"].iloc[-1] for tfs in data.values() for bars, _ in tfs.values())
        first = min(bars["time"].iloc[0] for tfs in data.values() for bars, _ in tfs.values())
        self.data_end = pd.Timestamp(last)
        self.data_start = pd.Timestamp(first)
        self.holdout_start = self.data_end - pd.DateOffset(months=self.cfg.holdout_months)

    # ----------------------------------------------------------- configs
    def enumerate(self) -> list[Candidate]:
        out = []
        for fam in self.families:
            cls = LIBRARY[fam]
            keys = list(cls.GRID)
            combos = list(itertools.product(*(cls.GRID[k] for k in keys)))
            if self.quick:
                combos = combos[: max(1, len(combos) // 3)]
            for tf in cls.TIMEFRAMES:
                for combo in combos:
                    params = {**cls.DEFAULTS, **dict(zip(keys, combo, strict=True))}
                    out.append(Candidate(fam, tf, params))
        return out

    def settings_for(self, c: Candidate, stress: bool = False) -> Settings:
        s = candidate_settings(self.s)
        cls = LIBRARY[c.family]
        s.strategy.name = c.family
        s.strategy.timeframes = {"entry": c.tf, "confirm": c.tf, "trend": c.tf}
        s.strategy.params = dict(c.params)
        for k, v in cls.exec_overrides(c.params).items():
            setattr(s.execution, k, v)
        for k, v in cls.SESSIONS.items():
            setattr(s.sessions, k, copy.deepcopy(v))
        for k, v in cls.NEWS.items():
            setattr(s.news, k, v)
        s.backtest.initial_equity = self.cfg.candidate_equity
        s.risk.risk_per_trade_pct = REF_RISK
        s.levels.enabled = False
        if stress:
            s.backtest.spread_mult = self.cfg.stress_spread_mult
            s.backtest.simulated_slippage_points *= self.cfg.stress_slippage_mult
        return s

    def run_candidate(self, c: Candidate, stress: bool = False) -> list[BTTrade]:
        s = self.settings_for(c, stress)
        strat = create_strategy(s.strategy, s.levels, self.clock)
        trades: list[BTTrade] = []
        for sym, tfs in self.data.items():
            if c.tf not in tfs:
                continue
            bars, spec = tfs[c.tf]
            if len(bars) < 300:
                continue
            res = BacktestEngine(s, strat, sym, spec=spec, clock=self.clock).run(bars, base_tf=c.tf)
            trades += res.trades
        return trades

    # ------------------------------------------------------------ helpers
    def _pre(self, trades: list[BTTrade]) -> list[BTTrade]:
        return [t for t in trades if pd.Timestamp(t.entry_time) < self.holdout_start]

    def _hold(self, trades: list[BTTrade]) -> list[BTTrade]:
        return [t for t in trades if pd.Timestamp(t.entry_time) >= self.holdout_start]

    def _days(self, trades: list[BTTrade]) -> float:
        if not trades:
            return 0.0
        a = min(pd.Timestamp(t.entry_time) for t in trades)
        b = max(pd.Timestamp(t.exit_time) for t in trades)
        return max((b - a).days, 1)

    def rules(self) -> PortfolioRules:
        r = PortfolioRules.from_risk(self.s.risk, self.s.backtest.initial_equity)
        r.risk_per_trade_pct = self.cfg.risk_per_trade_pct
        r.max_trades_per_day = max(r.max_trades_per_day, 50)
        return r

    def portfolio(self, trades: list[BTTrade]) -> PortfolioResult:
        return simulate_portfolio(
            sorted(trades, key=lambda t: t.entry_time), self.rules(), REF_RISK
        )

    def features_table(self, trades: list[BTTrade], tf: str) -> pd.DataFrame:
        rows = []
        by_sym: dict[str, list[BTTrade]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        for sym, ts in by_sym.items():
            key = (sym, tf)
            if key not in self._feat_cache:
                bars, spec = self.data[sym][tf]
                point = spec.point if spec else self.s.symbols[sym].backtest_spec().point
                self._feat_cache[key] = market_features(bars, point, self.clock, sym, self.calendar)
            f = features_at(
                self._feat_cache[key], pd.Series([t.signal_time for t in ts]), TF_MINUTES[tf]
            )
            f["side"] = [t.side for t in ts]
            f["r"] = [t.r_multiple for t in ts]
            f["signal_time"] = [pd.Timestamp(t.signal_time) for t in ts]
            f["exit_time"] = [pd.Timestamp(t.exit_time) for t in ts]
            rows.append(f)
        return (
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*FEATURES, "r"])
        )

    # --------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        cands = self.enumerate()
        n_runs = sum(sum(1 for tfs in self.data.values() if c.tf in tfs) for c in cands)
        if self.skipped:
            self.con.print(f"[yellow]Skipped (no economic calendar): {', '.join(self.skipped)} - "
                           "install it with 8_CALENDARIO_NOTIZIE.bat[/]")  # fmt: skip
        elif self.calendar is not None:
            self.con.print(f"Economic calendar: {len(self.calendar.events):,} events up to "
                           f"{self.calendar.last_time:%Y-%m-%d} (news blackout, ML features, news_drift)")  # fmt: skip
        self.con.print(f"[bold]Lab[/] {self.run_id}: {len(cands)} configurations x symbols = {n_runs} backtests. "
                       f"Vault (holdout) from [bold]{self.holdout_start:%Y-%m-%d}[/] — never used for selection.")  # fmt: skip
        with Progress(console=self.con, transient=True) as prog:
            task = prog.add_task("Searching strategies", total=len(cands))
            for c in cands:
                c.trades = self.run_candidate(c)
                prog.advance(task)

        # ---- metrics on pre-holdout data
        for c in cands:
            pre = self._pre(c.trades)
            c.metrics = r_metrics([t.r_multiple for t in pre], self._days(pre))
            per_sym: dict[str, list[float]] = {}
            for t in pre:
                per_sym.setdefault(t.symbol, []).append(t.r_multiple)
            c.n_symbols = len(per_sym)
            c.by_symbol_pos = sum(1 for v in per_sym.values() if np.mean(v) > 0)
        trials = [c for c in cands if c.metrics["n"] >= self.cfg.min_train_trades]
        srs = [c.metrics["sr"] for c in trials]
        sr_var = statistics.pvariance(srs) if len(srs) > 1 else 0.0
        for c in cands:
            pre = self._pre(c.trades)
            c.dsr = deflated_sharpe([t.r_multiple for t in pre], max(len(trials), 1), sr_var)

        wf = self.walk_forward(cands)
        self.apply_filters(cands)
        survivors = [c for c in cands if c.status == "SURVIVOR"]
        survivors.sort(key=lambda c: -c.metrics["t"])
        survivors = survivors[: self.cfg.max_survivors]
        if self.cfg.meta_enabled:
            for c in survivors:
                self.meta_label(c)
        holdout = self.evaluate_holdout(survivors)
        promoted = self.write_profiles(survivors, holdout)
        result = {"cands": cands, "trials": len(trials), "wf": wf, "survivors": survivors,
                  "holdout": holdout, "promoted": promoted}  # fmt: skip
        self.save(result)
        self.report(result)
        return result

    # ------------------------------------------------------- walk-forward
    def walk_forward(self, cands: list[Candidate]) -> dict[str, Any]:
        start = self.data_start + pd.DateOffset(months=self.cfg.wf_min_train_months)
        folds = []
        t0 = start
        while t0 < self.holdout_start:
            t1 = min(t0 + pd.DateOffset(months=self.cfg.wf_test_months), self.holdout_start)
            folds.append((t0, t1))
            t0 = t1
        oos: list[BTTrade] = []
        picks: list[tuple[str, str, str]] = []
        for a, b in folds:
            scored = []
            for c in cands:
                tr = [t.r_multiple for t in c.trades if pd.Timestamp(t.entry_time) < a]
                if len(tr) >= self.cfg.min_train_trades:
                    m = np.mean(tr)
                    scored.append((m * np.sqrt(len(tr)), c))
            scored.sort(key=lambda x: -x[0])
            for score, c in scored[: self.cfg.wf_top_k]:
                if score <= 0:
                    continue  # never pick a strategy that was losing up to that point
                picks.append((f"{a:%Y-%m}", c.label(), f"{score:.1f}"))
                oos += [t for t in c.trades if a <= pd.Timestamp(t.entry_time) < b]
        port = self.portfolio(oos) if oos else None
        return {"folds": len(folds), "trades": oos, "portfolio": port, "picks": picks,
                "metrics": r_metrics([t.r_multiple for t in oos], self._days(oos))}  # fmt: skip

    # ------------------------------------------------------------ filters
    def apply_filters(self, cands: list[Candidate]) -> None:
        cfg = self.cfg
        by_group: dict[tuple[str, str], list[Candidate]] = {}
        for c in cands:
            by_group.setdefault((c.family, c.tf), []).append(c)
        for c in cands:
            m = c.metrics
            if m["n"] < cfg.min_trades:
                c.status = f"few trades ({m['n']})"
            elif m["mean_r"] <= 0:
                c.status = "negative expectancy"
            elif c.dsr < cfg.dsr_min:
                c.status = f"luck? DSR {c.dsr:.2f}"
        for c in cands:
            if c.status:
                continue
            group = by_group[(c.family, c.tf)]
            nb = [
                o
                for o in group
                if o is not c and sum(o.params[k] != c.params[k] for k in c.params) == 1
            ]
            c.neighbors_pos = float(np.mean([o.metrics["mean_r"] > 0 for o in nb])) if nb else 1.0
            if c.neighbors_pos < cfg.neighbor_min_positive:
                c.status = f"fragile params ({c.neighbors_pos:.0%})"
        for c in cands:
            if c.status:
                continue
            stress = self._pre(self.run_candidate(c, stress=True))
            c.stress_mean_r = float(np.mean([t.r_multiple for t in stress])) if stress else 0.0
            c.status = (
                "SURVIVOR" if c.stress_mean_r > 0 else f"dies with costs x{cfg.stress_spread_mult}"
            )

    # -------------------------------------------------------------- meta
    def meta_label(self, c: Candidate) -> None:
        pre = self._pre(c.trades)
        table = self.features_table(pre, c.tf)
        model, rep = train_meta(
            table, min_keep=self.cfg.meta_min_keep, min_trades=self.cfg.meta_min_trades
        )
        c.meta = rep
        if model is not None:
            path = resolve_path(self.cfg.models_dir) / f"lab_{c.family}_{c.tf}_{c.short_id}.joblib"
            model.save(path)
            c.meta_path = str(path.relative_to(resolve_path(".")))
            c._model = model  # type: ignore[attr-defined]

    def _filtered_holdout(self, c: Candidate) -> list[BTTrade]:
        hold = self._hold(c.trades)
        model = getattr(c, "_model", None)
        if model is None or not hold:
            return hold
        table = self.features_table(hold, c.tf)
        keep = model.accept(table.fillna(table.median(numeric_only=True)))
        return [t for t, k in zip(hold, keep, strict=True) if k]

    # ----------------------------------------------------------- holdout
    def evaluate_holdout(self, survivors: list[Candidate]) -> dict[str, Any]:
        journal = resolve_path(self.s.general.log_dir) / "lab" / "journal.jsonl"
        reused = False
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("holdout_start") == str(self.holdout_start.date()) and e.get(
                    "holdout_evaluated"
                ):
                    reused = True
        per = {}
        all_hold: list[BTTrade] = []
        for c in survivors:
            h = self._filtered_holdout(c)
            per[c.cid] = r_metrics([t.r_multiple for t in h], self._days(h))
            all_hold += h
        port = self.portfolio(all_hold) if all_hold else None
        return {"per": per, "portfolio": port, "reused": reused,
                "metrics": r_metrics([t.r_multiple for t in all_hold], self._days(all_hold))}  # fmt: skip

    # ----------------------------------------------------------- outputs
    def write_profiles(self, survivors: list[Candidate], holdout: dict[str, Any]) -> list[str]:
        for old in PROFILES_DIR.glob("lab_*.yaml"):
            old.unlink()  # generated files: always reflect the latest lab run
        written = []
        for k, c in enumerate(survivors, start=1):
            hm = holdout["per"].get(c.cid, {})
            if hm.get("n", 0) < 5 or hm.get("mean_r", 0) <= 0:
                continue  # failed (or not testable on) the vault
            cls = LIBRARY[c.family]
            prof = {
                "general": {"magic_number": self.s.general.magic_number + 100 + k,
                            "comment_prefix": f"LAB{k}"},
                "strategy": {"name": c.family, "timeframes": {"entry": c.tf, "confirm": c.tf, "trend": c.tf},
                             "params": c.params, "meta_model": c.meta_path,
                             "meta_threshold": c.meta.threshold if c.meta and c.meta.adopted else 0.5},
                "execution": cls.exec_overrides(c.params),
                "sessions": copy.deepcopy(cls.SESSIONS),
                "news": dict(cls.NEWS),
                "risk": {"risk_per_trade_pct": self.cfg.risk_per_trade_pct},
                "levels": {"enabled": False},
                "health": {"enabled": True, "expected_mean_r": round(c.metrics["mean_r"], 4),
                           "expected_sd_r": round(c.metrics["sd_r"], 4), "config_id": c.cid},
            }  # fmt: skip
            name = f"lab_{c.family}_{c.tf}_{c.short_id}"
            header = (f"# Generated by the strategy lab, run {self.run_id}. Do not edit by hand:\n"
                      f"# re-run the lab instead. Pre-holdout: {c.metrics['n']} trades, "
                      f"{c.metrics['mean_r']:+.3f} R/trade. Holdout: {hm.get('n')} trades, "
                      f"{hm.get('mean_r', 0):+.3f} R/trade.\n")  # fmt: skip
            (PROFILES_DIR / f"{name}.yaml").write_text(
                header + yaml.safe_dump(prof, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            written.append(name)
        return written

    def save(self, result: dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in result["cands"]:
            rows.append({"config": c.cid, "status": c.status, **{f"pre_{k}": v for k, v in c.metrics.items()},
                         "dsr": c.dsr, "neighbors_pos": c.neighbors_pos, "stress_mean_r": c.stress_mean_r,
                         "symbols_positive": f"{c.by_symbol_pos}/{c.n_symbols}"})  # fmt: skip
        pd.DataFrame(rows).sort_values("pre_t", ascending=False).to_csv(
            self.out_dir / "configs.csv", index=False
        )
        wf_tr = result["wf"]["trades"]
        if wf_tr:
            pd.DataFrame([t.__dict__ for t in wf_tr]).to_csv(
                self.out_dir / "walk_forward_trades.csv", index=False
            )
        entry = {
            "run_id": self.run_id, "time": datetime.now().isoformat(timespec="seconds"),
            "data_start": str(self.data_start.date()), "data_end": str(self.data_end.date()),
            "holdout_start": str(self.holdout_start.date()), "configs": len(result["cands"]),
            "trials": result["trials"], "survivors": [c.cid for c in result["survivors"]],
            "holdout_evaluated": bool(result["survivors"]), "promoted": result["promoted"],
            "wf_mean_r": result["wf"]["metrics"]["mean_r"], "wf_trades": result["wf"]["metrics"]["n"],
            "holdout_mean_r": result["holdout"]["metrics"]["mean_r"],
        }  # fmt: skip
        journal = self.out_dir.parent / "journal.jsonl"
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------- report
    def report(self, result: dict[str, Any]) -> None:
        con = self.con
        cands: list[Candidate] = result["cands"]
        dt = Table(title="Data", box=box.SIMPLE)
        dt.add_column("Symbol")
        tfs = sorted({tf for tfs in self.data.values() for tf in tfs}, key=lambda x: TF_MINUTES[x])
        for tf in tfs:
            dt.add_column(tf, justify="right")
        for sym, d in self.data.items():
            dt.add_row(sym, *[(f"{len(d[tf][0]):,} ({d[tf][0]['time'].iloc[0]:%Y-%m})" if tf in d else "-")
                              for tf in tfs])  # fmt: skip
        con.print(dt, width=max(con.width, 130))

        top = sorted(cands, key=lambda c: -c.metrics.get("t", 0))[:15]
        tt = Table(title=f"Top 15 of {len(cands)} configurations (pre-holdout, pooled over symbols)",
                   box=box.SIMPLE_HEAVY)  # fmt: skip
        for col in ("Strategy", "Trades", "Tr/wk", "Win%", "PF", "R/trade", "t", "DSR", "Symbols+",
                    "Neighb.", "Stress R", "Status"):  # fmt: skip
            tt.add_column(col, justify="left" if col in ("Strategy", "Status") else "right")
        for c in top:
            m = c.metrics
            color = "green" if c.status == "SURVIVOR" else "red"
            tt.add_row(c.label(), str(m["n"]), f"{m['per_week']:.1f}", f"{m['win_pct']:.0f}",
                       f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "∞", f"{m['mean_r']:+.3f}",
                       f"{m['t']:+.1f}", f"{c.dsr:.2f}", f"{c.by_symbol_pos}/{c.n_symbols}",
                       "-" if np.isnan(c.neighbors_pos) else f"{c.neighbors_pos:.0%}",
                       "-" if np.isnan(c.stress_mean_r) else f"{c.stress_mean_r:+.3f}",
                       f"[{color}]{c.status}[/]")  # fmt: skip
        con.print(tt, width=max(con.width, 150))

        grave: dict[str, int] = {}
        for c in cands:
            key = c.status.split(" (")[0].split(" DSR")[0].split(" x")[0]
            grave[key] = grave.get(key, 0) + 1
        gt = Table(title="Outcome of every configuration", box=box.SIMPLE)
        gt.add_column("Outcome")
        gt.add_column("Configs", justify="right")
        for k, v in sorted(grave.items(), key=lambda kv: -kv[1]):
            gt.add_row(("[green]" if k == "SURVIVOR" else "") + k, str(v))
        con.print(gt)

        wf = result["wf"]
        self._portfolio_table("Walk-forward (the selection process, out-of-sample)", wf["metrics"],
                              wf["portfolio"], extra=f"{wf['folds']} folds")  # fmt: skip

        if result["survivors"]:
            st = Table(title="Survivors and ML filter (meta-labeling)", box=box.SIMPLE_HEAVY)
            for col in ("Strategy", "Pre R/trade", "Trades", "ML filter", "OOS R all → kept", "Kept",
                        "Top features", "Holdout R/trade", "Holdout n"):  # fmt: skip
                st.add_column(col)
            for c in result["survivors"]:
                mr = c.meta
                hm = result["holdout"]["per"].get(c.cid, {})
                ml = "[green]ON[/]" if mr and mr.adopted else f"off ({mr.reason})" if mr else "off"
                st.add_row(c.label(), f"{c.metrics['mean_r']:+.3f}", str(c.metrics["n"]), ml,
                           f"{mr.oos_mean_r_all:+.3f} → {mr.oos_mean_r_kept:+.3f}" if mr and mr.oos_trades else "-",
                           f"{mr.oos_kept}/{mr.oos_trades}" if mr and mr.oos_trades else "-",
                           ", ".join(list(mr.feature_importance)[:3]) if mr and mr.feature_importance else "-",
                           f"{hm.get('mean_r', 0):+.3f}", str(hm.get("n", 0)))  # fmt: skip
            con.print(st, width=max(con.width, 150))
            ho = result["holdout"]
            self._portfolio_table(
                "VAULT (holdout): survivors on never-seen data", ho["metrics"], ho["portfolio"]
            )
            if ho["reused"]:
                con.print("[bold yellow]⚠ This vault period was already evaluated by a previous run: "
                          "these holdout results are no longer independent. Wait for new data "
                          "(or change holdout_months) for a clean test.[/]")  # fmt: skip
        self.verdict(result)

    def _portfolio_table(self, title: str, m: dict[str, float], port: PortfolioResult | None,
                         extra: str = "") -> None:  # fmt: skip
        t = Table(title=title, box=box.SIMPLE)
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Trades", f"{m['n']}" + (f"  ({extra})" if extra else ""))
        if m["n"]:
            t.add_row("R per trade", f"{m['mean_r']:+.3f}  (t = {m['t']:+.1f})")
            t.add_row("Win rate / PF", f"{m['win_pct']:.1f}% / {m['pf']:.2f}")
            t.add_row("Trades per week", f"{m['per_week']:.1f}")
        if port is not None and port.stats.get("trades"):
            ps = port.stats
            t.add_row(f"Account @ {self.cfg.risk_per_trade_pct}% risk", f"{ps['return_pct']:+.1f}%  "
                      f"max DD {ps['max_drawdown_pct']:.1f}%  Sharpe {ps['sharpe']:.2f}")  # fmt: skip
        self.con.print(t)

    def verdict(self, result: dict[str, Any]) -> None:
        wf = result["wf"]["metrics"]
        ho = result["holdout"]["metrics"]
        if result["promoted"]:
            self.con.print(f"[bold green]✓ {len(result['promoted'])} strategies passed every test including the "
                           f"vault. Profiles written: {', '.join(result['promoted'])}.[/]\n"
                           "  Next: DRY-RUN on demo for a few weeks (7_LAB_DRYRUN.bat) and compare with the "
                           "expected R/trade (the health monitor does it automatically).")  # fmt: skip
        elif result["survivors"]:
            self.con.print(f"[bold yellow]Survivors found, but they failed the vault ({ho['mean_r']:+.3f} R/trade): "
                           "likely luck. No profile promoted.[/]")  # fmt: skip
        else:
            self.con.print("[bold red]No configuration passed the anti-luck filters. Nothing is promoted: "
                           "trading these would most likely lose money.[/]")  # fmt: skip
        if wf["n"] and wf["t"] < 1.5:
            self.con.print(f"[dim]Walk-forward of the whole selection process: {wf['mean_r']:+.3f} R/trade "
                           f"(t = {wf['t']:+.1f}) - not significantly positive.[/]")  # fmt: skip


def load_lab_data(settings: Settings, symbols: list[str], store: Any, conn: Any,
                  console: Console, timeframes: list[str]) -> LabData:  # fmt: skip
    """Download (incrementally) or read from cache every timeframe the lab needs."""
    from src.core.data_feed import DataFeed

    data: LabData = {}
    for sym in symbols:
        spec = None
        if conn is not None:
            if not conn.select_symbol(sym):
                console.print(f"[yellow]{sym}: not available, skipped[/]")
                continue
            spec = DataFeed(conn).symbol_spec(sym)
        for tf in timeframes:
            days = settings.lab.history_days.get(tf, 730)
            if conn is not None:
                bars = store.fetch_historical_data(sym, tf, days)
            else:
                bars = store.slice_days(store.load_bars(sym, tf), days)
            if len(bars) < 300:
                continue
            data.setdefault(sym, {})[tf] = (bars.iloc[:-1].reset_index(drop=True), spec)
        if sym in data:
            spans = ", ".join(f"{tf}:{len(b):,}" for tf, (b, _) in data[sym].items())
            console.print(f"{sym}: {spans}")
    return data


def needed_timeframes(families: list[str]) -> list[str]:
    tfs = {tf for f in families if f in LIBRARY for tf in LIBRARY[f].TIMEFRAMES}
    return sorted(tfs, key=lambda x: TF_MINUTES[x])

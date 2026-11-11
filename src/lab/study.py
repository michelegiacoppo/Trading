"""Price-action study: do the textbook candlestick rules work on YOUR data?

For every candlestick pattern, in every context:

* **location** - on a support/resistance zone (built from past swings) or in the middle of nowhere
* **trend** - with the trend, against it, or neutral
* **confirmation** - entering right after the pattern, or only after the next candle confirms

it simulates the same realistic trade (entry at the next open paying the spread, stop beyond
the pattern extreme + 0.2 ATR, target 2R, exit after ``timeout`` bars, stop first when both are
hit in the same candle) and measures the result in R. A **baseline** (same trade on any
candle) shows what "no edge" looks like. Results are split in an early (70%) and a late (30%)
period: a real effect must show up in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from src.config import Settings, resolve_path
from src.strategies import indicators as ind
from src.strategies.candles import LABELS_IT, PATTERNS, detect_all
from src.strategies.sr_levels import sr_zones

TARGET_R = 2.0


@dataclass
class StudyConfig:
    timeframes: tuple[str, ...] = ("M15", "H1", "H4")
    timeout_bars: int = 48
    target_r: float = TARGET_R
    zone_tol_atr: float = 0.5
    min_events: int = 50


def _simulate(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, spread: np.ndarray,
              entry_idx: np.ndarray, side: np.ndarray, stop: np.ndarray, target_r: float,
              timeout: int) -> np.ndarray:  # fmt: skip
    """R result of a bracket trade per event (NaN when it cannot be simulated)."""
    n = len(o)
    out = np.full(len(entry_idx), np.nan)
    for k, (e, sd, sl) in enumerate(zip(entry_idx, side, stop, strict=True)):
        if e >= n - 1:
            continue
        entry = o[e] + (spread[e] if sd > 0 else 0.0)  # longs buy the ask
        risk = sd * (entry - sl)
        if not np.isfinite(risk) or risk <= 0:
            continue
        tp = entry + sd * target_r * risk
        end = min(e + timeout, n)
        hh, ll, sp = h[e:end], lo[e:end], spread[e:end]
        if sd > 0:  # long: exits on the bid
            hit_sl = ll <= sl
            hit_tp = hh >= tp
        else:  # short: exits on the ask
            hit_sl = hh + sp >= sl
            hit_tp = ll + sp <= tp
        i_sl = np.argmax(hit_sl) if hit_sl.any() else 10**9
        i_tp = np.argmax(hit_tp) if hit_tp.any() else 10**9
        if i_sl == 10**9 and i_tp == 10**9:
            last = end - 1
            exit_px = c[last] + (spread[last] if sd < 0 else 0.0)
            out[k] = sd * (exit_px - entry) / risk
        elif i_sl <= i_tp:  # stop first on ties (conservative)
            out[k] = -1.0
        else:
            out[k] = target_r
    return out


def study_frame(bars: pd.DataFrame, point: float, cfg: StudyConfig) -> pd.DataFrame:
    """One row per pattern event with its context and R outcome."""
    df = bars.reset_index(drop=True)
    atr = ind.atr(df, 14)
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    spread = (df["spread"].to_numpy(dtype=float) if "spread" in df else np.zeros(len(df))) * point
    z = sr_zones(df, atr)
    ema50, ema200 = ind.ema(df["close"], 50), ind.ema(df["close"], 200)
    up = ((df["close"] > ema200) & (ema50 > ema50.shift(5))).to_numpy()
    dn = ((df["close"] < ema200) & (ema50 < ema50.shift(5))).to_numpy()
    tol = cfg.zone_tol_atr * atr
    low3, high3 = df["low"].rolling(3).min(), df["high"].rolling(3).max()
    at_sup = ((low3 <= z["support"] + tol) & (df["close"] > z["support"])).to_numpy()
    at_res = ((high3 >= z["resistance"] - tol) & (df["close"] < z["resistance"])).to_numpy()
    pats = detect_all(df, atr)
    atr_v = atr.to_numpy()
    rows = []
    events = {
        f"{name}|{d}": pats[name][0 if d > 0 else 1].to_numpy()
        for name in PATTERNS
        for d in (1, -1)
    }
    rng = np.random.default_rng(0)
    base = rng.random(len(df)) < 0.05  # baseline: ~5% random candles, both directions
    events["baseline|1"] = base
    events["baseline|-1"] = base
    for key, flag in events.items():
        name, d = key.split("|")
        sd = int(d)
        idx = np.flatnonzero(flag & np.isfinite(atr_v))
        idx = idx[idx < len(df) - 3]
        if not len(idx):
            continue
        for confirm in (0, 1):
            if confirm:
                nxt = idx + 1
                ok = (c[nxt] > h[idx]) if sd > 0 else (c[nxt] < lo[idx])
                ev, entry = idx[ok], nxt[ok] + 1
            else:
                ev, entry = idx, idx + 1
            if not len(ev):
                continue
            n_bars = 3 + confirm
            ext_lo = np.array([lo[max(0, e - n_bars + 1) : e + 1].min() for e in entry - 1])
            ext_hi = np.array([h[max(0, e - n_bars + 1) : e + 1].max() for e in entry - 1])
            stop = np.where(
                sd > 0, ext_lo - 0.2 * atr_v[entry - 1], ext_hi + 0.2 * atr_v[entry - 1]
            )
            r = _simulate(o, h, lo, c, spread, entry, np.full(len(entry), sd), stop, cfg.target_r,
                          cfg.timeout_bars)  # fmt: skip
            loc = np.where(at_sup[ev] if sd > 0 else at_res[ev], "S/R", "nessun livello")
            with_t = up[ev] if sd > 0 else dn[ev]
            against = dn[ev] if sd > 0 else up[ev]
            trend = np.where(with_t, "a favore", np.where(against, "contro", "neutro"))
            label = (
                "Candela qualsiasi (baseline)"
                if name == "baseline"
                else LABELS_IT[name][0 if sd > 0 else 1]
            )
            rows.append(pd.DataFrame({
                "time": df["time"].to_numpy()[ev], "pattern": label, "family": name, "side": sd,
                "location": loc, "trend": trend, "confirm": confirm, "r": r,
            }))  # fmt: skip
    return pd.concat(rows, ignore_index=True).dropna(subset=["r"]) if rows else pd.DataFrame()


def summarize(events: pd.DataFrame, min_events: int) -> pd.DataFrame:
    if events.empty:
        return events
    cut = events["time"].quantile(0.7)
    keys = ["pattern", "location", "trend", "confirm"]
    g = events.groupby(keys)
    out = (
        g["r"]
        .agg(n="size", mean_r="mean", sd_r="std", win=lambda s: (s > 0).mean() * 100)
        .reset_index()
    )
    early = events[events["time"] < cut].groupby(keys)["r"].mean().rename("early_r")
    late = events[events["time"] >= cut].groupby(keys)["r"].mean().rename("late_r")
    out = out.join(early, on=keys).join(late, on=keys)
    out["t"] = out["mean_r"] / (out["sd_r"] / np.sqrt(out["n"]))
    return out[out["n"] >= min_events].sort_values("t", ascending=False).reset_index(drop=True)


def run_study(
    settings: Settings, data: dict, console: Console, cfg: StudyConfig | None = None
) -> pd.DataFrame:
    cfg = cfg or StudyConfig()
    all_ev = []
    for sym, tfs in data.items():
        for tf in cfg.timeframes:
            if tf not in tfs:
                continue
            bars, spec = tfs[tf]
            point = spec.point if spec else settings.symbols[sym].backtest_spec().point
            ev = study_frame(bars, point, cfg)
            if len(ev):
                ev["symbol"], ev["tf"] = sym, tf
                all_ev.append(ev)
    if not all_ev:
        console.print("[red]No data for the study.[/]")
        return pd.DataFrame()
    events = pd.concat(all_ev, ignore_index=True)
    out_dir = resolve_path(settings.general.log_dir) / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_tables(events, cfg, console, out_dir)
    return events


def _table(df: pd.DataFrame, title: str) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    for col in ("Pattern", "Posizione", "Trend", "Conferma", "Eventi", "Win%", "R medio", "t",
                "R prima (70%)", "R dopo (30%)"):  # fmt: skip
        t.add_column(col, justify="left" if col in ("Pattern", "Posizione", "Trend") else "right")
    for _, r in df.iterrows():
        col = "green" if r["mean_r"] > 0 else "red"
        both = (r["early_r"] > 0) == (r["late_r"] > 0)
        t.add_row(r["pattern"], r["location"], r["trend"], "sì" if r["confirm"] else "no", f"{int(r['n'])}",
                  f"{r['win']:.0f}", f"[{col}]{r['mean_r']:+.3f}[/]", f"{r['t']:+.1f}",
                  f"{r['early_r']:+.3f}", f"{r['late_r']:+.3f}" + ("" if both else " ⚠"))  # fmt: skip
    return t


def report_tables(events: pd.DataFrame, cfg: StudyConfig, con: Console, out_dir: Path) -> None:
    con.print(f"[bold]Price-action study[/]: {len(events):,} pattern events, symbols "
              f"{events['symbol'].nunique()}, timeframes {', '.join(sorted(events['tf'].unique()))}. "
              f"Trade: stop beyond the pattern, target {cfg.target_r:g}R, spread included.")  # fmt: skip
    for tf in sorted(events["tf"].unique()):
        s = summarize(events[events["tf"] == tf], cfg.min_events)
        if s.empty:
            continue
        s.to_csv(out_dir / f"price_action_study_{tf}.csv", index=False)
        base = s[s["pattern"].str.startswith("Candela qualsiasi")]
        b = f"{base['mean_r'].mean():+.3f}" if len(base) else "n/a"
        con.print(_table(s.head(15), f"{tf} - the 15 best pattern/context combinations (baseline {b} R)"),
                  width=max(con.width, 150))  # fmt: skip
        con.print(_table(s.tail(5), f"{tf} - the 5 worst"), width=max(con.width, 150))
    # textbook rule check: reversal pattern ON a level vs in the middle of nowhere
    rev = events[events["family"].isin([n for n, v in PATTERNS.items() if v[3]])]
    if len(rev):
        t = Table(
            title="Textbook rule: reversal candles work better on support/resistance?",
            box=box.SIMPLE,
        )
        t.add_column("Timeframe")
        t.add_column("On S/R: R (n)", justify="right")
        t.add_column("No level: R (n)", justify="right")
        t.add_column("On S/R + trend + confirm: R (n)", justify="right")
        for tf, g in rev.groupby("tf"):
            sr, no = g[g["location"] == "S/R"]["r"], g[g["location"] != "S/R"]["r"]
            best = g[(g["location"] == "S/R") & (g["trend"] == "a favore") & (g["confirm"] == 1)][
                "r"
            ]
            t.add_row(tf, f"{sr.mean():+.3f} ({len(sr)})", f"{no.mean():+.3f} ({len(no)})",
                      f"{best.mean():+.3f} ({len(best)})" if len(best) else "-")  # fmt: skip
        con.print(t)
    con.print("[dim]A combination is interesting only if R is positive with t > 2 AND positive in both "
              "periods. Many combinations were tested: some will look good by luck - the lab "
              "(price_action family) applies the full anti-luck filters.[/]")  # fmt: skip
    con.print(f"[dim]Full tables: {out_dir}[/]")

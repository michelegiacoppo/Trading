"""Meta-labeling: an ML model that decides WHICH signals of a strategy to take.

The base strategy decides the direction; the meta-model estimates P(trade wins) from the
market context at signal time (:mod:`src.lab.features`) and filters out low-probability
signals. Validation is a purged, expanding walk-forward: a fold is predicted only by a
model trained on trades that were fully closed before the fold started (no leakage), and
its decision threshold is chosen on earlier folds only.

The filter is adopted only if, out-of-sample, it raises the expectancy while keeping a
meaningful share of the trades - otherwise the strategy trades unfiltered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.lab.features import FEATURES

log = logging.getLogger(__name__)
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def _model():  # noqa: ANN202
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          l2_regularization=1.0, min_samples_leaf=40,
                                          random_state=7)  # fmt: skip


def _best_threshold(prob: np.ndarray, r: np.ndarray, min_keep: float) -> float:
    best_t, best_score = 0.5, -np.inf
    for t in THRESHOLDS:
        keep = prob >= t
        if keep.mean() < min_keep or keep.sum() < 20:
            continue
        score = r[keep].mean() * np.sqrt(keep.sum())
        if score > best_score:
            best_t, best_score = t, score
    return best_t


@dataclass
class MetaReport:
    adopted: bool
    reason: str
    oos_trades: int = 0
    oos_kept: int = 0
    oos_mean_r_all: float = 0.0
    oos_mean_r_kept: float = 0.0
    threshold: float = 0.5
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class MetaModel:
    model: object
    threshold: float
    features: list[str]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x[self.features])[:, 1]

    def accept(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(x) >= self.threshold

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "threshold": self.threshold, "features": self.features}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> MetaModel:
        import joblib

        d = joblib.load(path)
        return cls(d["model"], float(d["threshold"]), list(d["features"]))


def train_meta(table: pd.DataFrame, n_folds: int = 5, min_keep: float = 0.4,
               min_trades: int = 200) -> tuple[MetaModel | None, MetaReport]:  # fmt: skip
    """``table``: one row per trade with FEATURES + ``r`` + ``signal_time`` + ``exit_time``."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return None, MetaReport(False, "scikit-learn not installed")
    t = table.dropna(subset=["r"]).sort_values("signal_time").reset_index(drop=True)
    if len(t) < min_trades:
        return None, MetaReport(False, f"only {len(t)} trades (< {min_trades})")
    x, y, r = t[FEATURES], (t["r"] > 0).astype(int).to_numpy(), t["r"].to_numpy()
    edges = np.linspace(0, len(t), n_folds + 1).astype(int)
    oos_prob = np.full(len(t), np.nan)
    kept_mask = np.zeros(len(t), dtype=bool)
    evaluated = np.zeros(len(t), dtype=bool)
    for k in range(1, n_folds):
        a, b = edges[k], edges[k + 1]
        test_start = t["signal_time"].iloc[a]
        train = (
            (t["exit_time"] < test_start).to_numpy().copy()
        )  # purge: trades closed before the fold
        train[a:] = False
        if train.sum() < 50 or len(set(y[train])) < 2:
            continue
        m = _model().fit(x[train], y[train])
        oos_prob[a:b] = m.predict_proba(x.iloc[a:b])[:, 1]
        prior = ~np.isnan(oos_prob[:a])
        thr = (
            _best_threshold(oos_prob[:a][prior], r[:a][prior], min_keep)
            if prior.sum() >= 30
            else 0.5
        )
        kept_mask[a:b] = oos_prob[a:b] >= thr
        evaluated[a:b] = True
    if evaluated.sum() < 30:
        return None, MetaReport(False, "not enough out-of-sample folds")
    r_all, r_kept = r[evaluated], r[evaluated & kept_mask]
    rep = MetaReport(False, "", int(evaluated.sum()), int(len(r_kept)), float(r_all.mean()),
                     float(r_kept.mean()) if len(r_kept) else 0.0)  # fmt: skip
    keep_frac = len(r_kept) / len(r_all)
    if len(r_kept) < 30 or keep_frac < min_keep:
        rep.reason = f"filter keeps too few trades ({keep_frac:.0%})"
        return None, rep
    if rep.oos_mean_r_kept <= rep.oos_mean_r_all + 0.02:
        rep.reason = "no out-of-sample improvement"
        return None, rep
    final = _model().fit(x, y)
    valid = ~np.isnan(oos_prob)
    rep.threshold = _best_threshold(oos_prob[valid], r[valid], min_keep)
    rep.adopted, rep.reason = True, "improves out-of-sample expectancy"
    try:
        from sklearn.inspection import permutation_importance

        imp = permutation_importance(final, x, y, n_repeats=3, random_state=0)
        rep.feature_importance = dict(sorted(zip(FEATURES, imp.importances_mean, strict=True),
                                             key=lambda kv: -kv[1])[:5])  # fmt: skip
    except Exception:  # pragma: no cover - importance is informative only
        pass
    return MetaModel(final, rep.threshold, list(FEATURES)), rep

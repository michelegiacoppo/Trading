"""Statistics for strategy selection that survive data-mining.

* :func:`r_metrics` - per-trade metrics in R
* :func:`probabilistic_sharpe` - PSR (Bailey & Lopez de Prado): probability that the true
  Sharpe exceeds a benchmark, accounting for sample size, skew and fat tails
* :func:`deflated_sharpe` - DSR: PSR against the Sharpe you would expect from the BEST of
  N random trials. The more configurations the lab tests, the higher the bar.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def r_metrics(r: np.ndarray | list[float], days: float | None = None) -> dict[str, float]:
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_r": 0.0, "sd_r": 0.0, "t": 0.0, "win_pct": 0.0, "pf": 0.0,
                "sr": 0.0, "per_week": 0.0, "total_r": 0.0}  # fmt: skip
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    mean = float(r.mean())
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    return {
        "n": n,
        "mean_r": mean,
        "sd_r": sd,
        "t": mean / (sd / math.sqrt(n)) if sd > 0 else 0.0,
        "win_pct": float((r > 0).mean() * 100),
        "pf": float(gains / losses) if losses > 0 else math.inf,
        "sr": mean / sd if sd > 0 else 0.0,
        "per_week": n / days * 7 if days else 0.0,
        "total_r": float(r.sum()),
    }


def _moments(r: np.ndarray) -> tuple[float, float]:
    """Skewness and (non-excess) kurtosis."""
    if len(r) < 3 or r.std() == 0:
        return 0.0, 3.0
    z = (r - r.mean()) / r.std()
    return float((z**3).mean()), float((z**4).mean())


def probabilistic_sharpe(r: np.ndarray | list[float], sr_benchmark: float = 0.0) -> float:
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = _moments(r)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent strategies with zero true skill."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(r: np.ndarray | list[float], n_trials: int, sr_variance: float) -> float:
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sr_variance))

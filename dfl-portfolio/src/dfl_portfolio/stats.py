"""Statistical comparison of two strategies' weekly return series."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def _sharpe(r: np.ndarray) -> float:
    return r.mean() / r.std(ddof=1) * np.sqrt(52)


def sharpe_diff_bootstrap(a: np.ndarray, b: np.ndarray, block: int = 8, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Circular block bootstrap of ``Sharpe(a) - Sharpe(b)`` (paired, same blocks).

    Returns the point estimate, a 95% percentile interval and a two-sided
    p-value from the bootstrap distribution centred at the estimate.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        diffs[k] = _sharpe(a[idx]) - _sharpe(b[idx])
    est = _sharpe(a) - _sharpe(b)
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(est)))
    return {"diff": est, "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)), "p_value": p}


def jobson_korkie_memmel(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Jobson-Korkie test with Memmel's (2003) correction for equal Sharpe ratios (iid returns)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    ma, mb, sa, sb = a.mean(), b.mean(), a.std(ddof=1), b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (2 * sa ** 2 * sb ** 2 - 2 * sa * sb * (sa * sb * rho)
             + 0.5 * ma ** 2 * sb ** 2 + 0.5 * mb ** 2 * sa ** 2
             - ma * mb / (sa * sb) * (sa * sb * rho) ** 2) / T
    z = (sb * ma - sa * mb) / np.sqrt(theta)
    return {"z": float(z), "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}


def paired_mean_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Paired t-test across seeds/folds (e.g. per-seed Sharpe of two methods)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return {"mean_diff": float(d.mean()), "t": np.nan, "p_value": np.nan}
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(2 * (1 - stats.t.cdf(abs(t), len(d) - 1)))}

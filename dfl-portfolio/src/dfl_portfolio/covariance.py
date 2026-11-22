"""Backward-looking covariance estimates (no look-ahead)."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def rolling_ledoit_wolf(returns: np.ndarray, window: int = 104, min_periods: int = 52, jitter: float = 1e-8) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance for every date.

    ``Sigma[t]`` is estimated from ``returns[max(0, t - window + 1): t + 1]``,
    i.e. it only uses information available at the end of week ``t``.  Dates
    with fewer than ``min_periods`` observations get NaN.
    """
    returns = np.asarray(returns, dtype=float)
    T, N = returns.shape
    sigmas = np.full((T, N, N), np.nan)
    for t in range(T):
        start = max(0, t - window + 1)
        block = returns[start: t + 1]
        if block.shape[0] < min_periods:
            continue
        cov = LedoitWolf().fit(block).covariance_
        sigmas[t] = 0.5 * (cov + cov.T) + jitter * np.eye(N)
    return sigmas


def rolling_mean(returns: np.ndarray, window: int = 156, min_periods: int = 52) -> np.ndarray:
    """Trailing sample mean of returns (the classic 'historical mean' forecast)."""
    returns = np.asarray(returns, dtype=float)
    T, N = returns.shape
    out = np.full((T, N), np.nan)
    for t in range(T):
        start = max(0, t - window + 1)
        block = returns[start: t + 1]
        if block.shape[0] >= min_periods:
            out[t] = block.mean(axis=0)
    return out

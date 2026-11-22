"""Backtesting with drift, turnover and proportional transaction costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .layers import PortfolioLayer


@dataclass
class BacktestResult:
    weights: np.ndarray  # (T, N) target weights chosen at each decision date
    gross: np.ndarray  # (T,) gross portfolio return over the following week
    net: np.ndarray  # (T,) net of transaction costs
    turnover: np.ndarray  # (T,) one-way turnover sum_i |w_t - w_drifted|


def simulate(weights: np.ndarray, next_returns: np.ndarray, cost: float = 0.001,
             w_start: Optional[np.ndarray] = None) -> BacktestResult:
    """Hold ``weights[t]`` over week ``t+1``; trade from drifted previous weights.

    ``w_start`` defaults to an all-cash start, so the first decision pays for
    the initial allocation (identical for every strategy).
    """
    weights = np.asarray(weights, dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)
    T, N = weights.shape
    drifted = np.zeros(N) if w_start is None else np.asarray(w_start, dtype=float)
    gross = np.zeros(T)
    turnover = np.zeros(T)
    for t in range(T):
        turnover[t] = np.abs(weights[t] - drifted).sum()
        gross[t] = weights[t] @ next_returns[t]
        grown = weights[t] * (1.0 + next_returns[t])
        drifted = grown / grown.sum() if grown.sum() > 0 else weights[t]
    net = gross - cost * turnover
    return BacktestResult(weights=weights, gross=gross, net=net, turnover=turnover)


def decide_sequential(mu: np.ndarray, sigmas: np.ndarray, next_returns: np.ndarray, layer: PortfolioLayer,
                      w_start: Optional[np.ndarray] = None) -> np.ndarray:
    """Run the layer week by week so the turnover term sees the *drifted* holdings.

    With ``layer.eta == 0`` decisions do not depend on holdings and are solved
    in one batch.
    """
    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    S_t = torch.as_tensor(sigmas, dtype=torch.float64)
    if layer.eta <= 0:
        with torch.no_grad():
            w, _ = layer(mu_t, S_t)
        return w.numpy()
    T, N = mu.shape
    out = np.zeros((T, N))
    held = np.full(N, 1.0 / N) if w_start is None else np.asarray(w_start, dtype=float)
    with torch.no_grad():
        for t in range(T):
            w, _ = layer(mu_t[t: t + 1], S_t[t: t + 1], w_prev=torch.as_tensor(held[None], dtype=torch.float64))
            out[t] = w[0].numpy()
            grown = out[t] * (1.0 + next_returns[t])
            held = grown / grown.sum()
    return out

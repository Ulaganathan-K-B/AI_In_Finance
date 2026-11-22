"""Performance, prediction and decision-quality metrics (weekly data)."""

from __future__ import annotations

from typing import Dict

import numpy as np

WEEKS = 52


def perf_metrics(net: np.ndarray, turnover: np.ndarray | None = None, lam: float | None = None) -> Dict[str, float]:
    """Annualised performance of a weekly net-return series.

    Sharpe uses a 0% risk-free rate and is ``mean / std * sqrt(52)``.
    Max drawdown is measured from an initial wealth of 1.
    """
    r = np.asarray(net, dtype=float)
    wealth = np.cumprod(1.0 + r)
    path = np.concatenate([[1.0], wealth])
    dd = path / np.maximum.accumulate(path) - 1.0
    downside = r[r < 0]
    sd = r.std(ddof=1)
    out = {
        "ann_return": wealth[-1] ** (WEEKS / len(r)) - 1.0,
        "ann_vol": sd * np.sqrt(WEEKS),
        "sharpe": r.mean() / sd * np.sqrt(WEEKS) if sd > 0 else np.nan,
        "sortino": r.mean() / np.sqrt((downside ** 2).sum() / len(r)) * np.sqrt(WEEKS) if len(downside) else np.nan,
        "max_drawdown": dd.min(),
        "cvar95_weekly": -np.sort(r)[: max(1, int(0.05 * len(r)))].mean(),
        "cum_return": wealth[-1] - 1.0,
    }
    if lam is not None:
        out["utility_weekly"] = r.mean() - lam * r.var()
    if turnover is not None:
        out["turnover_ann"] = np.asarray(turnover)[1:].mean() * WEEKS
    return out


def prediction_metrics(mu_hat: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Forecast accuracy and calibration.

    * ``mse`` and ``mae`` over all (date, asset) pairs;
    * ``rank_ic``: mean cross-sectional Spearman correlation per date;
    * ``calib_slope`` / ``calib_intercept``: pooled regression of realised on
      predicted return (perfect calibration = slope 1, intercept 0);
    * ``bias``: mean prediction minus mean realisation.
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    ra = mu_hat.argsort(axis=1).argsort(axis=1).astype(float)
    rb = y.argsort(axis=1).argsort(axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = denom > 0
    ics = (ra * rb).sum(1)[ok] / denom[ok]  # Spearman per date (ties broken arbitrarily)
    x, z = mu_hat.ravel(), y.ravel()
    slope, intercept = (np.polyfit(x, z, 1) if np.std(x) > 0 else (np.nan, np.nan))
    return {
        "mse": float(np.mean((mu_hat - y) ** 2)),
        "mae": float(np.mean(np.abs(mu_hat - y))),
        "rank_ic": float(np.mean(ics)) if len(ics) else np.nan,
        "calib_slope": float(slope),
        "calib_intercept": float(intercept),
        "bias": float(mu_hat.mean() - y.mean()),
        "pred_sd": float(mu_hat.std()),
    }


def expected_utility(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, lam: float) -> np.ndarray:
    """``mu_t' w_t - lam w_t' Sigma_t w_t`` per date (synthetic experiments, truth known)."""
    Sigma = np.broadcast_to(Sigma, (w.shape[0],) + Sigma.shape[-2:])
    return (w * mu).sum(-1) - lam * np.einsum("ti,tij,tj->t", w, Sigma, w)

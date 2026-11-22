r"""Controlled experiments on synthetic markets (truth known).

Because the true expected returns ``mu`` and covariance ``Sigma`` are known, a
decision ``w`` is scored by its *expected* utility
``U(w) = mu'w - lam w'Sigma w`` rather than a noisy realised return.  We report
the normalised regret

.. math::

    NR = \frac{\bar U(w^\star(\mu)) - \bar U(\hat w)}{\bar U(w^\star(\mu)) - \bar U(w_{EW})},

which is 0 for the oracle and 1 for a decision no better than 1/N.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .experiments import train_methods
from .features import FeatureScaler
from .layers import PortfolioLayer
from .metrics import expected_utility, prediction_metrics
from .synthetic import SyntheticMarket, SyntheticSpec
from .train import Split


def _to_split(d: Dict, scaler: FeatureScaler) -> Split:
    return Split(
        X=torch.as_tensor(scaler.transform(d["X"]), dtype=torch.float32),
        Y=torch.as_tensor(d["Y"], dtype=torch.float32),
        Sigma=torch.as_tensor(d["Sigma"], dtype=torch.float64),
        mu_true=torch.as_tensor(d["MU"], dtype=torch.float64),
    )


def _decide(layer: PortfolioLayer, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        w, _ = layer(torch.as_tensor(mu, dtype=torch.float64), torch.as_tensor(Sigma, dtype=torch.float64))
    return w.numpy()


def score(layer: PortfolioLayer, mu_hat: np.ndarray, d: Dict, lam: float) -> Dict[str, float]:
    N = d["MU"].shape[1]
    w_hat = _decide(layer, mu_hat, d["Sigma"])
    w_or = _decide(layer, d["MU"], d["Sigma"])
    w_ew = np.full_like(w_hat, 1.0 / N)
    u_hat = expected_utility(w_hat, d["MU"], d["Sigma"], lam).mean()
    u_or = expected_utility(w_or, d["MU"], d["Sigma"], lam).mean()
    u_ew = expected_utility(w_ew, d["MU"], d["Sigma"], lam).mean()
    pm_mu = prediction_metrics(mu_hat, d["MU"])
    pm_y = prediction_metrics(mu_hat, d["Y"])
    return {
        "norm_regret": (u_or - u_hat) / (u_or - u_ew),
        "regret": u_or - u_hat,
        "exp_utility": u_hat,
        "mse_mu": pm_mu["mse"],
        "mse_y": pm_y["mse"],
        "rank_ic_mu": pm_mu["rank_ic"],
        "calib_slope": pm_y["calib_slope"],
        "bias": pm_y["bias"],
        # against the true conditional mean (no sampling noise from realised returns)
        "calib_slope_mu": pm_mu["calib_slope"],
        "bias_mu": pm_mu["bias"],
        "pred_sd": pm_mu["pred_sd"],
    }


def _job(args):
    torch.set_num_threads(1)
    degree, n_train, seed, cfg = args
    spec = SyntheticSpec(**{**cfg.get("spec", {}), "degree": degree})
    market = SyntheticMarket(spec)
    train = market.sample(n_train, seed=seed)
    val = market.sample(cfg.get("n_val", 200), seed=10_000 + seed)
    test = market.sample(cfg.get("n_test", 2000), seed=20_000 + seed)
    shifted = {name: market.sample(cfg.get("n_test", 2000), seed=30_000 + seed, **kw)
               for name, kw in cfg.get("shifts", {}).items()}
    scaler = FeatureScaler().fit(train["X"])
    lc = cfg["layer"]
    layer = PortfolioLayer(lam=lc["lam"], lower=lc.get("lower", 0.0), upper=lc["upper"])
    trained = train_methods(cfg["methods"], _to_split(train, scaler), _to_split(val, scaler), layer,
                            spec.n_features, spec.n_assets, float(np.std(train["Y"])), seed, cfg.get("train", {}))
    rows, diags = [], []
    for name, res in trained.items():
        model = res["model"].eval()
        for test_name, d in [("test", test)] + list(shifted.items()):
            with torch.no_grad():
                mu_hat = model(_to_split(d, scaler).X).double().numpy()
            rows.append({"degree": degree, "n_train": n_train, "seed": seed, "method": name, "test": test_name,
                         **score(layer, mu_hat, d, lc["lam"])})
        diags.append({"degree": degree, "n_train": n_train, **res["diag"]})
    return rows, diags


def run_synthetic(cfg: Dict, processes: int = 2, verbose: bool = True):
    jobs = [(d, n, s, cfg) for d in cfg["degrees"] for n in cfg["n_train"] for s in range(cfg.get("seeds", 5))]
    t0 = time.time()
    rows, diags = [], []
    with mp.get_context("fork").Pool(processes) as pool:
        for i, (r, dg) in enumerate(pool.imap_unordered(_job, jobs)):
            rows.extend(r)
            diags.extend(dg)
            if verbose and (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {i + 1}/{len(jobs)} jobs ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(diags)

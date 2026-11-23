"""Experiment orchestration: walk-forward backtests on real data.

A *method* is a dict such as::

    {"name": "dfl_mlp", "model": "mlp", "loss": "utility", "init": "pf_mlp"}

Methods in one job are trained in the listed order so a decision-focused model
can be warm-started from its prediction-focused twin (``init``).  Everything
else - features, architecture, covariance, portfolio layer and backtest - is
shared, so differences come from the training objective alone.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .backtest import decide_sequential, simulate
from .covariance import rolling_ledoit_wolf, rolling_mean
from .data import load_weekly_returns
from .features import FEATURE_NAMES, FeatureScaler, build_features
from .layers import PortfolioLayer
from .metrics import perf_metrics, prediction_metrics
from .models import build_model
from .splits import walk_forward_folds
from .stats import jobson_korkie_memmel, paired_mean_test, sharpe_diff_bootstrap
from .train import Split, TrainConfig, fit_decision, fit_mse

MKT_VOL = FEATURE_NAMES.index("mkt_vol_13w")
TRAIN_KEYS = set(TrainConfig.__dataclass_fields__)


# --------------------------------------------------------------------------
# Data panel
# --------------------------------------------------------------------------
def prepare_panel(universe: str, cov_window: int = 104, hist_window: int = 156, data_dir=None,
                  returns: Optional[pd.DataFrame] = None) -> Dict:
    returns = load_weekly_returns(universe, data_dir) if returns is None else returns
    X, Y, valid, dates = build_features(returns)
    Sigma = rolling_ledoit_wolf(returns.to_numpy(), window=cov_window)
    hist = rolling_mean(returns.to_numpy(), window=hist_window)
    valid = valid & np.isfinite(Sigma).all(axis=(1, 2)) & np.isfinite(hist).all(axis=1)
    return {
        "universe": universe,
        "assets": list(returns.columns),
        "returns": returns,
        "dates": dates,
        "X": X,
        "Y": Y,
        "valid": valid,
        "Sigma": Sigma,
        "hist_mean": hist,
        "mkt_vol": X[:, 0, MKT_VOL],
    }


def _split(panel, idx, scaler, regime_threshold=None) -> Split:
    groups = None
    if regime_threshold is not None:
        groups = torch.as_tensor((panel["mkt_vol"][idx] > regime_threshold).astype(np.int64))
    return Split(
        X=torch.as_tensor(scaler.transform(panel["X"][idx]), dtype=torch.float32),
        Y=torch.as_tensor(panel["Y"][idx], dtype=torch.float32),
        Sigma=torch.as_tensor(panel["Sigma"][idx], dtype=torch.float64),
        groups=groups,
    )


# --------------------------------------------------------------------------
# Training a list of methods on one fold
# --------------------------------------------------------------------------
def make_layer(cfg: Dict) -> PortfolioLayer:
    lc = cfg["layer"]
    return PortfolioLayer(lam=lc["lam"], lower=lc.get("lower", 0.0), upper=lc["upper"], eta=lc.get("eta", 0.0))


def train_methods(methods: List[Dict], train: Split, val: Split, layer: PortfolioLayer, n_features: int,
                  n_assets: int, output_scale: float, seed: int, base: Dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for k, m in enumerate(methods):
        torch.manual_seed(seed * 1000 + k)
        mk = dict(m.get("model_kwargs", {}))
        scale = float(mk.pop("output_scale", output_scale))  # override only to mimic an unscaled network
        model = build_model(m["model"], n_features, n_assets, scale, **mk)
        if m.get("init"):
            model.load_state_dict(copy.deepcopy(out[m["init"]]["model"].state_dict()))
        params = dict(base)
        params.update({key: v for key, v in m.items() if key in TRAIN_KEYS})
        params.setdefault("lam", layer.lam)
        params["seed"] = seed * 1000 + k
        cfg = TrainConfig(**params)
        t0 = time.time()
        if cfg.loss == "mse":
            hist = fit_mse(model, train, val, cfg)
        else:
            hist = fit_decision(model, train, val, layer, cfg)
        out[m["name"]] = {
            "model": model,
            "diag": {
                "method": m["name"],
                "seed": seed,
                "epochs_run": len(hist.train_loss),
                "best_epoch": hist.best_epoch,
                "seconds": time.time() - t0,
                "zero_grad_first": hist.frac_zero_grad[0] if hist.frac_zero_grad else np.nan,
                "zero_grad_best": hist.frac_zero_grad[hist.best_epoch] if hist.frac_zero_grad else np.nan,
                "mean_free_first": hist.mean_free[0] if hist.mean_free else np.nan,
                "grad_norm_first": hist.grad_norm[0] if hist.grad_norm else np.nan,
                "train_loss_first": hist.train_loss[0] if hist.train_loss else np.nan,
                "train_loss_best": hist.train_loss[hist.best_epoch] if hist.train_loss else np.nan,
            },
            "history": hist.as_dict(),
        }
    return out


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
_PANEL: Dict = {}


def mp_context():
    """Multiprocessing start method that exists on every OS.

    Linux keeps the cheap ``fork``.  Windows has no ``fork`` and macOS's is unsafe with PyTorch, so they use
    ``spawn``.  Workers receive the data panel through ``_init_worker`` so both methods behave identically.
    """
    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def _init_worker(panel: Dict) -> None:
    global _PANEL
    _PANEL = panel


def _job(args):
    torch.set_num_threads(1)
    fold, seed, cfg = args
    panel = _PANEL
    scaler = FeatureScaler().fit(panel["X"][fold.train])
    threshold = float(np.median(panel["mkt_vol"][fold.train]))
    train = _split(panel, fold.train, scaler, threshold)
    val = _split(panel, fold.val, scaler, threshold)
    test = _split(panel, fold.test, scaler)
    output_scale = float(np.std(panel["Y"][fold.train]))
    layer = make_layer(cfg)
    trained = train_methods(cfg["methods"], train, val, layer, train.X.shape[-1], train.X.shape[1],
                            output_scale, seed, cfg.get("train", {}))
    preds, diags, hists = {}, [], {}
    for name, res in trained.items():
        res["model"].eval()
        with torch.no_grad():
            preds[name] = res["model"](test.X).double().numpy()
        d = dict(res["diag"])
        d["fold"] = fold.name
        diags.append(d)
        hists[name] = res["history"]
    return fold.name, seed, preds, diags, hists


def run_walkforward(cfg: Dict, out_dir: Path, processes: int = 2, panel: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """Train every method on every fold and seed, then backtest the stitched test period."""
    global _PANEL
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = panel or prepare_panel(cfg["universe"], cfg.get("cov_window", 104), cfg.get("hist_window", 156))
    _PANEL = panel
    wf = cfg["walkforward"]
    folds = walk_forward_folds(panel["dates"], panel["valid"], wf["first_test_year"], wf.get("last_test_year"),
                               val_weeks=wf.get("val_weeks", 52), embargo=wf.get("embargo", 1),
                               min_train=wf.get("min_train", 156), step_years=wf.get("step_years", 1))
    if cfg.get("frozen"):
        # distribution-shift test: train once, never refit
        f0 = folds[0]
        test_all = np.concatenate([f.test for f in folds])
        folds = [type(f0)(name=f"{f0.name}-frozen", train=f0.train, val=f0.val, test=test_all)]
    seeds = list(range(cfg.get("seeds", 3)))
    jobs = [(f, s, cfg) for f in folds for s in seeds]
    if verbose:
        for f in folds:
            print(f.summary(panel["dates"]))
        print(f"{len(jobs)} jobs on {processes} processes")
    t0 = time.time()
    if processes > 1:
        with mp_context().Pool(processes, initializer=_init_worker, initargs=(panel,)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_job, jobs)):
                results.append(r)
                if verbose and (i + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"  {i + 1}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)", flush=True)
    else:
        results = [_job(j) for j in jobs]
    if verbose:
        print(f"training finished in {time.time() - t0:.0f}s")

    test_idx = np.concatenate([f.test for f in folds])
    order = {f.name: i for i, f in enumerate(folds)}
    method_names = [m["name"] for m in cfg["methods"]]
    N = len(panel["assets"])
    preds = {m: np.zeros((len(seeds), len(test_idx), N)) for m in method_names}
    offsets = np.cumsum([0] + [len(f.test) for f in folds])
    diags, histories = [], {}
    for fold_name, seed, p, d, h in results:
        i = order[fold_name]
        for m in method_names:
            preds[m][seed, offsets[i]: offsets[i + 1]] = p[m]
        diags.extend(d)
        histories[f"{fold_name}|{seed}"] = h

    evaluation = evaluate(panel, cfg, test_idx, preds, seeds, out_dir)
    pd.DataFrame(diags).to_csv(out_dir / "train_diagnostics.csv", index=False)
    with open(out_dir / "histories.json", "w") as fh:
        json.dump(histories, fh)
    np.savez_compressed(out_dir / "predictions.npz", test_idx=test_idx, **preds)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    return evaluation


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _regimes(panel, test_idx) -> np.ndarray:
    """'turbulent' if trailing 13w market vol exceeds the 75th percentile of all earlier weeks."""
    vol = panel["mkt_vol"]
    labels = []
    for t in test_idx:
        past = vol[: t + 1]
        past = past[np.isfinite(past)]
        labels.append("turbulent" if vol[t] > np.quantile(past, 0.75) else "calm")
    return np.array(labels)


def evaluate(panel: Dict, cfg: Dict, test_idx: np.ndarray, preds: Dict[str, np.ndarray], seeds: List[int],
             out_dir: Path) -> Dict:
    layer = make_layer(cfg)
    cost = cfg.get("backtest_cost", 0.001)
    lam = cfg["layer"]["lam"]
    Y = panel["Y"][test_idx]
    S = panel["Sigma"][test_idx]
    dates = panel["dates"][test_idx]
    N = Y.shape[1]
    regimes = _regimes(panel, test_idx)

    series = {}  # (method, seed) -> BacktestResult
    pred_rows = []
    # baselines (deterministic)
    series[("equal_weight", 0)] = simulate(np.full((len(test_idx), N), 1.0 / N), Y, cost)
    series[("min_variance", 0)] = simulate(decide_sequential(np.zeros_like(Y), S, Y, layer), Y, cost)
    hist_mu = panel["hist_mean"][test_idx]
    series[("mv_hist_mean", 0)] = simulate(decide_sequential(hist_mu, S, Y, layer), Y, cost)
    pred_rows.append({"method": "mv_hist_mean", "seed": 0, **prediction_metrics(hist_mu, Y)})
    for m, P in preds.items():
        for s in seeds:
            w = decide_sequential(P[s], S, Y, layer)
            series[(m, s)] = simulate(w, Y, cost)
            pred_rows.append({"method": m, "seed": s, **prediction_metrics(P[s], Y)})

    rows, weekly = [], []
    for (m, s), res in series.items():
        rows.append({"method": m, "seed": s, **perf_metrics(res.net, res.turnover, lam=lam),
                     "gross_sharpe": perf_metrics(res.gross)["sharpe"],
                     "avg_max_weight": float(res.weights.max(axis=1).mean()),
                     "avg_n_holdings": float((res.weights > 1e-4).sum(axis=1).mean())})
        for regime in ["calm", "turbulent"]:
            mask = regimes == regime
            r = res.net[mask]
            rows[-1][f"{regime}_mean_ann"] = r.mean() * 52
            rows[-1][f"{regime}_utility"] = r.mean() - lam * r.var()
            rows[-1][f"{regime}_sharpe"] = r.mean() / r.std(ddof=1) * np.sqrt(52)
        weekly.append(pd.DataFrame({"date": dates, "method": m, "seed": s, "gross": res.gross, "net": res.net,
                                    "turnover": res.turnover, "regime": regimes}))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    pd.concat(weekly).to_csv(out_dir / "weekly_returns.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "prediction_metrics_per_seed.csv", index=False)

    summary = per_seed.drop(columns="seed").groupby("method", sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    pred_summary = pred_df.drop(columns="seed").groupby("method", sort=False).agg(["mean", "std"])
    pred_summary.columns = [f"{a}_{b}" for a, b in pred_summary.columns]
    summary = summary.join(pred_summary, how="left")
    summary.to_csv(out_dir / "summary.csv")

    # yearly net returns (seed-averaged)
    wk = pd.concat(weekly)
    wk["year"] = pd.to_datetime(wk["date"]).dt.year
    yearly = (wk.groupby(["method", "seed", "year"])["net"].apply(lambda r: np.prod(1 + r) - 1)
              .groupby(["method", "year"]).mean().unstack("year"))
    yearly.to_csv(out_dir / "yearly_returns.csv")

    # significance: each decision-focused method vs its prediction-focused twin
    tests = []
    for m in cfg["methods"]:
        ref = m.get("compare_to") or m.get("init")
        if not ref:
            continue
        a = per_seed[per_seed.method == m["name"]].sort_values("seed")
        b = per_seed[per_seed.method == ref].sort_values("seed")
        ra = np.mean([series[(m["name"], s)].net for s in seeds], axis=0)
        rb = np.mean([series[(ref, s)].net for s in seeds], axis=0)
        boot = sharpe_diff_bootstrap(ra, rb)
        jk = jobson_korkie_memmel(ra, rb)
        pt = paired_mean_test(a["sharpe"].to_numpy(), b["sharpe"].to_numpy())
        pu = paired_mean_test(a["utility_weekly"].to_numpy(), b["utility_weekly"].to_numpy())
        tests.append({"method": m["name"], "vs": ref,
                      "sharpe_diff_seedavg": boot["diff"], "boot_ci_low": boot["ci_low"],
                      "boot_ci_high": boot["ci_high"], "boot_p": boot["p_value"], "jkm_p": jk["p_value"],
                      "seed_sharpe_diff": pt["mean_diff"], "seed_sharpe_p": pt["p_value"],
                      "seed_utility_diff": pu["mean_diff"], "seed_utility_p": pu["p_value"]})
    pd.DataFrame(tests).to_csv(out_dir / "tests.csv", index=False)
    return {"summary": summary, "per_seed": per_seed, "tests": pd.DataFrame(tests), "yearly": yearly}

"""Splits, features, backtest accounting, metrics and a tiny end-to-end training run."""

import numpy as np
import pandas as pd
import torch

from dfl_portfolio.backtest import simulate
from dfl_portfolio.features import build_features
from dfl_portfolio.layers import PortfolioLayer
from dfl_portfolio.metrics import perf_metrics, prediction_metrics
from dfl_portfolio.models import build_model
from dfl_portfolio.splits import check_no_leakage, walk_forward_folds
from dfl_portfolio.stats import sharpe_diff_bootstrap
from dfl_portfolio.synthetic import SyntheticMarket, SyntheticSpec
from dfl_portfolio.train import Split, TrainConfig, fit_decision, fit_mse


def fake_returns(T=400, N=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-07", periods=T, freq="W-FRI")
    return pd.DataFrame(rng.normal(0.002, 0.03, (T, N)), index=dates, columns=list("ABCDE")[:N])


def test_features_use_only_past_information():
    r = fake_returns()
    X, Y, valid, dates = build_features(r)
    t = 200
    r2 = r.copy()
    r2.iloc[t + 1:] = 0.123  # change the future
    X2, _, _, _ = build_features(r2)
    assert np.allclose(X[: t + 1], X2[: t + 1], equal_nan=True)
    assert np.allclose(Y[t], r.iloc[t + 1].to_numpy())
    assert not valid[-1]  # no label for the last week


def test_walk_forward_has_no_overlap_or_leakage():
    r = fake_returns(T=600)
    X, Y, valid, dates = build_features(r)
    folds = walk_forward_folds(dates, valid, first_test_year=2005, val_weeks=52, embargo=1, min_train=100)
    assert len(folds) >= 3
    for f in folds:
        assert check_no_leakage(f)
        assert len(set(f.train) & set(f.test)) == 0
    for a, b in zip(folds[:-1], folds[1:]):
        assert a.test.max() < b.test.min()


def test_backtest_equal_weight_identity_and_costs():
    r = fake_returns(T=50).to_numpy()
    w = np.full_like(r, 1 / r.shape[1])
    res = simulate(w, r, cost=0.001)
    assert np.allclose(res.gross, r.mean(axis=1))
    assert np.isclose(res.turnover[0], 1.0)  # initial allocation from cash
    assert np.all(res.net <= res.gross + 1e-15)
    res0 = simulate(w, r, cost=0.0)
    assert np.allclose(res0.net, res0.gross)


def test_backtest_turnover_counts_drift():
    r = np.array([[0.10, -0.10], [0.0, 0.0]])
    w = np.array([[0.5, 0.5], [0.5, 0.5]])
    res = simulate(w, r, cost=0.0)
    # after week 1 holdings drift to 0.55/0.45 (normalised); rebalancing back costs 0.1 of turnover
    assert np.isclose(res.turnover[1], 0.1)


def test_metrics_basic():
    net = np.array([0.01, -0.02, 0.03, 0.0])
    m = perf_metrics(net, turnover=np.ones(4), lam=5.0)
    assert np.isclose(m["max_drawdown"], -0.02)
    assert np.isclose(m["cum_return"], np.prod(1 + net) - 1)
    p = prediction_metrics(np.array([[1.0, 2.0, 3.0]]), np.array([[0.1, 0.2, 0.3]]))
    assert np.isclose(p["rank_ic"], 1.0)


def test_bootstrap_identical_series():
    r = np.random.default_rng(0).normal(0.001, 0.02, 300)
    out = sharpe_diff_bootstrap(r, r, n_boot=200)
    assert out["diff"] == 0.0 and out["ci_low"] == 0.0 and out["ci_high"] == 0.0


def test_end_to_end_training_runs_and_improves_decision_loss():
    torch.manual_seed(0)
    market = SyntheticMarket(SyntheticSpec(degree=4, signal_sd=0.005))
    tr, va = market.sample(200, 0), market.sample(100, 1)

    def split(d):
        return Split(X=torch.tensor(d["X"], dtype=torch.float32), Y=torch.tensor(d["Y"], dtype=torch.float32),
                     Sigma=torch.tensor(d["Sigma"]))

    layer = PortfolioLayer(10.0, 0.0, 0.4)
    model = build_model("linear", 5, 10, float(np.std(tr["Y"])))
    fit_mse(model, split(tr), split(va), TrainConfig(epochs=30, patience=30, min_epochs=0))
    hist = fit_decision(model, split(tr), split(va), layer,
                        TrainConfig(loss="utility", lam=10.0, cost=0.0, epochs=40, patience=40, early_stop=False))
    assert hist.train_loss[-1] < hist.train_loss[0]
    assert hist.frac_zero_grad[0] < 0.5

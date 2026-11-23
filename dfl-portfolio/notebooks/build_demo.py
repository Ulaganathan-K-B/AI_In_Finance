"""Build notebooks/demo.ipynb (a 1-minute guided tour of the package).

    python notebooks/build_demo.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Decision-focused portfolio learning: guided tour

This notebook walks through the three pieces that make the project work, each on a
problem small enough to run in about a minute on a laptop CPU:

1. **The differentiable portfolio layer**: solve the constrained mean-variance problem and check its
   gradient.
2. **Why the preliminary notebook's DFL model never trained**: the zero-gradient (vertex) problem, and the fix.
3. **Prediction-focused vs decision-focused training** on a synthetic market where the true expected returns
   are known, so regret can be measured exactly.

The full experiments are run by `scripts/reproduce_all.sh`. The last cell shows their headline tables if they
are present in `results/`."""),
    code("""import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, pandas as pd
from dfl_portfolio import PortfolioLayer, mv_closed_form, build_model, TrainConfig, fit_mse, fit_decision
from dfl_portfolio.synthetic import SyntheticMarket, SyntheticSpec
from dfl_portfolio.synthetic_experiments import _to_split, score
from dfl_portfolio.features import FeatureScaler
torch.set_num_threads(2)
pd.set_option("display.precision", 4)"""),
    md("""## 1. The portfolio layer

$w^\\star(\\hat\\mu) = \\arg\\max_w\\ \\hat\\mu^\\top w - \\lambda\\, w^\\top\\Sigma w - \\eta\\lVert w - w_{t-1}\\rVert_1$
subject to $\\mathbf 1^\\top w = 1$ and $0 \\le w_i \\le u$.

The forward pass is a batched ADMM solver polished to the exact active set; the backward pass differentiates
the KKT conditions of that active set (implicit function theorem). The tests in `tests/test_layers.py` check
both against CVXPY and finite differences; here is the quick version against the closed-form solution that
exists when only the budget constraint binds."""),
    code("""torch.manual_seed(0)
n = 5
A = torch.randn(n, n, dtype=torch.float64)
Sigma = (A @ A.T / n + 0.5 * torch.eye(n, dtype=torch.float64)) * 1e-3
mu = (0.002 + 0.001 * torch.randn(n, dtype=torch.float64)).requires_grad_()
layer = PortfolioLayer(lam=5.0, lower=-10.0, upper=10.0)  # loose box -> only the budget binds
w, info = layer(mu[None], Sigma[None])
w_cf = mv_closed_form(mu[None], Sigma, 5.0)
print("max |w - closed form|      :", (w - w_cf).abs().max().item())
g_layer, = torch.autograd.grad(w[0, 0], mu)
g_cf, = torch.autograd.grad(mv_closed_form(mu[None], Sigma, 5.0)[0, 0], mu)
print("max |dw0/dmu - closed form|:", (g_layer - g_cf).abs().max().item())"""),
    md("""## 2. The zero-gradient problem in the preliminary notebook

With long-only weights and a 30% cap, the optimum sits at a *vertex* of the feasible set whenever the predicted
returns are large relative to $\\lambda\\Sigma$. At a vertex every weight is pinned to a bound, so small changes in
$\\hat\\mu$ do not move $w^\\star$ and the gradient is exactly zero. The notebook had $\\lambda = 1$, weekly
covariances of order $10^{-3}$ and an un-scaled network whose outputs were of order 0.1, which is far past that
point, so its loss stayed at the same value for 40 epochs.

Below: the share of dates with at most one free weight (zero gradient) as the scale of the predictions grows."""),
    code("""rng = np.random.default_rng(0)
N = 10
B = 500
F = rng.standard_normal((N, 3)) * 0.02
Sig = torch.as_tensor(F @ F.T + np.diag(np.full(N, 0.03 ** 2)))[None].expand(B, N, N).contiguous()
rows = []
for lam in [1.0, 5.0]:
    lay = PortfolioLayer(lam=lam, lower=0.0, upper=0.3)
    for scale in [0.001, 0.003, 0.01, 0.03, 0.1]:
        mu_hat = torch.as_tensor(rng.standard_normal((B, N)) * scale)
        with torch.no_grad():
            _, info = lay(mu_hat, Sig)
        rows.append({"lambda": lam, "prediction sd": scale, "zero-gradient share": info.frac_zero_grad,
                     "mean free weights": info.n_free.double().mean().item()})
pd.DataFrame(rows).pivot(index="prediction sd", columns="lambda", values="zero-gradient share")"""),
    md("""The fix used throughout the project: scale the network output by the standard deviation of the training
targets (so predictions start at the right order of magnitude), initialise the output layer near zero, warm-start
DFL from the MSE-trained model, use a larger $\\lambda$, and train on the mean-variance utility instead of the
mean return alone. `results/megacap10_notebook_replication` runs the notebook's set-up and the fixed one
side by side on the same 10 stocks.

## 3. Prediction-focused vs decision-focused on a synthetic market

The true expected return is a degree-$d$ polynomial of a linear index of the features. With $d=1$ a linear
model is correctly specified; with $d=6$ it is not. We train the same linear model with MSE (PF) and with the
realised mean-variance utility through the layer (DFL), and score both by **normalised regret** on 2,000 fresh
dates (0 = oracle that knows $\\mu$, 1 = no better than 1/N)."""),
    code("""def run(degree, seed=0, n_train=300):
    spec = SyntheticSpec(n_assets=10, n_features=5, degree=degree, signal_sd=0.005)
    mkt = SyntheticMarket(spec)
    tr, va, te = mkt.sample(n_train, seed=seed), mkt.sample(200, seed=10_000 + seed), mkt.sample(2000, seed=20_000 + seed)
    sc = FeatureScaler().fit(tr["X"])
    lay = PortfolioLayer(lam=10.0, lower=0.0, upper=0.4)
    s = float(np.std(tr["Y"]))
    torch.manual_seed(seed)
    pf = build_model("linear", 5, 10, s)
    fit_mse(pf, _to_split(tr, sc), _to_split(va, sc), TrainConfig(loss="mse", epochs=300, patience=50, seed=seed))
    dfl = build_model("linear", 5, 10, s)
    dfl.load_state_dict(pf.state_dict())   # warm start from the MSE solution
    fit_decision(dfl, _to_split(tr, sc), _to_split(va, sc), lay,
                 TrainConfig(loss="utility", lam=10.0, cost=0.0, epochs=200, patience=40, seed=seed))
    out = []
    for name, m in [("PF (MSE)", pf), ("DFL (utility)", dfl)]:
        with torch.no_grad():
            mu_hat = m(_to_split(te, sc).X).double().numpy()
        r = score(lay, mu_hat, te, 10.0)
        out.append({"degree": degree, "method": name, "norm. regret": r["norm_regret"],
                    "MSE vs true mu (x1e6)": r["mse_mu"] * 1e6, "calibration slope": r["calib_slope"]})
    return out

res = pd.DataFrame(run(1) + run(6))
res.set_index(["degree", "method"])"""),
    md("""Typically, at degree 1 PF is as good or better: MSE is the right loss when the model is correct. At degree 6
DFL has lower regret even though its MSE is no better and its forecasts are less calibrated. That is RQ1 and RQ3 in
miniature. `results/synthetic_main` repeats this over 5 degrees, 3 sample sizes and 20 seeds.

## 4. Headline results from the full runs (if present)"""),
    code("""from IPython.display import Markdown, display
tab = ROOT / "results" / "tables"
for name in ["synthetic_pooled_dfl_vs_pf", "sector9_main_summary", "sector9_main_tests"]:
    f = tab / f"{name}.md"
    if f.exists():
        display(Markdown(f"**{name}**\\n\\n" + f.read_text(encoding="utf-8")))
    else:
        print(f"{f} not found: run scripts/reproduce_all.sh first")"""),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("demo.ipynb")
nbf.write(nb, out)
print("wrote", out)

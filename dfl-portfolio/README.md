# Decision-focused learning for portfolio optimisation

AI4403 Project 4. This repository trains a return predictor *through* a differentiable mean-variance portfolio
layer and benchmarks it against the usual two-stage pipeline ("predict, then optimise"). It covers synthetic
markets, three real asset universes, ablations of the layer and robust variants.

```
features x_t ──► predictor f_θ ──► μ̂_t ──► portfolio layer ──► w_t ──► realised utility on r_{t+1}
                                     ▲          argmax μ̂ᵀw − λ wᵀΣ̂w − η‖w − w_{t−1}‖₁        │
two-stage (PF):  θ fitted by MSE(μ̂, r)│          s.t. 1ᵀw = 1, 0 ≤ w ≤ u                       │
decision-focused (DFL): θ fitted by ──┴──────────── gradient of realised utility ◄───────────────┘
```

* **Start with:** [`docs/REPORT.md`](docs/REPORT.md) is the write-up. It covers the method, every
  experiment, the answers to the three research questions, and limitations.
* **About the preliminary notebook:** [`docs/NOTEBOOK_REVIEW.md`](docs/NOTEBOOK_REVIEW.md) lists what was
  wrong with `01_data.ipynb` and where each issue is fixed here.
* **Guided tour:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs in about a minute.

## Results at a glance

**RQ1: lower prediction error does not mean better decisions.** In synthetic markets with known μ (20 seeds, paired):

* PF wins when the predictor is well specified and data are ample (regret +0.010 for DFL, p = 0.005).
* DFL wins under misspecification (−0.011 to −0.014, p < 0.01) with the same or *higher* MSE.
* DFL also wins with little data (−0.04 to −0.08 at n = 100).
* Under misspecification, test MSE ranks the models' decisions much less reliably (Kendall τ about 0.75 → 0.5).
* The gains do not survive a doubling of volatility at test time.

**RQ2: stability.** The failure mode is *vertex decisions*: when forecasts are large relative to λΣ, or caps are
tight, every weight sits on a bound and the gradient is exactly zero. This is why the notebook's DFL loss stayed
flat: re-running its recipe here, 96% of training dates had zero gradient. DFL is stable with:

* λ ≥ 5 (weekly units) and caps of 20–30%;
* output scaling and a warm start from MSE;
* early stopping on the decision loss;
* transaction costs in the *loss* rather than as an ℓ₁ penalty in the layer, which pins weights and starves the
  gradient.

Without refitting through a regime shift, plain DFL degrades more than PF, and adding a CVaR term fixes this.

**RQ3: calibration.** Pure DFL forecasts are biased and poorly calibrated (calibration slope 0.25 vs 0.72 for PF on
sector ETFs). The decision does not depend on the forecast level, so DFL never learns it. A hybrid loss with a small
MSE term restores calibration and keeps DFL's decision quality.

**Real data** (net of 10 bps costs, walk-forward, mean over seeds; `results/tables/real_data_overview.md`):

| Universe | 1/N | PF MLP | DFL MLP | Hybrid | DFL − PF Sharpe, 95% CI | Turnover/yr PF → DFL |
|---|---|---|---|---|---|---|
| FF12 industries 1996–2025 | 0.663 | 0.488 | 0.577 | 0.586 | **+0.09 [+0.01, +0.17]**, p = 0.02 | 24.6 → 13.3 |
| Sector ETFs 2005–2025 | 0.651 | 0.548 | 0.568 | 0.566 | +0.02 [−0.06, +0.12], p = 0.61 | 25.4 → 14.7 |
| 10 mega-caps 2017–2025 | 1.290 | 0.938 | 0.823 | 0.898 | −0.10 [−0.26, +0.05], p = 0.18 | 23.7 → 19.5 |

* DFL beats the two-stage pipeline on the long ex-ante universes, mainly by learning forecasts that trade less.
  Once the two-stage pipeline gets a turnover penalty in its optimiser, it catches up: on sector ETFs it then matches
  or beats DFL (`results/tables/turnover_penalty_dfl_vs_pf.md`). On FF12, DFL stays ahead (+0.04
  Sharpe) but no longer significantly (p = 0.23).
* On the hindsight-selected mega-caps (2–10 years of training data), DFL loses, though not significantly.
* No learned model beats 1/N net of costs. Weekly predictability from price features is very weak, so the results
  are about the training loss, not a trading strategy.
* Extension: CVaR-regularised DFL is the most robust learned model. It has the best Sharpe ratio, tail risk and
  turnover, both walk-forward and frozen, over 2007–2025.

| | |
|---|---|
| ![](results/figures/fig1_synthetic_regret_vs_degree.png) | ![](results/figures/fig2_mse_vs_regret.png) |
| ![](results/figures/fig4_wealth_ff12_main.png) | ![](results/figures/fig7_robust_extension.png) |

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+; CPU PyTorch is enough
python -m pytest -q                      # 18 tests, ~10 s (layer vs CVXPY, finite differences, no-leakage checks)
QUICK=1 bash scripts/reproduce_all.sh    # 1 seed, 3 test years per experiment: smoke test, ~5-10 min
bash scripts/reproduce_all.sh            # everything in the report: ~4 h on 2 CPU cores
```

Single experiments:

```bash
python scripts/run_walkforward.py configs/sector9.yaml            # main real-data experiment
python scripts/run_walkforward.py configs/robust_sector9.yaml --frozen
python scripts/run_synthetic.py   configs/synthetic.yaml          # RQ1
python scripts/run_synthetic.py   configs/synthetic_beta.yaml     # RQ3
python scripts/run_ablation.py    configs/ablation_sector9.yaml   # RQ2
python scripts/make_figures.py                                    # figures + tables from results/
```

Every run writes to `results/<name>/`:

| File | Contents |
|---|---|
| `summary.csv` | Every metric, mean and sd over seeds. |
| `metrics_per_seed.csv` | One row per method and seed. |
| `tests.csv` | DFL vs PF: block-bootstrap Sharpe-difference CI, Jobson-Korkie-Memmel test, paired seed tests. |
| `weekly_returns.csv` | Gross and net weekly returns, turnover and regime label for every method, seed and date. |
| `yearly_returns.csv` | Net return per calendar year. |
| `prediction_metrics_per_seed.csv` | Forecast metrics. |
| `train_diagnostics.csv`, `histories.json` | Epochs, best epoch, zero-gradient share and gradient norm. |
| `predictions.npz` | Raw forecasts. |
| `config.json` | The exact configuration used. |

To keep the archive small, the per-setting `weekly_returns.csv`, `predictions.npz` and `histories.json` of the
ablation runs (`results/ablation_sector9/*/`) are not included. Their summaries are, and
`scripts/run_ablation.py` regenerates the rest.

## Using the layer on its own

```python
import torch
from dfl_portfolio import PortfolioLayer

layer = PortfolioLayer(lam=5.0, lower=0.0, upper=0.3, eta=0.0)   # long-only, 30% cap
mu_hat = model(x)                                    # (B, N) predicted returns, requires grad
w, info = layer(mu_hat, Sigma)                       # Sigma: (B, N, N) or (N, N)
loss = -((w * r_next).sum(-1).mean() - 5.0 * (w * r_next).sum(-1).var())
loss.backward()                                      # exact gradient through the KKT system
print(info.frac_zero_grad)                           # share of samples at a vertex (zero gradient)
```

The forward pass is a batched ADMM solver followed by active-set polishing to the exact KKT point. The backward
pass solves the linear KKT system of that active set (the implicit function theorem), so gradients are exact rather
than smoothed. The tests check the forward pass against CVXPY (about 1e-11) and the gradients against finite
differences (about 1e-10) and the closed form (about 1e-13).

## Repository layout

```
src/dfl_portfolio/
  data.py          universes (sector ETFs, Fama-French 12 industries, the notebook's 10 stocks), weekly returns
  features.py      13 causal features per asset (momentum, volatility, relative momentum, market state)
  covariance.py    rolling Ledoit-Wolf covariance, rolling historical mean
  splits.py        expanding-window walk-forward folds with validation year and embargo; leakage check
  layers.py        the differentiable portfolio layer (ADMM + active-set polish + KKT backward)
  models.py        cross-sectional linear model and MLP shared across assets
  losses.py        utility, Sharpe, CVaR, Wasserstein-DRO, group-DRO, regret
  train.py         MSE training and decision-focused training (hybrid beta, warm start, early stopping)
  backtest.py      drift-aware turnover, proportional costs
  metrics.py       performance, forecast-accuracy and calibration metrics
  stats.py         block bootstrap, Jobson-Korkie-Memmel, paired tests
  synthetic.py     synthetic markets with controllable misspecification
  experiments.py, synthetic_experiments.py   experiment drivers
configs/           one YAML per experiment (see docs/REPORT.md, section 3)
scripts/           run_walkforward.py, run_synthetic.py, run_ablation.py, make_figures.py, reproduce_all.sh
tests/             layer correctness, pipeline/no-leakage tests
data/raw/          cached daily prices used for the reported results
results/           all outputs; results/figures and results/tables are generated by make_figures.py
```

## Data

| Universe | Assets | Weekly sample | Source |
|---|---|---|---|
| `sector9` | 9 SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) | 1999–2025 | Yahoo Finance, adjusted close |
| `ff12` | Fama-French 12 industry portfolios (value-weighted) | 1990–2025 | Kenneth French data library |
| `megacap10` | the notebook's 10 stocks (hindsight-selected, secondary only) | 2013–2025 | Yahoo Finance, adjusted close |

The cached CSVs in `data/raw` are the exact inputs behind the reported numbers. `scripts/download_data.py`
keeps them unless you pass `--force`.

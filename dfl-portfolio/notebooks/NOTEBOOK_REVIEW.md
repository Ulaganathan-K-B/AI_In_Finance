# Review of the preliminary notebook (`01_data.ipynb`)

**Verdict.** As a *preliminary* version it is acceptable: the data pipeline is clean, the split is
chronological, there is no look-ahead in the features, the baselines are the right ones (1/N, Markowitz,
historical mean → Markowitz), the MSE-vs-decision comparison is the right question, and the
`cvxpylayers` layer is set up correctly. It is **not yet usable for conclusions**. The decision-focused
model never trained, and the evaluation design cannot separate skill from luck. The table below lists each
issue and where this repository fixes it.

| # | Issue in the notebook | Why it matters | Fix in this repository |
|---|---|---|---|
| 1 | **The DFL model did not learn.** Training loss is `-0.005510` at epochs 10, 20, 30, 40 and 50. | With λ = 1, a weekly covariance of order 1e-3 and an un-normalised network whose outputs are about 0.1, the predicted-return term dominates the risk term by about two orders of magnitude. Every solution is therefore a vertex of the feasible set, such as 0.3/0.3/0.3/0.1 (cell 27 shows one). At a vertex all weights sit on bounds, so ∂w*/∂μ̂ = 0 and the gradient is exactly zero. | Network output scaled to the target std with a near-zero initial output layer (`models.py`). DFL is warm-started from the MSE model. λ = 5 on sector ETFs. The layer reports the share of zero-gradient samples (`LayerInfo.frac_zero_grad`), which is logged every epoch. `configs/megacap10_notebook_replication.yaml` re-runs the notebook set-up next to the fixed one. |
| 2 | The DFL loss is the **mean realised return only**. | The risk term in the layer does nothing for the training signal, so the model is pushed towards the most extreme corner. It also optimises something different from what the layer optimises. | `losses.py` provides the realised mean-variance utility (the same objective as the layer, net of costs), plus Sharpe, CVaR, Wasserstein-DRO, group-DRO and regret losses. |
| 3 | **One 85-week test window, one seed.** "NN + Markowitz: 77% cumulative return" comes from a single window. | With 85 weekly returns the annualised Sharpe ratio has a standard error of about 0.8, so a 95% interval for a single strategy is roughly ±1.5. The window (May 2024–Dec 2025) was a strong mega-cap rally in which every long-only rule looks good. | Expanding-window walk-forward with yearly refits (21 test years on sector ETFs, 30 on FF12), 3–5 seeds, a block-bootstrap Sharpe-difference CI, the Jobson-Korkie-Memmel test and paired seed tests (`stats.py`). |
| 4 | **The NN forecasts are worse than the historical mean** (test MSE 0.001609 vs 0.001557) but the NN portfolio does much better. | This is a sign that the portfolio result is luck or a vertex artefact, not forecasting skill. It is also a nice example of RQ1, but it needs controlled experiments to interpret. | Synthetic markets with known μ, where regret and MSE are exact (`synthetic.py`, `results/synthetic_main`). |
| 5 | **No transaction costs, no turnover.** | Weekly re-optimisation can trade 50% of the book every week; at 10 bps that is about 2.5% a year. | `backtest.py` uses drift-aware turnover, 10 bps proportional costs and starts from cash. The layer has an optional turnover penalty η‖w − w_prev‖₁, and DFL is trained net of costs. |
| 6 | **Dense 120 → 64 → 32 → 10 network on 393 samples,** raw (unstandardised) inputs, 100 full-batch epochs, no early stopping. | It has about 10k parameters for about 4k targets, so it overfits (validation loss rises after epoch 60). There is no parameter sharing across assets. | Cross-sectional models shared across assets (`SharedLinear`, `SharedMLP` with a linear skip): 13 standardised features per asset, mini-batches, weight decay, early stopping on a validation year, and a scaler fitted on training rows only. |
| 7 | **The covariance is frozen** at its 2015–2022 estimate for the whole test. | The decision then ignores the 2022 and 2025 volatility regimes. | 104-week rolling Ledoit-Wolf covariance, re-estimated every week from past data only (`covariance.py`). |
| 8 | **Hindsight-selected universe**: the 10 largest US stocks of 2025 (including NVDA and META). | This introduces survivorship and look-ahead bias, so any long-only strategy on these names looks good. | The main results use universes that are fixed ex ante: 9 SPDR sector ETFs (1999–) and the 12 Fama-French industry portfolios (1990–). The 10 mega-caps are kept as a secondary test with that caveat. |
| 9 | The headline "Sharpe" in the printouts is CAGR / volatility. | That ratio is not the Sharpe ratio, and with compounding it overstates high-volatility strategies. | The conventional Sharpe ratio (mean / sd · √52, risk-free rate 0) is used throughout. The notebook's own last table already did this. |
| 10 | Small things | The final "week" (2026-01-02) contains only three trading days, which is harmless for one observation but worth knowing. The two Markowitz cells (19 and 34) are the same strategy. `cvxpylayers` is called one sample at a time, which is slow. | The same short final week is present here: it affects only the last 2025 decision. A batched custom layer solves 1,200 problems in about 0.1 s with exact gradients, checked against CVXPY and finite differences in `tests/test_layers.py`. |

## What happened when the notebook set-up was re-run here

Two runs train three models each, at the notebook's λ = 1 and 30% cap, under the same walk-forward protocol as
everything else:

* `results/megacap10_notebook_replication` uses the notebook's 10 stocks, 2017–2025, 5 seeds.
* `results/sector9_notebook_replication` is a control on 9 sector ETFs, 2005–2025, 3 seeds.

The three models are:

* `pf_mlp`: an MSE-trained shared MLP, then the layer.
* `dfl_notebook_style`: the notebook's recipe. It uses an un-normalised network (no output scaling, default
  initialisation, no skip connection, no dropout), the mean-return loss and 100 epochs without early stopping.
* `dfl_fixed`: the recipe used in this repository. It uses output scaling, a small initialisation, a warm start from
  `pf_mlp`, the utility loss and early stopping.

| | 10 mega-caps | 9 sector ETFs |
|---|---|---|
| Training dates with zero gradient at the start: notebook recipe / fixed recipe | 96% / 34% | 98% / 52% |
| Net Sharpe: PF / DFL notebook recipe / DFL fixed | 1.020 / 1.231 / 1.035 | 0.400 / 0.452 / 0.398 |
| Net Sharpe: MV with historical mean / 1/N | 1.303 / 1.290 | 0.613 / 0.651 |

1. **The diagnosis is confirmed.** With the notebook's recipe almost every training date is a vertex solution with
   zero gradient, and in many epochs the whole gradient is exactly zero. That is the flat loss in cell 38.
2. **The recipe's win on mega-caps is not learning.** A model that gets a gradient from 2–4% of dates ends up as a
   corner portfolio holding 4 names at the 30% cap. On stocks chosen because they were 2025's winners, any
   concentrated bet on past winners pays. The historical-mean MV portfolio does the same thing and does better.
   On sector ETFs the edge shrinks to +0.05 Sharpe and is not significant (p = 0.33).
3. **λ = 1 is too low for weekly data.** Even the fixed recipe leaves 34–52% of dates without gradient, and every
   λ = 1 strategy on sector ETFs trails 1/N by about 0.2 Sharpe. The main experiments use λ = 5 (see
   `docs/REPORT.md`, §4.2).

Tables: `results/tables/notebook_replication_diagnostics.md`, `results/tables/megacap10_notebook_replication_*.md`,
`results/tables/sector9_notebook_replication_*.md`. Figure: `results/figures/fig8_notebook_diagnostics.png`.

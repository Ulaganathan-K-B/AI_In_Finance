# Decision-focused learning for constrained mean-variance portfolios

*AI4403, Project 4: report.* All numbers in this report are produced by `scripts/reproduce_all.sh`. Tables are in
`results/tables`, figures in `results/figures`.

## 1. Problem

A portfolio manager forecasts next-week returns μ̂_t from features x_t, then picks weights by solving

  w*(μ̂_t, Σ̂_t) = argmax_w μ̂_tᵀw − λ wᵀΣ̂_t w − η‖w − w_{t−1}‖₁  subject to 1ᵀw = 1 and 0 ≤ w ≤ u.

The **two-stage** or **prediction-focused (PF)** approach fits the forecaster by mean-squared error and then plugs
its forecasts into the optimiser. **Decision-focused learning (DFL)** instead fits the forecaster to maximise the
*realised* quality of the decisions it induces, w*(f_θ(x_t))ᵀr_{t+1} net of costs and risk, by differentiating
through the optimiser. The project asks three questions:

* **RQ1.** When does lower prediction error fail to give lower decision regret?
* **RQ2.** Which constraints or risk penalties keep end-to-end training stable, including under regime shift?
* **RQ3.** Does DFL trade forecast calibration for decision utility?

The extension adds risk-constrained (CVaR) and distributionally robust (Wasserstein, group-DRO) decision losses and
tests them under regime shift.

## 2. Method

### 2.1 The differentiable portfolio layer (`layers.py`)

The problem above is a convex QP with Q = 2λΣ̂, a budget equality and box constraints. The η term is handled exactly
by the proximal operator of the ℓ₁ norm.

* **Forward pass.** A batched ADMM solver works on a copy scaled by the mean diagonal of Q, with a pre-factorised KKT
  matrix and over-relaxation (α = 1.6). It is followed by up to 25 rounds of primal-dual active-set polishing.
  Polishing identifies, for each asset, whether it sits at the lower bound, at the upper bound, at w_{t−1} (the kink
  of the ℓ₁ term) or is free, and then solves the equality-constrained QP on that set exactly.
* **Backward pass.** Given the active set, w* solves a small linear KKT system: free rows [Q | 1], fixed rows equal to
  identity, and the budget row. That system is solved with `torch.linalg.solve` *inside autograd*, so
  ∂w*/∂μ̂ and ∂w*/∂Σ̂ come out by the implicit function theorem and are **exact**. They are not the smoothed
  interior-point derivatives that `cvxpylayers` returns.
* **Degenerate cases.** When every coordinate is fixed (a vertex), the budget multiplier ν is not unique. It is pinned
  to the midpoint of its feasible interval, which makes the system non-singular. The gradient is then exactly
  zero, which is correct, and `LayerInfo.frac_zero_grad` reports how often this happens.
* **Verification** (`tests/test_layers.py`). The forward pass matches CVXPY/CLARABEL to about 1e-11 for five
  parameter sets, with and without η. Gradients match central finite differences to about 1e-10, match the
  budget-only closed form to about 1e-13, and are shift-invariant. They are exactly zero at a vertex and agree in
  direction with `cvxpylayers` (cosine above 0.95). The layer solves 1,200 nine-asset problems in about 0.1 s.

### 2.2 Predictors (`models.py`, `features.py`)

The predictors are cross-sectional and **share parameters across assets**. One function maps an asset's 13 features
to its next-week return, plus a learned per-asset intercept. The features are:

* returns over the last 1, 4, 13 and 26 weeks;
* 52-week momentum skipping the latest 4 weeks;
* 4-, 13- and 52-week volatility;
* 13- and 52-week momentum relative to the cross-section;
* three market-state features (4- and 13-week market momentum, 13-week market volatility).

All features use data up to week t only. They are standardised and clipped at ±5 with statistics from the training
rows only. Two models are used: a linear model and an MLP (13 → 32 → 16 → 1, ReLU, dropout 0.1) with a linear skip
connection. Outputs are multiplied by the standard deviation of the training targets, and the output layer is
initialised near zero. Initial forecasts are therefore close to zero, so the layer starts in its interior with
non-zero gradients. This is fix #1 for the preliminary notebook.

### 2.3 Training (`train.py`, `losses.py`)

* **PF.** Mini-batch Adam (batch 64, lr 1e-3, weight decay 1e-4), with early stopping on validation MSE
  (patience 40, at most 300 epochs).
* **DFL.** Full-batch, chronological training. The loss is the negative realised mean-variance utility of the
  layer's decisions over the training window, −(mean(R) − λ·var(R)), where R_t = w_tᵀr_{t+1} minus 10 bps × turnover.
  Turnover is measured against the previous row's decision, so DFL is trained net of costs. DFL is warm-started
  from the PF model and early-stopped on the validation-year decision loss. The ADMM state is warm-started across
  epochs.
* **Hybrid.** L = decision loss + β·MSE. β = 0 gives pure DFL and β → ∞ gives PF.
* **Other decision losses** (extension): negative Sharpe; mean + κ·CVaR₀.₉ of losses (Rockafellar-Uryasev); a
  Wasserstein-type DRO utility, utility − ε·mean‖w_t‖₂, which is the worst case over return perturbations
  ‖δ‖₂ ≤ ε; and group-DRO, the worst utility across calm and turbulent regimes split at the training-median
  market volatility.

### 2.4 Evaluation protocol (`splits.py`, `backtest.py`, `stats.py`)

* **Walk-forward.** For each test year Y, the model trains on all weeks up to Y−1 minus a 52-week validation year,
  with a 1-week embargo, and is then tested on the 52 weeks of Y. It is refit every year. `check_no_leakage` asserts
  that train, validation and test windows are disjoint and in time order.
* **Inputs at each decision date.** The decision at date t uses μ̂_t from the model, Σ̂_t from a 104-week rolling
  Ledoit-Wolf estimate, and w_{t−1}. All three use data up to t only.
* **Backtest.** Weights drift with returns between rebalances. Turnover is ‖w_t − w_{t−1}^{drifted}‖₁, costs are
  10 bps per unit of turnover, and the portfolio starts in cash.
* **Baselines.** 1/N; long-only minimum variance with the same cap; and MV with the 156-week historical mean
  through the same layer.
* **Seeds.** 5 seeds for the main experiments (3 for FF12 and the ablation). We report mean ± sd over seeds.
* **Tests for DFL vs PF.**
  * A circular block bootstrap (8-week blocks, 5,000 draws) of the Sharpe difference of seed-averaged returns.
  * The Jobson-Korkie-Memmel test.
  * A paired t-test of per-seed Sharpe ratios and utilities. This last test captures *training* variance, not market
    variance, so it is the weakest of the three.
* **Regimes.** A test week is *turbulent* if trailing 13-week market volatility is above the 75th percentile of all
  earlier weeks. The label uses only past data.

### 2.5 Synthetic markets (`synthetic.py`)

For RQ1 and RQ3 we need the true μ. Asset features are Gaussian. The true expected return is
μ_{t,i} = m + s·((z_{t,i}+3)^d − c_d)/s_d with z = xᵀb, a degree-d polynomial of a linear index, after
Elmachtoub and Grigas (2022). It is standardised so the cross-sectional signal sd is s for every d: only the *shape*
changes with d, not the signal-to-noise ratio. Returns follow a one-factor model, so Σ is known. A decision is
scored by its *expected* utility under the true (μ, Σ) on 2,000 fresh dates, and we report

  normalised regret NR = (U(oracle) − U(ŵ)) / (U(oracle) − U(1/N)),

which is 0 for the oracle and 1 for no better than 1/N. Settings: 10 assets, 5 features, s = 0.5%/week, factor
vol 2%, idiosyncratic vol 3%, λ = 10, cap 40%. The shift test doubles all volatilities at test time, both
in the returns and in the Σ given to the layer, while μ stays unchanged.

## 3. Experiments

| Config | Question | Data | Design |
|---|---|---|---|
| `synthetic.yaml` | RQ1 | synthetic | degree d ∈ {1,2,4,6,8} × n_train ∈ {100,300,1000} × 20 seeds; PF/DFL × linear/MLP; DFL without warm start; vol×2 shift |
| `synthetic_beta.yaml` | RQ3 | synthetic | hybrid β ∈ {0,1,10,…,10⁴,∞}, d ∈ {1,6}, 10 seeds |
| `synthetic_lambda.yaml` | RQ2 | synthetic | λ ∈ {1,2,5,10,25,50} × cap ∈ {0.2,0.4,1.0}, d = 6, 5 seeds, vol×2 shift |
| `sector9.yaml` | main benchmark | 9 sector ETFs | walk-forward 2005–2025 (21 years), 5 seeds, λ = 5, cap 30% |
| `ff12.yaml` | robustness | FF12 industries | walk-forward 1996–2025 (30 years), 3 seeds, λ = 5, cap 25% |
| `ff12_eta.yaml` | cost-aware baseline | FF12 industries | as `ff12.yaml` with a turnover penalty η = 0.001 in the layer, PF vs DFL |
| `megacap10.yaml` | secondary | notebook's 10 stocks | walk-forward 2017–2025, 5 seeds |
| `megacap10_notebook_replication.yaml`, `sector9_notebook_replication.yaml` | notebook diagnosis | both | notebook recipe vs fixed recipe, λ = 1, cap 30% |
| `ablation_sector9.yaml` | RQ2 | sector ETFs | one knob at a time: λ, cap, η; walk-forward (3 seeds) and frozen model (5 seeds) |
| `robust_sector9.yaml` (+ `--frozen`) | extension | sector ETFs | utility / Sharpe / CVaR / Wasserstein-DRO / group-DRO losses; walk-forward 2007–25 and frozen (trained on data up to 2005, never refit) |

## 4. Results

### 4.1 RQ1: when does lower prediction error fail to give lower regret?

`results/tables/synthetic_pooled_dfl_vs_pf.md`, `synthetic_dfl_vs_pf_by_n.md`,
`synthetic_kendall_mse_regret.md`; Figs. 1–2. There are 20 seeds per cell. Each comparison is paired: the same
seed means the same training data and the same test set.

* **Well specified, enough data: PF wins.** With degree 1 and n ≥ 300, DFL's normalised regret is higher by 0.010
  (80 pairs, p = 0.005). When the model class contains the truth, MSE is the efficient, lowest-variance way to fit
  it, and a decision loss only adds noise.
* **Misspecified: DFL wins.** With degree ≥ 4 and n ≥ 300, DFL lowers regret by 0.011 (240 pairs, p = 0.003; DFL
  better in 58% of pairs), and by 0.014 at degree ≥ 6. Its test MSE is the same or *higher*: the median MSE ratio
  of DFL to PF is 1.00–1.11. Across all sample sizes, 24–32% of misspecified pairs have DFL with both a higher MSE
  and a lower regret, against 9% when well specified.
* **Little data (n = 100): DFL wins at every degree.** The gain is 0.04–0.08 at degree ≥ 2 (p < 0.005), with
  equal MSE. A plausible reading: with 100 dates the MSE fit is noisy in the directions that matter for the
  decision, and fine-tuning on the decision loss, with early stopping on validation *decisions*, removes some of
  that noise.
* **MSE stops ranking decisions.** Take the five trained models on the same data and rank them by test MSE and by
  regret. At degree 1 the two rankings agree well: Kendall τ = 0.72–0.79 for n ≥ 300. At degree 4–8 they agree
  less: τ = 0.36–0.71, and as low as 0.47–0.57 for n = 1000. Picking a model by forecast accuracy then often picks
  the wrong one for the decision.
* **The gains are regime-specific.** When test volatility doubles (vol×2), DFL's advantage at n ≥ 300 disappears:
  +0.004 for the linear model (p = 0.04, PF better) and +0.001 for the MLP (n.s.). DFL learns the decision boundary
  of the training risk regime. This links to RQ2.

**On real data the same pattern appears.** On sector ETFs and FF12 the DFL models have *higher* test MSE and,
on sector ETFs, a lower rank IC than their PF twins (`results/tables/sector9_main_summary.md`), yet they give
higher realised utility. On FF12 they also give a significantly higher net Sharpe ratio (§4.4). The preliminary
notebook shows the same thing by accident: its network had a higher MSE than the historical mean (0.001609 vs
0.001557), yet its portfolio did much better. Without controlled experiments, that cannot be told apart from luck.

**Why it happens.** The layer only uses μ̂ through the solution w*(μ̂), and three properties of that map make
MSE a poor proxy.

* **Shift invariance.** Under the budget constraint, adding a constant to every μ̂_i leaves w* unchanged. The
  level of the forecasts, which matters for MSE, is irrelevant for the decision.
* **Locally constant regions.** At a bound, a coordinate's forecast can move within an interval without changing
  w*. Forecast errors there cost MSE but no utility.
* **Unequal costs.** Errors on assets the optimiser is close to buying, or that are highly correlated with held
  assets, cost much more utility than errors elsewhere. MSE weighs all errors equally.

A misspecified model cannot be right everywhere, so how it spends its errors matters. DFL spends them where they do
not move the decision. A well-specified model with enough data can be right everywhere, and then MSE is the more
efficient, lower-variance loss.

### 4.2 RQ2: what keeps end-to-end training stable?

The main failure mode is **vertex decisions**. When forecasts are large relative to λΣ, or the box is tight, w*
sits at a vertex where all weights are at bounds and ∂w*/∂μ̂ = 0. Training then sees no signal on those dates.
On the remaining dates it fits noise from a handful of points.

* **Synthetic λ × cap grid** (Fig. 9, `results/tables/synthetic_lambda_cap.md`). The share of zero-gradient
  samples at the start of DFL training is 61% at λ = 1 with cap 0.2. It is 32% at λ = 1 without a cap and below
  10% for every λ ≥ 5.
  * *Loose cap, moderate λ.* This is where DFL's gain is largest: it lowers normalised regret by 0.070 at λ = 5
    with no cap, and by 0.054 at λ = 10.
  * *Tight cap.* The gain turns *negative*: with cap 0.2, PF is as good or better at every λ, by up to 0.03.
  * *High λ.* The gain vanishes at λ ≥ 25, where all decisions collapse towards minimum variance and forecasts
    hardly matter.
  * *Volatility shift.* DFL's gain moves to low λ (+0.05 at λ = 1–2 without a cap). At λ = 10 without a cap, PF
    is better under the shift (−0.016) although DFL was better in-distribution (+0.054). DFL's advantage is
    specific to the risk regime it was trained in.
* **Sector-ETF ablation** (Fig. 6, `results/tables/ablation_sector9.md`):
  * *Risk aversion λ.* At λ = 1–2, 31–54% of training dates start with a zero gradient, DFL does not beat PF, and
    the frozen DFL model loses 0.03–0.06 Sharpe against frozen PF. From λ = 5 the zero-gradient share at the start
    of training falls to 1–16%. At the selected epoch the shares are 33–55% for λ ≤ 2 and 6–26% for λ ≥ 5: DFL
    pushes decisions towards corners as it trains. At λ = 5 frozen DFL still trails frozen PF (0.477 vs 0.508). At
    λ = 10 and 25 it beats it (0.568 vs 0.506 and 0.607 vs 0.582) with less than half the turnover (7.4 vs 16.6 and
    2.9 vs 9.6 per year). A stronger risk term both regularises the decision map and keeps it differentiable.
  * *Position cap.* Tighter caps give higher net Sharpe for both PF and DFL (0.58–0.59 at 20% vs 0.39–0.40 with no
    cap). DFL beats PF on utility at every cap in the walk-forward runs, but its Sharpe edge is positive only at caps
    of 20–30%. Caps act as a robust-optimisation constraint: they bound how much a wrong forecast can move the
    portfolio.
  * *Turnover penalty η.* This is the most important knob for real-data performance, and it changes the comparison
    (`results/tables/turnover_penalty_dfl_vs_pf.md`).
    * *Effect on PF.* η = 0.001, the same 10 bps as the trading cost, cuts PF turnover from 25 to 5.5 per year and
      lifts PF's net Sharpe from 0.545 to 0.639.
    * *Effect on DFL.* DFL also trades less (14 → 3.6 per year), but its Sharpe rises less (0.569 → 0.607).
    * *Comparison.* With any η > 0, PF matches or beats DFL on Sharpe (−0.002 to −0.028; not significant), and DFL's
      utility edge is no longer significant (seed p = 0.06–0.69).
    * *Gradient sparsity.* η also makes DFL's training signal sparse. At the selected epoch, 55%, 70% and 86% of
      training dates have zero gradient for η = 0.0005, 0.001 and 0.002, against 26% at η = 0. This is because
      coordinates stick at the kink w_i = w_{t−1,i} of the ℓ₁ term.

    The practical reading: put the transaction cost in DFL's *loss*, where it is differentiable, rather than in the
    layer. But a two-stage pipeline with a cost-aware layer is a strong baseline, and DFL's real-data edge at η = 0
    is largely cost-awareness that the two-stage pipeline lacks. The best strategy in the whole ablation is the
    simplest one: the historical-mean MV portfolio with η = 0.001 (Sharpe 0.711 walk-forward, 0.673 frozen), ahead of
    1/N (0.651).
* **Warm start is essential.** DFL trained from scratch on synthetic data (`dfl_linear_scratch`) has a normalised
  regret 0.065 higher than warm-started DFL (worse in 87% of 300 cases), and 0.03–0.07 higher than plain PF.
  Starting from the PF solution puts the model in a region where the layer is differentiable and the forecasts have
  the right scale. Early stopping on the validation decision loss then stops it before it overfits. The best epoch
  is around 30–40 of a 300-epoch budget.
* **The notebook's recipe** combined λ = 1, an un-scaled network and the mean-return loss. At the start of training,
  96% of training dates had zero gradient (`results/tables/notebook_replication_diagnostics.md`, Fig. 8a). With
  the fixed recipe at the same λ the share is 34%.

**Recommendation for stable DFL.**

* Use λ ≥ 5 (weekly units) or an equivalent risk budget, and a position cap of 20–30%.
* Scale the outputs and warm-start from MSE.
* Train on utility net of costs: put the costs in the loss, not as an ℓ₁ term in the layer.
* Early-stop on the validation decision loss.
* Monitor the zero-gradient share. If it is above about 30%, the training signal is mostly noise.
* Compare against a two-stage pipeline that is given the same cost-awareness (η > 0 in its layer).

### 4.3 RQ3: does DFL trade calibration for utility?

**Yes, but most of the trade is unnecessary, and a small MSE term recovers calibration.**

**Synthetic** (`results/tables/synthetic_beta.md`, Fig. 3; n = 300, 10 seeds, measured against the true μ):

| β (weight on MSE) | 0 (DFL) | 1 | 100 | 1e4 | ∞ (PF) |
|---|---|---|---|---|---|
| Degree 1: regret − regret(PF) | +0.015 | +0.014 | +0.009 | +0.008 | 0 |
| Degree 1: calibration slope | 0.96 | 0.95 | 0.99 | 0.98 | 1.04 |
| Degree 1: bias (bp/week) | −3.6 | +1.7 | +2.0 | +1.9 | +1.1 |
| Degree 6: regret − regret(PF) | +0.003 | +0.000 | −0.000 | −0.000 | 0 |
| Degree 6: calibration slope | 0.90 | 0.90 | 0.95 | 0.92 | 1.07 |
| Degree 6: bias (bp/week) | −5.9 | −0.4 | +0.6 | +1.2 | −1.2 |

* The clearest cost of pure DFL is the forecast *level*. Its bias is −3.6 and −5.9 bp/week, which is 7–12% of
  the cross-sectional signal sd. A weight of β = 1 on MSE removes it entirely and leaves regret unchanged.
* The calibration *slope* moves from above 1 for PF (forecasts shrunk towards the mean, as MSE does under noise) to
  below 1 for DFL (forecasts more dispersed than the truth). β = 100–1000 lands closest to 1.
* At degree 1, intermediate β interpolates the regret between DFL and PF. At degree 6 all β give the same regret, so
  there calibration comes for free.

**Real data** (`results/tables/real_data_overview.md`). The pooled calibration slope of realised on predicted
returns falls under DFL in both long samples:

| | PF MLP | DFL MLP | Hybrid MLP (β = 100) |
|---|---|---|---|
| Sector ETFs: calibration slope | 0.72 | 0.25 | 0.63 |
| Sector ETFs: net Sharpe | 0.548 | 0.568 | 0.566 |
| FF12: calibration slope | 0.27 | −0.23 | 0.25 |
| FF12: net Sharpe | 0.488 | 0.577 | 0.586 |

The hybrid keeps PF-level calibration and DFL-level decision quality on both universes. On FF12 its Sharpe gain
over PF (+0.10, bootstrap p = 0.007) is as large as pure DFL's. On the hindsight mega-cap universe neither DFL nor
the hybrid helps (§4.4).

**Why calibration is lost.**

* *Level.* Shift invariance means DFL gets no gradient for the forecast level, so the level drifts. In the synthetic
  runs the bias of pure DFL is −3.6 to −5.9 bp/week against the true μ, while PF and every hybrid stay within
  ±2.5 bp.
* *Dispersion.* DFL sets the dispersion of the forecasts to whatever produces good decisions under the risk term and
  costs, not to the conditional mean. On real data that means smaller, smoother forecasts that trade less.

DFL's outputs therefore behave as decision scores rather than forecasts. If the forecasts are used anywhere else
(risk reports, other optimisers, communicating views), use the hybrid.

### 4.4 Real-data benchmark

`results/tables/real_data_overview.md`, Figs. 4, 5 and 10. Net Sharpe ratios are the mean over seeds; tests compare
seed-averaged weekly returns.

| Universe | 1/N | MV hist. mean | PF MLP | DFL MLP | Hybrid | DFL − PF (95% CI) | bootstrap p / JKM p |
|---|---|---|---|---|---|---|---|
| Sector ETFs 2005–25 | 0.651 | 0.647 | 0.548 | 0.568 | 0.566 | +0.02 [−0.06, +0.12] | 0.61 / 0.62 |
| FF12 industries 1996–25 | 0.663 | 0.623 | 0.488 | 0.577 | 0.586 | +0.09 [+0.01, +0.17] | 0.02 / 0.04 |
| 10 mega-caps 2017–25 | 1.290 | 1.071 | 0.938 | 0.823 | 0.898 | −0.10 [−0.26, +0.05] | 0.18 / 0.24 |

1. **DFL beats the two-stage pipeline on the two long, ex-ante universes**, and the gain comes mostly from trading
   less.
   * On FF12 (30 test years, 4–33 years of training data) DFL beats PF by +0.09 to +0.11 Sharpe on every seed,
     significant on both tests for the linear model, the MLP and the hybrid.
   * On sector ETFs (21 test years) DFL improves utility on every seed (paired p < 0.001), but the Sharpe gain
     (+0.02) is well within the bootstrap noise.
   * On the 10 mega-caps DFL is worse, though not significantly over time (p = 0.18). This universe has only 2–10
     years of training data and is dominated by a few trending winners. We cannot separate a sample-size explanation
     from a universe effect. The synthetic experiments do *not* show DFL losing with little data (§4.1).
   * The mechanism is costs. Trained net of costs, DFL halves turnover (25 → 13–15 per year on sector ETFs and
     FF12). Gross of costs, PF is ahead on sector ETFs (0.70 vs 0.66) and DFL only slightly ahead on FF12 (0.66 vs
     0.64). The decision-aware loss is the only part of the pipeline that sees the cost of a jumpy forecast.
   * When the layer itself is made cost-aware with a turnover penalty η (§4.2), the two-stage pipeline catches up.
     On sector ETFs it then matches or beats DFL. On FF12 (`configs/ff12_eta.yaml`, η = 0.001) DFL stays ahead
     (0.642 vs 0.608, +0.035), but the gap is no longer significant over time (bootstrap p = 0.23; seed p = 0.04).
     With η every strategy improves. The historical-mean MV and minimum-variance portfolios (0.69–0.70) then beat
     both learned models on FF12 too (`results/tables/turnover_penalty_dfl_vs_pf.md`, `ff12_eta_summary.md`).
2. **No learned model beats 1/N net of costs on any universe**, and simple minimum variance is the best strategy on
   sector ETFs (0.695). Weekly cross-sectional return predictability from price features is very weak. Rank ICs are
   0.01–0.05, and PF test MSE is within 1% of the historical mean's. This matches DeMiguel, Garlappi and Uppal (2009).
   The comparison that the project asks about, DFL vs PF with the same model, features and layer, is
   still informative: it isolates the effect of the training loss.
3. **Regimes** (Fig. 5). In turbulent weeks PF does better than DFL on sector ETFs (Sharpe 0.74 vs 0.60), and in
   calm weeks DFL does better (0.60 vs 0.49). On FF12, DFL is ahead in both regimes.

### 4.5 Extension: risk-constrained and robust decision losses

Fig. 7 and `results/tables/robust_extension.md` cover sector ETFs, tested 2007–2025, with 5 seeds. In *frozen*
mode the model is trained once on 2000–2005 (validated on 2006) and never refit, so it faces the 2008, 2020 and 2022
regimes unseen.

| Loss | Walk-forward Sharpe | Frozen Sharpe | Turnover/yr (WF) | CVaR95 weekly (WF) | Turbulent-week utility, ann. (WF) |
|---|---|---|---|---|---|
| PF (MSE) | 0.513 | 0.508 | 25.8 | 5.79% | −0.155 |
| DFL utility | 0.524 | 0.477 | 13.4 | 5.31% | −0.157 |
| DFL Sharpe | 0.493 | 0.485 | 17.3 | 5.59% | −0.169 |
| **DFL + CVaR₀.₉ (κ = 1)** | **0.568** | **0.531** | **10.3** | **5.06%** | **−0.116** |
| DFL + Wasserstein DRO (ε = 0.003) | 0.525 | 0.510 | 14.8 | 5.41% | −0.160 |
| DFL + group DRO (calm/turbulent) | 0.499 | 0.486 | 16.3 | 5.45% | −0.154 |

* **CVaR-regularised DFL is the most robust learned strategy.** Walk-forward, it has the best Sharpe ratio, CVaR,
  turnover and turbulent-week utility of all learned models. Frozen, it keeps the best Sharpe ratio, CVaR and
  turnover. Against PF its utility gain is significant across seeds (p < 0.001), although the Sharpe difference
  over time is not (bootstrap p = 0.38). Minimum variance still has the lowest tail risk overall (CVaR 4.88%).
* **Plain DFL degrades most when it is not refit** (0.524 → 0.477, against 0.513 → 0.508 for PF). The decision
  loss fits the training regime's decision boundary more tightly than MSE does.
  * The Wasserstein penalty discourages concentration and cuts the degradation to 0.015.
  * The CVaR term starts higher and remains the best learned model after the shift (0.531).

  This is the robust-DFL argument of the extension: under a regime shift, the decision loss needs a risk or
  robustness term.
* The Sharpe loss is scale-free, so it does not penalise variance in absolute terms. It was among the worst
  learned models in both modes.
* Group-DRO did not help. Its two groups split the training weeks at median market volatility, so the worst group
  is almost always the high-volatility half. The objective then fits the few crisis weeks that dominate that half.
  We did not tune the group definition.

### 4.6 The preliminary notebook, re-run

`results/tables/notebook_replication_diagnostics.md`, Fig. 8. Both recipes use the notebook's λ = 1 and a 30% cap,
with the same walk-forward protocol and costs as everything else.

| | 10 mega-caps 2017–25 (5 seeds) | 9 sector ETFs 2005–25 (3 seeds) |
|---|---|---|
| Zero-gradient share at start of training: notebook recipe / fixed recipe | 96% / 34% | 98% / 52% |
| Net Sharpe: PF MLP | 1.020 | 0.400 |
| Net Sharpe: DFL, notebook recipe | 1.231 (vs PF +0.21, bootstrap p = 0.03) | 0.452 (vs PF +0.05, p = 0.33) |
| Net Sharpe: DFL, fixed recipe | 1.035 | 0.398 |
| Net Sharpe: MV with historical mean / 1/N | 1.303 / 1.290 | 0.613 / 0.651 |
| Average holdings, notebook recipe | 4.0 of 10 | 4.0 of 9 |

* **The diagnosis is confirmed.** With the notebook's recipe, 96–98% of training dates sit at a vertex with exactly
  zero gradient. In many epochs the gradient norm hits zero (Fig. 8b), which is the flat loss seen in the notebook.
  The fixed recipe reduces this, but at λ = 1 still leaves 34–52% of dates without gradient. λ = 1 is too low for
  weekly data, and this is the main reason the main experiments use λ = 5.
* **The recipe's apparent success on mega-caps is not evidence of learning.** A model that sees a gradient on 2–4%
  of dates ends up as a near-constant corner portfolio: 4 names at the 30% cap. On the hindsight-selected 2017–2025
  mega-caps any concentrated bet on past winners pays. The historical-mean MV portfolio does the same thing and does
  better (1.303), and 1/N does about as well (1.290). On the ex-ante sector universe the recipe's edge over PF
  shrinks to +0.05 and is not significant. Every λ = 1 strategy there trails 1/N by 0.2 Sharpe.
* **What to keep from the notebook:** the data pipeline, the chronological split and the baselines. **What to
  change:** see `docs/NOTEBOOK_REVIEW.md`.

## 5. Answers in brief

* **RQ1.** Lower prediction error fails to give lower regret when the predictor is misspecified, and when data are
  scarce. Well-specified models with ample data are better trained by MSE. Under misspecification DFL lowers regret
  with equal or higher MSE, and MSE becomes a poor guide for choosing between models (Kendall τ with regret falls
  from about 0.75 to about 0.5). On real data DFL has higher MSE and lower IC but better net utility. The
  mechanism is that the decision ignores the forecast level and is locally insensitive at the bounds, so forecast
  errors matter only in the few directions that move the portfolio.
* **RQ2.** Training is stable when the decision map is differentiable on most training dates. That needs:
  * enough risk aversion (λ ≥ 5 weekly) and a position cap of 20–30%;
  * output scaling and a warm start from MSE;
  * early stopping on the decision loss;
  * costs in the loss rather than as an ℓ₁ penalty in the layer, which pins weights at a kink and leaves 55–86% of
    dates without gradient.

  Low λ, or a large-output network, puts most dates at a vertex with zero gradient: 96% for the notebook's recipe.
  Under a regime shift without refitting, plain DFL degrades more than PF, and a CVaR term fixes this.
* **Benchmark.** Against an MSE-trained twin with the same features, model and layer, DFL wins on the long ex-ante
  universes: significantly on FF12 (+0.09 Sharpe) and on utility on sector ETFs. The gain is mostly learned
  cost-awareness, though. Give the two-stage pipeline a turnover penalty in its optimiser and the gap closes on
  sector ETFs and becomes insignificant on FF12. No learned model beats 1/N or a cost-penalised historical-mean MV
  portfolio.
* **RQ3.** Yes. Pure DFL produces biased, poorly calibrated forecasts (calibration slope 0.25 vs 0.72 on sector
  ETFs, −0.23 vs 0.27 on FF12) because the decision is blind to the forecast level and rewards decision-friendly
  dispersion. A hybrid loss with a small MSE weight (β = 100) restores calibration at no measurable cost in utility.

## 6. Limitations

* Predictability is weak and the features are price-only. All learned models trail 1/N net of costs. The
  conclusions are about the *training loss*, holding everything else fixed, not about a profitable strategy.
* The real-data DFL vs PF gains are significant on one of three universes (FF12). On sector ETFs they are
  significant across seeds but not over time. On the hindsight universe they are negative. Against a cost-aware
  two-stage baseline (η > 0) none of them is significant.
* Weekly frequency, long-only, 9–12 assets. Larger universes would need sparse or low-rank Σ̂ in the layer. The
  KKT system is (N+1)×(N+1) per sample, which is fine up to a few hundred assets.
* Σ̂ is estimated, not learned. The layer supports gradients in Σ (tested), but we did not train a covariance model
  end to end.
* The hyper-parameters of DFL (lr, patience, β = 100, κ = 1, ε = 0.003) were set once from a pilot on 2010–2012 and
  not tuned per universe. Validation-based tuning could change the margins in either direction.
* Seeds capture training randomness only. Market randomness is captured by the bootstrap over time, which is the
  test to trust for real-data claims.

## 7. Reproducibility

`python scripts/reproduce_all.py` (Windows, macOS, Linux) or `bash scripts/reproduce_all.sh` re-runs everything from
the cached prices in `data/raw`. It takes about 4 hours on 2 CPU cores. `--quick` (or `QUICK=1` with the bash script)
runs a smoke test in 5-10 minutes. Every configuration is in `configs/` and copied to
`results/<name>/config.json`. Seeds are fixed. `python -m pytest` runs 18 tests: the layer against CVXPY and finite
differences, the features with no look-ahead, the folds with no leakage, and the backtest identities.

## References

* Amos, B. & Kolter, J. Z. (2017). OptNet: Differentiable optimization as a layer in neural networks. *ICML*.
* Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S. & Kolter, J. Z. (2019). Differentiable convex optimization layers. *NeurIPS*.
* Elmachtoub, A. N. & Grigas, P. (2022). Smart "predict, then optimize". *Management Science* 68(1).
* Donti, P., Amos, B. & Kolter, J. Z. (2017). Task-based end-to-end model learning in stochastic optimization. *NeurIPS*.
* Butler, A. & Kwon, R. H. (2023). Integrating prediction in mean-variance portfolio optimization. *Quantitative Finance* 23(3).
* Uysal, A. S., Li, X. & Mulvey, J. M. (2023). End-to-end risk budgeting portfolio optimization with neural networks. *Annals of Operations Research*.
* Costa, G. & Iyengar, G. N. (2023). Distributionally robust end-to-end portfolio construction. *Quantitative Finance* 23(10).
* Mandi, J., Kotary, J., Berden, S., et al. (2024). Decision-focused learning: foundations, state of the art, benchmark and future opportunities. *JAIR* 80.
* Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3).
* Sagawa, S., Koh, P. W., Hashimoto, T. & Liang, P. (2020). Distributionally robust neural networks for group shifts. *ICLR*.
* Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Analysis* 88(2).
* DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies* 22(5).
* Jobson, J. D. & Korkie, B. (1981); Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters* 1.
* Politis, D. N. & Romano, J. P. (1992). A circular block-resampling procedure for stationary data.

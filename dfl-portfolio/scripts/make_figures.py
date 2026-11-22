"""Build every figure and the headline tables from ``results/``.

    python scripts/make_figures.py

Figures go to ``results/figures``; tables (markdown + csv) to ``results/tables``.
Missing experiment folders are skipped with a message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUFFIX = "_quick" if "--quick" in sys.argv else ""  # QUICK=1 smoke runs write to *_quick folders
FIG = RES / ("figures" + SUFFIX)
TAB = RES / ("tables" + SUFFIX)
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---- style: validated reference categorical palette (fixed order) + recessive greys
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GREYS = ["#8a8984", "#b4b3ad", "#52514e"]
SEQ3 = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue ramp (validated --ordinal)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
# diverging blue <-> red with a neutral grey midpoint; sequential = one blue hue
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#c22f2e"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})

# colour follows the entity, everywhere
STYLE = {
    "pf_linear": dict(color=BLUE, ls="--", label="PF linear (MSE)"),
    "pf_mlp": dict(color=BLUE, ls="-", label="PF MLP (MSE)"),
    "dfl_linear": dict(color=ORANGE, ls="--", label="DFL linear"),
    "dfl_mlp": dict(color=ORANGE, ls="-", label="DFL MLP"),
    "hybrid_mlp": dict(color=AQUA, ls="-", label="Hybrid MLP (beta=100)"),
    "equal_weight": dict(color=GREYS[0], ls=":", label="1/N"),
    "min_variance": dict(color=GREYS[1], ls="-", label="Min-variance"),
    "mv_hist_mean": dict(color=GREYS[2], ls="-.", label="MV, historical mean"),
    "dfl_linear_scratch": dict(color=YELLOW, ls="--", label="DFL linear, no warm start"),
}
LABEL = {k: v["label"] for k, v in STYLE.items()}
LABEL.update({
    "dfl_mlp_sharpe": "DFL Sharpe loss", "dfl_mlp_cvar": "DFL + CVaR", "dfl_mlp_dro": "DFL + Wasserstein DRO",
    "dfl_mlp_groupdro": "DFL + group DRO", "dfl_notebook_style": "DFL as in notebook", "dfl_fixed": "DFL (fixed)",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def R(name: str) -> Path:
    """Result folder for an experiment, honouring --quick."""
    return RES / (name + SUFFIX)


def have(*names):
    ok = all((RES / n.replace("/", SUFFIX + "/", 1)).exists() for n in names)
    if not ok:
        print("skipping (missing):", names)
    return ok


def fmt_table(df: pd.DataFrame, name: str, floatfmt: str = ".3f"):
    df.to_csv(TAB / f"{name}.csv")
    multi = isinstance(df.index, pd.MultiIndex)
    out = df.reset_index() if multi else df
    out = out.astype(object).where(out.notna(), None)  # blank cells instead of "nan"
    with open(TAB / f"{name}.md", "w") as fh:
        fh.write(out.to_markdown(floatfmt=floatfmt, index=not multi, missingval=""))
    print("wrote", TAB / f"{name}.md")


# --------------------------------------------------------------------------
# Synthetic: RQ1
# --------------------------------------------------------------------------
def fig_synthetic():
    if not have("synthetic_main/results.csv"):
        return
    df = pd.read_csv(R("synthetic_main") / "results.csv")
    test = df[df.test == "test"]
    ns = sorted(test.n_train.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.6), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        sub = test[test.n_train == n]
        for m in ["pf_linear", "dfl_linear", "pf_mlp", "dfl_mlp"]:
            g = sub[sub.method == m].groupby("degree")["norm_regret"]
            mean, se = g.mean(), g.std() / np.sqrt(g.count())
            st = STYLE[m]
            ax.plot(mean.index, mean.values, color=st["color"], ls=st["ls"], marker="o", ms=4, label=st["label"])
            ax.fill_between(mean.index, mean - se, mean + se, color=st["color"], alpha=0.1, lw=0)
        ax.set_title(f"{n} training dates")
        ax.set_xlabel("degree of misspecification")
        ax.set_xticks(sorted(sub.degree.unique()))
    np.atleast_1d(axes)[0].set_ylabel("normalised regret (0 = oracle, 1 = 1/N)")
    np.atleast_1d(axes)[-1].legend(loc="upper left")
    fig.suptitle("Normalised regret on 2,000 test dates (mean and 95% CI over 20 seeds): DFL helps under "
                 "misspecification and with little data", x=0.01, y=1.03, ha="left")
    save(fig, "fig1_synthetic_regret_vs_degree.png")

    # table: regret and MSE by degree for n=300
    t = (test.groupby(["n_train", "degree", "method"])[["norm_regret", "mse_mu", "rank_ic_mu", "calib_slope"]]
         .mean().unstack("method"))
    fmt_table(t.xs("norm_regret", axis=1, level=0), "synthetic_norm_regret", ".4f")
    # paired DFL - PF difference with t-test per (n, degree)
    rows = []
    for (n, d), sub in test.groupby(["n_train", "degree"]):
        for pf, dfl in [("pf_linear", "dfl_linear"), ("pf_mlp", "dfl_mlp")]:
            a = sub[sub.method == dfl].sort_values("seed").norm_regret.to_numpy()
            b = sub[sub.method == pf].sort_values("seed").norm_regret.to_numpy()
            dd = a - b
            p = stats.ttest_rel(a, b).pvalue if len(dd) > 1 else np.nan
            ma = sub[sub.method == dfl].mse_mu.mean() / sub[sub.method == pf].mse_mu.mean()
            rows.append({"n_train": n, "degree": d, "model": pf.split("_")[1], "regret_PF": b.mean(),
                         "regret_DFL": a.mean(), "DFL_minus_PF": dd.mean(), "p_value": p,
                         "MSE_ratio_DFL_over_PF": ma})
    fmt_table(pd.DataFrame(rows).set_index(["n_train", "degree", "model"]), "synthetic_dfl_vs_pf", ".4f")

    # shift: vol x2
    if "vol_x2" in set(df.test):
        sh = df[df.test == "vol_x2"]
        t2 = sh.groupby(["n_train", "degree", "method"])["norm_regret"].mean().unstack("method")
        fmt_table(t2, "synthetic_vol_shift_norm_regret", ".4f")

    # Fig 2 (RQ1): does a lower MSE identify the better decision?
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    tau = (test.groupby(["degree", "n_train", "seed"])
           .apply(lambda g: stats.kendalltau(g.mse_mu, g.norm_regret).statistic, include_groups=False)
           .groupby(["degree", "n_train"]).agg(["mean", "sem"]))
    for n, color in zip(ns, SEQ3):
        t_ = tau.xs(n, level="n_train")
        axes[0].errorbar(t_.index, t_["mean"], yerr=1.96 * t_["sem"], color=color, marker="o", ms=5, capsize=3,
                         label=f"{n} training dates")
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylim(-0.4, 1.05)
    axes[0].set_xticks(sorted(test.degree.unique()))
    axes[0].set_xlabel("degree of misspecification")
    axes[0].set_ylabel("Kendall tau(MSE, regret) across the 5 models\n(1 = MSE ranks decisions perfectly)")
    axes[0].set_title("How well test MSE ranks the models' decisions", loc="left")
    axes[0].legend(loc="lower left", fontsize=8)
    pairs = paired_dfl_vs_pf(test)
    degs = sorted(pairs.degree.unique())
    xpos = np.arange(len(degs))
    width = 0.26
    for k, (n, color) in enumerate(zip(ns, SEQ3)):
        g = pairs[pairs.n_train == n].groupby("degree")["d_regret"].agg(["mean", "sem", "size"]).reindex(degs)
        axes[1].bar(xpos + (k - 1) * width, g["mean"], width=width * 0.92, color=color, yerr=1.96 * g["sem"],
                    ecolor=INK2, capsize=2, label=f"{n} training dates")
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks(xpos, [str(d) for d in degs])
    axes[1].set_xlabel("degree of misspecification")
    axes[1].set_ylabel("regret(DFL) - regret(PF)\n(below 0 = DFL better)")
    npairs = int(pairs.groupby(["n_train", "degree"]).size().iloc[0])
    axes[1].set_title(f"Paired difference, mean and 95% CI ({npairs} pairs per bar)", loc="left")
    axes[1].grid(axis="x", visible=False)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig2_mse_vs_regret.png")

    pooled = pairs.groupby("degree").apply(lambda g: pd.Series({
        "pairs": len(g), "mean_regret_diff": g.d_regret.mean(), "ci95": 1.96 * g.d_regret.sem(),
        "share_dfl_wins": g.dfl_wins.mean(), "share_higher_mse_lower_regret": g.paradox.mean(),
        "t_test_p": stats.ttest_1samp(g.d_regret, 0).pvalue, "wilcoxon_p": stats.wilcoxon(g.d_regret).pvalue,
    }), include_groups=False)
    fmt_table(pooled, "synthetic_pooled_dfl_vs_pf", ".4f")
    by_n = pairs.groupby(["n_train", "degree"]).apply(lambda g: pd.Series({
        "pairs": len(g), "mean_regret_diff": g.d_regret.mean(), "share_dfl_wins": g.dfl_wins.mean(),
        "mse_ratio_dfl_over_pf": g.mse_ratio.median(), "t_test_p": stats.ttest_1samp(g.d_regret, 0).pvalue,
    }), include_groups=False)
    fmt_table(by_n, "synthetic_dfl_vs_pf_by_n", ".4f")
    fmt_table(tau["mean"].unstack("n_train"), "synthetic_kendall_mse_regret", ".3f")


def paired_dfl_vs_pf(test: pd.DataFrame) -> pd.DataFrame:
    """One row per (degree, n_train, seed, model family): DFL minus PF."""
    rows = []
    for (d, n, seed), g in test.groupby(["degree", "n_train", "seed"]):
        g = g.set_index("method")
        for pf, dfl in [("pf_linear", "dfl_linear"), ("pf_mlp", "dfl_mlp")]:
            if pf in g.index and dfl in g.index:
                dr = g.loc[dfl, "norm_regret"] - g.loc[pf, "norm_regret"]
                dm = g.loc[dfl, "mse_mu"] / g.loc[pf, "mse_mu"]
                rows.append({"degree": d, "n_train": n, "seed": seed, "model": pf.split("_")[1], "d_regret": dr,
                             "mse_ratio": dm, "dfl_wins": dr < 0, "paradox": (dm > 1) and (dr < 0)})
    return pd.DataFrame(rows)


def fig_beta():
    """RQ3: hybrid loss L = decision loss + beta * MSE, from beta = 0 (pure DFL) to beta = inf (pure PF)."""
    if not have("synthetic_beta/results.csv"):
        return
    df = pd.read_csv(R("synthetic_beta") / "results.csv")
    df = df[df.test == "test"]
    order = ["dfl_linear", "hyb_b1", "hyb_b10", "hyb_b100", "hyb_b1000", "hyb_b10000", "pf_linear"]
    betas = ["0\n(DFL)", "1", "10", "100", "1e3", "1e4", "inf\n(PF)"]
    degs = sorted(df.degree.unique())
    colors = [AQUA, YELLOW]
    slope = "calib_slope_mu" if "calib_slope_mu" in df else "calib_slope"
    bias = "bias_mu" if "bias_mu" in df else "bias"
    ref, ref_sym = ("true mu", "mu") if slope.endswith("_mu") else ("realised returns", "r")
    # paired against pure PF (same seed = same data), so seed-to-seed noise cancels
    pf = df[df.method == "pf_linear"].set_index(["degree", "seed"])["norm_regret"]
    df = df.assign(d_regret=df.norm_regret.to_numpy() - pf.loc[list(zip(df.degree, df.seed))].to_numpy())
    panels = [("d_regret", "regret minus regret of pure PF\n(below 0 = better decisions than PF)", 1),
              (slope, f"calibration slope vs {ref} (1 = calibrated)", 1),
              (bias, f"forecast bias, mean(mu_hat - {ref_sym})  (x 1e-4)", 1e4)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = np.arange(len(order))
    for d, color in zip(degs, colors):
        g = df[df.degree == d].groupby("method")
        for ax, (col, yl, mult) in zip(axes, panels):
            m = g[col].mean().loc[order] * mult
            se = g[col].sem().loc[order] * mult
            ax.errorbar(x, m, yerr=1.96 * se, color=color, marker="o", ms=5, capsize=3, label=f"degree {d}")
    for ax, (col, yl, mult) in zip(axes, panels):
        ax.set_xticks(x, betas, fontsize=8)
        ax.set_xlabel("beta (weight on the MSE term)")
        ax.set_ylabel(yl)
    axes[0].axhline(0.0, color=MUTED, lw=1)
    axes[1].axhline(1.0, color=MUTED, lw=1)
    axes[2].axhline(0.0, color=MUTED, lw=1)
    axes[2].legend(fontsize=8, loc="lower right")
    axes[0].set_title("Decision quality", loc="left")
    axes[1].set_title("Calibration", loc="left")
    axes[2].set_title("Level of the forecasts", loc="left")
    fig.suptitle("Hybrid loss from pure DFL (beta = 0) to pure PF (beta = inf): synthetic market, "
                 "300 training dates, mean and 95% CI over seeds", x=0.01, y=1.03, ha="left")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig3_beta_frontier.png")
    t = df.groupby(["degree", "method"])[["norm_regret", "d_regret", "mse_mu", slope, bias, "rank_ic_mu",
                                          "pred_sd"]].mean()
    t = t.reindex(pd.MultiIndex.from_product([degs, order], names=["degree", "method"]))
    fmt_table(t, "synthetic_beta", ".5f")


def fig_synthetic_lambda():
    if not have("synthetic_lambda/results.csv"):
        return
    df = pd.read_csv(R("synthetic_lambda") / "results.csv")
    dg = pd.read_csv(R("synthetic_lambda") / "train_diagnostics.csv")
    rows = []
    for (lam, up), sub in df.groupby(["lam", "upper"]):
        for test in ["test", "vol_x2"]:
            s = sub[sub.test == test]
            a = s[s.method == "dfl_linear"].sort_values("seed").norm_regret.to_numpy()
            b = s[s.method == "pf_linear"].sort_values("seed").norm_regret.to_numpy()
            rows.append({"lam": lam, "upper": up, "test": test, "gain": (b - a).mean(),
                         "dfl_seed_sd": a.std(ddof=1), "pf_regret": b.mean(), "dfl_regret": a.mean()})
    g = pd.DataFrame(rows)
    zg = dg[dg.method == "dfl_linear"].groupby(["lam", "upper"])["zero_grad_first"].mean()
    lams, ups = sorted(g.lam.unique()), sorted(g.upper.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    for ax, test, title in [(axes[0], "test", "(a) Regret reduction, same distribution"),
                            (axes[1], "vol_x2", "(b) Regret reduction, test volatility x2")]:
        M = g[g.test == test].pivot(index="upper", columns="lam", values="gain").loc[ups, lams].to_numpy()
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * lim else INK)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    Z = zg.unstack("lam").loc[ups, lams].to_numpy()
    im = axes[2].imshow(Z, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            axes[2].text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if Z[i, j] > 0.6 else INK)
    axes[2].set_title("(c) Zero-gradient share at start of DFL training")
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.05, pad=0.03)
    for ax in axes:
        ax.set_xticks(range(len(lams)), [f"{x:g}" for x in lams])
        ax.set_yticks(range(len(ups)), [f"{x:g}" for x in ups])
        ax.set_xlabel("risk aversion lambda")
        ax.set_ylabel("position cap")
    fig.suptitle("Where DFL helps: regret(PF) - regret(DFL), red = DFL better, blue = PF better "
                 "(synthetic, degree 6, 300 training dates, mean of 5 seeds)", x=0.01, y=1.04, ha="left")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig9_synthetic_lambda_cap.png")
    fmt_table(g.set_index(["lam", "upper", "test"]), "synthetic_lambda_cap", ".4f")


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
MAIN_COLS = {"sharpe_mean": "Sharpe", "sharpe_std": "Sharpe sd (seeds)", "gross_sharpe_mean": "Gross Sharpe", "ann_return_mean": "Ann. return",
             "ann_vol_mean": "Ann. vol", "max_drawdown_mean": "Max DD", "cvar95_weekly_mean": "CVaR95 (wk)",
             "turnover_ann_mean": "Turnover / yr", "utility_weekly_mean": "Utility (wk)",
             "mse_mean": "MSE (x1e-4)", "rank_ic_mean": "Rank IC", "calib_slope_mean": "Calib. slope"}


def real_tables(name: str):
    if not have(f"{name}/summary.csv"):
        return None
    s = pd.read_csv(R(name) / "summary.csv", index_col=0)
    s = s.copy()
    if "mse_mean" in s:
        s["mse_mean"] = s["mse_mean"] * 1e4
    t = s[[c for c in MAIN_COLS if c in s]].rename(columns=MAIN_COLS)
    t.index = [LABEL.get(i, i) for i in t.index]
    fmt_table(t, f"{name}_summary", ".4f")
    if (R(name) / "tests.csv").exists():
        tests = pd.read_csv(R(name) / "tests.csv")
        tests["method"] = tests.method.map(lambda x: LABEL.get(x, x))
        tests["vs"] = tests.vs.map(lambda x: LABEL.get(x, x))
        fmt_table(tests.set_index(["method", "vs"]), f"{name}_tests", ".4f")
    return s


def overview_table():
    """One table across universes: the numbers quoted in the README."""
    rows = []
    for name, uni in [("sector9_main", "Sector ETFs 2005-25"), ("ff12_main", "FF12 industries 1996-25"),
                      ("megacap10_main", "10 mega-caps 2017-25")]:
        if not R(name).exists() or not (R(name) / "summary.csv").exists():
            continue
        sm = pd.read_csv(R(name) / "summary.csv", index_col=0)
        ts = pd.read_csv(R(name) / "tests.csv").set_index("method")
        for m in ["equal_weight", "mv_hist_mean", "pf_mlp", "dfl_mlp", "hybrid_mlp"]:
            if m not in sm.index:
                continue
            r = {"universe": uni, "method": LABEL.get(m, m), "net Sharpe": sm.loc[m, "sharpe_mean"],
                 "gross Sharpe": sm.loc[m, "gross_sharpe_mean"], "turnover/yr": sm.loc[m, "turnover_ann_mean"],
                 "utility (ann.)": sm.loc[m, "utility_weekly_mean"] * 52,
                 "calib. slope": sm.loc[m, "calib_slope_mean"] if "calib_slope_mean" in sm else np.nan}
            if m in ts.index:
                r.update({"dSharpe vs PF": ts.loc[m, "sharpe_diff_seedavg"],
                          "95% CI": f"[{ts.loc[m, 'boot_ci_low']:+.2f}, {ts.loc[m, 'boot_ci_high']:+.2f}]",
                          "bootstrap p": ts.loc[m, "boot_p"], "JKM p": ts.loc[m, "jkm_p"]})
            rows.append(r)
    if rows:
        fmt_table(pd.DataFrame(rows).set_index(["universe", "method"]), "real_data_overview", ".3f")


def eta_table():
    """Does DFL's real-data edge survive a turnover penalty in the layer (a cost-aware two-stage baseline)?"""
    rows = []
    for name, uni, eta in [("ff12_main", "FF12 industries 1996-25", 0.0), ("ff12_eta", "FF12 industries 1996-25", 0.001)]:
        if not (R(name) / "metrics_per_seed.csv").exists():
            continue
        ps = pd.read_csv(R(name) / "metrics_per_seed.csv")
        ts = pd.read_csv(R(name) / "tests.csv").set_index("method")
        for m in ["pf_mlp", "dfl_mlp"]:
            g = ps[ps.method == m]
            r = {"universe": uni, "eta": eta, "method": LABEL.get(m, m), "seeds": len(g),
                 "net Sharpe": g.sharpe.mean(), "utility (ann.)": g.utility_weekly.mean() * 52,
                 "turnover/yr": g.turnover_ann.mean()}
            if m == "dfl_mlp" and m in ts.index:
                r.update({"dSharpe vs PF": ts.loc[m, "sharpe_diff_seedavg"], "bootstrap p": ts.loc[m, "boot_p"],
                          "seed utility p": ts.loc[m, "seed_utility_p"]})
            rows.append(r)
    abl = R("ablation_sector9")
    for eta in [0.0, 0.0005, 0.001, 0.002]:
        d = abl / f"eta={eta}_walkforward"
        if not (d / "metrics_per_seed.csv").exists():
            continue
        ps = pd.read_csv(d / "metrics_per_seed.csv")
        ts = pd.read_csv(d / "tests.csv").set_index("method")
        for m in ["pf_mlp", "dfl_mlp"]:
            g = ps[ps.method == m]
            r = {"universe": "Sector ETFs 2005-25", "eta": eta, "method": LABEL.get(m, m), "seeds": len(g),
                 "net Sharpe": g.sharpe.mean(), "utility (ann.)": g.utility_weekly.mean() * 52,
                 "turnover/yr": g.turnover_ann.mean()}
            if m == "dfl_mlp" and m in ts.index:
                r.update({"dSharpe vs PF": ts.loc[m, "sharpe_diff_seedavg"], "bootstrap p": ts.loc[m, "boot_p"],
                          "seed utility p": ts.loc[m, "seed_utility_p"]})
            rows.append(r)
    if rows:
        fmt_table(pd.DataFrame(rows).set_index(["universe", "eta", "method"]), "turnover_penalty_dfl_vs_pf", ".4f")


def fig_wealth(name: str, title: str, methods, fname: str):
    if not have(f"{name}/weekly_returns.csv"):
        return
    wk = pd.read_csv(R(name) / "weekly_returns.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for m in methods:
        sub = wk[wk.method == m]
        if sub.empty:
            continue
        r = sub.groupby("date")["net"].mean()  # equal capital in every seed's strategy
        wealth = np.cumprod(1 + r)
        st = STYLE.get(m, dict(color=MUTED, ls="-", label=m))
        ax.plot(wealth.index, wealth.values, color=st["color"], ls=st["ls"], lw=1.6 if m in STYLE else 1.2,
                label=st["label"])
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("net wealth (log scale, start = 1)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    # bars: Sharpe mean +/- seed sd
    ps = pd.read_csv(R(name) / "metrics_per_seed.csv")
    g = ps.groupby("method")["sharpe"].agg(["mean", "std"]).reindex(methods).dropna(how="all")
    ax = axes[1]
    y = np.arange(len(g))
    colors = [STYLE.get(m, {}).get("color", MUTED) for m in g.index]
    ax.barh(y, g["mean"], height=0.6, color=colors, xerr=g["std"].fillna(0), ecolor=INK2, capsize=3)
    ax.set_yticks(y, [LABEL.get(m, m) for m in g.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("net Sharpe ratio (mean +/- sd over seeds)")
    ax.grid(axis="y", visible=False)
    for yi, v, e in zip(y, g["mean"], g["std"].fillna(0)):
        ax.text(v + e + 0.015, yi, f"{v:.2f}", fontsize=8, color=INK2, va="center")
    ax.set_xlim(0, (g["mean"] + g["std"].fillna(0)).max() * 1.15)
    save(fig, fname)


def fig_regimes(name: str, fname: str, methods):
    if not have(f"{name}/metrics_per_seed.csv"):
        return
    ps = pd.read_csv(R(name) / "metrics_per_seed.csv")
    g = ps.groupby("method")[["calm_utility", "turbulent_utility", "calm_sharpe", "turbulent_sharpe"]].agg(
        ["mean", "std"]).reindex(methods).dropna(how="all")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    y = np.arange(len(g))
    for ax, regime in zip(axes, ["calm", "turbulent"]):
        col = f"{regime}_utility"
        colors = [STYLE.get(m, {}).get("color", MUTED) for m in g.index]
        ax.barh(y, g[(col, "mean")] * 52, height=0.6, color=colors, xerr=g[(col, "std")].fillna(0) * 52,
                ecolor=INK2, capsize=3)
        ax.set_yticks(y, [LABEL.get(m, m) for m in g.index], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{regime} weeks")
        ax.set_xlabel("annualised mean-variance utility (net)")
        ax.grid(axis="y", visible=False)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("Performance by market regime (turbulent = trailing market vol above its 75th percentile)",
                 x=0.01, y=1.03, ha="left")
    save(fig, fname)


def fig_ablation():
    if not have("ablation_sector9/ablation_summary.csv"):
        return
    a = pd.read_csv(R("ablation_sector9") / "ablation_summary.csv")
    # zero-gradient share at the selected (best) epoch; the first epoch has no previous decision, so with eta > 0
    # only the best-epoch value reflects the turnover term
    zg = []
    for _, r in a.iterrows():
        f = R("ablation_sector9") / f"{r.knob}={r.value}_{r['mode']}" / "train_diagnostics.csv"
        if f.exists() and r.method == "dfl_mlp":
            d = pd.read_csv(f)
            zg.append(d[d.method == "dfl_mlp"].zero_grad_best.mean())
        else:
            zg.append(np.nan)
    a["zero_grad_best"] = zg
    knobs = [("lam", "risk aversion lambda"), ("upper", "position cap"), ("eta", "turnover penalty eta")]
    fig, axes = plt.subplots(4, 3, figsize=(13, 11.5))
    for j, (k, xl) in enumerate(knobs):
        sub = a[a.knob == k]
        for mode, ls in [("walkforward", "-"), ("frozen", "--")]:
            s = sub[sub["mode"] == mode]
            pf = s[s.method == "pf_mlp"].set_index("value")
            dfl = s[s.method == "dfl_mlp"].set_index("value")
            x = np.arange(len(pf))
            axes[0, j].plot(x, (dfl.utility_mean - pf.utility_mean) * 52, color=ORANGE, ls=ls, marker="o", ms=4,
                            label=f"DFL - PF, {mode}")
            axes[1, j].plot(x, dfl.sharpe_mean - pf.sharpe_mean, color=ORANGE, ls=ls, marker="o", ms=4,
                            label=f"DFL - PF, {mode}")
            axes[2, j].plot(x, pf.turnover, color=BLUE, ls=ls, marker="o", ms=4, label=f"PF, {mode}")
            axes[2, j].plot(x, dfl.turnover, color=ORANGE, ls=ls, marker="o", ms=4, label=f"DFL, {mode}")
            axes[3, j].plot(x, dfl.zero_grad_best, color=ORANGE, ls=ls, marker="o", ms=4, label=f"DFL, {mode}")
            for ax in axes[:, j]:
                ax.set_xticks(x, [f"{v:g}" for v in pf.index])
        axes[0, j].axhline(0, color=MUTED, lw=1)
        axes[1, j].axhline(0, color=MUTED, lw=1)
        axes[3, j].set_xlabel(xl)
    axes[0, 0].set_ylabel("utility gain, DFL - PF\n(annualised)")
    axes[1, 0].set_ylabel("net Sharpe gain, DFL - PF")
    axes[2, 0].set_ylabel("turnover per year")
    axes[3, 0].set_ylabel("share of training dates with\nzero gradient (selected epoch)")
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle("Ablation of the portfolio layer (sector ETFs; solid = walk-forward, dashed = frozen model)",
                 x=0.01, y=1.03, ha="left")
    fig.tight_layout()
    save(fig, "fig6_ablation_layer.png")
    fmt_table(a.set_index(["knob", "value", "mode", "method"]), "ablation_sector9", ".4f")


def fig_notebook_diagnostics():
    name = "megacap10_notebook_replication"
    if not have(f"{name}/histories.json"):
        return
    with open(R(name) / "histories.json") as fh:
        hist = json.load(fh)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    nb_color, fix_color = MAGENTA, ORANGE
    for key, h in hist.items():
        for m, color in [("dfl_notebook_style", nb_color), ("dfl_fixed", fix_color)]:
            if m not in h or not h[m]["frac_zero_grad"]:
                continue
            axes[0].plot(h[m]["frac_zero_grad"], color=color, alpha=0.3, lw=1)
            axes[1].plot(np.array(h[m]["grad_norm"]) + 1e-12, color=color, alpha=0.3, lw=1)
    for ax in axes[:2]:
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("share of training dates with zero gradient")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) Vertex solutions block the gradient", loc="left")
    axes[1].set_ylabel("gradient norm")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Gradient size", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=nb_color, label="notebook recipe (return loss, unscaled init)"),
               Line2D([], [], color=fix_color, label="fixed recipe (utility loss, scaled init, warm start)")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    # (c) net Sharpe on the hindsight universe vs an ex-ante universe
    methods = ["equal_weight", "mv_hist_mean", "pf_mlp", "dfl_notebook_style", "dfl_fixed"]
    colors = {"equal_weight": GREYS[0], "mv_hist_mean": GREYS[2], "pf_mlp": BLUE, "dfl_notebook_style": nb_color,
              "dfl_fixed": fix_color}
    unis = [(name, "10 mega-caps\n(hindsight)"), ("sector9_notebook_replication", "9 sector ETFs\n(ex ante)")]
    ax = axes[2]
    width = 0.16
    shown = False
    for k, (u, ulabel) in enumerate(unis):
        f = R(u) / "metrics_per_seed.csv"
        if not f.exists():
            continue
        ps = pd.read_csv(f)
        g = ps.groupby("method").agg(sh=("sharpe", "mean"), sd=("sharpe", "std"), hold=("avg_n_holdings", "mean"))
        for j, m in enumerate(methods):
            if m not in g.index:
                continue
            x = k + (j - 2) * width
            ax.bar(x, g.loc[m, "sh"], width=width * 0.9, color=colors[m], yerr=np.nan_to_num(g.loc[m, "sd"]),
                   ecolor=INK2, capsize=2, label=LABEL.get(m, m) if not shown else None)
        shown = True
    ax.set_xticks(range(len(unis)), [u[1] for u in unis])
    ax.set_ylabel("net Sharpe ratio (lambda = 1, cap 30%)")
    ax.set_title("(c) Out-of-sample result depends on the universe", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    fig.suptitle("The notebook's decision-focused set-up (both recipes at lambda = 1, 10 mega-caps): "
                 "96% of training dates give no gradient",
                 x=0.01, y=1.03, ha="left")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig8_notebook_diagnostics.png")
    rows = []
    for u, ulabel in unis:
        f = R(u) / "train_diagnostics.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        t = d.groupby("method")[["zero_grad_first", "zero_grad_best", "mean_free_first", "grad_norm_first",
                                 "epochs_run", "best_epoch"]].mean()
        ps = pd.read_csv(R(u) / "metrics_per_seed.csv").groupby("method")[
            ["sharpe", "utility_weekly", "turnover_ann", "avg_n_holdings", "avg_max_weight"]].mean()
        t = ps.join(t, how="left")
        t["universe"] = ulabel.replace("\n", " ")
        rows.append(t.reset_index())
    if rows:
        fmt_table(pd.concat(rows).set_index(["universe", "method"]), "notebook_replication_diagnostics", ".3f")


def fig_robust():
    rows = []
    for name, mode in [("robust_sector9", "walk-forward"), ("robust_sector9_frozen", "frozen (trained 2000-05)")]:
        if not have(f"{name}/metrics_per_seed.csv"):
            continue
        ps = pd.read_csv(R(name) / "metrics_per_seed.csv")
        g = ps.groupby("method").agg(
            sharpe=("sharpe", "mean"), sharpe_sd=("sharpe", "std"), utility=("utility_weekly", "mean"),
            utility_sd=("utility_weekly", "std"), turb_utility=("turbulent_utility", "mean"),
            max_dd=("max_drawdown", "mean"), cvar95=("cvar95_weekly", "mean"),
            turnover=("turnover_ann", "mean"))
        g["mode"] = mode
        rows.append(g)
    if not rows:
        return
    t = pd.concat(rows).reset_index().set_index(["mode", "method"])
    fmt_table(t, "robust_extension", ".4f")
    methods = ["pf_mlp", "dfl_mlp", "dfl_mlp_sharpe", "dfl_mlp_cvar", "dfl_mlp_dro", "dfl_mlp_groupdro"]
    modes = [("walk-forward", "o", "walk-forward (refit yearly, 2007-25)"),
             ("frozen (trained 2000-05)", "D", "frozen (trained 2000-05, never refit)")]
    panels = [("sharpe", "net Sharpe ratio  (higher = better)", 1),
              ("turb_utility", "utility in turbulent weeks, annualised\n(higher = better)", 52),
              ("cvar95", "CVaR 95%, weekly loss  (lower = better)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    y = np.arange(len(methods))
    for ax, (col, xl, mult) in zip(axes, panels):
        vals = {}
        for mode, marker, _ in modes:
            if mode in t.index.get_level_values(0):
                vals[mode] = t.loc[mode].reindex(methods)[col].to_numpy() * mult
        if len(vals) == 2:
            a, b = vals[modes[0][0]], vals[modes[1][0]]
            for yi, (x0, x1) in enumerate(zip(a, b)):
                ax.plot([x0, x1], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        for mode, marker, label in modes:
            if mode not in vals:
                continue
            colors = [STYLE.get(m, {}).get("color", ORANGE) if m in STYLE else ORANGE for m in methods]
            face = colors if marker == "o" else [SURFACE] * len(methods)
            ax.scatter(vals[mode], y, s=60, marker=marker, facecolors=face, edgecolors=colors, linewidths=1.8,
                       zorder=3, label=label)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABEL.get(m, m) for m in methods], fontsize=9)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=mk, ls="", color=INK2, markerfacecolor=INK2 if mk == "o" else SURFACE,
                      markersize=7, label=lab) for _, mk, lab in modes]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Extension: risk-aware and robust decision losses on sector ETFs (mean over 5 seeds)",
                 x=0.01, y=1.03, ha="left")
    save(fig, "fig7_robust_extension.png")


def fig_yearly(name, fname):
    if not have(f"{name}/yearly_returns.csv"):
        return
    y = pd.read_csv(R(name) / "yearly_returns.csv", index_col=0)
    if not {"dfl_mlp", "pf_mlp"} <= set(y.index):
        return
    diff = (y.loc["dfl_mlp"] - y.loc["pf_mlp"]) * 100
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = [ORANGE if v >= 0 else BLUE for v in diff.values]
    ax.bar(diff.index.astype(str), diff.values, color=colors, width=0.6)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("DFL MLP - PF MLP\nnet return (pp)")
    ax.set_title("Year-by-year difference, sector ETFs (orange = DFL ahead, blue = PF ahead)", loc="left")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    save(fig, fname)


if __name__ == "__main__":
    fig_synthetic()
    fig_beta()
    fig_synthetic_lambda()
    main_methods = ["equal_weight", "min_variance", "mv_hist_mean", "pf_linear", "dfl_linear", "pf_mlp", "dfl_mlp",
                    "hybrid_mlp"]
    for name, title in [("sector9_main", "Sector ETFs, walk-forward 2005-2025 (net of 10 bps costs)"),
                        ("megacap10_main", "Mega-cap stocks, walk-forward 2017-2025 (net of 10 bps costs)"),
                        ("ff12_main", "FF12 industries, walk-forward 1996-2025 (net of 10 bps costs)")]:
        real_tables(name)
        fig_wealth(name, title, main_methods, f"fig4_wealth_{name}.png")
    overview_table()
    eta_table()
    real_tables("ff12_eta")
    real_tables("megacap10_notebook_replication")
    real_tables("sector9_notebook_replication")
    real_tables("robust_sector9")
    real_tables("robust_sector9_frozen")
    fig_regimes("sector9_main", "fig5_regimes_sector9.png", main_methods)
    fig_yearly("sector9_main", "fig10_yearly_diff_sector9.png")
    fig_ablation()
    fig_notebook_diagnostics()
    fig_robust()

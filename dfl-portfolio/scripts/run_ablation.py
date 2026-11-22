"""Research question 2: one-knob-at-a-time ablation of the portfolio layer on real data.

For every value in ``settings`` the base configuration is re-run as a walk-forward
and as a frozen model (trained once before the 2007-09 crisis, never refit).

    python scripts/run_ablation.py configs/ablation_sector9.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dfl_portfolio.config import deep_merge, load_config  # noqa: E402
from dfl_portfolio.experiments import prepare_panel, run_walkforward  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--skip-frozen", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    settings = cfg.pop("settings")
    panel = prepare_panel(cfg["universe"], cfg.get("cov_window", 104), cfg.get("hist_window", 156))
    out_root = ROOT / "results" / "ablation_sector9"
    rows = []
    for key, values in settings.items():
        for v in values:
            for mode in (["walkforward"] if args.skip_frozen else ["walkforward", "frozen"]):
                sub = deep_merge(cfg, {"layer": {key: v}})
                if mode == "frozen":
                    sub["frozen"] = True
                    sub["walkforward"]["first_test_year"] = 2007
                    sub["seeds"] = 5
                tag = f"{key}={v}_{mode}"
                print("==", tag, flush=True)
                ev = run_walkforward(sub, out_root / tag, processes=args.processes, panel=panel, verbose=False)
                ps = ev["per_seed"]
                diag = pd.read_csv(out_root / tag / "train_diagnostics.csv")
                for method in ["pf_mlp", "dfl_mlp", "equal_weight", "mv_hist_mean"]:
                    sel = ps[ps.method == method]
                    dsel = diag[diag.method == method]
                    rows.append({
                        "knob": key, "value": v, "mode": mode, "method": method,
                        "sharpe_mean": sel.sharpe.mean(), "sharpe_std": sel.sharpe.std(),
                        "utility_mean": sel.utility_weekly.mean(), "utility_std": sel.utility_weekly.std(),
                        "turnover": sel.turnover_ann.mean(), "max_dd": sel.max_drawdown.mean(),
                        "turbulent_utility": sel.turbulent_utility.mean(),
                        "zero_grad_first": dsel.zero_grad_first.mean() if len(dsel) else float("nan"),
                        "mean_free_first": dsel.mean_free_first.mean() if len(dsel) else float("nan"),
                        "best_epoch": dsel.best_epoch.mean() if len(dsel) else float("nan"),
                    })
                pd.DataFrame(rows).to_csv(out_root / "ablation_summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())

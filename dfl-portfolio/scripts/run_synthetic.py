"""Synthetic experiments (misspecification grid, beta sweep, lambda x cap grid).

    python scripts/run_synthetic.py configs/synthetic.yaml
    python scripts/run_synthetic.py configs/synthetic_beta.yaml
    python scripts/run_synthetic.py configs/synthetic_lambda.yaml   # loops over 'settings'
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dfl_portfolio.config import deep_merge, load_config  # noqa: E402
from dfl_portfolio.synthetic_experiments import run_synthetic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["seeds"] = 1
        cfg["degrees"] = cfg["degrees"][:2]
        cfg["n_train"] = cfg["n_train"][:1]
        cfg["name"] += "_quick"
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    settings = cfg.pop("settings", None)
    all_rows, all_diags = [], []
    if settings:
        keys = list(settings)
        for values in itertools.product(*[settings[k] for k in keys]):
            sub = deep_merge(cfg, {"layer": dict(zip(keys, values))})
            sub["layer"].setdefault("lower", 0.0)
            print("setting", dict(zip(keys, values)), flush=True)
            rows, diags = run_synthetic(sub, processes=args.processes, verbose=False)
            for k, v in zip(keys, values):
                rows[k], diags[k] = v, v
            all_rows.append(rows)
            all_diags.append(diags)
    else:
        rows, diags = run_synthetic(cfg, processes=args.processes)
        all_rows, all_diags = [rows], [diags]
    rows, diags = pd.concat(all_rows), pd.concat(all_diags)
    rows.to_csv(out / "results.csv", index=False)
    diags.to_csv(out / "train_diagnostics.csv", index=False)
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    group = [c for c in ["lam", "upper", "degree", "n_train", "test"] if c in rows]
    pd.set_option("display.width", 200)
    print(rows.groupby(group + ["method"])["norm_regret"].mean().unstack("method").round(4).to_string())
    print(f"results written to {out}")

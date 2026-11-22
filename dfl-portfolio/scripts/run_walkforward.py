"""Walk-forward backtest for one configuration.

    python scripts/run_walkforward.py configs/sector9.yaml
    python scripts/run_walkforward.py configs/robust_sector9.yaml --frozen
    python scripts/run_walkforward.py configs/sector9.yaml --quick     # 1 seed, 3 folds (smoke test)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dfl_portfolio.config import load_config  # noqa: E402
from dfl_portfolio.experiments import run_walkforward  # noqa: E402

COLS = ["sharpe_mean", "sharpe_std", "ann_return_mean", "ann_vol_mean", "max_drawdown_mean",
        "turnover_ann_mean", "utility_weekly_mean", "mse_mean", "rank_ic_mean"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--frozen", action="store_true", help="train once on the first fold, never refit")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.frozen:
        cfg["frozen"] = True
        cfg["name"] = cfg["name"] + "_frozen"
    if args.quick:
        cfg["seeds"] = 1
        y = cfg["walkforward"]["first_test_year"]
        cfg["walkforward"]["last_test_year"] = y + 2
        cfg["name"] = cfg["name"] + "_quick"
    out = Path(args.out) if args.out else ROOT / "results" / cfg["name"]
    ev = run_walkforward(cfg, out, processes=args.processes)
    pd.set_option("display.width", 200)
    print(ev["summary"][[c for c in COLS if c in ev["summary"]]].round(4).to_string())
    print(ev["tests"].round(4).to_string())
    print(f"results written to {out}")

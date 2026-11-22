"""Download and cache daily prices for the pre-defined universes.

    python scripts/download_data.py                 # all universes (keeps cached files)
    python scripts/download_data.py sector9 ff12    # a subset
    python scripts/download_data.py --force         # re-download (Yahoo data can change slightly)

The repository ships the cached CSVs used for the reported results, so by default
nothing is re-downloaded and the results reproduce exactly.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dfl_portfolio.data import UNIVERSES, default_data_dir, download_universe, load_weekly_returns  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("universes", nargs="*", default=list(UNIVERSES))
    ap.add_argument("--force", action="store_true", help="re-download even if a cached file exists")
    args = ap.parse_args()
    for u in args.universes:
        path = Path(default_data_dir()) / f"{u}_daily.csv"
        if args.force or not path.exists():
            path = download_universe(u)
        r = load_weekly_returns(u)
        print(f"{u:10s} -> {path}  weekly {r.shape}  {r.index.min().date()} .. {r.index.max().date()}")

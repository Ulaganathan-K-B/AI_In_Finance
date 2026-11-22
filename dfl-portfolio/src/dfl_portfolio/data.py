"""Data download, caching and weekly-return construction.

Three universes are pre-defined:

* ``sector9``   - the nine original SPDR sector ETFs (inception Dec-1998).  No
  survivorship bias: the universe was fixed in 1998, not chosen with hindsight.
* ``megacap10`` - the ten large-cap stocks used in the preliminary notebook.
  Kept for continuity; it is hindsight-selected (see README).
* ``ff12``      - Kenneth French's 12 value-weighted industry portfolios
  (daily, from 1926).  Used as a long-history robustness check.

Prices are cached as CSV files in ``data/raw`` so every experiment can be
re-run offline once the download has happened.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

UNIVERSES: Dict[str, dict] = {
    "sector9": {
        "source": "yahoo",
        "tickers": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
        "start": "1999-01-01",
        "end": "2026-01-01",
        "description": "Nine original SPDR sector ETFs",
    },
    "megacap10": {
        "source": "yahoo",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "JNJ", "KO"],
        "start": "2013-01-01",
        "end": "2026-01-01",
        "description": "Ten mega-cap stocks from the preliminary notebook (hindsight-selected)",
    },
    "ff12": {
        "source": "french",
        "dataset": "12_Industry_Portfolios_daily",
        "start": "1990-01-01",
        "end": "2026-01-01",
        "description": "Fama-French 12 value-weighted industry portfolios",
    },
}

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"


def default_data_dir() -> Path:
    """Repository-level ``data/raw`` directory."""
    return Path(__file__).resolve().parents[2] / "data" / "raw"


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _download_yahoo(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="all")


def _download_french(name: str, start: str, end: str) -> pd.DataFrame:
    """Download a Ken French daily portfolio file; return a total-return price index."""
    import urllib.request

    url = FRENCH_URL.format(name=name)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    # The first table in the file is "Average Value Weighted Returns -- Daily".
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(",") and "NoDur" in line)
    rows = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    columns = [c.strip() for c in lines[header_idx].split(",")][1:]
    frame = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=columns).astype(float)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.replace([-99.99, -999.0], np.nan) / 100.0
    frame = frame.loc[start:end]
    return (1.0 + frame.fillna(0.0)).cumprod()


def download_universe(universe: str, data_dir: Optional[Path] = None) -> Path:
    """Download daily prices for ``universe`` and cache them as CSV."""
    spec = UNIVERSES[universe]
    data_dir = Path(data_dir or default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    if spec["source"] == "yahoo":
        prices = _download_yahoo(spec["tickers"], spec["start"], spec["end"])
    elif spec["source"] == "french":
        prices = _download_french(spec["dataset"], spec["start"], spec["end"])
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown source {spec['source']}")
    path = data_dir / f"{universe}_daily.csv"
    prices.to_csv(path)
    return path


# --------------------------------------------------------------------------
# Loading / transformation
# --------------------------------------------------------------------------
def load_daily_prices(universe: str, data_dir: Optional[Path] = None, download_if_missing: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir or default_data_dir())
    path = data_dir / f"{universe}_daily.csv"
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found; run scripts/download_data.py first")
        download_universe(universe, data_dir)
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """Load any wide CSV of prices (date index, one column per asset)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def to_weekly_returns(daily_prices: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """Week-ending (Friday) simple returns from daily prices.

    Assets with missing history at the start are dropped from the leading rows
    only (rows with any NaN at the beginning are removed), so that the panel is
    balanced.  This is appropriate for the fixed universes used here.
    """
    weekly = daily_prices.resample(rule).last()
    returns = weekly.pct_change()
    returns = returns.iloc[1:]
    first_full = returns.dropna().index.min()
    returns = returns.loc[first_full:]
    if returns.isna().any().any():
        returns = returns.ffill().fillna(0.0)
    return returns


def load_weekly_returns(universe: str, data_dir: Optional[Path] = None, download_if_missing: bool = True) -> pd.DataFrame:
    return to_weekly_returns(load_daily_prices(universe, data_dir, download_if_missing))

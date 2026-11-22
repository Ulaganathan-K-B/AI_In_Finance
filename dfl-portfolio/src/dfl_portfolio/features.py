"""Feature construction for cross-sectional return prediction.

Every feature for decision date ``t`` uses returns up to and including week
``t`` only.  The prediction target for sample ``t`` is the return of week
``t + 1``.  Features are computed per asset, plus a few market-wide features
that are appended to every asset (so one shared model can serve all assets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

ASSET_FEATURES = [
    "ret_1w",
    "mom_4w",
    "mom_13w",
    "mom_26w",
    "mom_52w_skip4",
    "vol_4w",
    "vol_13w",
    "vol_52w",
    "rel_mom_13w",
    "rel_mom_52w",
]
MARKET_FEATURES = ["mkt_mom_4w", "mkt_mom_13w", "mkt_vol_13w"]
FEATURE_NAMES = ASSET_FEATURES + MARKET_FEATURES
MAX_LOOKBACK = 52


def _cum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return np.exp(np.log1p(returns).rolling(window).sum()) - 1.0


def build_features(returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return ``(X, Y, valid, dates)``.

    ``X`` has shape (T, N, P), ``Y[t] = returns[t + 1]`` has shape (T, N) and
    ``valid[t]`` is True when all look-backs are available and the target exists.
    """
    r = returns.astype(float)
    feats = {
        "ret_1w": r,
        "mom_4w": _cum(r, 4),
        "mom_13w": _cum(r, 13),
        "mom_26w": _cum(r, 26),
        # 12-1 style momentum: 52-week return excluding the most recent 4 weeks
        "mom_52w_skip4": _cum(r.shift(4), 48),
        "vol_4w": r.rolling(4).std(),
        "vol_13w": r.rolling(13).std(),
        "vol_52w": r.rolling(52).std(),
    }
    feats["rel_mom_13w"] = feats["mom_13w"].sub(feats["mom_13w"].mean(axis=1), axis=0)
    feats["rel_mom_52w"] = feats["mom_52w_skip4"].sub(feats["mom_52w_skip4"].mean(axis=1), axis=0)

    market = r.mean(axis=1)
    mkt = pd.DataFrame(
        {
            "mkt_mom_4w": np.exp(np.log1p(market).rolling(4).sum()) - 1.0,
            "mkt_mom_13w": np.exp(np.log1p(market).rolling(13).sum()) - 1.0,
            "mkt_vol_13w": market.rolling(13).std(),
        }
    )

    T, N = r.shape
    X = np.zeros((T, N, len(FEATURE_NAMES)), dtype=np.float64)
    for k, name in enumerate(ASSET_FEATURES):
        X[:, :, k] = feats[name].to_numpy()
    for j, name in enumerate(MARKET_FEATURES):
        X[:, :, len(ASSET_FEATURES) + j] = mkt[name].to_numpy()[:, None]

    Y = np.full((T, N), np.nan)
    Y[:-1] = r.to_numpy()[1:]

    valid = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(Y).all(axis=1)
    return X, Y, valid, r.index


@dataclass
class FeatureScaler:
    """Pooled (over assets and dates) z-scoring fitted on training rows only."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray, clip: float = 5.0) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.clip(Z, -clip, clip)


def feature_names() -> List[str]:
    return list(FEATURE_NAMES)

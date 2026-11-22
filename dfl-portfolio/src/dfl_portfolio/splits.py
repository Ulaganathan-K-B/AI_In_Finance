"""Walk-forward (expanding window) splits with an embargo.

Sample ``t`` is a decision made at the end of week ``t`` whose label is the
return of week ``t + 1``.  With ``embargo >= 1`` the last training label
(week ``t_train + 1``) always precedes the first validation decision date, and
the same holds between validation and test.  Nothing is ever shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def summary(self, dates: pd.DatetimeIndex) -> str:
        def rng(idx):
            return f"{dates[idx[0]].date()}..{dates[idx[-1]].date()} (n={len(idx)})"

        return f"{self.name}: train {rng(self.train)} | val {rng(self.val)} | test {rng(self.test)}"


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    valid: np.ndarray,
    first_test_year: int,
    last_test_year: int | None = None,
    val_weeks: int = 52,
    embargo: int = 1,
    min_train: int = 156,
    step_years: int = 1,
) -> List[Fold]:
    """Expanding-window folds; each test block is ``step_years`` calendar years."""
    dates = pd.DatetimeIndex(dates)
    positions = np.flatnonzero(valid)
    years = dates[positions].year
    last_test_year = last_test_year or int(years.max())
    folds: List[Fold] = []
    for year in range(first_test_year, last_test_year + 1, step_years):
        in_test = (years >= year) & (years < year + step_years)
        test = positions[in_test]
        if len(test) == 0:
            continue
        test_start = test[0]
        val_end = test_start - embargo  # exclusive
        val = positions[(positions < val_end) & (positions >= val_end - val_weeks)]
        if len(val) == 0:
            continue
        train_end = val[0] - embargo
        train = positions[positions < train_end]
        if len(train) < min_train:
            continue
        label = str(year) if step_years == 1 else f"{year}-{year + step_years - 1}"
        folds.append(Fold(name=label, train=train, val=val, test=test))
    return folds


def check_no_leakage(fold: Fold) -> bool:
    """Labels (t+1) of an earlier block must not reach the next block's decisions."""
    return bool(fold.train.max() + 1 < fold.val.min() and fold.val.max() + 1 < fold.test.min())

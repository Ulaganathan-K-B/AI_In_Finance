r"""Synthetic markets with a controllable degree of model misspecification.

Following Elmachtoub & Grigas (2022), expected returns are a polynomial of a
linear index of asset features:

.. math::

    z_{t,i} = x_{t,i}^\top b / \lVert b \rVert, \qquad
    \mu_{t,i} = m + s\,\frac{(z_{t,i} + 3)^{d} - c_d}{s_d},

where ``d`` is the *degree*.  ``d = 1`` is exactly linear (a linear predictor is
well specified); larger ``d`` makes the truth increasingly convex, so a linear
predictor becomes misspecified.  ``c_d`` and ``s_d`` standardise the
polynomial so the cross-sectional signal has standard deviation ``s`` for every
degree, which keeps the signal-to-noise ratio fixed while only the *shape*
changes.

Returns are ``r_t = mu_t + beta f_t + e_t`` with a market factor ``f`` and
idiosyncratic noise ``e`` (Gaussian or Student-t), so the true covariance is
known: ``Sigma = sigma_f^2 beta beta' + sigma_e^2 I``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SyntheticSpec:
    n_assets: int = 10
    n_features: int = 5
    degree: int = 1
    signal_sd: float = 0.01
    mean: float = 0.002
    factor_vol: float = 0.02
    idio_vol: float = 0.03
    noise: str = "normal"  # normal | t
    t_df: float = 5.0
    structure_seed: int = 12345


def _poly_moments(degree: int) -> tuple[float, float]:
    z = np.random.default_rng(0).standard_normal(2_000_000)
    h = (z + 3.0) ** degree
    return float(h.mean()), float(h.std())


class SyntheticMarket:
    def __init__(self, spec: SyntheticSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.structure_seed)
        self.b = rng.standard_normal(spec.n_features)
        self.b /= np.linalg.norm(self.b)
        self.betas = rng.uniform(0.5, 1.5, spec.n_assets)
        self.c_d, self.s_d = _poly_moments(spec.degree)

    def sigma(self, vol_scale: float = 1.0) -> np.ndarray:
        s = self.spec
        cov = s.factor_vol ** 2 * np.outer(self.betas, self.betas) + s.idio_vol ** 2 * np.eye(s.n_assets)
        return cov * vol_scale ** 2

    def mu_of(self, X: np.ndarray, degree: int | None = None) -> np.ndarray:
        s = self.spec
        if degree is None or degree == s.degree:
            c_d, s_d, d = self.c_d, self.s_d, s.degree
        else:
            (c_d, s_d), d = _poly_moments(degree), degree
        z = X @ self.b
        return s.mean + s.signal_sd * ((z + 3.0) ** d - c_d) / s_d

    def sample(self, T: int, seed: int, vol_scale: float = 1.0, degree: int | None = None) -> Dict[str, np.ndarray]:
        s = self.spec
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((T, s.n_assets, s.n_features))
        MU = self.mu_of(X, degree)
        if s.noise == "t":
            scale = np.sqrt((s.t_df - 2.0) / s.t_df)
            f = rng.standard_t(s.t_df, T) * scale
            e = rng.standard_t(s.t_df, (T, s.n_assets)) * scale
        else:
            f = rng.standard_normal(T)
            e = rng.standard_normal((T, s.n_assets))
        noise = (s.factor_vol * f[:, None] * self.betas[None, :] + s.idio_vol * e) * vol_scale
        Y = MU + noise
        Sigma = np.broadcast_to(self.sigma(vol_scale), (T, s.n_assets, s.n_assets)).copy()
        return {"X": X, "Y": Y, "MU": MU, "Sigma": Sigma}

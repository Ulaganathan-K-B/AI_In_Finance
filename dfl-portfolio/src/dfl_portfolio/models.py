"""Return predictors.

All models are *cross-sectional* and share parameters across assets: the same
function maps an asset's feature vector to its predicted next-week return,
optionally plus a learned per-asset intercept.  Sharing parameters makes the
model far more sample-efficient than one dense network over all assets (the
preliminary notebook used 120 inputs -> 10 outputs on ~400 samples).

Inputs are standardised features; outputs are in return units through a fixed
``output_scale`` (the training-target standard deviation), so the network
itself works with O(1) numbers.
"""

from __future__ import annotations

import torch
from torch import nn


class SharedLinear(nn.Module):
    def __init__(self, n_features: int, n_assets: int, output_scale: float = 0.03, asset_bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(n_features, 1)
        nn.init.normal_(self.lin.weight, std=1e-3)
        nn.init.zeros_(self.lin.bias)
        self.asset_bias = nn.Parameter(torch.zeros(n_assets)) if asset_bias else None
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, P)
        out = self.lin(x).squeeze(-1)
        if self.asset_bias is not None:
            out = out + self.asset_bias
        return out * self.output_scale


class SharedMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden=(32, 16),
        dropout: float = 0.1,
        output_scale: float = 0.03,
        asset_bias: bool = True,
        skip_linear: bool = True,
        small_init: bool = True,
    ):
        super().__init__()
        layers = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        # Linear skip connection ("wide & deep"): the MLP only has to learn the
        # non-linear part on top of a linear model.
        self.skip = nn.Linear(n_features, 1, bias=False) if skip_linear else None
        if small_init:
            # Small output init: initial predictions are close to zero, so the
            # portfolio layer starts in its interior instead of at a vertex.
            nn.init.normal_(self.head.weight, std=1e-3)
            nn.init.zeros_(self.head.bias)
            if self.skip is not None:
                nn.init.normal_(self.skip.weight, std=1e-3)
        self.asset_bias = nn.Parameter(torch.zeros(n_assets)) if asset_bias else None
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.body(x)).squeeze(-1)
        if self.skip is not None:
            out = out + self.skip(x).squeeze(-1)
        if self.asset_bias is not None:
            out = out + self.asset_bias
        return out * self.output_scale


def build_model(kind: str, n_features: int, n_assets: int, output_scale: float, **kwargs) -> nn.Module:
    if kind == "linear":
        return SharedLinear(n_features, n_assets, output_scale=output_scale,
                            asset_bias=kwargs.get("asset_bias", True))
    if kind == "mlp":
        return SharedMLP(
            n_features,
            n_assets,
            hidden=tuple(kwargs.get("hidden", (32, 16))),
            dropout=kwargs.get("dropout", 0.1),
            output_scale=output_scale,
            asset_bias=kwargs.get("asset_bias", True),
            skip_linear=kwargs.get("skip_linear", True),
            small_init=kwargs.get("small_init", True),
        )
    raise ValueError(f"unknown model kind {kind!r}")

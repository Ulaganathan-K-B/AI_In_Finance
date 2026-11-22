"""Training loops: prediction-focused (MSE), decision-focused and hybrid.

Both trainers share features, architecture and the portfolio layer; only the
training objective differs.  That is the comparison the project asks for.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .layers import PortfolioLayer
from .losses import decision_loss, realised_returns, regret


@dataclass
class Split:
    """Tensors for one block of consecutive decision dates."""

    X: torch.Tensor  # (S, N, P) standardised features, float32
    Y: torch.Tensor  # (S, N) next-period returns, float32
    Sigma: torch.Tensor  # (S, N, N) covariance estimates, float64
    groups: Optional[torch.Tensor] = None  # (S,) regime labels (group DRO)
    mu_true: Optional[torch.Tensor] = None  # (S, N) synthetic only: true expected returns

    def __len__(self) -> int:
        return self.X.shape[0]


@dataclass
class TrainConfig:
    loss: str = "mse"  # mse | utility | sharpe | return | cvar | dro | groupdro | regret
    beta: float = 0.0  # weight of MSE in a hybrid decision loss
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    batch_size: Optional[int] = 64  # dates per mini-batch (MSE only); DFL is full-batch, chronological
    min_epochs: int = 20  # no early stopping before this epoch
    grad_clip: float = 1.0
    lam: float = 10.0  # risk aversion of the decision loss (defaults to the layer's)
    cost: float = 0.001  # proportional cost per unit turnover in the training loss
    kappa: float = 1.0
    alpha: float = 0.9
    radius: float = 0.0
    early_stop: bool = True  # select the epoch with the best validation objective
    seed: int = 0


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_metric: List[float] = field(default_factory=list)
    frac_zero_grad: List[float] = field(default_factory=list)
    mean_free: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    best_epoch: int = 0

    def as_dict(self) -> Dict[str, list]:
        return {k: getattr(self, k) for k in ["train_loss", "val_metric", "frac_zero_grad", "mean_free", "grad_norm"]}


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(X)


# --------------------------------------------------------------------------
# Prediction-focused
# --------------------------------------------------------------------------
def fit_mse(model: nn.Module, train: Split, val: Split, cfg: TrainConfig) -> History:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = History()
    best, best_state, wait = np.inf, copy.deepcopy(model.state_dict()), 0
    S = len(train)
    bs = cfg.batch_size or S
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(S, generator=gen)
        losses = []
        for start in range(0, S, bs):
            idx = perm[start: start + bs]
            opt.zero_grad()
            loss = ((model(train.X[idx]) - train.Y[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        val_mse = float(((predict(model, val.X) - val.Y) ** 2).mean())
        hist.train_loss.append(float(np.mean(losses)))
        hist.val_metric.append(val_mse)
        if val_mse < best - 1e-12:
            best, best_state, wait, hist.best_epoch = val_mse, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= cfg.patience and epoch >= cfg.min_epochs:
                break
    model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------
# Decision-focused (and hybrid)
# --------------------------------------------------------------------------
class DecisionObjective:
    """Evaluates a decision loss for a split; keeps ADMM warm starts between epochs."""

    def __init__(self, layer: PortfolioLayer, cfg: TrainConfig):
        self.layer = layer
        self.cfg = cfg
        self.warm: Dict[str, tuple] = {}
        self.w_prev: Dict[str, torch.Tensor] = {}

    def __call__(self, name: str, mu: torch.Tensor, split: Split):
        cfg = self.cfg
        w_prev = self.w_prev.get(name) if self.layer.eta > 0 else None
        w, info = self.layer(mu, split.Sigma, w_prev=w_prev, warm=self.warm.get(name))
        self.warm[name] = info.warm
        if self.layer.eta > 0:
            # previous decision for the turnover term = last epoch's decision one
            # step earlier (a fixed point at convergence; exact at test time)
            shifted = torch.cat([w[:1].detach(), w[:-1].detach()], dim=0)
            self.w_prev[name] = shifted
        R = realised_returns(w, split.Y, cost=cfg.cost)
        if cfg.loss == "regret":
            with torch.no_grad():
                w_or, _ = self.layer(split.Y.to(torch.float64), split.Sigma)
            loss = regret(w, w_or, split.Y, split.Sigma, cfg.lam)
        else:
            loss = decision_loss(cfg.loss, R, w, cfg.lam, groups=split.groups, kappa=cfg.kappa,
                                 alpha=cfg.alpha, radius=cfg.radius)
        if cfg.beta > 0:
            loss = loss + cfg.beta * ((mu - split.Y.to(mu.dtype)) ** 2).mean()
        return loss, w, info


def fit_decision(model: nn.Module, train: Split, val: Split, layer: PortfolioLayer, cfg: TrainConfig) -> History:
    """Full-batch, chronological decision-focused training with early stopping.

    Early stopping uses the same decision objective on the validation block.
    """
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    objective = DecisionObjective(layer, cfg)
    hist = History()
    best, best_state, wait = np.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        mu = model(train.X).to(torch.float64)
        loss, _, info = objective("train", mu, train)
        loss.backward()
        hist.grad_norm.append(_grad_norm(model))
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        hist.train_loss.append(loss.item())
        hist.frac_zero_grad.append(info.frac_zero_grad)
        hist.mean_free.append(float(info.n_free.double().mean()))

        with torch.no_grad():
            model.eval()
            mu_val = model(val.X).to(torch.float64)
            val_loss, _, _ = objective("val", mu_val, val)
            # beta part excluded from model selection: select on the decision objective
            if cfg.beta > 0:
                val_loss = val_loss - cfg.beta * ((mu_val - val.Y.to(torch.float64)) ** 2).mean()
        v = float(val_loss)
        hist.val_metric.append(v)
        if not cfg.early_stop:
            best_state, hist.best_epoch = copy.deepcopy(model.state_dict()), epoch
            continue
        if v < best - 1e-12:
            best, best_state, wait, hist.best_epoch = v, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= cfg.patience and epoch >= cfg.min_epochs:
                break
    model.load_state_dict(best_state)
    return hist

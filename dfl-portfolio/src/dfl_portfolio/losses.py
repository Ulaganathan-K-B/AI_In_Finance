"""Decision losses on realised portfolio returns (all are minimised).

``R`` is the vector of realised (net-of-cost) portfolio returns of one batch,
in chronological order.  Mean-variance utility needs a variance, which one
realised return cannot provide, so every loss here is defined over a batch of
decision dates (the full training window in our full-batch training).
"""

from __future__ import annotations

from typing import Optional

import torch


def realised_returns(w: torch.Tensor, y: torch.Tensor, cost: float = 0.0, w_before: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Portfolio returns ``w_t' r_{t+1}`` minus proportional costs on ``|w_t - w_{t-1}|``.

    Rows are consecutive decision dates, so ``w_{t-1}`` is the previous row
    (drift is ignored during training; the backtest accounts for it).
    """
    gross = (w * y.to(w.dtype)).sum(-1)
    if cost <= 0:
        return gross
    prev = torch.cat([(w_before if w_before is not None else w[:1]).detach(), w[:-1]], dim=0)
    return gross - cost * (w - prev).abs().sum(-1)


def neg_utility(R: torch.Tensor, lam: float) -> torch.Tensor:
    return -(R.mean() - lam * R.var(unbiased=False))


def neg_sharpe(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -R.mean() / (R.std(unbiased=False) + eps)


def cvar(losses: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """Mean of the worst ``1 - alpha`` share of ``losses`` (Rockafellar-Uryasev at the optimum)."""
    k = max(1, int(round((1.0 - alpha) * losses.numel())))
    return torch.topk(losses, k).values.mean()


def cvar_loss(R: torch.Tensor, kappa: float = 1.0, alpha: float = 0.9) -> torch.Tensor:
    """Risk-constrained objective: ``-mean(R) + kappa * CVaR_alpha(-R)``."""
    return -R.mean() + kappa * cvar(-R, alpha)


def dro_utility(R: torch.Tensor, w: torch.Tensor, lam: float, radius: float) -> torch.Tensor:
    """Utility robust to adversarial return perturbations ``||delta_t||_2 <= radius``.

    ``sup_{||d|| <= eps} -w'(r + d) = -w'r + eps ||w||_2`` (dual norm), so the
    worst case of the mean term adds ``radius * mean_t ||w_t||_2``, which
    penalises concentrated decisions (a Wasserstein-type robustification).
    """
    return neg_utility(R, lam) + radius * w.norm(dim=-1).mean()


def group_dro_utility(R: torch.Tensor, groups: torch.Tensor, lam: float) -> torch.Tensor:
    """Worst-group utility loss across regimes (e.g. calm vs turbulent weeks)."""
    losses = []
    for g in torch.unique(groups):
        Rg = R[groups == g]
        if Rg.numel() >= 4:
            losses.append(neg_utility(Rg, lam))
    return torch.stack(losses).max()


def regret(w_hat: torch.Tensor, w_oracle: torch.Tensor, y: torch.Tensor, Sigma: torch.Tensor, lam: float) -> torch.Tensor:
    """Per-sample regret ``U(w*(r); r) - U(w_hat; r)`` with ``U(w; r) = r'w - lam w'Sigma w``."""
    y = y.to(w_hat.dtype)

    def U(w):
        quad = torch.einsum("bi,bij,bj->b", w, Sigma.to(w.dtype), w)
        return (w * y).sum(-1) - lam * quad

    return (U(w_oracle) - U(w_hat)).mean()


def decision_loss(
    kind: str,
    R: torch.Tensor,
    w: torch.Tensor,
    lam: float,
    groups: Optional[torch.Tensor] = None,
    kappa: float = 1.0,
    alpha: float = 0.9,
    radius: float = 0.0,
) -> torch.Tensor:
    if kind == "utility":
        return neg_utility(R, lam)
    if kind == "sharpe":
        return neg_sharpe(R)
    if kind == "return":  # the preliminary notebook's loss (mean return only)
        return -R.mean()
    if kind == "cvar":
        return cvar_loss(R, kappa=kappa, alpha=alpha)
    if kind == "dro":
        return dro_utility(R, w, lam, radius)
    if kind == "groupdro":
        if groups is None:
            raise ValueError("groupdro needs regime labels")
        return group_dro_utility(R, groups, lam)
    raise ValueError(f"unknown decision loss {kind!r}")

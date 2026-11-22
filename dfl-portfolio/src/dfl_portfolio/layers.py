r"""Fast, exact, differentiable mean-variance portfolio layer.

We solve, for every sample ``b`` in a batch,

.. math::

    w^\star_b = \arg\min_w \tfrac12 w^\top Q_b w - c_b^\top w
                 + \eta \lVert w - p_b \rVert_1
    \quad \text{s.t.}\quad \mathbf 1^\top w = 1,\; \ell \le w \le u,

with :math:`Q_b = 2\lambda\Sigma_b` and :math:`c_b = \hat\mu_b`.  This is the
handbook's layer :math:`\max_w \hat\mu^\top w - \lambda w^\top\hat\Sigma w
- \eta\lVert w-w_{t-1}\rVert_1` written as a minimisation.

How it works
------------
1. **Forward (no gradient).**  A batched ADMM splits the problem into an
   equality-constrained quadratic step (a pre-factorised KKT solve) and a
   separable proximal step (soft-threshold around ``p`` then clip to the box).
2. **Polish.**  From the ADMM solution we read off the *state* of each
   coordinate - at the lower bound, at the upper bound, pinned at ``p`` (the
   kink of the turnover term) or free - and refine it with a few primal-dual
   active-set iterations until the KKT conditions hold exactly.
3. **Backward.**  With the state fixed, :math:`w^\star` is the solution of a
   linear system (stationarity rows for free coordinates, ``w_i = bound`` rows
   for fixed ones, and the budget row).  We solve that system with
   ``torch.linalg.solve`` *with* autograd enabled, so the gradient PyTorch
   produces is exactly the implicit-function-theorem gradient of the KKT
   conditions (OptNet / cvxpylayers), at a fraction of the cost.

The unit tests compare the forward pass with CVXPY and the gradient with
cvxpylayers, finite differences and the closed form of the equality-only case.

A coordinate pinned at a bound or at ``p`` has zero derivative with respect to
``c``; if fewer than two coordinates are free, the whole decision is locally
constant in ``c`` and the gradient vanishes.  ``info["n_free"]`` reports this,
which is the diagnostic used for research question 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

LO, HI, AT_P, FREE_BELOW, FREE_ABOVE, FREE = 0, 1, 2, 3, 4, 5
DTYPE = torch.float64


def _as_batch(x, B: int, n: int, like: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=DTYPE, device=like.device)
    if x.dim() == 0:
        return x.expand(B, n).clone()
    if x.dim() == 1:
        return x.expand(B, n).clone()
    return x


# --------------------------------------------------------------------------
# ADMM forward pass
# --------------------------------------------------------------------------
@torch.no_grad()
def admm_solve(
    Q: torch.Tensor,
    c: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    eta: torch.Tensor,
    p: torch.Tensor,
    rho: float = 1.0,
    alpha: float = 1.6,
    max_iter: int = 4000,
    eps: float = 1e-6,
    check_every: int = 10,
    warm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Batched ADMM on a scaled copy of the problem.  Returns ``(z, u, iters)``."""
    B, n = c.shape
    scale = torch.diagonal(Q, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)  # (B,)
    Qs = Q / scale[:, None, None]
    cs = c / scale[:, None]
    etas = eta / scale  # (B,)

    eye = torch.eye(n, dtype=DTYPE, device=c.device)
    K = torch.zeros(B, n + 1, n + 1, dtype=DTYPE, device=c.device)
    K[:, :n, :n] = Qs + rho * eye
    K[:, :n, n] = 1.0
    K[:, n, :n] = 1.0
    Kinv = torch.linalg.inv(K)
    A = Kinv[:, :n, :n]
    a = Kinv[:, :n, n]

    if warm is not None:
        z, u = warm[0].clone(), warm[1].clone()
    else:
        z = torch.clamp(torch.full_like(c, 1.0 / n), lo, hi)
        u = torch.zeros_like(c)
    thr = (etas / rho)[:, None]
    it = 0
    for it in range(1, max_iter + 1):
        q = cs + rho * (z - u)
        x = torch.bmm(A, q.unsqueeze(-1)).squeeze(-1) + a
        xh = alpha * x + (1.0 - alpha) * z
        v = xh + u
        d = v - p
        v = p + torch.sign(d) * torch.clamp(d.abs() - thr, min=0.0)
        z_new = torch.maximum(torch.minimum(v, hi), lo)
        u = u + xh - z_new
        if it % check_every == 0:
            r_prim = (x - z_new).abs().amax()
            r_dual = (rho * (z_new - z)).abs().amax()
            z = z_new
            if r_prim < eps and r_dual < eps:
                break
        else:
            z = z_new
    return z, u, it


# --------------------------------------------------------------------------
# Polishing: primal-dual active set on the coordinate states
# --------------------------------------------------------------------------
def _initial_state(z, lo, hi, p, eta_pos, tol):
    state = torch.full(z.shape, FREE, dtype=torch.long, device=z.device)
    with_eta = eta_pos[:, None].expand_as(z)
    state = torch.where(with_eta & (z < p), torch.full_like(state, FREE_BELOW), state)
    state = torch.where(with_eta & (z >= p), torch.full_like(state, FREE_ABOVE), state)
    near_p = with_eta & ((z - p).abs() <= tol) & (p > lo + tol) & (p < hi - tol)
    state = torch.where(near_p, torch.full_like(state, AT_P), state)
    state = torch.where(z <= lo + tol, torch.full_like(state, LO), state)
    state = torch.where(z >= hi - tol, torch.full_like(state, HI), state)
    return state


def _nu_interval(Q, c, lo, hi, eta, p, state, w):
    """Range of budget multipliers compatible with the fixed coordinates' sign conditions."""
    e = eta[:, None].expand_as(w)
    h = torch.bmm(Q, w.unsqueeze(-1)).squeeze(-1) - c
    inf = torch.full_like(w, float("inf"))
    low = torch.where((state == LO) & (lo < p), e - h, -inf)
    low = torch.maximum(low, torch.where((state == LO) & (lo >= p), -e - h, -inf))
    low = torch.maximum(low, torch.where(state == AT_P, -e - h, -inf))
    high = torch.where((state == HI) & (hi > p), -e - h, inf)
    high = torch.minimum(high, torch.where((state == HI) & (hi <= p), e - h, inf))
    high = torch.minimum(high, torch.where(state == AT_P, e - h, inf))
    return low.amax(-1), high.amin(-1)


def _solve_state(Q, c, lo, hi, eta, p, state):
    """Solve the linear KKT system implied by ``state`` (differentiable).

    Free coordinates contribute stationarity rows, fixed ones ``w_i = value``
    rows, and the last row is the budget.  If *every* coordinate is fixed (a
    vertex, or "no trade" under a turnover penalty) the budget multiplier is
    not identified by the system; we then pin it to the middle of its feasible
    interval, which leaves ``w`` (and its zero gradient) unchanged.
    """
    B, n = c.shape
    fixed = (state == LO) | (state == HI) | (state == AT_P)
    fixed_val = torch.where(state == LO, lo, torch.where(state == HI, hi, p))
    sigma = torch.where(state == FREE_BELOW, 1.0, torch.where(state == FREE_ABOVE, -1.0, 0.0)).to(DTYPE)
    all_fixed = fixed.all(dim=-1)

    eye = torch.eye(n, dtype=DTYPE, device=c.device).expand(B, n, n)
    ones_col = torch.ones(B, n, 1, dtype=DTYPE, device=c.device)
    zeros_col = torch.zeros(B, n, 1, dtype=DTYPE, device=c.device)
    free_rows = torch.cat([Q, ones_col], dim=-1)
    fixed_rows = torch.cat([eye, zeros_col], dim=-1)
    top = torch.where(fixed[:, :, None], fixed_rows, free_rows)
    budget_row = torch.cat([torch.ones(B, 1, n, dtype=DTYPE, device=c.device),
                            torch.zeros(B, 1, 1, dtype=DTYPE, device=c.device)], dim=-1)
    nu_row = torch.cat([torch.zeros(B, 1, n, dtype=DTYPE, device=c.device),
                        torch.ones(B, 1, 1, dtype=DTYPE, device=c.device)], dim=-1)
    bottom = torch.where(all_fixed[:, None, None], nu_row, budget_row)
    M = torch.cat([top, bottom], dim=1)

    last = torch.ones(B, dtype=DTYPE, device=c.device)
    if bool(all_fixed.any()):
        with torch.no_grad():
            wf = torch.where(fixed, fixed_val, torch.zeros_like(fixed_val))
            low, high = _nu_interval(Q.detach(), c.detach(), lo, hi, eta, p, state, wf)
            nu0 = torch.where(torch.isfinite(low) & torch.isfinite(high), 0.5 * (low + high),
                              torch.where(torch.isfinite(low), low, torch.where(torch.isfinite(high), high, 0.0)))
        last = torch.where(all_fixed, nu0, last)
    rhs_top = torch.where(fixed, fixed_val, c + eta[:, None] * sigma)
    rhs = torch.cat([rhs_top, last[:, None]], dim=1)
    y = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
    return y[:, :n], y[:, n]


@torch.no_grad()
def _update_state(Q, c, lo, hi, eta, p, state, w, nu, tol_w, tol_g):
    """One primal-dual active-set update.  Returns ``(new_state, violated)``."""
    e = eta[:, None].expand_as(w)
    g = torch.bmm(Q, w.unsqueeze(-1)).squeeze(-1) - c + nu[:, None]
    new = state.clone()

    # Primal violations of free coordinates.
    free_any = (state == FREE) | (state == FREE_BELOW) | (state == FREE_ABOVE)
    below_lo = free_any & (w < lo - tol_w)
    above_hi = free_any & (w > hi + tol_w)
    cross_up = (state == FREE_BELOW) & (w > p + tol_w)
    cross_down = (state == FREE_ABOVE) & (w < p - tol_w)
    p_interior = (p > lo + tol_w) & (p < hi - tol_w)
    new = torch.where(cross_up | cross_down, torch.where(p_interior, torch.full_like(new, AT_P), new), new)
    new = torch.where(below_lo, torch.full_like(new, LO), new)
    new = torch.where(above_hi, torch.full_like(new, HI), new)

    # Dual violations of fixed coordinates.
    lo_below_p = lo < p
    viol_lo = (state == LO) & torch.where(lo_below_p, g - e < -tol_g, g + e < -tol_g)
    hi_above_p = hi > p
    viol_hi = (state == HI) & torch.where(hi_above_p, g + e > tol_g, g - e > tol_g)
    viol_p_down = (state == AT_P) & (g > e + tol_g)
    viol_p_up = (state == AT_P) & (g < -e - tol_g)

    has_eta = e > 0
    free_from_lo = torch.where(has_eta, torch.where(lo_below_p, FREE_BELOW, FREE_ABOVE), FREE)
    free_from_hi = torch.where(has_eta, torch.where(hi_above_p, FREE_ABOVE, FREE_BELOW), FREE)
    new = torch.where(viol_lo, free_from_lo, new)
    new = torch.where(viol_hi, free_from_hi, new)
    new = torch.where(viol_p_down, torch.full_like(new, FREE_BELOW), new)
    new = torch.where(viol_p_up, torch.full_like(new, FREE_ABOVE), new)

    violated = (
        below_lo | above_hi | cross_up | cross_down | viol_lo | viol_hi | viol_p_down | viol_p_up
    ).any(dim=-1)
    violated = violated | ((w.sum(-1) - 1.0).abs() > 10 * tol_w)
    return new, violated


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------
@dataclass
class LayerInfo:
    n_free: torch.Tensor  # number of free coordinates per sample
    valid: torch.Tensor  # polish succeeded (exact KKT point, exact gradient)
    admm_iters: int
    warm: Tuple[torch.Tensor, torch.Tensor]

    @property
    def frac_zero_grad(self) -> float:
        """Share of samples whose decision is locally constant in mu (<= 1 free coordinate)."""
        return float((self.n_free <= 1).double().mean())


def solve_box_qp(
    Q: torch.Tensor,
    c: torch.Tensor,
    lo,
    hi,
    eta=0.0,
    p: Optional[torch.Tensor] = None,
    warm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rho: float = 1.0,
    max_iter: int = 4000,
    eps: float = 1e-6,
    polish_iters: int = 25,
) -> Tuple[torch.Tensor, LayerInfo]:
    """Solve the batch of QPs; differentiable with respect to ``Q`` and ``c``."""
    if c.dim() == 1:
        w, info = solve_box_qp(Q.unsqueeze(0) if Q.dim() == 2 else Q, c.unsqueeze(0), lo, hi, eta,
                               None if p is None else p.reshape(1, -1), warm, rho, max_iter, eps, polish_iters)
        return w[0], info
    c = c.to(DTYPE)
    B, n = c.shape
    Q = Q.to(DTYPE)
    if Q.dim() == 2:
        Q = Q.expand(B, n, n)
    lo_t = _as_batch(lo, B, n, c)
    hi_t = _as_batch(hi, B, n, c)
    eta_t = torch.as_tensor(eta, dtype=DTYPE, device=c.device)
    eta_t = eta_t.expand(B).clone() if eta_t.dim() == 0 else eta_t
    p_t = torch.zeros(B, n, dtype=DTYPE, device=c.device) if p is None else p.to(DTYPE).detach()

    Qd, cd = Q.detach(), c.detach()
    z, u, iters = admm_solve(Qd, cd, lo_t, hi_t, eta_t, p_t, rho=rho, max_iter=max_iter, eps=eps, warm=warm)

    scale_w = 1e-7
    scale_g = 1e-10 + 1e-7 * (cd.abs().amax(-1, keepdim=True) + torch.diagonal(Qd, dim1=-2, dim2=-1).amax(-1, keepdim=True))
    state = _initial_state(z, lo_t, hi_t, p_t, eta_t > 0, tol=1e-6)
    valid = torch.zeros(B, dtype=torch.bool, device=c.device)
    with torch.no_grad():
        for _ in range(polish_iters):
            w_try, nu_try = _solve_state(Qd, cd, lo_t, hi_t, eta_t, p_t, state)
            state_new, violated = _update_state(Qd, cd, lo_t, hi_t, eta_t, p_t, state, w_try, nu_try, scale_w, scale_g)
            valid = ~violated
            if bool(valid.all()):
                break
            state = torch.where(violated[:, None], state_new, state)

    w_exact, _ = _solve_state(Q, c, lo_t, hi_t, eta_t, p_t, state)  # with autograd
    w = torch.where(valid[:, None], w_exact, z)
    n_free = ((state == FREE) | (state == FREE_BELOW) | (state == FREE_ABOVE)).sum(-1)
    return w, LayerInfo(n_free=n_free, valid=valid, admm_iters=iters, warm=(z, u))


class PortfolioLayer:
    r"""``w* = argmax mu'w - lam w'Sigma w - eta |w - w_prev|_1`` s.t. ``1'w = 1, lower <= w <= upper``."""

    def __init__(self, lam: float = 5.0, lower: float = 0.0, upper: float = 1.0, eta: float = 0.0, **solver_kwargs):
        self.lam = float(lam)
        self.lower = lower
        self.upper = upper
        self.eta = float(eta)
        self.solver_kwargs = solver_kwargs

    def __call__(self, mu: torch.Tensor, Sigma: torch.Tensor, w_prev: Optional[torch.Tensor] = None, warm=None):
        Q = 2.0 * self.lam * Sigma.to(DTYPE)
        eta = self.eta if w_prev is not None else 0.0
        return solve_box_qp(Q, mu, self.lower, self.upper, eta=eta, p=w_prev, warm=warm, **self.solver_kwargs)


def mv_closed_form(mu: torch.Tensor, Sigma: torch.Tensor, lam: float) -> torch.Tensor:
    r"""Budget-only mean-variance solution (no bounds), used to test gradients.

    :math:`w^\star = \frac{\Sigma^{-1}1}{1^\top\Sigma^{-1}1}
    + \frac{1}{2\lambda}\Big(\Sigma^{-1} - \frac{\Sigma^{-1}11^\top\Sigma^{-1}}{1^\top\Sigma^{-1}1}\Big)\mu`.
    """
    n = mu.shape[-1]
    ones = torch.ones(n, dtype=mu.dtype)
    Sinv = torch.linalg.inv(Sigma)
    Sinv1 = Sinv @ ones
    denom = ones @ Sinv1
    M = Sinv - torch.outer(Sinv1, Sinv1) / denom
    return Sinv1 / denom + (M @ mu.T).T / (2.0 * lam)

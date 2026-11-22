"""The portfolio layer: forward pass vs CVXPY, gradients vs closed form / finite differences / cvxpylayers."""

import numpy as np
import pytest
import torch

from dfl_portfolio.layers import PortfolioLayer, mv_closed_form

cp = pytest.importorskip("cvxpy")


def random_problem(seed, n=7, B=12):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(B, n, n)) * 0.03
    Sig = A @ A.transpose(0, 2, 1) / n + np.eye(n) * 2e-4
    mu = rng.normal(0.002, 0.008, size=(B, n))
    wprev = rng.dirichlet(np.ones(n), size=B)
    return mu, Sig, wprev


def cvxpy_solve(mu, Sig, lam, lo, up, eta=0.0, wprev=None):
    n = len(mu)
    x = cp.Variable(n)
    obj = mu @ x - lam * cp.quad_form(x, Sig)
    if eta > 0:
        obj = obj - eta * cp.norm1(x - wprev)
    # scale the objective: weekly returns are O(1e-3) and solvers' absolute tolerances would dominate otherwise
    cp.Problem(cp.Maximize(1e3 * obj), [cp.sum(x) == 1, x >= lo, x <= up]).solve(
        solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return x.value


@pytest.mark.parametrize("lam,up,eta", [(5.0, 1.0, 0.0), (5.0, 0.3, 0.0), (1.0, 0.3, 0.0),
                                        (20.0, 0.25, 0.0005), (5.0, 0.3, 0.002)])
def test_forward_matches_cvxpy(lam, up, eta):
    mu, Sig, wprev = random_problem(0)
    layer = PortfolioLayer(lam, 0.0, up, eta)
    w, info = layer(torch.tensor(mu), torch.tensor(Sig), torch.tensor(wprev) if eta > 0 else None)
    w = w.numpy()
    assert info.valid.all()
    assert np.allclose(w.sum(1), 1.0, atol=1e-10)
    assert w.min() >= -1e-10 and w.max() <= up + 1e-10
    for b in range(len(mu)):
        ref = cvxpy_solve(mu[b], Sig[b], lam, 0.0, up, eta, wprev[b])
        assert np.abs(w[b] - ref).max() < 1e-6


def test_budget_only_closed_form_and_gradient():
    mu, Sig, _ = random_problem(1, B=1)
    lam = 5.0
    v = torch.tensor(np.random.default_rng(2).normal(size=mu.shape[1]))
    m1 = torch.tensor(mu[0], requires_grad=True)
    w1, _ = PortfolioLayer(lam, -np.inf, np.inf)(m1[None], torch.tensor(Sig[0]))
    g1, = torch.autograd.grad((w1[0] * v).sum(), m1)
    m2 = torch.tensor(mu[0], requires_grad=True)
    w2 = mv_closed_form(m2[None], torch.tensor(Sig[0]), lam)
    g2, = torch.autograd.grad((w2[0] * v).sum(), m2)
    assert torch.allclose(w1, w2, atol=1e-12)
    assert torch.allclose(g1, g2, atol=1e-9)
    # shift invariance under the budget constraint: adding a constant to mu changes nothing
    w3, _ = PortfolioLayer(lam, -np.inf, np.inf)((m1 + 0.05)[None].detach(), torch.tensor(Sig[0]))
    assert torch.allclose(w1.detach(), w3, atol=1e-12)
    assert abs(float(g1.sum())) < 1e-9  # gradient is orthogonal to the all-ones direction


def finite_difference(mu, Sig, layer, v, wprev=None, h=1e-8):
    fd = np.zeros_like(mu)
    for i in range(len(mu)):
        e = np.zeros_like(mu)
        e[i] = h
        kw = {} if wprev is None else {"w_prev": torch.tensor(wprev)[None]}
        wp, _ = layer(torch.tensor(mu + e)[None], torch.tensor(Sig), **kw)
        wm, _ = layer(torch.tensor(mu - e)[None], torch.tensor(Sig), **kw)
        fd[i] = float(((wp[0] - wm[0]) * v).sum()) / (2 * h)
    return fd


@pytest.mark.parametrize("eta", [0.0, 0.001])
def test_gradient_matches_finite_differences(eta):
    mu, Sig, wprev = random_problem(3, B=8)
    layer = PortfolioLayer(5.0, 0.0, 0.4, eta)
    v = torch.tensor(np.random.default_rng(4).normal(size=mu.shape[1]))
    checked_nonzero = 0
    for b in range(len(mu)):
        m = torch.tensor(mu[b], requires_grad=True)
        kw = {} if eta == 0 else {"w_prev": torch.tensor(wprev[b])[None]}
        w, info = layer(m[None], torch.tensor(Sig[b]), **kw)
        g, = torch.autograd.grad((w[0] * v).sum(), m)
        fd = finite_difference(mu[b], Sig[b], layer, v, None if eta == 0 else wprev[b])
        assert np.abs(g.numpy() - fd).max() < 1e-5 * max(1.0, np.abs(fd).max())
        checked_nonzero += int(np.abs(fd).max() > 0)
    assert checked_nonzero > 0


def test_gradient_wrt_sigma():
    mu, Sig, _ = random_problem(5, B=1)
    S = torch.tensor(Sig[0], requires_grad=True)
    layer = PortfolioLayer(5.0, 0.0, 1.0)
    v = torch.tensor(np.random.default_rng(6).normal(size=mu.shape[1]))
    w, _ = layer(torch.tensor(mu[0])[None], S)
    gS, = torch.autograd.grad((w[0] * v).sum(), S)
    i, j, h = 1, 2, 1e-9
    E = np.zeros_like(Sig[0])
    E[i, j] = E[j, i] = h
    wp, _ = layer(torch.tensor(mu[0])[None], torch.tensor(Sig[0] + E))
    wm, _ = layer(torch.tensor(mu[0])[None], torch.tensor(Sig[0] - E))
    fd = float(((wp[0] - wm[0]) * v).sum()) / (2 * h)
    assert abs(float(gS[i, j] + gS[j, i]) - fd) < 1e-4 * max(1.0, abs(fd))


def test_vertex_solution_has_zero_gradient():
    """Huge predicted returns push the solution to a vertex: the decision is then locally constant."""
    mu, Sig, _ = random_problem(7, B=1)
    m = torch.tensor(mu[0] * 100, requires_grad=True)
    w, info = PortfolioLayer(1.0, 0.0, 0.3)(m[None], torch.tensor(Sig[0]))
    g, = torch.autograd.grad(w[0, 0] + 2 * w[0, 1], m)
    assert int(info.n_free[0]) <= 1
    assert float(g.abs().max()) == 0.0


def test_agrees_with_cvxpylayers_in_direction():
    """cvxpylayers' gradients are only approximate (interior-point smoothing), so compare directions."""
    pytest.importorskip("cvxpylayers")
    from cvxpylayers.torch import CvxpyLayer

    rng = np.random.default_rng(8)
    n = 6
    A = rng.normal(size=(n, n)) * 0.03
    Sig = A @ A.T / n + np.eye(n) * 2e-4
    x, m = cp.Variable(n), cp.Parameter(n)
    L = np.linalg.cholesky(Sig)
    prob = cp.Problem(cp.Maximize(1e3 * (m @ x - 5.0 * cp.sum_squares(L.T @ x))), [cp.sum(x) == 1, x >= 0])
    ref_layer = CvxpyLayer(prob, parameters=[m], variables=[x])
    mu = rng.normal(0.002, 0.002, size=n)
    v = torch.tensor(rng.normal(size=n))
    ma = torch.tensor(mu, requires_grad=True)
    wa, = ref_layer(ma)
    ga, = torch.autograd.grad((wa * v).sum(), ma)
    mb = torch.tensor(mu, requires_grad=True)
    wb, _ = PortfolioLayer(5.0, 0.0, 1.0)(mb[None], torch.tensor(Sig))
    gb, = torch.autograd.grad((wb[0] * v).sum(), mb)
    assert np.abs(wa.detach().numpy() - wb[0].detach().numpy()).max() < 1e-4
    cos = float(ga @ gb / (ga.norm() * gb.norm() + 1e-12))
    assert cos > 0.95

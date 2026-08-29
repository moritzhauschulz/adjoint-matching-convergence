"""Self-consistency (fixed-point) operator for adjoint sampling.

    P(u)(t, x) = -sigma(t) * E[ grad_g(X_T^{u, x}) | X_t = x ]

where X^{u,x} solves  dX_s = sigma(s) u(X_s, s) ds + sigma(s) dB_s,  X_t = x,
and grad_g is the terminal-cost gradient whose RAM-loss target is
-sigma(t) grad_g(X_1)  (arXiv:2504.11713).  The optimal control u* is the fixed
point:  P(u*) = u*.

Sign convention: this module uses the code's convention (target = -sigma*grad_g,
so u*(T, x) = -sigma(T) grad_g(x)).  Project_Notes.tm writes the operator without
the leading minus and with a correspondingly sign-flipped u*; the residual
||P(u*) - u*|| is identical either way as long as P and u* use one convention.

`fixed_point_residual_field` gives  ||P(u*)(t, x) - u*(t, x)||  on an (x-grid) x
(time-grid) mesh — a basic sanity check that should be << 1 everywhere for any
experiment where u* is known in closed form.  It only needs `u_star_fn` and
`grad_g_fn`; everything else is the shared SDE machinery.
"""

import math

import torch
from torch import Tensor


@torch.no_grad()
def _rollout(control_fn, x: Tensor, t_start_idx: int, ts: Tensor, sigma_fn, device) -> Tensor:
    """Euler-Maruyama from ts[t_start_idx] to ts[-1] under `control_fn`."""
    for i in range(t_start_idx, ts.shape[0] - 1):
        t_curr = ts[i].item()
        dt = (ts[i + 1] - ts[i]).item()
        t_vec = torch.full((x.shape[0],), t_curr, device=device)
        s = sigma_fn(t_vec).unsqueeze(-1)
        x = x + s * control_fn(x, t_vec) * dt + s * math.sqrt(dt) * torch.randn_like(x)
    return x


@torch.no_grad()
def operator_field(
    control_fn,
    grad_g_fn,
    x_states: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    d: int,
    n_mc: int,
    device,
) -> Tensor:
    """P(control_fn)(t_start, x) for each state x in `x_states`.

    Args:
        control_fn: callable(x [N, d], t [N]) -> drift [N, d]
        grad_g_fn:  callable(x_T [N, d]) -> grad_g(x_T) [N, d]
        x_states:   [n_pts, d] starting states at time `t_start`
        ts:         [K+1] uniform time grid covering [0, T]
        n_mc:       Monte-Carlo rollouts per starting state

    Returns:
        [n_pts, d] tensor  -sigma(t_start) * mean_mc[ grad_g(X_T) ]  (signed).

    At t_start = T the rollout is empty, X_T = x, so the result is
    -sigma(T) grad_g(x) exactly (no MC noise).
    """
    n_pts = x_states.shape[0]
    t_start_idx = int((ts - t_start).abs().argmin().item())
    x = x_states.repeat_interleave(n_mc, dim=0)             # [n_pts * n_mc, d]
    x = _rollout(control_fn, x, t_start_idx, ts, sigma_fn, device)
    gg = grad_g_fn(x)                                        # [n_pts * n_mc, d]
    sigma_t = float(sigma_fn(torch.full((1,), t_start, device=device)).item())
    return -sigma_t * gg.view(n_pts, n_mc, d).mean(dim=1)   # [n_pts, d]


@torch.no_grad()
def operator_grid_field(
    control_fn,
    grad_g_fn,
    xs: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    d: int,
    n_mc: int,
    device,
) -> Tensor:
    """`operator_field` for a 1-D grid: states are (xs[i], 0, ..., 0).  -> [n_grid, d]."""
    n_grid = xs.shape[0]
    x_states = torch.zeros(n_grid, d, device=device)
    x_states[:, 0] = xs
    return operator_field(control_fn, grad_g_fn, x_states, t_start, ts, sigma_fn, d, n_mc, device)


def _apply_chunked(fn, x: Tensor, t_vec: Tensor, chunk: int) -> Tensor:
    """fn(x, t_vec) split into row-chunks of size `chunk` (memory cap)."""
    if x.shape[0] <= chunk:
        return fn(x, t_vec)
    return torch.cat(
        [fn(x[j:j + chunk], t_vec[j:j + chunk]) for j in range(0, x.shape[0], chunk)],
        dim=0,
    )


@torch.no_grad()
def operator_grid_field_all_times(
    control_fn,
    grad_g_fn,
    xs: Tensor,
    ts: Tensor,
    sigma_fn,
    d: int,
    n_mc: int,
    device,
    chunk: int = 1_000_000,
) -> Tensor:
    """P(control_fn)(t_k, x) for EVERY time slice t_k and grid point x, in ONE
    batched Euler-Maruyama pass.  Returns [K+1, n_grid, d] (signed).

    Same result (statistically) as calling `operator_grid_field` once per t_k,
    but with K network invocations instead of O(K²/2): walker (k, g, m) starts at
    (t_k, xs[g]); it is stepped only from global step k onward, so at global
    step i all *active* walkers share global time t_i (hence one sigma(t_i) and
    one control_fn call per step).  Walkers are ordered k-major, so "active at
    step i" is the contiguous prefix x[:(i+1)·n_grid·n_mc] — a plain slice.

    At t_K = T the k=K walkers are never stepped ⇒ exact -σ(T)∇g(x).
    """
    Kp1 = ts.shape[0]
    K = Kp1 - 1
    n_grid = xs.shape[0]
    gm = n_grid * n_mc

    x0 = torch.zeros(n_grid, d, device=device)
    x0[:, 0] = xs
    # [K+1, n_grid, n_mc, d] flattened k-major -> [(K+1)·n_grid·n_mc, d]
    x = x0[None, :, None, :].expand(Kp1, n_grid, n_mc, d).reshape(Kp1 * gm, d).clone()

    for i in range(K):
        n_active = (i + 1) * gm
        xi = x[:n_active]
        dt_i = float((ts[i + 1] - ts[i]).item())
        t_vec = torch.full((n_active,), float(ts[i].item()), device=device)
        s = sigma_fn(t_vec).unsqueeze(-1)
        u = _apply_chunked(control_fn, xi, t_vec, chunk)
        noise = torch.randn_like(xi)
        x[:n_active] = xi + s * u * dt_i + s * math.sqrt(dt_i) * noise

    gg = grad_g_fn(x).view(Kp1, n_grid, n_mc, d).mean(dim=2)   # [K+1, n_grid, d]
    sigma_k = sigma_fn(ts).reshape(Kp1, 1, 1)                  # σ(t_k)
    return -sigma_k * gg


@torch.no_grad()
def fixed_point_residual_field(
    u_star_fn,
    grad_g_fn,
    xs: Tensor,
    ts: Tensor,
    sigma_fn,
    d: int,
    n_mc: int,
    device,
) -> dict:
    """||P(u*)(t_k, x) - u*(t_k, x)|| over the full (x-grid) x (time-grid) mesh.

    Args:
        u_star_fn: callable(x [N, d], t [N]) -> u*(x, t) [N, d]  (analytic optimal control)
        grad_g_fn: callable(x_T [N, d]) -> grad_g(x_T) [N, d]
        xs:        [n_grid] 1-D evaluation grid
        ts:        [K+1] uniform time grid

    Returns dict of CPU tensors:
        residual : [K+1, n_grid]      ||P(u*) - u*||
        p_u_star : [K+1, n_grid, d]   signed P(u*)
        u_star   : [K+1, n_grid, d]   signed u*
    """
    K = ts.shape[0] - 1
    n_grid = xs.shape[0]
    x_states = torch.zeros(n_grid, d, device=device)
    x_states[:, 0] = xs

    residual = torch.zeros(K + 1, n_grid)
    p_field = torch.zeros(K + 1, n_grid, d)
    u_field = torch.zeros(K + 1, n_grid, d)

    for k in range(K + 1):
        t_k = float(ts[k].item())
        t_vec = torch.full((n_grid,), t_k, device=device)
        p = operator_field(u_star_fn, grad_g_fn, x_states, t_k, ts, sigma_fn, d, n_mc, device)
        us = u_star_fn(x_states, t_vec)
        p_field[k] = p.cpu()
        u_field[k] = us.cpu()
        residual[k] = (p - us).norm(dim=-1).cpu()

    return {"residual": residual, "p_u_star": p_field, "u_star": u_field}

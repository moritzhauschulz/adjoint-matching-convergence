"""Noise schedules, the base-process bridge, Euler-Maruyama step, and eval helpers.

Notation follows arXiv:2504.11713.
Base process: dX_t = σ(t) dB_t, X_0 = 0.  Marginal variance: ν_t = ∫_0^t σ(s)² ds.
"""

import math
import random
import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Noise schedules  σ(t)
# ---------------------------------------------------------------------------

def sigma_constant(sigma: float = 1.0):
    """σ(t) = sigma  (constant schedule)."""
    def fn(t: Tensor) -> Tensor:
        return torch.full_like(t, sigma)
    return fn


def sigma_linear(sigma: float = 1.0, floor: float = 0.0):
    """σ(t) = sigma · (floor + (1 − floor)(1 − t))  — linear from sigma at t=0
    down to  floor·sigma  at t=1.  floor=0 ⇒ vanishing linear; floor=1 ⇒ constant.
    """
    b = 1.0 - floor
    def fn(t: Tensor) -> Tensor:
        return sigma * (1.0 - b * t)
    return fn


# ---------------------------------------------------------------------------
# Marginal variance  ν_t = ∫_0^t σ(s)² ds
# ---------------------------------------------------------------------------

def nu_constant(sigma: float = 1.0):
    """ν_t = σ² t  for constant schedule."""
    def fn(t: Tensor) -> Tensor:
        return sigma ** 2 * t
    return fn


def nu_linear(sigma: float = 1.0, floor: float = 0.0):
    """ν_t = ∫_0^t σ² ds for σ(t) = sigma·(1 − b·t),  b = 1 − floor.

    ν_t = σ² (1 − (1 − b t)³) / (3 b)   (→ σ² t as b → 0).
    """
    b = 1.0 - floor
    def fn(t: Tensor) -> Tensor:
        if abs(b) < 1e-9:
            return sigma ** 2 * t
        return sigma ** 2 * (1.0 - (1.0 - b * t) ** 3) / (3.0 * b)
    return fn


# ---------------------------------------------------------------------------
# Schedule dispatch
# ---------------------------------------------------------------------------

def make_noise_schedule(name: str = "constant", sigma: float = 1.0, floor: float = 0.0):
    """Return (sigma_fn, nu_fn, nu_1) for the named noise schedule.

    name = "constant":  σ(t) = sigma,                             ν_1 = sigma²
    name = "linear":    σ(t) = sigma·(floor + (1−floor)(1−t)),    ν_1 = sigma²·(1 + a + a²)/3,  a = floor
                        floor=0 ⇒ vanishing linear (σ(1)=0);  floor=1 ⇒ constant.
    """
    if name == "constant":
        return sigma_constant(sigma), nu_constant(sigma), float(sigma ** 2)
    if name == "linear":
        a = float(floor)
        nu_1 = float(sigma ** 2 * (1.0 + a + a ** 2) / 3.0)
        return sigma_linear(sigma, a), nu_linear(sigma, a), nu_1
    raise ValueError(f"unknown noise schedule: {name!r} (expected 'constant' or 'linear')")


# ---------------------------------------------------------------------------
# Base process bridge  p_{t|1}^base(·|X_1)
# ---------------------------------------------------------------------------

def sample_base_conditional(x1: Tensor, t: Tensor, nu_t: Tensor, nu_1: float) -> Tensor:
    """Sample X_t ~ p_{t|1}^base(·|X_1) = N(ν_t/ν_1 · x1, ν_{t|1} · I).

    ν_{t|1} = ν_t (ν_1 − ν_t) / ν_1
    """
    extra = [1] * (x1.dim() - 1)
    alpha = (nu_t / nu_1).reshape(-1, *extra)               # [B, 1, ...]
    var = (nu_t * (nu_1 - nu_t) / nu_1).clamp(min=0.0)
    std = var.sqrt().reshape(-1, *extra)                     # [B, 1, ...]
    return alpha * x1 + std * torch.randn_like(x1)


# ---------------------------------------------------------------------------
# Euler-Maruyama step
# ---------------------------------------------------------------------------

def euler_maruyama_step(x: Tensor, u: Tensor, sigma: Tensor, dt: float) -> Tensor:
    """X_{n+1} = X_n + σ(t_n) u(X_n, t_n) dt + σ(t_n) √dt ε_n,  ε_n ~ N(0, I)."""
    s = sigma.reshape(-1, *([1] * (x.dim() - 1)))
    return x + s * u * dt + s * (dt ** 0.5) * torch.randn_like(x)


def linspace_time(steps: int, device=None) -> Tensor:
    """Uniform grid 0 = t_0 < t_1 < … < t_N = 1."""
    return torch.linspace(0.0, 1.0, steps + 1, device=device)


def sigma_int_from_nu(nu_fn, nu_1: float):
    """Σ_t = ∫_t^1 σ(s)² ds = ν_1 − ν_t  as a callable (schedule-agnostic)."""
    return lambda t: nu_1 - nu_fn(t)


# ---------------------------------------------------------------------------
# Evaluation helpers (uncontrolled by autograd)
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_paths(control_fn, n_paths: int, ts: Tensor, d: int, sigma_fn, device) -> Tensor:
    """Euler–Maruyama rollout of `control_fn` from X_0 = 0 on the grid `ts`.

    Returns the trajectory stack [len(ts), n_paths, d].
    """
    x = torch.zeros(n_paths, d, device=device)
    steps = [x.clone()]
    for i in range(ts.shape[0] - 1):
        t_vec = ts[i].expand(n_paths)
        dt = (ts[i + 1] - ts[i]).item()
        s = sigma_fn(t_vec).unsqueeze(-1)
        x = x + s * control_fn(x, t_vec) * dt + s * math.sqrt(dt) * torch.randn_like(x)
        steps.append(x.clone())
    return torch.stack(steps, dim=0)


@torch.no_grad()
def rel_l2(u_theta, x_samples: Tensor, t_val: float, u_star_fn) -> float:
    """‖u_θ(·,t) − u*(·,t)‖₂ / ‖u*(·,t)‖₂  over `x_samples` (NaN if u* ≈ 0)."""
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star = u_star_fn(x_samples, t)
    den = u_star.pow(2).sum(-1).mean().item()
    if den < 1e-12:
        return float("nan")
    return math.sqrt((u_hat - u_star).pow(2).sum(-1).mean().item() / den)

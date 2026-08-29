"""Noise schedules, base process distributions, and Euler-Maruyama step.

Notation follows arXiv:2504.11713.
Base process: dX_t = σ(t) dB_t, X_0 = 0.
Marginal variance: ν_t = ∫_0^t σ(s)² ds.
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
# Base process: log-density, score, and bridge conditional
# ---------------------------------------------------------------------------

def log_p1_base(x: Tensor, nu_1: float) -> Tensor:
    """log p_1^base(x) = −d/2 log(2π ν_1) − ‖x‖² / (2 ν_1)."""
    d = x.shape[-1]
    return -0.5 * d * math.log(2 * math.pi * nu_1) - x.pow(2).sum(-1) / (2.0 * nu_1)


def grad_log_p1_base(x: Tensor, nu_1: float) -> Tensor:
    """∇_x log p_1^base(x) = −x / ν_1."""
    return -x / nu_1


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

def euler_maruyama_step(
    x: Tensor, u: Tensor, sigma: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """X_{n+1} = X_n + σ(t_n) u(X_n, t_n) dt + σ(t_n) √dt ε_n.

    Returns (x_next, eps) where eps ~ N(0, I) is stored for the backward pass.
    """
    extra = [1] * (x.dim() - 1)
    s = sigma.reshape(-1, *extra)
    eps = torch.randn_like(x)
    return x + s * u * dt + s * (dt ** 0.5) * eps, eps


def linspace_time(steps: int, device=None) -> Tensor:
    """Uniform grid 0 = t_0 < t_1 < … < t_N = 1."""
    return torch.linspace(0.0, 1.0, steps + 1, device=device)

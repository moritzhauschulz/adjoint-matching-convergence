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


# ---------------------------------------------------------------------------
# Marginal variance  ν_t = ∫_0^t σ(s)² ds
# ---------------------------------------------------------------------------

def nu_constant(sigma: float = 1.0):
    """ν_t = σ² t  for constant schedule."""
    def fn(t: Tensor) -> Tensor:
        return sigma ** 2 * t
    return fn


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

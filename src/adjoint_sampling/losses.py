"""Loss functions for adjoint sampling (arXiv:2504.11713).

L_RAM  — main training objective (Algorithm 1 inner loop).
L_AM   — reference objective; requires controlled trajectories.
L_SOC  — ground-truth objective; evaluation only.
"""

import torch
import torch.nn as nn
from torch import Tensor

from adjoint_sampling.utils import sample_base_conditional, linspace_time


def ram_loss(
    u_theta: nn.Module,
    x1: Tensor,
    grad_g: Tensor,
    sigma_fn,
    nu_fn,
    nu_1: float,
) -> Tensor:
    """Reciprocal Adjoint Matching loss (L_RAM, Algorithm 1 inner loop).

    L_RAM = E_{t, X_t ~ p_{t|1}^base(·|X_1), X_1 ~ p_1^ū}
              [ λ(t)/2 · ‖u_θ(X_t, t) + σ(t) ∇g(X_1)‖² ]

    λ(t) = 1/σ(t)²

    Args:
        u_theta:  drift network u_θ(x, t)
        x1:       terminal samples from replay buffer, [B, d]
        grad_g:   ∇g(X_1) from replay buffer, [B, d]
        sigma_fn: σ(t), callable Tensor -> Tensor
        nu_fn:    ν_t = ∫_0^t σ²ds, callable Tensor -> Tensor
        nu_1:     ν_1 (scalar float)
    """
    batch_size = x1.shape[0]
    device = x1.device

    t = torch.rand(batch_size, device=device).clamp(1e-4, 1 - 1e-4)

    nu_t = nu_fn(t)
    x_t = sample_base_conditional(x1, t, nu_t, nu_1)

    u = u_theta(x_t, t)                              # [B, d]

    sigma_t = sigma_fn(t)                            # [B]
    target = -sigma_t.unsqueeze(-1) * grad_g         # [B, d]
    lam = 1.0 / sigma_t.pow(2)                       # [B]

    sq_err = (u - target).pow(2).sum(dim=-1)         # [B]
    return (lam * 0.5 * sq_err).mean()


def am_loss(
    u_theta: nn.Module,
    trajectory: list[Tensor],
    grad_g: Tensor,
    sigma_fn,
    steps: int,
) -> Tensor:
    """Adjoint Matching loss (L_AM) computed from a stopgrad forward trajectory.

    L_AM = E_{X ~ p^ū} [ ∫_0^1 ½ ‖u_θ(X_t, t) + σ(t) ∇g(X_1)‖² dt ]

    Discretised as a mean over time steps. Requires the full trajectory stored
    by Sampler.sample_trajectory(); more expensive than L_RAM.

    Args:
        trajectory: list of [B, d] tensors {X_n}, length steps+1
        grad_g:     ∇g(X_1), [B, d]
    """
    dt = 1.0 / steps
    ts = linspace_time(steps, device=grad_g.device)
    total = torch.tensor(0.0, device=grad_g.device)

    for n in range(steps):
        x_n = trajectory[n].detach()
        t_n = ts[n].expand(x_n.shape[0])
        sigma_n = sigma_fn(t_n).unsqueeze(-1)        # [B, 1]
        target = -sigma_n * grad_g                   # [B, d]
        u = u_theta(x_n, t_n)
        total = total + 0.5 * (u - target).pow(2).sum(-1).mean() * dt

    return total


@torch.no_grad()
def soc_objective(
    u_theta: nn.Module,
    trajectory: list[Tensor],
    g_fn,
    sigma_fn,
    steps: int,
) -> Tensor:
    """Stochastic optimal control objective L_SOC (evaluation only).

    L_SOC(u) = E_{p^u} [ ∫_0^1 ½ ‖u(X_t,t)‖² dt  +  g(X_1) ]

    Args:
        trajectory: list of [B, d] tensors {X_n}, length steps+1
        g_fn:       terminal cost function g : ℝ^d → ℝ^B
    """
    dt = 1.0 / steps
    ts = linspace_time(steps, device=trajectory[0].device)
    running = torch.tensor(0.0, device=trajectory[0].device)

    for n in range(steps):
        x_n = trajectory[n]
        t_n = ts[n].expand(x_n.shape[0])
        u = u_theta(x_n, t_n)
        running = running + 0.5 * u.pow(2).sum(-1).mean() * dt

    terminal = g_fn(trajectory[-1]).mean()
    return running + terminal

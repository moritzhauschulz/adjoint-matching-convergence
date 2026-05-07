"""Backward adjoint ODE solver for the lean adjoint state ã(t; X).

dã(t; X)/dt = −(∇_x b(X_t, t))ᵀ ã(t; X),   ã(1; X) = ∇g(X_1)

where b(X_t, t) = σ(t) u_θ(X_t, t) is the controlled drift.

Used to compute L_AM gradients without backpropagating through the SDE solver.
Note: main training uses L_RAM which avoids this solve entirely (see losses.py).
"""

import torch
import torch.nn as nn
from torch import Tensor

from adjoint_sampling.utils import linspace_time


class AdjointSolver:

    def __init__(self, sigma_fn, steps: int):
        self.sigma_fn = sigma_fn
        self.steps = steps
        self.dt = 1.0 / steps

    def solve(
        self,
        u_theta: nn.Module,
        trajectory: list[Tensor],   # {X_n}, length steps+1, from sampler
        terminal_grad: Tensor,       # ∇g(X_1), shape [B, d]
    ) -> list[Tensor]:
        """Euler integration of the adjoint ODE backward from t=1 to t=0.

        ã(t_{n-1}) = ã(t_n) + (∇_x b)ᵀ ã(t_n) · Δt

        (∇_x b)ᵀ ã is computed via VJP — never materialises the d×d Jacobian.

        Returns:
            List of adjoint states [ã(t_N), ã(t_{N-1}), ..., ã(t_0)].
        """
        ts = linspace_time(self.steps, device=terminal_grad.device)
        a = terminal_grad.clone()
        adjoints = [a]

        for n in reversed(range(self.steps)):
            x_n = trajectory[n].detach().requires_grad_(True)
            t_n = ts[n].expand(x_n.shape[0])
            sigma_n = self.sigma_fn(t_n)

            # b(x, t_n) = σ(t_n) u_θ(x, t_n)
            def b_fn(x: Tensor) -> Tensor:
                s = sigma_n.reshape(-1, *([1] * (x.dim() - 1)))
                return s * u_theta(x, t_n)

            # VJP: returns (∂b/∂x)ᵀ a  ≡  ∇_x (bᵀ a)
            with torch.enable_grad():
                _, vjp_val = torch.autograd.functional.vjp(b_fn, x_n, v=a)

            # Backward Euler step (integrating dã/dt = -(∇_x b)ᵀ ã, reversed):
            # ã(t_{n-1}) = ã(t_n) + (∇_x b)ᵀ ã · Δt
            a = (a + vjp_val * self.dt).detach()
            adjoints.append(a)

        return adjoints  # length steps+1, from t=1 to t=0

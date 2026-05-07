"""Forward SDE sampler under the controlled process p^u (Algorithm 1 outer loop).

dX_t = σ(t) u_θ(X_t, t) dt + σ(t) dB_t,   X_0 = 0,   t ∈ [0, 1].

Runs entirely under torch.no_grad() — corresponds to stopgrad(u_θ) in Algorithm 1.
"""

import torch
import torch.nn as nn
from torch import Tensor

from adjoint_sampling.utils import euler_maruyama_step, linspace_time


class Sampler:

    def __init__(self, sigma_fn, steps: int):
        """
        Args:
            sigma_fn: callable(t: Tensor) -> Tensor, the noise schedule σ(t).
            steps: number of Euler-Maruyama steps N.
        """
        self.sigma_fn = sigma_fn
        self.steps = steps
        self.dt = 1.0 / steps

    @torch.no_grad()
    def sample(self, u_theta: nn.Module, batch_size: int, d: int, device) -> Tensor:
        """Sample X_1 ~ p_1^(stopgrad(u_θ)).

        Returns:
            x1: [batch_size, d]
        """
        ts = linspace_time(self.steps, device=device)
        x = torch.zeros(batch_size, d, device=device)
        for n in range(self.steps):
            t = ts[n].expand(batch_size)
            u = u_theta(x, t)
            sigma = self.sigma_fn(t)
            x, _ = euler_maruyama_step(x, u, sigma, self.dt)
        return x

    @torch.no_grad()
    def sample_trajectory(
        self, u_theta: nn.Module, batch_size: int, d: int, device
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Sample a full trajectory and store noise — needed for L_AM.

        Returns:
            xs:   list of [batch_size, d] tensors, length N+1 (t_0 to t_N)
            epss: list of [batch_size, d] noise tensors, length N
        """
        ts = linspace_time(self.steps, device=device)
        x = torch.zeros(batch_size, d, device=device)
        xs = [x]
        epss: list[Tensor] = []
        for n in range(self.steps):
            t = ts[n].expand(batch_size)
            u = u_theta(x, t)
            sigma = self.sigma_fn(t)
            x, eps = euler_maruyama_step(x, u, sigma, self.dt)
            xs.append(x)
            epss.append(eps)
        return xs, epss

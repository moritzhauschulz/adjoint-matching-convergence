"""Bimodal Gaussian-mixture reward and its analytic optimal control.

Shared by the `right_to_left_convergence_bimodal` and
`right_to_left_convergence_bimodal_same_bm` experiments.

Reward (log-density up to a constant, p^{u*}(x) ∝ e^{r(x)}):

    r(x) = log( w₁ exp(-λ₁/2 ‖x-μ₁‖²) + w₂ exp(-λ₂/2 ‖x-μ₂‖²) )

Full adjoint sampling objective (arXiv:2504.11713):  g = log p₁^base − r,
so  ∇g(x) = −x/ν₁ − ∇r(x)  and the base measure cancels in p^{u*}.

Analytic optimal control via the Doob h-transform  u*(t,x) = σ(t) ∇_x log h(t,x)
with  h(t,x) = E_base[e^{-g(X₁)} | X_t = x].  Completing the square gives a
two-component Gaussian h; see `optimal_control` for the effective parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianMixtureTarget:
    """Parameters of the two-component reward.  μᵢ are scalars, broadcast over d."""

    w1: float
    lambda1: float
    mu1: float
    w2: float
    lambda2: float
    mu2: float

    # -- reward gradient ----------------------------------------------------

    def grad_r(self, x: Tensor) -> Tensor:
        """∇r(x).  Log-space softmax over the two components avoids 0/0.  [B,d]→[B,d]."""
        sq1 = ((x - self.mu1) ** 2).sum(-1, keepdim=True)
        sq2 = ((x - self.mu2) ** 2).sum(-1, keepdim=True)
        log_a1 = math.log(self.w1) - 0.5 * self.lambda1 * sq1
        log_a2 = math.log(self.w2) - 0.5 * self.lambda2 * sq2
        log_sum = torch.logaddexp(log_a1, log_a2)
        n1 = (log_a1 - log_sum).exp()
        n2 = (log_a2 - log_sum).exp()
        return -(self.lambda1 * (x - self.mu1) * n1 + self.lambda2 * (x - self.mu2) * n2)

    def grad_g(self, x: Tensor, nu_1: float) -> Tensor:
        """∇g(x) = −x/ν₁ − ∇r(x)  (RAM-loss terminal gradient; target is −σ∇g)."""
        return -x / nu_1 - self.grad_r(x)

    def grad_g_fn(self, nu_1: float):
        """Return the callable  x ↦ ∇g(x)  bound to `nu_1`."""
        return lambda x: self.grad_g(x, nu_1)

    # -- analytic optimal control ----------------------------------------------

    def optimal_control(self, x: Tensor, t: Tensor, sigma_fn, sigma_int_fn,
                        nu_1: float, d: int) -> Tensor:
        """u*(t,x), schedule-agnostic for an x-independent σ(t).

        Effective parameters (require λᵢ > 1/ν₁):

            λᵢ* = λᵢ − 1/ν₁,   μᵢ* = λᵢ μᵢ / λᵢ*,
            κᵢ  = d·λᵢ μᵢ² / (2 ν₁ λᵢ*)

        κᵢ is the mode-dependent constant left over from completing the square in
        −λᵢ(x₁−μᵢ)²/2 + x₁²/(2ν₁); it shifts the mixture weights and cancels only
        for symmetric mixtures (μ₁²=μ₂², equal λ).  Σ_t = ∫_t^1 σ² ds comes from
        `sigma_int_fn`.  x: [B,d], t: [B] → [B,d].
        """
        l1s = self.lambda1 - 1.0 / nu_1
        l2s = self.lambda2 - 1.0 / nu_1
        m1s = self.lambda1 * self.mu1 / l1s
        m2s = self.lambda2 * self.mu2 / l2s
        k1 = d * self.lambda1 * self.mu1 ** 2 / (2.0 * nu_1 * l1s)
        k2 = d * self.lambda2 * self.mu2 ** 2 / (2.0 * nu_1 * l2s)

        sigma_int = sigma_int_fn(t)
        denom1 = (1.0 + l1s * sigma_int).unsqueeze(-1)
        denom2 = (1.0 + l2s * sigma_int).unsqueeze(-1)
        sq1 = ((x - m1s) ** 2).sum(-1, keepdim=True)
        sq2 = ((x - m2s) ** 2).sum(-1, keepdim=True)
        log_a1 = math.log(self.w1) + k1 - 0.5 * d * denom1.log() - l1s * sq1 / (2.0 * denom1)
        log_a2 = math.log(self.w2) + k2 - 0.5 * d * denom2.log() - l2s * sq2 / (2.0 * denom2)
        log_sum = torch.logaddexp(log_a1, log_a2)
        n1 = (log_a1 - log_sum).exp()
        n2 = (log_a2 - log_sum).exp()

        numer = (l1s * (x - m1s) / denom1 * n1
                 + l2s * (x - m2s) / denom2 * n2)
        return -sigma_fn(t).unsqueeze(-1) * numer

    def optimal_control_fn(self, sigma_fn, sigma_int_fn, nu_1: float, d: int):
        """Return the callable  (x, t) ↦ u*(t,x)."""
        return lambda x, t: self.optimal_control(x, t, sigma_fn, sigma_int_fn, nu_1, d)

    # -- terminal distribution p^{u*}(x₁) ∝ e^{r(x₁)} -------------------------

    def terminal_mixture(self) -> tuple[float, float, float, float, float, float]:
        """(α₁, μ₁, v₁, α₂, μ₂, v₂) for  p^{u*}(x₁) = Σ αᵢ N(μᵢ, vᵢ),  αᵢ ∝ wᵢ/√λᵢ."""
        a1 = self.w1 / math.sqrt(self.lambda1)
        a2 = self.w2 / math.sqrt(self.lambda2)
        z = a1 + a2
        return a1 / z, self.mu1, 1.0 / self.lambda1, a2 / z, self.mu2, 1.0 / self.lambda2

    def terminal_pdf(self, x: np.ndarray) -> np.ndarray:
        """p^{u*}(x) evaluated on a numpy grid (for plotting)."""
        a1, m1, v1, a2, m2, v2 = self.terminal_mixture()
        x = np.asarray(x)
        p1 = a1 * np.exp(-0.5 * (x - m1) ** 2 / v1) / np.sqrt(2 * np.pi * v1)
        p2 = a2 * np.exp(-0.5 * (x - m2) ** 2 / v2) / np.sqrt(2 * np.pi * v2)
        return p1 + p2

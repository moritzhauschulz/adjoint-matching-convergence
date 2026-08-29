"""Gaussian baseline experiment for adjoint sampling (arXiv:2504.11713).

Target: μ(x) = N(x; m, c·I).  Optimal control is available in closed form
(see src/adjoint_sampling/CLAUDE.md §9), allowing exact convergence analysis.

Run:
    python experiments/gaussian_baseline/run.py experiment=gaussian_baseline
    python experiments/gaussian_baseline/run.py experiment=gaussian_baseline target.c=2.0 target.d=8
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
import hydra
from omegaconf import DictConfig

from adjoint_sampling import DriftMLP, Sampler, ReplayBuffer, ram_loss, soc_objective, utils


# ---------------------------------------------------------------------------
# Gaussian target: energy, grad, and analytic optimal control
# ---------------------------------------------------------------------------

def grad_g(x1: Tensor, m: Tensor, c: float, nu_1: float) -> Tensor:
    """∇g(X_1) = −X_1/ν_1 + (X_1 − m)/c  (§9.2)."""
    return x1 * (1.0 / c - 1.0 / nu_1) - m / c


def terminal_cost(x1: Tensor, m: Tensor, c: float, nu_1: float) -> Tensor:
    """g(x) = log p_1^base(x) + E(x)  (per sample, shape [B])."""
    d = x1.shape[-1]
    log_base = -0.5 * d * math.log(2 * math.pi * nu_1) - x1.pow(2).sum(-1) / (2.0 * nu_1)
    energy = (x1 - m).pow(2).sum(-1) / (2.0 * c)
    return log_base + energy


def optimal_control(
    x_t: Tensor, t: Tensor, m: Tensor, c: float, sigma_fn, nu_fn, nu_1: float, eps: float = 1e-6
) -> Tensor:
    """u*(x_t, t) = σ(t) [(1/ν_t − 1/Σ_t) x_t + α_t m / Σ_t]  (§9.4).

    Clamps ν_t, Σ_t away from zero to avoid t → 0 singularity.
    """
    nu_t = nu_fn(t).clamp(min=eps)           # [B]
    alpha_t = nu_t / nu_1                     # [B]
    beta_t = (nu_t * (nu_1 - nu_t) / nu_1).clamp(min=0.0)
    Sigma_t = (alpha_t ** 2 * c + beta_t).clamp(min=eps)   # [B], scalar variance

    sigma_t = sigma_fn(t)                     # [B]

    A = (1.0 / nu_t - 1.0 / Sigma_t)         # [B]
    b = alpha_t / Sigma_t                     # [B]  (coefficient for m)

    A = A.unsqueeze(-1)                        # [B, 1]
    b = b.unsqueeze(-1)                        # [B, 1]
    sigma_t = sigma_t.unsqueeze(-1)            # [B, 1]

    return sigma_t * (A * x_t + b * m)


def sample_optimal_marginal(
    t: Tensor, m: Tensor, c: float, nu_fn, nu_1: float, eps: float = 1e-6
) -> Tensor:
    """Sample x_t ~ p_t^* = N(μ_t, Σ_t I)  (§9.3).

    μ_t = α_t m,  Σ_t = α_t² c + β_t
    """
    nu_t = nu_fn(t).clamp(min=eps)
    alpha_t = (nu_t / nu_1).unsqueeze(-1)             # [B, 1]
    beta_t = (nu_t * (nu_1 - nu_t) / nu_1).clamp(min=0.0)
    Sigma_t = (alpha_t.squeeze(-1) ** 2 * c + beta_t).clamp(min=eps)
    std_t = Sigma_t.sqrt().unsqueeze(-1)               # [B, 1]
    d = m.shape[0]
    x = alpha_t * m + std_t * torch.randn(t.shape[0], d, device=t.device)
    return x


# ---------------------------------------------------------------------------
# Evaluation metrics  (§9.5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def control_mse(
    u_theta: nn.Module,
    m: Tensor, c: float,
    sigma_fn, nu_fn, nu_1: float,
    n_samples: int, n_t_points: int,
    device,
) -> float:
    """E_{x_t ~ p_t^*, t ~ U[0,1]} [ ‖u_θ(x_t,t) − u*(x_t,t)‖² ]."""
    ts = torch.rand(n_t_points, device=device).clamp(1e-4, 1 - 1e-4)
    total = 0.0
    for t_val in ts:
        t = t_val.expand(n_samples)
        x_t = sample_optimal_marginal(t, m, c, nu_fn, nu_1)
        u_hat = u_theta(x_t, t)
        u_star = optimal_control(x_t, t, m, c, sigma_fn, nu_fn, nu_1)
        total += (u_hat - u_star).pow(2).sum(-1).mean().item()
    return total / n_t_points


@torch.no_grad()
def w2_squared(x1_samples: Tensor, m: Tensor, c: float) -> float:
    """W₂²(p_1^{u_θ}, μ) for isotropic Gaussians  (§9.5).

    W₂² = ‖μ̂ − m‖² + d (√σ̂ − √c)²
    """
    mu_hat = x1_samples.mean(dim=0)
    sigma_hat_sq = x1_samples.var(dim=0).mean().item()   # mean scalar variance
    mean_term = (mu_hat - m).pow(2).sum().item()
    d = m.shape[0]
    var_term = d * (sigma_hat_sq ** 0.5 - c ** 0.5) ** 2
    return mean_term + var_term


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    utils.seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    # --- target ---
    d = cfg.target.d
    m = torch.full((d,), cfg.target.m, device=device)
    c = float(cfg.target.c)

    # --- noise schedule ---
    sigma_fn = utils.sigma_constant(cfg.sigma)
    nu_fn = utils.nu_constant(cfg.sigma)
    nu_1 = float(cfg.sigma ** 2)

    # --- components ---
    net = DriftMLP(
        d=d,
        hidden_dim=cfg.network.hidden_dim,
        n_layers=cfg.network.n_layers,
        t_emb_dim=cfg.network.t_emb_dim,
    ).to(device)

    sampler = Sampler(sigma_fn, steps=cfg.sampler.steps)
    buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    def g_fn(x1):
        return terminal_cost(x1, m, c, nu_1)

    def grad_g_fn(x1):
        return grad_g(x1, m, c, nu_1)

    # --- training loop (Algorithm 1) ---
    for outer_it in range(cfg.training.outer_iterations):

        # Outer loop: sample X_1 ~ p_1^(stopgrad(u_θ)), compute ∇g, fill buffer
        x1 = sampler.sample(net, cfg.algorithm.n_outer, d, device)
        gg = grad_g_fn(x1)
        buffer.add(x1, gg)

        # Inner loop: L_RAM gradient steps
        for _ in range(cfg.algorithm.n_inner_steps):
            x1_b, gg_b = buffer.sample(cfg.algorithm.n_inner, device=device)
            loss = ram_loss(net, x1_b, gg_b, sigma_fn, nu_fn, nu_1)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Logging
        if outer_it % cfg.logging.log_every == 0:
            print(f"[{outer_it:4d}] ram_loss={loss.item():.4f}")
            if cfg.logging.wandb:
                wandb.log({"ram_loss": loss.item(), "outer_it": outer_it})

        # Evaluation
        if outer_it % cfg.eval.every == 0:
            cmse = control_mse(
                net, m, c, sigma_fn, nu_fn, nu_1,
                n_samples=cfg.eval.n_control_mse,
                n_t_points=cfg.eval.n_mse_t_points,
                device=device,
            )

            xs = sampler.sample_trajectory(net, cfg.eval.n_rollout, d, device)
            x1_eval = xs[-1]
            lsoc = soc_objective(net, xs, g_fn, sigma_fn, steps=cfg.sampler.steps).item()
            w2sq = w2_squared(x1_eval, m, c)

            print(
                f"  [eval] control_mse={cmse:.4f}  "
                f"L_SOC={lsoc:.4f}  "
                f"W2²={w2sq:.4f}"
            )
            if cfg.logging.wandb:
                wandb.log({
                    "control_mse": cmse,
                    "L_SOC": lsoc,
                    "W2_squared": w2sq,
                    "outer_it": outer_it,
                })

    print("Training complete.")


if __name__ == "__main__":
    main()

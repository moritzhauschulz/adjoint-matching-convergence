"""Right-to-left convergence experiment for adjoint sampling.

Verifies the theoretical contraction bound: the learned control u_θ should
converge to u* from right to left (i.e., error decreases as t → 1).

Setting: quadratic reward r(x) = λ/2‖x‖², simplified objective (no log p₁^base),
with analytic optimal control available via Riccati ODE (Section 4.1 of notes).

Run:
    python experiments/right_to_left_convergence/run.py experiment=right_to_left_convergence
    python experiments/right_to_left_convergence/run.py experiment=right_to_left_convergence target.lambda_=2.0
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Hydra changes cwd to its output dir; keep the experiment directory importable.
sys.path.insert(0, str(Path(__file__).parent))

from adjoint_sampling import DriftMLP, Sampler, ReplayBuffer, ram_loss, utils
from plotting import plot_convergence, save_snapshots


# ---------------------------------------------------------------------------
# Analytic optimal control and Riccati variance
# ---------------------------------------------------------------------------

def riccati_coefficient(t: Tensor, lambda_: float, sigma_fn) -> Tensor:
    """a(t) = λ / (1 + λ σ₀² (1 − t)).

    Scalar closed-form for constant σ.
    """
    sigma_0 = sigma_fn(t)          # [B] or scalar
    return lambda_ / (1.0 + lambda_ * sigma_0 ** 2 * (1.0 - t))


def optimal_control(x: Tensor, t: Tensor, lambda_: float, sigma_fn) -> Tensor:
    """u*(t, x) = −σ(t) a(t) x   (Section 4.1 Lemma)."""
    sigma_t = sigma_fn(t).unsqueeze(-1)                   # [B, 1]
    a_t = riccati_coefficient(t, lambda_, sigma_fn).unsqueeze(-1)  # [B, 1]
    return -sigma_t * a_t * x


def riccati_variance(ts: Tensor, lambda_: float, sigma_fn) -> Tensor:
    """Integrate dV/dt = −2σ²(t) a(t) V + σ²(t), V₀ = 0 on the grid ts.

    Returns V_t for each t in ts (shape [len(ts)]).
    Uses forward Euler on the provided grid.
    """
    n = ts.shape[0]
    Vs = torch.zeros(n, device=ts.device)
    # Vs[0] = 0 (initial condition X_0 = 0 ⟹ V_0 = 0)
    for i in range(n - 1):
        dt = (ts[i + 1] - ts[i]).item()
        t_i = ts[i].unsqueeze(0)          # [1]
        sigma2 = sigma_fn(t_i).pow(2).item()
        a_i = riccati_coefficient(t_i, lambda_, sigma_fn).item()
        dV = -2.0 * sigma2 * a_i * Vs[i].item() + sigma2
        Vs[i + 1] = Vs[i] + dV * dt
    return Vs


def sample_optimal_marginal(t_idx: int, Vs: Tensor, d: int, n_samples: int, device) -> Tensor:
    """Sample X_t ~ N(0, V_t I) using precomputed Riccati variance grid."""
    V_t = Vs[t_idx].clamp(min=1e-8).item()
    return math.sqrt(V_t) * torch.randn(n_samples, d, device=device)


# ---------------------------------------------------------------------------
# Evaluation metrics  (Section 4.1 / §6 of CLAUDE.md)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rel_l2(
    u_theta: nn.Module,
    t_val: float,
    Vt_val: float,
    lambda_: float,
    sigma_fn,
    d: int,
    n_samples: int,
    device,
) -> float:
    """RelL₂(t) = ‖u_θ − u*‖_{L₂(P^{u*})} / ‖u*‖_{L₂(P^{u*})}."""
    t = torch.full((n_samples,), t_val, device=device)
    std = math.sqrt(max(Vt_val, 1e-8))
    x_t = std * torch.randn(n_samples, d, device=device)

    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, sigma_fn)

    num = (u_hat - u_star).pow(2).sum(-1).mean().item()
    den = u_star.pow(2).sum(-1).mean().item()
    if den < 1e-12:
        return float("nan")
    return math.sqrt(num / den)


@torch.no_grad()
def abs_l2(
    u_theta: nn.Module,
    t_val: float,
    Vt_val: float,
    lambda_: float,
    sigma_fn,
    d: int,
    n_samples: int,
    device,
) -> float:
    """AbsL₂(t) = (E_{P^{u*}} ‖u_θ(t, X_t) − u*(t, X_t)‖²)^{1/2}."""
    t = torch.full((n_samples,), t_val, device=device)
    std = math.sqrt(max(Vt_val, 1e-8))
    x_t = std * torch.randn(n_samples, d, device=device)

    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, sigma_fn)

    return math.sqrt((u_hat - u_star).pow(2).sum(-1).mean().item())


@torch.no_grad()
def abs_linf(
    u_theta: nn.Module,
    t_val: float,
    Vt_val: float,
    lambda_: float,
    sigma_fn,
    d: int,
    n_samples: int,
    device,
) -> float:
    """AbsL∞(t) ≈ max over samples of ‖u_θ(t, x) − u*(t, x)‖."""
    t = torch.full((n_samples,), t_val, device=device)
    std = math.sqrt(max(Vt_val, 1e-8))
    x_t = std * torch.randn(n_samples, d, device=device)

    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, sigma_fn)

    return (u_hat - u_star).norm(dim=-1).max().item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    utils.seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    d = cfg.target.d
    lambda_ = float(cfg.target.lambda_)

    sigma_fn = utils.sigma_constant(cfg.sigma)
    nu_fn = utils.nu_constant(cfg.sigma)
    nu_1 = float(cfg.sigma ** 2)

    net = DriftMLP(
        d=d,
        hidden_dim=cfg.network.hidden_dim,
        n_layers=cfg.network.n_layers,
        t_emb_dim=cfg.network.t_emb_dim,
    ).to(device)

    sampler = Sampler(sigma_fn, steps=cfg.sampler.steps)
    buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)

    # Precompute Riccati variance on evaluation grid
    K = cfg.eval.n_time_slices
    ts_eval = torch.linspace(0.0, 1.0, K + 1, device=device)
    Vs_eval = riccati_variance(ts_eval, lambda_, sigma_fn)

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    def grad_g_fn(x1: Tensor) -> Tensor:
        return lambda_ * x1

    snapshots: list[dict] = []
    ts_list = ts_eval.tolist()

    for outer_it in range(cfg.training.outer_iterations):

        # Outer: sample X_1 under stopgrad(u_θ), compute ∇g_eff = λ X_1
        x1 = sampler.sample(net, cfg.algorithm.n_outer, d, device)
        gg = grad_g_fn(x1)
        buffer.add(x1, gg)

        # Inner: RAM gradient steps
        for _ in range(cfg.algorithm.n_inner_steps):
            x1_b, gg_b = buffer.sample(cfg.algorithm.n_inner, device=device)
            loss = ram_loss(net, x1_b, gg_b, sigma_fn, nu_fn, nu_1)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if outer_it % cfg.logging.log_every == 0:
            print(f"[{outer_it:4d}] ram_loss={loss.item():.4f}")
            if cfg.logging.wandb:
                wandb.log({"ram_loss": loss.item(), "outer_it": outer_it})

        if outer_it % cfg.eval.every == 0:
            rl2_vals, al2_vals, al_inf_vals = [], [], []
            wandb_metrics: dict = {}

            for k in range(K + 1):
                t_val = ts_eval[k].item()
                Vt_val = Vs_eval[k].item()
                rl2 = rel_l2(net, t_val, Vt_val, lambda_, sigma_fn, d,
                             cfg.eval.n_metric_samples, device)
                al2 = abs_l2(net, t_val, Vt_val, lambda_, sigma_fn, d,
                             cfg.eval.n_metric_samples, device)
                al_inf = abs_linf(net, t_val, Vt_val, lambda_, sigma_fn, d,
                                  cfg.eval.n_metric_samples, device)
                rl2_vals.append(rl2)
                al2_vals.append(al2)
                al_inf_vals.append(al_inf)
                wandb_metrics[f"rel_l2/t{k:03d}"] = rl2
                wandb_metrics[f"abs_l2/t{k:03d}"] = al2
                wandb_metrics[f"abs_linf/t{k:03d}"] = al_inf

            snapshots.append({
                "outer_it": outer_it,
                "rel_l2": rl2_vals,
                "abs_l2": al2_vals,
                "abs_linf": al_inf_vals,
            })

            # Summarise: print first, middle, last slice
            indices = [0, K // 2, K]
            summary = "  [eval] " + "  ".join(
                f"t={ts_eval[k].item():.2f} RelL2={rl2_vals[k]:.4f}"
                for k in indices
            )
            print(summary)

            if cfg.logging.wandb:
                wandb.log({"outer_it": outer_it, **wandb_metrics})

    print("Training complete.")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_snapshots({"snapshots": snapshots, "ts": ts_list}, output_dir)
    plot_convergence(snapshots, ts_list, output_dir)


if __name__ == "__main__":
    main()

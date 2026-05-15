"""Right-to-left convergence experiment — bimodal log-mixture reward.

Uses the full adjoint sampling objective (arXiv:2504.11713) including the
log p₁^base term, so the terminal distribution matches the Boltzmann target
p^{u*}(x₁) ∝ exp(r(x₁)) exactly.

Target reward:
    r(x) = log(w₁ exp(-λ₁/2‖x-μ₁‖²) + w₂ exp(-λ₂/2‖x-μ₂‖²))

Full terminal cost (§3 of CLAUDE.md):
    g(x) = log p₁^base(x) − r(x),   ∇g(x) = −x/ν₁ − ∇r(x)

Analytic optimal control via Feynman-Kac (§4 of CLAUDE.md):
    λᵢ* = λᵢ − 1/ν₁,   μᵢ* = λᵢ μᵢ / λᵢ*

Run:
    python experiments/right_to_left_convergence_bimodal/run.py experiment=right_to_left_convergence_bimodal
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))

from adjoint_sampling import DriftMLP, Sampler, ReplayBuffer, ram_loss, utils
from plotting import (plot_convergence, plot_heatmaps, plot_optimal_control,
                      plot_contraction_heatmaps, plot_control_evolution,
                      plot_terminal_distributions, plot_terminal_evolution,
                      plot_inner_steps, plot_inner_convergence,
                      save_snapshots)


# ---------------------------------------------------------------------------
# Analytic optimal control (full Feynman-Kac)  §4 of CLAUDE.md
# ---------------------------------------------------------------------------

def sigma_integral(t: Tensor, sigma_fn) -> Tensor:
    """Σ_t = ∫_t^1 σ(s)² ds = σ₀²(1−t) for constant σ."""
    return sigma_fn(t) ** 2 * (1.0 - t)


def A_component(x: Tensor, t: Tensor,
                w: float, lambda_: float, mu_i: float,
                sigma_fn, d: int) -> Tensor:
    """A(t,x;w,λ,μ) = w/(1+λΣ_t)^{d/2} exp(−λ‖x−μ‖²/(2(1+λΣ_t))).

    Used for both simplified and full objectives by passing the appropriate
    (λ, μ) — for the full objective these are the effective (λ*, μ*).
    x: [B, d], t: [B] → [B].
    """
    Sigma_t = sigma_integral(t, sigma_fn)          # [B]
    denom = 1.0 + lambda_ * Sigma_t                # [B]
    sq_dist = ((x - mu_i) ** 2).sum(-1)            # [B]
    log_A = (math.log(w)
             - 0.5 * d * denom.log()
             - lambda_ * sq_dist / (2.0 * denom))
    return log_A.exp()                              # [B]


def optimal_control(x: Tensor, t: Tensor,
                    w1: float, lambda1: float, mu1: float,
                    w2: float, lambda2: float, mu2: float,
                    sigma_fn, nu_1: float, d: int) -> Tensor:
    """u*(t,x) for the full objective g = log p₁^base − r.

    Effective parameters: λᵢ* = λᵢ − 1/ν₁,  μᵢ* = λᵢ μᵢ / λᵢ*.
    Uses log-space normalisation so the result is well-defined even when x
    is far from both modes (avoids 0/0 from Gaussian underflow).
    Requires λᵢ > 1/ν₁ so that λᵢ* > 0.

    x: [B, d], t: [B] → [B, d]
    """
    lambda1_star = lambda1 - 1.0 / nu_1
    lambda2_star = lambda2 - 1.0 / nu_1
    mu1_star = lambda1 * mu1 / lambda1_star
    mu2_star = lambda2 * mu2 / lambda2_star

    Sigma_t = sigma_integral(t, sigma_fn)
    denom1 = (1.0 + lambda1_star * Sigma_t).unsqueeze(-1)  # [B, 1]
    denom2 = (1.0 + lambda2_star * Sigma_t).unsqueeze(-1)
    sq1 = ((x - mu1_star) ** 2).sum(-1, keepdim=True)      # [B, 1]
    sq2 = ((x - mu2_star) ** 2).sum(-1, keepdim=True)
    log_A1 = math.log(w1) - 0.5 * d * denom1.log() - lambda1_star * sq1 / (2.0 * denom1)
    log_A2 = math.log(w2) - 0.5 * d * denom2.log() - lambda2_star * sq2 / (2.0 * denom2)
    log_sum = torch.logaddexp(log_A1, log_A2)
    n1 = (log_A1 - log_sum).exp()                          # softmax weight ∈ [0,1]
    n2 = (log_A2 - log_sum).exp()
    numer = (lambda1_star * (x - mu1_star) / denom1 * n1
             + lambda2_star * (x - mu2_star) / denom2 * n2)
    sigma_t = sigma_fn(t).unsqueeze(-1)
    return -sigma_t * numer                                 # [B, d]


def grad_r(x1: Tensor,
           w1: float, lambda1: float, mu1: float,
           w2: float, lambda2: float, mu2: float) -> Tensor:
    """∇r(x₁) at t=T (Σ_T=0): A_i(T,x) = w_i exp(−λᵢ‖x−μᵢ‖²/2).

    Uses log-space normalisation to avoid 0/0 when x is far from both modes.
    x1: [B, d] → [B, d]
    """
    sq1 = ((x1 - mu1) ** 2).sum(-1, keepdim=True)
    sq2 = ((x1 - mu2) ** 2).sum(-1, keepdim=True)
    log_A1 = math.log(w1) - 0.5 * lambda1 * sq1
    log_A2 = math.log(w2) - 0.5 * lambda2 * sq2
    log_sum = torch.logaddexp(log_A1, log_A2)
    n1 = (log_A1 - log_sum).exp()                  # softmax weight ∈ [0,1]
    n2 = (log_A2 - log_sum).exp()
    return -(lambda1 * (x1 - mu1) * n1 + lambda2 * (x1 - mu2) * n2)


def terminal_mixture_params(
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
) -> tuple[float, float, float, float, float, float]:
    """Return (α₁, μ₁, v₁, α₂, μ₂, v₂) for p^{u*}(x₁) = Σ αᵢ N(μᵢ, vᵢ).

    Full objective: p^{u*} ∝ exp(r(x₁)), which normalises to a mixture of
    N(μᵢ, 1/λᵢ) with weights αᵢ ∝ wᵢ/√λᵢ.  No σ₀ dependence.
    """
    a1 = w1 / math.sqrt(lambda1)
    a2 = w2 / math.sqrt(lambda2)
    norm = a1 + a2
    return a1 / norm, mu1, 1.0 / lambda1, a2 / norm, mu2, 1.0 / lambda2


# ---------------------------------------------------------------------------
# Evaluation metrics  §6 of CLAUDE.md
# ---------------------------------------------------------------------------

@torch.no_grad()
def rel_l2(u_theta: nn.Module, x_samples: Tensor, t_val: float,
           w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d) -> float:
    """RelL₂(t) using pre-simulated x_samples ~ P^{u*}_t."""
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star_val = optimal_control(x_samples, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, nu_1, d)
    num = (u_hat - u_star_val).pow(2).sum(-1).mean().item()
    den = u_star_val.pow(2).sum(-1).mean().item()
    if den < 1e-12:
        return float("nan")
    return math.sqrt(num / den)


@torch.no_grad()
def abs_l2(u_theta: nn.Module, x_samples: Tensor, t_val: float,
           w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d) -> float:
    """AbsL₂(t) using pre-simulated x_samples."""
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star_val = optimal_control(x_samples, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, nu_1, d)
    return math.sqrt((u_hat - u_star_val).pow(2).sum(-1).mean().item())


@torch.no_grad()
def _error_field(u_theta: nn.Module, xs: Tensor, t_val: float,
                 w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d) -> Tensor:
    """‖u_θ(t,x)−u*(t,x)‖ for each x in the 1-D grid xs, shape [n_grid]."""
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_hat = u_theta(x_t, t)
    u_star_val = optimal_control(x_t, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, nu_1, d)
    return (u_hat - u_star_val).norm(dim=-1)


@torch.no_grad()
def abs_linf(u_theta: nn.Module, t_val: float,
             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d,
             xs: Tensor) -> tuple[float, list[float]]:
    """AbsL∞(t) = max_x ‖u_θ−u*‖. Returns (scalar_max, per_point_errors)."""
    field = _error_field(u_theta, xs, t_val, w1, lambda1, mu1, w2, lambda2, mu2,
                         sigma_fn, nu_1, d)
    return field.max().item(), field.tolist()


@torch.no_grad()
def control_field(u_theta: nn.Module, t_val: float, d: int, xs: Tensor) -> list[float]:
    """u_θ(t,x) for each x in xs. Returns scalar for d=1, norm for d>1."""
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_hat = u_theta(x_t, t)
    if d == 1:
        return u_hat[:, 0].tolist()
    return u_hat.norm(dim=-1).tolist()


@torch.no_grad()
def optimal_control_field(t_val: float,
                          w1, lambda1, mu1, w2, lambda2, mu2,
                          sigma_fn, nu_1, d: int, xs: Tensor) -> list[float]:
    """u*(t,x) for each x in xs. Returns scalar for d=1, norm for d>1."""
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_star_val = optimal_control(x_t, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, nu_1, d)
    if d == 1:
        return u_star_val[:, 0].tolist()
    return u_star_val.norm(dim=-1).tolist()


# ---------------------------------------------------------------------------
# Sample path generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_paths(control_fn, n_paths: int, ts: Tensor,
                   d: int, sigma_fn, device) -> Tensor:
    """Simulate n_paths Euler-Maruyama trajectories from X_0=0.

    SDE: dX = σ(t) u(X,t) dt + σ(t) dB.
    Returns [K+1, n_paths, d] tensor (full state at each time step).
    Used for metric sampling (L₂) and as the base for euler_maruyama_paths.
    """
    x = torch.zeros(n_paths, d, device=device)
    snapshots = [x.clone()]
    for i in range(ts.shape[0] - 1):
        t_vec = ts[i].expand(n_paths)
        dt = (ts[i + 1] - ts[i]).item()
        u = control_fn(x, t_vec)
        sigma_t = sigma_fn(t_vec).unsqueeze(-1)
        x = x + sigma_t * u * dt + sigma_t * math.sqrt(dt) * torch.randn_like(x)
        snapshots.append(x.clone())
    return torch.stack(snapshots, dim=0)   # [K+1, n_paths, d]


@torch.no_grad()
def euler_maruyama_paths(control_fn, n_paths: int, ts: Tensor,
                         d: int, sigma_fn, device) -> list[list[float]]:
    """Sample n_paths trajectories, returning [n_paths][K+1] first-coord values."""
    all_steps = simulate_paths(control_fn, n_paths, ts, d, sigma_fn, device)
    traj = all_steps[:, :, 0].tolist()   # [K+1][n_paths]
    return [[traj[k][b] for k in range(len(traj))] for b in range(n_paths)]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    utils.seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    d = cfg.target.d
    w1 = float(cfg.target.w1)
    w2 = 1.0 - w1
    lambda1 = float(cfg.target.lambda1)
    mu1 = float(cfg.target.mu1)
    lambda2 = float(cfg.target.lambda2)
    mu2 = float(cfg.target.mu2)

    sigma_fn = utils.sigma_constant(cfg.sigma)
    nu_fn = utils.nu_constant(cfg.sigma)
    nu_1 = float(cfg.sigma ** 2)

    assert lambda1 > 1.0 / nu_1 and lambda2 > 1.0 / nu_1, (
        f"Full objective requires λᵢ > 1/ν₁; got λ1={lambda1}, λ2={lambda2}, ν₁={nu_1:.3f}. "
        f"Increase λ or decrease σ."
    )

    net = DriftMLP(
        d=d,
        hidden_dim=cfg.network.hidden_dim,
        n_layers=cfg.network.n_layers,
        t_emb_dim=cfg.network.t_emb_dim,
    ).to(device)

    sampler = Sampler(sigma_fn, steps=cfg.sampler.steps)
    buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)

    K = cfg.eval.n_time_slices
    ts_eval = torch.linspace(0.0, 1.0, K + 1, device=device)

    # x-grid centred at (μ₁+μ₂)/2
    x_center = (mu1 + mu2) / 2.0
    xs_linf = torch.linspace(x_center - cfg.eval.linf_x_range,
                             x_center + cfg.eval.linf_x_range,
                             cfg.eval.n_linf_grid, device=device)
    xs_list = xs_linf.tolist()

    # Ground-truth u*(t,x) field — computed once
    with torch.no_grad():
        u_star_field: list[list[float]] = [
            optimal_control_field(ts_eval[k].item(),
                                  w1, lambda1, mu1, w2, lambda2, mu2,
                                  sigma_fn, nu_1, d, xs_linf)
            for k in range(K + 1)
        ]

    # Pre-simulate metric paths under u* once — marginal is fixed across training
    u_star_fn = lambda x, t: optimal_control(
        x, t, w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d)
    metric_paths = simulate_paths(u_star_fn, cfg.eval.n_metric_samples,
                                  ts_eval, d, sigma_fn, device)
    # metric_paths: [K+1, n_metric_samples, d]

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    def grad_g_fn(x1: Tensor) -> Tensor:
        """∇g(x₁) = −x₁/ν₁ − ∇r(x₁)  (full adjoint sampling objective)."""
        return -x1 / nu_1 - grad_r(x1, w1, lambda1, mu1, w2, lambda2, mu2)

    snapshots: list[dict] = []
    ts_list = ts_eval.tolist()
    prev_al_inf_vals: list[float] | None = None
    prev_tiled_al_inf_vals: list[float] | None = None

    for outer_it in range(cfg.training.outer_iterations):

        x1 = sampler.sample(net, cfg.algorithm.n_outer, d, device)
        gg = grad_g_fn(x1)
        buffer.add(x1, gg)

        # Reset Adam first moments so each inner loop starts with a clean gradient
        # direction. Second moments (per-parameter step sizes) are preserved.
        for state in optim.state.values():
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
        for group in optim.param_groups:
            group['lr'] = cfg.training.lr

        n_max = cfg.algorithm.n_inner_steps
        warmup_steps = max(1, int(cfg.training.inner_warmup_frac * n_max))
        lr_base = cfg.training.lr
        lr_min_val = cfg.training.lr_min
        record_curve = (outer_it % cfg.eval.inner_curve_every == 0)

        loss_window: list[float] = []
        inner_loss_curve: list[float] = []
        inner_steps_taken = 0
        current_lr = lr_base * 1e-4
        for inner_step in range(n_max):
            if inner_step < warmup_steps:
                current_lr = lr_base * (1e-4 + (1.0 - 1e-4) * inner_step / warmup_steps)
            else:
                progress = (inner_step - warmup_steps) / max(1, n_max - warmup_steps)
                current_lr = lr_min_val + 0.5 * (lr_base - lr_min_val) * (1.0 + math.cos(math.pi * progress))
            for group in optim.param_groups:
                group['lr'] = current_lr

            x1_b, gg_b = buffer.sample(cfg.algorithm.n_inner, device=device)
            loss = ram_loss(net, x1_b, gg_b, sigma_fn, nu_fn, nu_1)
            optim.zero_grad()
            loss.backward()
            optim.step()
            inner_steps_taken += 1

            if record_curve:
                inner_loss_curve.append(loss.item())

            if cfg.algorithm.inner_tol > 0:
                loss_window.append(loss.item())
                if len(loss_window) > cfg.algorithm.inner_patience:
                    loss_window.pop(0)
                if (len(loss_window) == cfg.algorithm.inner_patience
                        and max(loss_window) - min(loss_window) < cfg.algorithm.inner_tol):
                    break

        if outer_it % cfg.logging.log_every == 0:
            print(f"[{outer_it:4d}] ram_loss={loss.item():.4f}  "
                  f"lr={current_lr:.2e}  inner_steps={inner_steps_taken}")
            if cfg.logging.wandb:
                wandb.log({"ram_loss": loss.item(), "lr": current_lr,
                           "inner_steps": inner_steps_taken, "outer_it": outer_it})

        if outer_it < cfg.eval.first_k or outer_it % cfg.eval.every == 0:
            rl2_vals, al2_vals, al_inf_vals = [], [], []
            error_fields: list[list[float]] = []
            wandb_metrics: dict = {}

            for k in range(K + 1):
                t_val = ts_eval[k].item()
                x_samples = metric_paths[k]   # [n_metric_samples, d]
                rl2 = rel_l2(net, x_samples, t_val,
                             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d)
                al2 = abs_l2(net, x_samples, t_val,
                             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d)
                al_inf, field = abs_linf(net, t_val,
                                         w1, lambda1, mu1, w2, lambda2, mu2,
                                         sigma_fn, nu_1, d, xs_linf)
                rl2_vals.append(rl2)
                al2_vals.append(al2)
                al_inf_vals.append(al_inf)
                error_fields.append(field)
                wandb_metrics[f"rel_l2/t{k:03d}"] = rl2
                wandb_metrics[f"abs_l2/t{k:03d}"] = al2
                wandb_metrics[f"abs_linf/t{k:03d}"] = al_inf
                if prev_al_inf_vals is not None and prev_al_inf_vals[k] > 1e-12:
                    wandb_metrics[f"contr_fact/t{k:03d}"] = al_inf_vals[k] / prev_al_inf_vals[k]

            if prev_al_inf_vals is not None:
                contr_fact = [
                    (al_inf_vals[k] / prev_al_inf_vals[k])
                    if prev_al_inf_vals[k] > 1e-12 else float("nan")
                    for k in range(K + 1)
                ]
            else:
                contr_fact = [float("nan")] * (K + 1)

            tiled_al_inf_vals: list[float] = [0.0] * (K + 1)
            running = 0.0
            for k in range(K, -1, -1):
                running = max(running, al_inf_vals[k])
                tiled_al_inf_vals[k] = running

            ef_arr = np.array(error_fields)
            tiled_ef_arr = np.maximum.accumulate(ef_arr[::-1])[::-1]
            tiled_error_fields = tiled_ef_arr.tolist()

            if prev_tiled_al_inf_vals is not None:
                tiled_contr_fact = [
                    (tiled_al_inf_vals[k] / prev_tiled_al_inf_vals[k])
                    if prev_tiled_al_inf_vals[k] > 1e-12 else float("nan")
                    for k in range(K + 1)
                ]
            else:
                tiled_contr_fact = [float("nan")] * (K + 1)

            for k in range(K + 1):
                wandb_metrics[f"tiled_al_inf/t{k:03d}"] = tiled_al_inf_vals[k]
                if prev_tiled_al_inf_vals is not None and prev_tiled_al_inf_vals[k] > 1e-12:
                    wandb_metrics[f"tiled_contr_fact/t{k:03d}"] = tiled_contr_fact[k]

            snap_u_theta_field: list[list[float]] = [
                control_field(net, ts_eval[k].item(), d, xs_linf)
                for k in range(K + 1)
            ]
            n_sp = cfg.eval.n_sample_paths
            snap_paths_theta = (
                euler_maruyama_paths(net, n_sp, ts_eval, d, sigma_fn, device)
                if n_sp > 0 else None
            )

            snapshots.append({
                "outer_it": outer_it,
                "inner_steps": inner_steps_taken,
                "inner_loss_curve": inner_loss_curve if record_curve else None,
                "rel_l2": rl2_vals,
                "abs_l2": al2_vals,
                "abs_linf": al_inf_vals,
                "contr_fact": contr_fact,
                "error_fields": error_fields,
                "tiled_al_inf": tiled_al_inf_vals,
                "tiled_contr_fact": tiled_contr_fact,
                "tiled_error_fields": tiled_error_fields,
                "u_theta_field": snap_u_theta_field,
                "paths_theta": snap_paths_theta,
            })
            prev_al_inf_vals = al_inf_vals
            prev_tiled_al_inf_vals = tiled_al_inf_vals

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

    with torch.no_grad():
        u_theta_field: list[list[float]] = [
            control_field(net, ts_eval[k].item(), d, xs_linf)
            for k in range(K + 1)
        ]

    n_sp = cfg.eval.n_sample_paths
    if n_sp > 0:
        paths_star = euler_maruyama_paths(u_star_fn, n_sp, ts_eval, d, sigma_fn, device)
        paths_theta = euler_maruyama_paths(net, n_sp, ts_eval, d, sigma_fn, device)
    else:
        paths_star = paths_theta = None

    # Terminal distribution: p^{u*} ∝ exp(r(x₁)) = mixture of N(μᵢ, 1/λᵢ)
    alpha1, m1s, v1s, alpha2, m2s, v2s = terminal_mixture_params(
        w1, lambda1, mu1, w2, lambda2, mu2)

    def target_pdf_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        p1 = alpha1 * np.exp(-0.5 * (x - m1s) ** 2 / v1s) / np.sqrt(2 * np.pi * v1s)
        p2 = alpha2 * np.exp(-0.5 * (x - m2s) ** 2 / v2s) / np.sqrt(2 * np.pi * v2s)
        return p1 + p2

    target_params = dict(w1=w1, lambda1=lambda1, mu1=mu1, w2=w2, lambda2=lambda2, mu2=mu2)

    save_snapshots({"snapshots": snapshots, "ts": ts_list, "xs": xs_list,
                    "u_star_field": u_star_field, "u_theta_field": u_theta_field,
                    "paths_star": paths_star, "paths_theta": paths_theta,
                    "target_params": target_params, "d": d},
                   output_dir)
    plot_convergence(snapshots, ts_list, output_dir)
    plot_inner_steps(snapshots, output_dir)
    plot_inner_convergence(snapshots, output_dir)
    plot_optimal_control(ts_list, xs_list, u_star_field, d, output_dir,
                         u_theta_field, paths_star, paths_theta)
    plot_control_evolution(snapshots, ts_list, xs_list, d, u_star_field, output_dir)
    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d, output_dir)
    plot_terminal_evolution(snapshots, target_pdf_fn, d, output_dir)
    plot_heatmaps(snapshots, ts_list, xs_list, output_dir)
    plot_contraction_heatmaps(snapshots, ts_list, xs_list, output_dir)


if __name__ == "__main__":
    main()

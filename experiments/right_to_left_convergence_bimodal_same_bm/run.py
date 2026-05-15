"""Right-to-left convergence — bimodal reward, same-BM evaluation.

Extends the bimodal experiment by directly estimating the LHS of the §2.1
contraction bound using shared Brownian motion for X^{u_θ} and X^{u*}:

    LHS_SBM(t, x) = E_B[ ‖σ(t)∇r(X_T^{u_θ,x}) − σ(t)∇r(X_T^{u*,x})‖ ]

where both SDEs use the same BM sample B_{t:T}.  This quantity → 0 as t → T
(right-to-left), which is what the bound predicts.

Run:
    python experiments/right_to_left_convergence_bimodal_same_bm/run.py \
        experiment=right_to_left_convergence_bimodal_same_bm
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
from plotting import (
    plot_convergence, plot_heatmaps, plot_optimal_control,
    plot_contraction_heatmaps, plot_control_evolution,
    plot_terminal_distributions, plot_terminal_evolution,
    plot_inner_steps, plot_inner_convergence,
    plot_same_bm_lhs_curves, plot_same_bm_lhs_heatmaps,
    plot_tiled_same_bm_lhs, plot_same_vs_diff_bm_lhs,
    plot_sbm_ratio,
    save_snapshots,
)


# ---------------------------------------------------------------------------
# Analytic optimal control — full Feynman-Kac (same as bimodal experiment)
# ---------------------------------------------------------------------------

def sigma_integral(t: Tensor, sigma_fn) -> Tensor:
    """Σ_t = ∫_t^1 σ(s)² ds = σ₀²(1−t) for constant σ."""
    return sigma_fn(t) ** 2 * (1.0 - t)


def optimal_control(x: Tensor, t: Tensor,
                    w1: float, lambda1: float, mu1: float,
                    w2: float, lambda2: float, mu2: float,
                    sigma_fn, nu_1: float, d: int) -> Tensor:
    """u*(t,x) for the full objective g = log p₁^base − r.

    Effective parameters: λᵢ* = λᵢ − 1/ν₁,  μᵢ* = λᵢ μᵢ / λᵢ*.
    Log-space normalisation avoids 0/0 for x far from both modes.
    Requires λᵢ > 1/ν₁.  x: [B, d], t: [B] → [B, d].
    """
    lambda1_star = lambda1 - 1.0 / nu_1
    lambda2_star = lambda2 - 1.0 / nu_1
    mu1_star = lambda1 * mu1 / lambda1_star
    mu2_star = lambda2 * mu2 / lambda2_star

    Sigma_t = sigma_integral(t, sigma_fn)
    denom1 = (1.0 + lambda1_star * Sigma_t).unsqueeze(-1)
    denom2 = (1.0 + lambda2_star * Sigma_t).unsqueeze(-1)
    sq1 = ((x - mu1_star) ** 2).sum(-1, keepdim=True)
    sq2 = ((x - mu2_star) ** 2).sum(-1, keepdim=True)
    log_A1 = math.log(w1) - 0.5 * d * denom1.log() - lambda1_star * sq1 / (2.0 * denom1)
    log_A2 = math.log(w2) - 0.5 * d * denom2.log() - lambda2_star * sq2 / (2.0 * denom2)
    log_sum = torch.logaddexp(log_A1, log_A2)
    n1 = (log_A1 - log_sum).exp()
    n2 = (log_A2 - log_sum).exp()
    numer = (lambda1_star * (x - mu1_star) / denom1 * n1
             + lambda2_star * (x - mu2_star) / denom2 * n2)
    sigma_t = sigma_fn(t).unsqueeze(-1)
    return -sigma_t * numer


def grad_r(x1: Tensor,
           w1: float, lambda1: float, mu1: float,
           w2: float, lambda2: float, mu2: float) -> Tensor:
    """∇r(x₁).  Log-space normalisation avoids 0/0.  x1: [B, d] → [B, d]."""
    sq1 = ((x1 - mu1) ** 2).sum(-1, keepdim=True)
    sq2 = ((x1 - mu2) ** 2).sum(-1, keepdim=True)
    log_A1 = math.log(w1) - 0.5 * lambda1 * sq1
    log_A2 = math.log(w2) - 0.5 * lambda2 * sq2
    log_sum = torch.logaddexp(log_A1, log_A2)
    n1 = (log_A1 - log_sum).exp()
    n2 = (log_A2 - log_sum).exp()
    return -(lambda1 * (x1 - mu1) * n1 + lambda2 * (x1 - mu2) * n2)


def terminal_mixture_params(
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
) -> tuple[float, float, float, float, float, float]:
    """(α₁, μ₁, v₁, α₂, μ₂, v₂) for p^{u*}(x₁) = Σ αᵢ N(μᵢ, vᵢ).

    Full objective: p^{u*} ∝ exp(r(x₁)), αᵢ ∝ wᵢ/√λᵢ.
    """
    a1 = w1 / math.sqrt(lambda1)
    a2 = w2 / math.sqrt(lambda2)
    norm = a1 + a2
    return a1 / norm, mu1, 1.0 / lambda1, a2 / norm, mu2, 1.0 / lambda2


# ---------------------------------------------------------------------------
# Standard evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def rel_l2(u_theta: nn.Module, x_samples: Tensor, t_val: float,
           w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d) -> float:
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
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star_val = optimal_control(x_samples, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, nu_1, d)
    return math.sqrt((u_hat - u_star_val).pow(2).sum(-1).mean().item())


@torch.no_grad()
def _error_field(u_theta: nn.Module, xs: Tensor, t_val: float,
                 w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d) -> Tensor:
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
    field = _error_field(u_theta, xs, t_val, w1, lambda1, mu1, w2, lambda2, mu2,
                         sigma_fn, nu_1, d)
    return field.max().item(), field.tolist()


@torch.no_grad()
def control_field(u_theta: nn.Module, t_val: float, d: int, xs: Tensor) -> list[float]:
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
# Path simulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_paths(control_fn, n_paths: int, ts: Tensor,
                   d: int, sigma_fn, device) -> Tensor:
    """Simulate n_paths EM trajectories from X_0=0.  Returns [K+1, n_paths, d]."""
    x = torch.zeros(n_paths, d, device=device)
    snapshots = [x.clone()]
    for i in range(ts.shape[0] - 1):
        t_vec = ts[i].expand(n_paths)
        dt = (ts[i + 1] - ts[i]).item()
        u = control_fn(x, t_vec)
        sigma_t = sigma_fn(t_vec).unsqueeze(-1)
        x = x + sigma_t * u * dt + sigma_t * math.sqrt(dt) * torch.randn_like(x)
        snapshots.append(x.clone())
    return torch.stack(snapshots, dim=0)


@torch.no_grad()
def euler_maruyama_paths(control_fn, n_paths: int, ts: Tensor,
                         d: int, sigma_fn, device) -> list[list[float]]:
    all_steps = simulate_paths(control_fn, n_paths, ts, d, sigma_fn, device)
    traj = all_steps[:, :, 0].tolist()
    return [[traj[k][b] for k in range(len(traj))] for b in range(n_paths)]


# ---------------------------------------------------------------------------
# Same-BM LHS metric (core new function)
# ---------------------------------------------------------------------------

@torch.no_grad()
def same_bm_lhs_field(
    u_fn,
    v_fn,
    xs: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_bm_samples: int,
    device,
) -> Tensor:
    """Estimate LHS_SBM(t, x) = E_B[‖σ(t)∇r(X_T^{u,x}) − σ(t)∇r(X_T^{v,x})‖]
    for each x in xs, using the same BM for both X^u and X^v.

    u_fn = u_θ (learned),  v_fn = u* (optimal).

    xs: [n_grid] 1-D grid values
    t_start: evaluation time t (start of the shared BM segment)
    ts: [K+1] uniform time grid covering [0, T]
    Returns [n_grid] tensor of per-point LHS estimates.

    At t_start = T: the loop runs zero steps, both processes stay at x,
    ∇r differs by 0, and the result is exactly 0.
    """
    n_grid = xs.shape[0]

    # Find the index in ts closest to t_start
    t_start_idx = int((ts - t_start).abs().argmin().item())

    # Build starting states on the 1-D x-grid: [n_grid, d]
    x_start = torch.zeros(n_grid, d, device=device)
    x_start[:, 0] = xs

    # Expand for BM replication: [n_grid * n_bm_samples, d]
    xu = x_start.repeat_interleave(n_bm_samples, dim=0)
    xv = xu.clone()

    # Simulate both SDEs from t_start to T with shared noise
    for i in range(t_start_idx, ts.shape[0] - 1):
        t_curr = ts[i].item()
        dt = (ts[i + 1] - ts[i]).item()
        N = xu.shape[0]
        t_vec = torch.full((N,), t_curr, device=device)
        sigma_step = sigma_fn(t_vec).unsqueeze(-1)   # [N, 1]
        noise = torch.randn_like(xu)                  # same noise for u and v

        uu = u_fn(xu, t_vec)
        uv = v_fn(xv, t_vec)

        xu = xu + sigma_step * uu * dt + sigma_step * math.sqrt(dt) * noise
        xv = xv + sigma_step * uv * dt + sigma_step * math.sqrt(dt) * noise

    # σ(t) at the evaluation starting time
    sigma_t = sigma_fn(torch.full((1,), t_start, device=device)).item()

    gr_u = grad_r(xu, w1, lambda1, mu1, w2, lambda2, mu2)  # [n_grid*n_bm, d]
    gr_v = grad_r(xv, w1, lambda1, mu1, w2, lambda2, mu2)

    diff = (sigma_t * (gr_u - gr_v)).norm(dim=-1)          # [n_grid * n_bm_samples]
    return diff.view(n_grid, n_bm_samples).mean(dim=1)      # [n_grid]


@torch.no_grad()
def diff_bm_lhs_field(
    u_fn,
    v_fn,
    xs: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_bm_samples: int,
    device,
) -> Tensor:
    """Estimate E_{B1,B2}[‖σ(t)∇r(X_T^{u,x}_{B1}) − σ(t)∇r(X_T^{v,x}_{B2})‖]
    for each x in xs, using INDEPENDENT BMs for X^u and X^v.

    Identical to same_bm_lhs_field except that each SDE draws its own noise,
    so the noise terms in the difference process do NOT cancel.  This gives a
    larger, non-contracting quantity and serves as the comparison baseline to
    isolate the role of the shared BM in producing the contraction.
    Returns [n_grid] tensor of per-point estimates.
    """
    n_grid = xs.shape[0]
    t_start_idx = int((ts - t_start).abs().argmin().item())

    x_start = torch.zeros(n_grid, d, device=device)
    x_start[:, 0] = xs

    xu = x_start.repeat_interleave(n_bm_samples, dim=0)
    xv = xu.clone()

    for i in range(t_start_idx, ts.shape[0] - 1):
        t_curr = ts[i].item()
        dt = (ts[i + 1] - ts[i]).item()
        N = xu.shape[0]
        t_vec = torch.full((N,), t_curr, device=device)
        sigma_step = sigma_fn(t_vec).unsqueeze(-1)
        noise_u = torch.randn_like(xu)   # independent BM for u
        noise_v = torch.randn_like(xv)   # independent BM for v

        uu = u_fn(xu, t_vec)
        uv = v_fn(xv, t_vec)

        xu = xu + sigma_step * uu * dt + sigma_step * math.sqrt(dt) * noise_u
        xv = xv + sigma_step * uv * dt + sigma_step * math.sqrt(dt) * noise_v

    sigma_t = sigma_fn(torch.full((1,), t_start, device=device)).item()
    gr_u = grad_r(xu, w1, lambda1, mu1, w2, lambda2, mu2)
    gr_v = grad_r(xv, w1, lambda1, mu1, w2, lambda2, mu2)

    diff = (sigma_t * (gr_u - gr_v)).norm(dim=-1)
    return diff.view(n_grid, n_bm_samples).mean(dim=1)


# ---------------------------------------------------------------------------
# Main
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
        f"Full objective requires λᵢ > 1/ν₁; got λ1={lambda1}, λ2={lambda2}, ν₁={nu_1:.3f}."
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

    x_center = (mu1 + mu2) / 2.0
    xs_linf = torch.linspace(x_center - cfg.eval.linf_x_range,
                             x_center + cfg.eval.linf_x_range,
                             cfg.eval.n_linf_grid, device=device)
    xs_list = xs_linf.tolist()

    # Same-BM grid (independent of linf grid, typically smaller)
    do_same_bm = bool(cfg.eval.same_bm_eval)
    if do_same_bm:
        xs_sbm = torch.linspace(x_center - cfg.eval.linf_x_range,
                                x_center + cfg.eval.linf_x_range,
                                cfg.eval.n_same_bm_grid, device=device)
        xs_sbm_list = xs_sbm.tolist()
        n_sbm = int(cfg.eval.n_same_bm_samples)
        sbm_every = int(cfg.eval.same_bm_every)
    else:
        xs_sbm = xs_sbm_list = None
        n_sbm = sbm_every = 0

    u_star_fn = lambda x, t: optimal_control(
        x, t, w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d)

    # u* field on linf grid — computed once
    with torch.no_grad():
        u_star_field: list[list[float]] = [
            optimal_control_field(ts_eval[k].item(),
                                  w1, lambda1, mu1, w2, lambda2, mu2,
                                  sigma_fn, nu_1, d, xs_linf)
            for k in range(K + 1)
        ]

    # Metric paths under u* — fixed throughout training
    metric_paths = simulate_paths(u_star_fn, cfg.eval.n_metric_samples,
                                  ts_eval, d, sigma_fn, device)

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

        # Reset Adam first moments; preserve second moments (per-param step sizes)
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
                x_samples = metric_paths[k]
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

            # -------------------------------------------------------------------
            # Same-BM LHS evaluation
            # -------------------------------------------------------------------
            _EPS_RATIO = 1e-12
            if do_same_bm and outer_it % sbm_every == 0:
                sbm_fields: list[list[float]] = []
                sbm_error_fields: list[list[float]] = []   # ‖u_θ−u*‖ on xs_sbm grid
                for k in range(K + 1):
                    t_val = ts_eval[k].item()
                    lhs = same_bm_lhs_field(
                        u_fn=net,
                        v_fn=u_star_fn,
                        xs=xs_sbm,
                        t_start=t_val,
                        ts=ts_eval,
                        sigma_fn=sigma_fn,
                        d=d,
                        w1=w1, lambda1=lambda1, mu1=mu1,
                        w2=w2, lambda2=lambda2, mu2=mu2,
                        n_bm_samples=n_sbm,
                        device=device,
                    )
                    sbm_fields.append(lhs.tolist())
                    wandb_metrics[f"sbm_lhs_mean/t{k:03d}"] = float(np.mean(lhs.tolist()))
                    wandb_metrics[f"sbm_lhs_max/t{k:03d}"] = float(np.max(lhs.tolist()))

                    # pointwise error ‖u_θ(t,x) − u*(t,x)‖ on same sbm grid
                    err_field = _error_field(net, xs_sbm, t_val,
                                             w1, lambda1, mu1, w2, lambda2, mu2,
                                             sigma_fn, nu_1, d)
                    sbm_error_fields.append(err_field.tolist())

                # tiled_same_bm_lhs[k] = sup_{s >= t_k, x} LHS_SBM(s, x)
                tiled_sbm: list[float] = [0.0] * (K + 1)
                running_sbm = 0.0
                for k in range(K, -1, -1):
                    running_sbm = max(running_sbm, max(sbm_fields[k]))
                    tiled_sbm[k] = running_sbm
                for k in range(K + 1):
                    wandb_metrics[f"tiled_sbm_lhs/t{k:03d}"] = tiled_sbm[k]

                # tiled_al_inf_sbm[k] = sup_{s >= t_k, x on sbm grid} ‖u_θ−u*‖(s,x)
                tiled_al_inf_sbm: list[float] = [0.0] * (K + 1)
                running_err = 0.0
                for k in range(K, -1, -1):
                    running_err = max(running_err, max(sbm_error_fields[k]))
                    tiled_al_inf_sbm[k] = running_err

                # pointwise ratio: LHS_SBM(t,x) / ‖u_θ−u*‖(t,x)
                sbm_ratio_fields: list[list[float]] = []
                for k in range(K + 1):
                    lhs_k = np.array(sbm_fields[k])
                    err_k = np.maximum(np.array(sbm_error_fields[k]), _EPS_RATIO)
                    sbm_ratio_fields.append((lhs_k / err_k).tolist())

                # tiled ratio: tiled_sbm[k] / tiled_al_inf_sbm[k]
                tiled_sbm_ratio: list[float] = [
                    tiled_sbm[k] / max(tiled_al_inf_sbm[k], _EPS_RATIO)
                    for k in range(K + 1)
                ]
                for k in range(K + 1):
                    wandb_metrics[f"tiled_sbm_ratio/t{k:03d}"] = tiled_sbm_ratio[k]

                # Different-BM version: independent noise for X^u and X^v
                dbm_fields: list[list[float]] = []
                for k in range(K + 1):
                    t_val = ts_eval[k].item()
                    lhs_d = diff_bm_lhs_field(
                        u_fn=net, v_fn=u_star_fn,
                        xs=xs_sbm, t_start=t_val, ts=ts_eval,
                        sigma_fn=sigma_fn, d=d,
                        w1=w1, lambda1=lambda1, mu1=mu1,
                        w2=w2, lambda2=lambda2, mu2=mu2,
                        n_bm_samples=n_sbm, device=device,
                    )
                    dbm_fields.append(lhs_d.tolist())
                    wandb_metrics[f"dbm_lhs_mean/t{k:03d}"] = float(np.mean(lhs_d.tolist()))

                # pointwise diff-BM ratio
                dbm_ratio_fields: list[list[float]] = []
                for k in range(K + 1):
                    lhs_d_k = np.array(dbm_fields[k])
                    err_k = np.maximum(np.array(sbm_error_fields[k]), _EPS_RATIO)
                    dbm_ratio_fields.append((lhs_d_k / err_k).tolist())

                tiled_dbm: list[float] = [0.0] * (K + 1)
                running_dbm = 0.0
                for k in range(K, -1, -1):
                    running_dbm = max(running_dbm, max(dbm_fields[k]))
                    tiled_dbm[k] = running_dbm
                for k in range(K + 1):
                    wandb_metrics[f"tiled_dbm_lhs/t{k:03d}"] = tiled_dbm[k]

                # tiled diff-BM ratio
                tiled_dbm_ratio: list[float] = [
                    tiled_dbm[k] / max(tiled_al_inf_sbm[k], _EPS_RATIO)
                    for k in range(K + 1)
                ]
                for k in range(K + 1):
                    wandb_metrics[f"tiled_dbm_ratio/t{k:03d}"] = tiled_dbm_ratio[k]
            else:
                sbm_fields = None
                sbm_error_fields = None
                sbm_ratio_fields = None
                tiled_sbm = None
                tiled_sbm_ratio = None
                dbm_fields = None
                dbm_ratio_fields = None
                tiled_dbm = None
                tiled_dbm_ratio = None

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
                "same_bm_lhs_fields": sbm_fields,
                "sbm_error_fields": sbm_error_fields,
                "sbm_ratio_fields": sbm_ratio_fields,
                "tiled_same_bm_lhs": tiled_sbm,
                "tiled_sbm_ratio": tiled_sbm_ratio,
                "diff_bm_lhs_fields": dbm_fields,
                "dbm_ratio_fields": dbm_ratio_fields,
                "tiled_diff_bm_lhs": tiled_dbm,
                "tiled_dbm_ratio": tiled_dbm_ratio,
                "u_theta_field": snap_u_theta_field,
                "paths_theta": snap_paths_theta,
            })
            prev_al_inf_vals = al_inf_vals
            prev_tiled_al_inf_vals = tiled_al_inf_vals

            indices = [0, K // 2, K]
            sbm_summary = ""
            if sbm_fields is not None:
                sbm_means = [float(np.mean(sbm_fields[k])) for k in indices]
                sbm_summary = "  SBM: " + "  ".join(
                    f"t={ts_eval[k].item():.2f} {sbm_means[i]:.4f}"
                    for i, k in enumerate(indices)
                ) + f"  TiledSBM(0)={tiled_sbm[0]:.4f}"
            summary = "  [eval] " + "  ".join(
                f"t={ts_eval[k].item():.2f} RelL2={rl2_vals[k]:.4f}"
                for k in indices
            ) + sbm_summary
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

    alpha1, m1s, v1s, alpha2, m2s, v2s = terminal_mixture_params(
        w1, lambda1, mu1, w2, lambda2, mu2)

    def target_pdf_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        p1 = alpha1 * np.exp(-0.5 * (x - m1s) ** 2 / v1s) / np.sqrt(2 * np.pi * v1s)
        p2 = alpha2 * np.exp(-0.5 * (x - m2s) ** 2 / v2s) / np.sqrt(2 * np.pi * v2s)
        return p1 + p2

    target_params = dict(w1=w1, lambda1=lambda1, mu1=mu1, w2=w2, lambda2=lambda2, mu2=mu2)

    save_snapshots({"snapshots": snapshots, "ts": ts_list, "xs": xs_list,
                    "xs_sbm": xs_sbm_list, "u_star_field": u_star_field,
                    "u_theta_field": u_theta_field,
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
    if do_same_bm and xs_sbm_list:
        plot_same_bm_lhs_curves(snapshots, ts_list, output_dir)
        plot_same_bm_lhs_heatmaps(snapshots, ts_list, xs_sbm_list, output_dir)
        plot_tiled_same_bm_lhs(snapshots, ts_list, output_dir)
        plot_same_vs_diff_bm_lhs(snapshots, ts_list, output_dir)
        plot_sbm_ratio(snapshots, ts_list, output_dir)


if __name__ == "__main__":
    main()

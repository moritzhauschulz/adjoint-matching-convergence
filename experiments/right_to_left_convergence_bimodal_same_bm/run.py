"""Right-to-left convergence — bimodal reward, "Sanity Check" evaluation.

Trains a control u_θ (objective `ram` or `am`, see `algorithm.objective`) and,
at each eval, measures the §"Sanity Check" quantities from Project_Notes.tm
against the analytic optimal control u*:

  * ‖P(u*)(t,x) − u*(t,x)‖               fixed-point residual (u* self-consistency)
  * ‖P(u_θ)(t,x) − u*(t,x)‖               operator error (MC only on the u_θ rollout)
  * the two ratios of that over ‖u_θ − u*‖_{[t,T]}   (→ 0 as t → T)
  * the path-based analogues along u_θ trajectories

where P(u)(t,x) = −σ(t) E[∇g(X_T^{u,x}) | X_t = x] is the self-consistency
operator, ∇g(x) = −x/ν₁ − ∇r(x), and P(u*) = u*.  Project_Notes.tm writes P
without the leading minus (and a sign-flipped u*); the code keeps −σ∇g so u*
is the fixed point of the RAM target.  See this experiment's CLAUDE.md.

Run:
    python experiments/right_to_left_convergence_bimodal_same_bm/run.py \
        experiment=right_to_left_convergence_bimodal_same_bm
"""

import copy
import logging
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

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from adjoint_sampling import DriftMLP, Sampler, ReplayBuffer, ram_loss, am_loss, utils
from adjoint_sampling.operator import (
    operator_grid_field_all_times as _operator_grid_field_all_times,
    operator_field as _operator_field_at_points,
    fixed_point_residual_field,
)
from plotting import (
    plot_optimal_control,
    plot_operator_vs_next_control,
    plot_terminal_distributions,
    plot_fixed_point_residual,
    plot_sbm_ratio,
    plot_sbm_ratio_learned,
    plot_path_u_vs_Tu,
    plot_learned_pointwise_over_tiled,
    plot_path_u_vs_ustar_per_t_over_tiled,
    set_suffix_percentile,
    save_snapshots,
)


# ---------------------------------------------------------------------------
# Analytic optimal control — full Feynman-Kac (same as bimodal experiment)
# ---------------------------------------------------------------------------

def make_sigma_int_fn(nu_fn, nu_1: float):
    """Σ_t = ∫_t^1 σ(s)² ds = ν_1 − ν_t  (schedule-agnostic)."""
    def fn(t: Tensor) -> Tensor:
        return nu_1 - nu_fn(t)
    return fn


def optimal_control(x: Tensor, t: Tensor,
                    w1: float, lambda1: float, mu1: float,
                    w2: float, lambda2: float, mu2: float,
                    sigma_fn, sigma_int_fn, nu_1: float, d: int) -> Tensor:
    """u*(t,x) for the full adjoint sampling objective g = log p₁^base − r.

    Doob h-transform: u*(t,x) = σ(t) ∇_x log h(t,x), schedule-agnostic for an
    x-independent σ(t).  Effective parameters: λᵢ* = λᵢ − 1/ν₁, μᵢ* = λᵢ μᵢ / λᵢ*;
    Σ_t = ∫_t^1 σ(s)² ds is supplied by `sigma_int_fn`.

    Completing the square in −λᵢ(x₁−μᵢ)²/2 + x₁²/(2ν₁) leaves a mode-dependent
    constant κᵢ = λᵢ‖μᵢ‖²/(2ν₁λᵢ*) = d·λᵢμᵢ²/(2ν₁λᵢ*) (scalar μᵢ broadcast over d
    axes) that MUST enter log Aᵢ — it shifts the mixture weights unless the modes
    are symmetric (μ₁²=μ₂², equal λ), in which case it cancels in the softmax.

    Log-space normalisation avoids 0/0.  Requires λᵢ > 1/ν₁.  x: [B, d], t: [B] → [B, d].
    """
    lambda1_star = lambda1 - 1.0 / nu_1
    lambda2_star = lambda2 - 1.0 / nu_1
    mu1_star = lambda1 * mu1 / lambda1_star
    mu2_star = lambda2 * mu2 / lambda2_star
    kappa1 = d * lambda1 * mu1 ** 2 / (2.0 * nu_1 * lambda1_star)
    kappa2 = d * lambda2 * mu2 ** 2 / (2.0 * nu_1 * lambda2_star)
    Sigma_t = sigma_int_fn(t)
    denom1 = (1.0 + lambda1_star * Sigma_t).unsqueeze(-1)
    denom2 = (1.0 + lambda2_star * Sigma_t).unsqueeze(-1)
    sq1 = ((x - mu1_star) ** 2).sum(-1, keepdim=True)
    sq2 = ((x - mu2_star) ** 2).sum(-1, keepdim=True)
    log_A1 = math.log(w1) + kappa1 - 0.5 * d * denom1.log() - lambda1_star * sq1 / (2.0 * denom1)
    log_A2 = math.log(w2) + kappa2 - 0.5 * d * denom2.log() - lambda2_star * sq2 / (2.0 * denom2)
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

    Full adjoint sampling objective: p^{u*} ∝ exp(r(x₁)), αᵢ ∝ wᵢ/√λᵢ.
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
           w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d) -> float:
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star_val = optimal_control(x_samples, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, sigma_int_fn, nu_1, d)
    num = (u_hat - u_star_val).pow(2).sum(-1).mean().item()
    den = u_star_val.pow(2).sum(-1).mean().item()
    if den < 1e-12:
        return float("nan")
    return math.sqrt(num / den)


def _suffix_sup(vals, pct: float) -> list[float]:
    """Robust ‖·‖_{[t_k,T]}: for each k, the `pct`-th percentile of vals[k:]
    (== the suffix-maximum at pct=100)."""
    v = np.asarray(vals, dtype=float)
    if pct >= 100.0:
        return np.maximum.accumulate(v[::-1])[::-1].tolist()
    return [float(np.percentile(v[k:], pct)) for k in range(len(v))]


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
                          sigma_fn, sigma_int_fn, nu_1, d: int, xs: Tensor) -> list[float]:
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_star_val = optimal_control(x_t, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, sigma_int_fn, nu_1, d)
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
# Self-consistency operator  P(u)(t,x) = -σ(t) E_u[∇g(X_T^{u,x})]
# ---------------------------------------------------------------------------

def _grad_g_bimodal(nu_1: float,
                    w1: float, lambda1: float, mu1: float,
                    w2: float, lambda2: float, mu2: float):
    """∇g(x) = −x/ν₁ − ∇r(x)  as a callable x_T -> ∇g(x_T)  (full AS objective)."""
    def grad_g(x_T: Tensor) -> Tensor:
        return -x_T / nu_1 - grad_r(x_T, w1, lambda1, mu1, w2, lambda2, mu2)
    return grad_g


@torch.no_grad()
def operator_field_all_times(
    u_fn, xs: Tensor, ts: Tensor, sigma_fn, nu_1: float, d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_mc: int, device,
) -> Tensor:
    """P(u)(t_k, x) for every time slice AND grid point in one batched EM pass
    → [K+1, n_grid, d].  Binds ∇g(x) = −x/ν₁ − ∇r(x) and defers to
    ``adjoint_sampling.operator``.
    """
    grad_g = _grad_g_bimodal(nu_1, w1, lambda1, mu1, w2, lambda2, mu2)
    return _operator_grid_field_all_times(u_fn, grad_g, xs, ts, sigma_fn, d, n_mc, device)


@torch.no_grad()
def operator_field_at_points(
    u_fn, x_states: Tensor, t_start: float, ts: Tensor, sigma_fn, nu_1: float, d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_mc: int, device,
) -> Tensor:
    """P(u)(t_start, x) at pre-built states `x_states` [n_pts, d] → [n_pts, d].
    Used to evaluate P(u_n) at the sampled path states of u^{n+1}.
    """
    grad_g = _grad_g_bimodal(nu_1, w1, lambda1, mu1, w2, lambda2, mu2)
    return _operator_field_at_points(u_fn, grad_g, x_states, t_start, ts, sigma_fn, d, n_mc, device)


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

    sigma_schedule = str(cfg.get("sigma_schedule", "constant"))
    sigma_floor = float(cfg.get("sigma_floor", 0.0))
    sigma_fn, nu_fn, nu_1 = utils.make_noise_schedule(
        sigma_schedule, float(cfg.sigma), sigma_floor)
    sigma_int_fn = make_sigma_int_fn(nu_fn, nu_1)   # Σ_t = ∫_t^1 σ² ds = ν_1 − ν_t
    log.info(f"noise schedule: {sigma_schedule} (σ scale = {cfg.sigma}, "
             f"floor = {sigma_floor}), ν₁ = {nu_1:.4f}")

    assert lambda1 > 1.0 / nu_1 and lambda2 > 1.0 / nu_1, (
        f"Full objective requires λᵢ > 1/ν₁; got λ1={lambda1}, λ2={lambda2}, "
        f"ν₁={nu_1:.3f} (schedule={sigma_schedule})."
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

    # Training objective: "ram" (default, reciprocal AM via base bridge) or "am"
    # (reference L_AM along stopgrad controlled trajectories).  See §"Training".
    objective = str(cfg.algorithm.get("objective", "ram")).lower()
    assert objective in ("ram", "am"), f"algorithm.objective must be ram|am, got {objective}"
    log.info(f"training objective: {objective}")

    K = cfg.eval.n_time_slices
    ts_eval = torch.linspace(0.0, 1.0, K + 1, device=device)

    # Robust "sup" for every tiled / sup-norm: p-th percentile instead of exact max,
    # applied over x (the grid) AND over the [t,T] time suffix.  100 ⇒ exact max.
    tiled_sup_pct = float(cfg.eval.get("tiled_sup_percentile", 100.0))
    if tiled_sup_pct < 100.0:
        log.info(f"tiled/sup norms clamped at the {tiled_sup_pct:g}th percentile")

    def _sup_over_x(a):
        """Robust sup over the last axis of a torch tensor. → tensor with that axis reduced."""
        if tiled_sup_pct >= 100.0:
            return a.max(dim=-1).values
        return torch.quantile(a, tiled_sup_pct / 100.0, dim=-1)

    x_center = (mu1 + mu2) / 2.0
    half_width = float(cfg.eval.eval_x_range)
    xs_ctrl = torch.linspace(x_center - half_width, x_center + half_width,
                             cfg.eval.n_ctrl_grid, device=device)   # control-heatmap grid
    xs_ctrl_list = xs_ctrl.tolist()

    # Operator P evaluation grid + cadence.  The §"Sanity Check" metrics
    # (‖P(u_θ) − u*‖ with the analytic-u* substitution and its ratio forms) all
    # live on this grid.
    do_op_eval = bool(cfg.eval.op_eval)
    if do_op_eval:
        n_op_grid = int(cfg.eval.n_op_grid)
        xs_op = torch.linspace(x_center - half_width, x_center + half_width,
                               n_op_grid, device=device)
        xs_op_list = xs_op.tolist()
        n_op_mc = int(cfg.eval.n_op_mc_samples)
        op_every = int(cfg.eval.op_every)
    else:
        xs_op = xs_op_list = None
        n_op_mc = op_every = 0

    u_star_fn = lambda x, t: optimal_control(
        x, t, w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d)

    # u*(t_k, x) on the control-heatmap grid — computed once
    with torch.no_grad():
        u_star_field: list[list[float]] = [
            optimal_control_field(ts_eval[k].item(),
                                  w1, lambda1, mu1, w2, lambda2, mu2,
                                  sigma_fn, sigma_int_fn, nu_1, d, xs_ctrl)
            for k in range(K + 1)
        ]

    # u*-sampled states for the RelL2 log line — fixed throughout training
    metric_paths = simulate_paths(u_star_fn, cfg.eval.n_metric_samples,
                                  ts_eval, d, sigma_fn, device)

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    # ∇g(x) = −x/ν₁ − ∇r(x)  (full adjoint sampling objective g = log p₁^base − r)
    grad_g_fn = _grad_g_bimodal(nu_1, w1, lambda1, mu1, w2, lambda2, mu2)

    # ---------------------------------------------------------------------------
    # Basic sanity check: is the ANALYTIC u* a fixed point of the operator?
    #   ‖ P(u*)(t,x) - u*(t,x) ‖ ≪ 1   with   P(u)(t,x) = -σ(t) E[∇g(X_T^{u,x})]
    # Independent of training — computed once on a fixed (x, t) grid.
    # ---------------------------------------------------------------------------
    # Minimal-output mode: only produce the two §"Sanity Check" ratio curves.
    sanity_ratio_only = bool(cfg.eval.get("sanity_ratio_only", False))

    fixed_point_data: dict | None = None
    if bool(cfg.eval.get("fixed_point_check", True)):
        n_fp_mc = int(cfg.eval.get("n_fixed_point_mc", 512))
        n_fp_grid = int(cfg.eval.get("n_fixed_point_grid", 121))
        xs_fp = torch.linspace(x_center - half_width, x_center + half_width,
                               n_fp_grid, device=device)
        fp = fixed_point_residual_field(u_star_fn, grad_g_fn, xs_fp, ts_eval,
                                        sigma_fn, d, n_fp_mc, device)
        resid = fp["residual"]
        fixed_point_data = {
            "xs": xs_fp.tolist(),
            "residual": resid.tolist(),                                   # [K+1][n_grid]
            "p_u_star": (fp["p_u_star"][..., 0] if d == 1
                        else fp["p_u_star"].norm(dim=-1)).tolist(),       # signed (d=1) / norm
            "u_star": (fp["u_star"][..., 0] if d == 1
                       else fp["u_star"].norm(dim=-1)).tolist(),
            "n_mc": n_fp_mc,
        }
        log.info(f"[fixed-point check] ‖P(u*) − u*‖ on {n_fp_grid}×{K + 1} grid "
                 f"(n_mc={n_fp_mc}):  max={resid.max().item():.4g}  "
                 f"mean={resid.mean().item():.4g}  "
                 f"max@t<1={resid[:-1].max().item():.4g}")
        if cfg.logging.wandb:
            wandb.log({"fixed_point/resid_max": resid.max().item(),
                       "fixed_point/resid_mean": resid.mean().item()})

    snapshots: list[dict] = []
    ts_list = ts_eval.tolist()
    prev_net_for_op: nn.Module | None = None   # u_θ^{n-1} copy for the path-based P(u_{n-1}) eval

    # outer_it = -1 is a pre-training pass: it records the freshly-initialised
    # network as snapshot "iteration 0", so that the learned control at
    # iteration n can be compared against P(iteration n-1) (u_θ^n ≈ P(u_θ^{n-1})).
    # snap_it = outer_it + 1 is the recorded iteration index.
    for outer_it in range(-1, cfg.training.outer_iterations):
        snap_it = outer_it + 1

        if outer_it < 0:
            inner_steps_taken = 0
        else:
            if objective == "am":
                # L_AM reference objective: regress u_θ along stopgrad controlled
                # trajectories onto -σ(t)∇g(X_T).  Lean adjoint ≡ ∇g(X_1) here (the
                # base drift is 0), so am_loss's constant target is exact.  NOTE:
                # am_loss carries NO λ(t)=1/σ(t)² weight (ram_loss does) — kept
                # deliberately for now; reconcile before quantitative comparison.
                traj_xs, _ = sampler.sample_trajectory(
                    net, cfg.algorithm.n_outer, d, device)
                am_traj = torch.stack(traj_xs, dim=0)              # [steps+1, n_outer, d]
                am_gg = grad_g_fn(am_traj[-1])                     # [n_outer, d]
            else:
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

            loss_window: list[float] = []
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

                if objective == "am":
                    idx = torch.randint(cfg.algorithm.n_outer,
                                        (cfg.algorithm.n_inner,), device=device)
                    mb_traj = [am_traj[n][idx] for n in range(am_traj.shape[0])]
                    loss = am_loss(net, mb_traj, am_gg[idx], sigma_fn, sampler.steps)
                else:
                    x1_b, gg_b = buffer.sample(cfg.algorithm.n_inner, device=device)
                    loss = ram_loss(net, x1_b, gg_b, sigma_fn, nu_fn, nu_1)
                optim.zero_grad()
                loss.backward()
                optim.step()
                inner_steps_taken += 1

                if cfg.algorithm.inner_tol > 0:
                    loss_window.append(loss.item())
                    if len(loss_window) > cfg.algorithm.inner_patience:
                        loss_window.pop(0)
                    if (len(loss_window) == cfg.algorithm.inner_patience
                            and max(loss_window) - min(loss_window) < cfg.algorithm.inner_tol):
                        break

            if outer_it % cfg.logging.log_every == 0:
                log.info(f"[{snap_it:4d}] {objective}_loss={loss.item():.4f}  "
                         f"lr={current_lr:.2e}  inner_steps={inner_steps_taken}")
                if cfg.logging.wandb:
                    wandb.log({f"{objective}_loss": loss.item(), "lr": current_lr,
                               "inner_steps": inner_steps_taken, "outer_it": snap_it})

        if snap_it < cfg.eval.first_k or snap_it % cfg.eval.every == 0:
            wandb_metrics: dict = {}

            # RelL2(t) = ‖u_θ − u*‖_2 / ‖u*‖_2 on u*-sampled states — logged only.
            rl2_vals = [
                rel_l2(net, metric_paths[k], ts_eval[k].item(),
                       w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d)
                for k in range(K + 1)
            ]
            for k in range(K + 1):
                wandb_metrics[f"rel_l2/t{k:03d}"] = rl2_vals[k]

            _EPS_RATIO = 1e-12

            # -------------------------------------------------------------------
            # Operator T evaluation: T(u_θ^n)(t,x) = -σ(t) E[∇g(X_T^{u_θ^n,x})]
            # The §"Sanity Check" metrics live here (analytic-u* substitution):
            #   ‖P(u_θ^n)(t,x) - u*(t,x)‖ = op_error_fields  (MC only on u_θ).
            # -------------------------------------------------------------------
            if do_op_eval and snap_it % op_every == 0:
                # ---- batched over ALL (t_k, x) nodes on xs_op ----------------
                n_op = xs_op.shape[0]
                x_grid = torch.zeros(n_op, d, device=device)
                x_grid[:, 0] = xs_op
                x_all = x_grid[None].expand(K + 1, n_op, d).reshape((K + 1) * n_op, d)
                t_all = ts_eval[:, None].expand(K + 1, n_op).reshape(-1)

                # T(u_θ^n)(t_k,x) for every slice in one batched EM pass
                Tu_all = operator_field_all_times(
                    net, xs_op, ts_eval, sigma_fn, nu_1, d,
                    w1, lambda1, mu1, w2, lambda2, mu2, n_op_mc, device,
                )                                               # [K+1, n_op, d]
                with torch.no_grad():
                    u_star_all = optimal_control(
                        x_all, t_all, w1, lambda1, mu1, w2, lambda2, mu2,
                        sigma_fn, sigma_int_fn, nu_1, d).view(K + 1, n_op, d)
                    u_theta_all = net(x_all, t_all).view(K + 1, n_op, d)

                def _scalar(z):   # [K+1, n_op, d] -> [K+1, n_op]
                    return z[..., 0] if d == 1 else z.norm(dim=-1)

                Tu_s = _scalar(Tu_all)
                u_star_s = _scalar(u_star_all)
                u_theta_s = _scalar(u_theta_all)
                op_err_all = (Tu_s - u_star_s).abs()             # [K+1, n_op]
                u_err_all = (u_theta_s - u_star_s).abs()

                u_star_op_fields: list[list[float]] = u_star_s.tolist()
                T_u_fields: list[list[float]] = Tu_s.tolist()
                u_theta_op_fields: list[list[float]] = u_theta_s.tolist()
                op_error_fields: list[list[float]] = op_err_all.tolist()
                u_error_op_fields: list[list[float]] = u_err_all.tolist()
                op_sup_error: list[float] = _sup_over_x(op_err_all).tolist()
                u_sup_error_op: list[float] = _sup_over_x(u_err_all).tolist()
                for k in range(K + 1):
                    wandb_metrics[f"op_sup_error/t{k:03d}"] = op_sup_error[k]
                    wandb_metrics[f"u_sup_error_op/t{k:03d}"] = u_sup_error_op[k]

                # Tiled operator sup error: ‖P(u_θ^n) − u*‖_{[t_k,T]}  (robust suffix-sup)
                tiled_op_sup_error: list[float] = _suffix_sup(op_sup_error, tiled_sup_pct)
                for k in range(K + 1):
                    wandb_metrics[f"tiled_op_sup_error/t{k:03d}"] = tiled_op_sup_error[k]

                # -----------------------------------------------------------
                # §"Sanity Check" ratios (analytic-u* substitution):
                #   numerator   = ‖E[σ(t)∇g(X_T^{u_n,x})] − u*(x,t)‖ = ‖P(u_n)−u*‖
                #                 = op_error_fields (pointwise) / tiled_op_sup_error (tiled)
                #   denominator = ‖u_n − u*‖_{[t,T]}   (NO ‖σ‖_{[t,T]} factor)
                # both → 0 as t → T.
                # -----------------------------------------------------------
                # ‖u_n-u*‖_{[t_k,T]} : robust suffix-sup of the sup_x control error
                tiled_u_sup_error_op: list[float] = _suffix_sup(u_sup_error_op, tiled_sup_pct)

                sanity_ratio_denom: list[float] = [
                    max(tiled_u_sup_error_op[k], _EPS_RATIO) for k in range(K + 1)
                ]
                sanity_ratio_pointwise: list[list[float]] = [
                    (np.asarray(op_error_fields[k]) / sanity_ratio_denom[k]).tolist()
                    for k in range(K + 1)
                ]
                sanity_ratio_tiled: list[float] = [
                    tiled_op_sup_error[k] / sanity_ratio_denom[k] for k in range(K + 1)
                ]
                for k in range(K + 1):
                    wandb_metrics[f"sanity_ratio_tiled/t{k:03d}"] = sanity_ratio_tiled[k]

                # -----------------------------------------------------------
                # Path-based metrics evaluated along u^{n+1}_θ trajectories.
                # Both require the same trajectory sample; compute together.
                # -----------------------------------------------------------
                n_path_sp = int(cfg.eval.n_path_op_samples)
                theta_traj = simulate_paths(
                    net, n_path_sp, ts_eval, d, sigma_fn, device
                )  # [K+1, n_path_sp, d]

                # path_u_vs_ustar: E[‖u^{n+1}_θ(t,X) − u*(t,X)‖]
                path_u_vs_ustar: list[float] = []
                for k in range(K + 1):
                    x_states_k = theta_traj[k]            # [n_path_sp, d]
                    t_val_k = ts_eval[k].item()
                    t_vec_k = torch.full((n_path_sp,), t_val_k, device=device)
                    with torch.no_grad():
                        u_curr_k = net(x_states_k, t_vec_k)        # [n_path_sp, d]
                        u_star_k = u_star_fn(x_states_k, t_vec_k)  # [n_path_sp, d]
                    err_k = (u_curr_k - u_star_k).norm(dim=-1).mean().item()
                    path_u_vs_ustar.append(err_k)
                    wandb_metrics[f"path_u_vs_ustar/t{k:03d}"] = err_k

                # path_u_vs_Tu and path_u_prev_vs_ustar: both need prev network
                if prev_net_for_op is not None:
                    path_u_vs_Tu: list[float] = []
                    path_u_prev_vs_ustar: list[float] = []
                    for k in range(K + 1):
                        x_states_k = theta_traj[k]            # [n_path_sp, d]
                        t_val_k = ts_eval[k].item()
                        t_vec_k = torch.full((n_path_sp,), t_val_k, device=device)
                        with torch.no_grad():
                            u_curr_k = net(x_states_k, t_vec_k)        # [n_path_sp, d]
                            u_prev_k = prev_net_for_op(x_states_k, t_vec_k)  # [n_path_sp, d]
                            u_star_k2 = u_star_fn(x_states_k, t_vec_k) # [n_path_sp, d]
                        Tu_k = operator_field_at_points(
                            prev_net_for_op, x_states_k, t_val_k, ts_eval,
                            sigma_fn, nu_1, d,
                            w1, lambda1, mu1, w2, lambda2, mu2,
                            n_op_mc, device,
                        )  # [n_path_sp, d]
                        path_u_vs_Tu.append((u_curr_k - Tu_k).norm(dim=-1).mean().item())
                        path_u_prev_vs_ustar.append((u_prev_k - u_star_k2).norm(dim=-1).mean().item())
                        wandb_metrics[f"path_u_vs_Tu/t{k:03d}"] = path_u_vs_Tu[-1]
                        wandb_metrics[f"path_u_prev_vs_ustar/t{k:03d}"] = path_u_prev_vs_ustar[-1]
                else:
                    path_u_vs_Tu = None
                    path_u_prev_vs_ustar = None

                # Save copy of current network for next iteration's T(u_n) eval
                prev_net_for_op = copy.deepcopy(net)
                prev_net_for_op.eval()

            else:
                T_u_fields = None
                op_error_fields = None
                op_sup_error = None
                tiled_op_sup_error = None
                u_error_op_fields = None
                u_sup_error_op = None
                u_theta_op_fields = None
                u_star_op_fields = None
                path_u_vs_Tu = None
                path_u_vs_ustar = None
                path_u_prev_vs_ustar = None
                sanity_ratio_pointwise = None
                sanity_ratio_tiled = None

            n_sp = cfg.eval.n_sample_paths
            snap_paths_theta = (
                euler_maruyama_paths(net, n_sp, ts_eval, d, sigma_fn, device)
                if n_sp > 0 else None
            )

            snapshots.append({
                "outer_it": snap_it,
                "rel_l2": rl2_vals,
                "sanity_ratio_pointwise": sanity_ratio_pointwise,
                "sanity_ratio_tiled": sanity_ratio_tiled,
                "T_u_fields": T_u_fields,
                "op_error_fields": op_error_fields,
                "op_sup_error": op_sup_error,
                "tiled_op_sup_error": tiled_op_sup_error,
                "u_error_op_fields": u_error_op_fields,
                "u_sup_error_op": u_sup_error_op,
                "u_theta_op_fields": u_theta_op_fields,
                "u_star_op_fields": u_star_op_fields,
                "path_u_vs_Tu": path_u_vs_Tu,
                "path_u_vs_ustar": path_u_vs_ustar,
                "path_u_prev_vs_ustar": path_u_prev_vs_ustar,
                "paths_theta": snap_paths_theta,
            })

            indices = [0, K // 2, K]
            op_summary = ""
            if op_error_fields is not None:
                op_means = [float(np.mean(op_error_fields[k])) for k in indices]
                op_summary = "  ‖P(u_θ)-u*‖: " + "  ".join(
                    f"t={ts_eval[k].item():.2f} {op_means[i]:.4f}"
                    for i, k in enumerate(indices)
                ) + f"  tiled(0)={tiled_op_sup_error[0]:.4f}"
            summary = "  [eval] " + "  ".join(
                f"t={ts_eval[k].item():.2f} RelL2={rl2_vals[k]:.4f}"
                for k in indices
            ) + op_summary
            log.info(summary)

            if cfg.logging.wandb:
                wandb.log({"outer_it": snap_it, **wandb_metrics})

    log.info("Training complete.")
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    with torch.no_grad():
        u_theta_field: list[list[float]] = [
            control_field(net, ts_eval[k].item(), d, xs_ctrl)
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
        """p^{u*}(x) = α₁ N(m₁s, v₁s) + α₂ N(m₂s, v₂s)  (terminal distribution)."""
        x = np.asarray(x)
        p1 = alpha1 * np.exp(-0.5 * (x - m1s) ** 2 / v1s) / np.sqrt(2 * np.pi * v1s)
        p2 = alpha2 * np.exp(-0.5 * (x - m2s) ** 2 / v2s) / np.sqrt(2 * np.pi * v2s)
        return p1 + p2

    save_snapshots({"snapshots": snapshots, "ts": ts_list,
                    "xs": xs_ctrl_list, "xs_op": xs_op_list, "d": d,
                    "u_star_field": u_star_field, "u_theta_field": u_theta_field,
                    "paths_star": paths_star, "paths_theta": paths_theta,
                    "target_params": dict(w1=w1, lambda1=lambda1, mu1=mu1,
                                          w2=w2, lambda2=lambda2, mu2=mu2),
                    "sigma": float(cfg.sigma), "sigma_schedule": sigma_schedule,
                    "sigma_floor": sigma_floor, "objective": objective,
                    "tiled_sup_percentile": tiled_sup_pct,
                    "fixed_point_check": fixed_point_data},
                   output_dir)
    set_suffix_percentile(tiled_sup_pct)   # robust tiled sup in the plots too
    d_conv = output_dir / "convergence"
    d_ctrl = output_dir / "control"
    d_term = output_dir / "terminal"
    d_sbm = output_dir / "same_bm"
    d_op = output_dir / "operator"

    # Only the 9 figures referenced in Project_Notes.tm are produced (see plotting.py).
    d_sbm.mkdir(parents=True, exist_ok=True)
    plot_sbm_ratio(snapshots, ts_list, d_sbm)
    plot_sbm_ratio_learned(snapshots, ts_list, d_sbm)
    if fixed_point_data is not None:
        plot_fixed_point_residual(fixed_point_data, ts_list, d, d_sbm)

    if sanity_ratio_only:
        log.info("sanity_ratio_only: wrote same_bm/{sbm_ratio, sbm_ratio_learned, "
                 "u_star_fixed_point_residual}.png; all other plots skipped.")
        return

    for _d in (d_conv, d_ctrl, d_term, d_op):
        _d.mkdir(parents=True, exist_ok=True)

    plot_optimal_control(ts_list, xs_ctrl_list, u_star_field, d, d_ctrl, u_theta_field)
    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d, d_term)
    if do_op_eval and xs_op_list:
        plot_operator_vs_next_control(snapshots, ts_list, xs_op_list, d, d_ctrl)
        plot_path_u_vs_Tu(snapshots, ts_list, d_op)
        plot_learned_pointwise_over_tiled(snapshots, ts_list, xs_op_list, d_conv)
        plot_path_u_vs_ustar_per_t_over_tiled(snapshots, ts_list, d_conv)


if __name__ == "__main__":
    main()

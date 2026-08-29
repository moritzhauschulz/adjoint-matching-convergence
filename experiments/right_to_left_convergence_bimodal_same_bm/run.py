"""Right-to-left convergence — bimodal reward, sanity-check evaluation.

Extends the bimodal experiment by directly estimating the LHS of the §2.1
contraction bound ("Sanity Check" in Project_Notes.tm), with the norm OUTSIDE
the Monte-Carlo expectation:

    LHS_SBM(t, x) = ‖ E_B[ σ(t)∇g(X_T^{u_θ,x}) − σ(t)∇g(X_T^{u*,x}) ] ‖

where ∇g(x) = −x/ν₁ − ∇r(x).  This equals ‖T(u_θ)(t,x) − T(u*)(t,x)‖ =
‖T(u_θ)(t,x) − u*(t,x)‖ exactly (u* is the fixed point).  The two rollouts
still share the BM sample B_{t:T} purely as a variance-reduction device — by
linearity of expectation the coupling does not change the value being
estimated, only the MC variance.  The quantity → 0 as t → T (right-to-left),
which is what the bound predicts.

NB: Project_Notes.tm writes this operator with ∇r rather than ∇g; the code
keeps ∇g so that u* is the fixed point of the trained RAM iteration (target
−σ∇g).  Flagged as a notes inconsistency — see this experiment's CLAUDE.md.

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
    operator_grid_field as _operator_grid_field,
    operator_grid_field_all_times as _operator_grid_field_all_times,
    operator_field as _operator_field_at_points,
    fixed_point_residual_field,
)
from plotting import (
    plot_convergence, plot_heatmaps, plot_optimal_control,
    plot_contraction_heatmaps, plot_control_evolution, plot_operator_vs_next_control,
    plot_terminal_distributions, plot_terminal_evolution,
    plot_inner_steps, plot_inner_convergence,
    plot_same_bm_lhs_curves, plot_same_bm_lhs_heatmaps,
    plot_tiled_same_bm_lhs,
    plot_sbm_ratio, plot_sbm_ratio_learned, plot_sanity_ratio_heatmap, plot_fixed_point_residual,
    set_suffix_percentile,
    plot_operator_error_curves, plot_operator_vs_learned, plot_u_vs_Tu_prev,
    plot_u_vs_Tu_norm, plot_Tu_vs_u_next_heatmap, plot_path_u_vs_Tu,
    plot_learned_error_heatmap, plot_learned_tiled_sup_error,
    plot_learned_pointwise_over_tiled, plot_learned_tiled_over_tiled,
    plot_path_u_vs_ustar, plot_path_u_vs_ustar_per_t_over_tiled,
    plot_path_u_vs_ustar_tiled_over_tiled,
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


@torch.no_grad()
def abs_l2(u_theta: nn.Module, x_samples: Tensor, t_val: float,
           w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d) -> float:
    t = torch.full((x_samples.shape[0],), t_val, device=x_samples.device)
    u_hat = u_theta(x_samples, t)
    u_star_val = optimal_control(x_samples, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, sigma_int_fn, nu_1, d)
    return math.sqrt((u_hat - u_star_val).pow(2).sum(-1).mean().item())


@torch.no_grad()
def _error_field(u_theta: nn.Module, xs: Tensor, t_val: float,
                 w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d) -> Tensor:
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_hat = u_theta(x_t, t)
    u_star_val = optimal_control(x_t, t, w1, lambda1, mu1, w2, lambda2, mu2,
                                 sigma_fn, sigma_int_fn, nu_1, d)
    return (u_hat - u_star_val).norm(dim=-1)


def _grid_sup(field, pct: float):
    """Robust sup over a 1-D grid: `pct`-th percentile (== exact max at pct=100).
    Accepts a torch tensor, numpy array, or list."""
    a = field.detach().cpu().numpy() if hasattr(field, "detach") else np.asarray(field, dtype=float)
    return float(a.max()) if pct >= 100.0 else float(np.percentile(a, pct))


def _suffix_sup(vals, pct: float) -> list[float]:
    """Robust ‖·‖_{[t_k,T]}: for each k, the `pct`-th percentile of vals[k:]
    (== the suffix-maximum at pct=100)."""
    v = np.asarray(vals, dtype=float)
    if pct >= 100.0:
        return np.maximum.accumulate(v[::-1])[::-1].tolist()
    return [float(np.percentile(v[k:], pct)) for k in range(len(v))]


@torch.no_grad()
def abs_linf(u_theta: nn.Module, t_val: float,
             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d,
             xs: Tensor, sup_pct: float = 100.0) -> tuple[float, list[float]]:
    field = _error_field(u_theta, xs, t_val, w1, lambda1, mu1, w2, lambda2, mu2,
                         sigma_fn, sigma_int_fn, nu_1, d)
    return _grid_sup(field, sup_pct), field.tolist()


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
# Both-rollout operator difference with a shared Brownian motion
# ---------------------------------------------------------------------------

@torch.no_grad()
def operator_diff_shared_bm_field(
    u_fn,
    v_fn,
    xs: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    d: int,
    nu_1: float,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_bm_samples: int,
    device,
) -> Tensor:
    """‖E_B[σ(t)∇g(X_T^{u,x}) − σ(t)∇g(X_T^{v,x})]‖ for each x in xs, with BOTH
    processes rolled out under a SHARED Brownian path (norm outside the MC mean).

    This is the "both-rollout" form of the §"Sanity Check" tiled bound LHS —
    kept only as a variance-reference for comparison against the analytic-u*
    substitution (§5.0/§5.1: ‖P(u_θ) − u*‖ = op_error_fields). By linearity of
    expectation the two have the same population value; the shared BM just lowers
    the MC variance of this estimator when u ≈ v.

    ∇g(x) = −x/ν₁ − ∇r(x).  u_fn = u_θ (learned), v_fn = u* (optimal).
    Returns [n_grid]; exactly 0 at t_start = T (zero rollout steps).
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

    # ∇g(x) = −x/ν₁ − ∇r(x)
    gg_u = -xu / nu_1 - grad_r(xu, w1, lambda1, mu1, w2, lambda2, mu2)  # [n_grid*n_bm, d]
    gg_v = -xv / nu_1 - grad_r(xv, w1, lambda1, mu1, w2, lambda2, mu2)

    # Norm OUTSIDE the MC expectation over shared BM draws:
    #   ‖ E_B[ σ(t)(∇g(X_T^{u,x}) − ∇g(X_T^{v,x})) ] ‖
    diff = (sigma_t * (gg_u - gg_v)).view(n_grid, n_bm_samples, -1)  # [n_grid, n_bm, d]
    return diff.mean(dim=1).norm(dim=-1)                            # [n_grid]


# ---------------------------------------------------------------------------
# Analytic operator T(u)(t,x) = -σ(t) E_u[∇g(X_T^{u,x})]
# ---------------------------------------------------------------------------

def _grad_g_bimodal(nu_1: float,
                    w1: float, lambda1: float, mu1: float,
                    w2: float, lambda2: float, mu2: float):
    """∇g(x) = −x/ν₁ − ∇r(x)  as a callable x_T -> ∇g(x_T)  (full AS objective)."""
    def grad_g(x_T: Tensor) -> Tensor:
        return -x_T / nu_1 - grad_r(x_T, w1, lambda1, mu1, w2, lambda2, mu2)
    return grad_g


@torch.no_grad()
def operator_field(
    u_fn,
    xs: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    nu_1: float,
    d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_mc: int,
    device,
) -> Tensor:
    """T(u)(t,x) = -σ(t) E_u[∇g(X_T^{u,x})] on a 1-D grid `xs`  → [n_grid, d].

    Thin wrapper over ``adjoint_sampling.operator.operator_grid_field`` binding
    ∇g(x) = −x/ν₁ − ∇r(x).  At t_start = T the result is -σ(T)∇g(x) exactly,
    which equals u*(T,x).
    """
    grad_g = _grad_g_bimodal(nu_1, w1, lambda1, mu1, w2, lambda2, mu2)
    return _operator_grid_field(u_fn, grad_g, xs, t_start, ts, sigma_fn, d, n_mc, device)


@torch.no_grad()
def operator_field_all_times(
    u_fn, xs: Tensor, ts: Tensor, sigma_fn, nu_1: float, d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_mc: int, device,
) -> Tensor:
    """T(u)(t_k,x) for every time slice AND grid point in one batched EM pass.
    → [K+1, n_grid, d].  Statistically equivalent to `operator_field` per t_k.
    """
    grad_g = _grad_g_bimodal(nu_1, w1, lambda1, mu1, w2, lambda2, mu2)
    return _operator_grid_field_all_times(u_fn, grad_g, xs, ts, sigma_fn, d, n_mc, device)


@torch.no_grad()
def operator_field_at_points(
    u_fn,
    x_states: Tensor,
    t_start: float,
    ts: Tensor,
    sigma_fn,
    nu_1: float,
    d: int,
    w1: float, lambda1: float, mu1: float,
    w2: float, lambda2: float, mu2: float,
    n_mc: int,
    device,
) -> Tensor:
    """`operator_field` for pre-built states `x_states` [n_pts, d]  → [n_pts, d].

    Used to evaluate T(u_n) at the sampled path states of u^{n+1}.
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
    xs_linf = torch.linspace(x_center - cfg.eval.linf_x_range,
                             x_center + cfg.eval.linf_x_range,
                             cfg.eval.n_linf_grid, device=device)
    xs_list = xs_linf.tolist()

    # Operator T evaluation setup.  The sanity-check metrics (§5.0/§5.1) — the
    # analytic-u* substitution ‖P(u_θ) − u*‖ and its tiled / ratio forms, plus a
    # both-rollout shared-BM comparison — all live on this grid and cadence.
    do_op_eval = bool(cfg.eval.op_eval)
    if do_op_eval:
        n_op_grid = int(cfg.eval.get("n_op_grid", cfg.eval.get("n_same_bm_grid", 60)))
        xs_op = torch.linspace(x_center - cfg.eval.linf_x_range,
                               x_center + cfg.eval.linf_x_range,
                               n_op_grid, device=device)
        xs_op_list = xs_op.tolist()
        n_op_mc = int(cfg.eval.n_op_mc_samples)
        op_every = int(cfg.eval.op_every)
        # Both-rollout shared-BM comparison (MC count / cadence follow op-eval)
        do_bothroll = bool(cfg.eval.get("both_rollout_compare", True))
        n_bothroll = int(cfg.eval.get("n_both_rollout_samples", n_op_mc))
    else:
        xs_op = xs_op_list = None
        n_op_mc = op_every = 0
        do_bothroll = False
        n_bothroll = 0

    u_star_fn = lambda x, t: optimal_control(
        x, t, w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d)

    # u* field on linf grid — computed once
    with torch.no_grad():
        u_star_field: list[list[float]] = [
            optimal_control_field(ts_eval[k].item(),
                                  w1, lambda1, mu1, w2, lambda2, mu2,
                                  sigma_fn, sigma_int_fn, nu_1, d, xs_linf)
            for k in range(K + 1)
        ]

    # Metric paths under u* — fixed throughout training
    metric_paths = simulate_paths(u_star_fn, cfg.eval.n_metric_samples,
                                  ts_eval, d, sigma_fn, device)

    # σ(t) on the eval grid — stored for reference (the §5.3 / sanity-check ratio
    # denominators no longer carry any ‖σ‖ factor, per the latest notes).
    with torch.no_grad():
        _sigma_grid = [float(v) for v in sigma_fn(ts_eval).tolist()]

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    def grad_g_fn(x1: Tensor) -> Tensor:
        """∇g(x₁) = −x₁/ν₁ − ∇r(x₁)  (full adjoint sampling objective g = log p₁^base − r)."""
        return -x1 / nu_1 - grad_r(x1, w1, lambda1, mu1, w2, lambda2, mu2)

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
        xs_fp = torch.linspace(x_center - cfg.eval.linf_x_range,
                               x_center + cfg.eval.linf_x_range,
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
    prev_al_inf_vals: list[float] | None = None
    prev_tiled_al_inf_vals: list[float] | None = None
    prev_T_u_fields: list[list[float]] | None = None   # T(u_θ^{n-1}) on xs_op grid
    prev_net_for_op: nn.Module | None = None            # u_θ^{n-1} network copy for path-based T eval

    # outer_it = -1 is a pre-training pass: it records the freshly-initialised
    # network as snapshot "iteration 0", so that the learned control at
    # iteration n can be compared against P(iteration n-1) (u_θ^n ≈ P(u_θ^{n-1})).
    # snap_it = outer_it + 1 is the recorded iteration index.
    for outer_it in range(-1, cfg.training.outer_iterations):
        snap_it = outer_it + 1

        if outer_it < 0:
            record_curve = False
            inner_loss_curve = []
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
            record_curve = (outer_it % cfg.eval.inner_curve_every == 0)

            loss_window: list[float] = []
            inner_loss_curve = []
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
                log.info(f"[{snap_it:4d}] {objective}_loss={loss.item():.4f}  "
                         f"lr={current_lr:.2e}  inner_steps={inner_steps_taken}")
                if cfg.logging.wandb:
                    wandb.log({f"{objective}_loss": loss.item(), "lr": current_lr,
                               "inner_steps": inner_steps_taken, "outer_it": snap_it})

        if snap_it < cfg.eval.first_k or snap_it % cfg.eval.every == 0:
            rl2_vals, al2_vals, al_inf_vals = [], [], []
            error_fields: list[list[float]] = []
            wandb_metrics: dict = {}

            for k in range(K + 1):
                t_val = ts_eval[k].item()
                x_samples = metric_paths[k]
                rl2 = rel_l2(net, x_samples, t_val,
                             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d)
                al2 = abs_l2(net, x_samples, t_val,
                             w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, sigma_int_fn, nu_1, d)
                al_inf, field = abs_linf(net, t_val,
                                         w1, lambda1, mu1, w2, lambda2, mu2,
                                         sigma_fn, sigma_int_fn, nu_1, d, xs_linf,
                                         sup_pct=tiled_sup_pct)
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

            tiled_al_inf_vals: list[float] = _suffix_sup(al_inf_vals, tiled_sup_pct)

            ef_arr = np.array(error_fields)                      # [K+1, n_linf_grid]
            if tiled_sup_pct >= 100.0:
                tiled_ef_arr = np.maximum.accumulate(ef_arr[::-1])[::-1]
            else:
                tiled_ef_arr = np.stack([
                    np.percentile(ef_arr[k:], tiled_sup_pct, axis=0) for k in range(K + 1)
                ])
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

                # Both-rollout shared-BM comparison: ‖E_B[σ(∇g(X_T^{u_θ}) − ∇g(X_T^{u*}))]‖
                if do_bothroll:
                    bothroll_fields: list[list[float]] = []
                    for k in range(K + 1):
                        t_val = ts_eval[k].item()
                        br = operator_diff_shared_bm_field(
                            u_fn=net, v_fn=u_star_fn, xs=xs_op, t_start=t_val,
                            ts=ts_eval, sigma_fn=sigma_fn, d=d, nu_1=nu_1,
                            w1=w1, lambda1=lambda1, mu1=mu1,
                            w2=w2, lambda2=lambda2, mu2=mu2,
                            n_bm_samples=n_bothroll, device=device,
                        )
                        bothroll_fields.append(br.tolist())
                    br_sup = [_grid_sup(bothroll_fields[k], tiled_sup_pct) for k in range(K + 1)]
                    tiled_bothroll: list[float] = _suffix_sup(br_sup, tiled_sup_pct)
                    for k in range(K + 1):
                        wandb_metrics[f"tiled_bothroll_lhs/t{k:03d}"] = tiled_bothroll[k]
                else:
                    bothroll_fields = None
                    tiled_bothroll = None

                # u_θ^n vs T(u_θ^{n-1}) on xs_op: how well does the outer step implement T?
                # Uses the batched signed u_θ^n values (u_theta_s) computed above.
                if prev_T_u_fields is not None:
                    prev_Tu = torch.tensor(prev_T_u_fields, device=device)   # [K+1, n_op]
                    u_vs_Tu_err = (u_theta_s - prev_Tu).abs()                # [K+1, n_op]
                    u_vs_Tu_fields: list[list[float]] = u_vs_Tu_err.tolist()
                    u_vs_Tu_sup: list[float] = _sup_over_x(u_vs_Tu_err).tolist()
                    for k in range(K + 1):
                        wandb_metrics[f"u_vs_Tu_sup/t{k:03d}"] = u_vs_Tu_sup[k]
                else:
                    u_vs_Tu_fields = None
                    u_vs_Tu_sup = None

                prev_T_u_fields = T_u_fields

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
                u_vs_Tu_fields = None
                u_vs_Tu_sup = None
                u_star_op_fields = None
                path_u_vs_Tu = None
                path_u_vs_ustar = None
                path_u_prev_vs_ustar = None
                sanity_ratio_pointwise = None
                sanity_ratio_tiled = None
                bothroll_fields = None
                tiled_bothroll = None

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
                "outer_it": snap_it,
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
                "sanity_ratio_pointwise": sanity_ratio_pointwise,
                "sanity_ratio_tiled": sanity_ratio_tiled,
                "bothroll_lhs_fields": bothroll_fields,
                "tiled_bothroll_lhs": tiled_bothroll,
                "T_u_fields": T_u_fields,
                "op_error_fields": op_error_fields,
                "op_sup_error": op_sup_error,
                "tiled_op_sup_error": tiled_op_sup_error,
                "u_error_op_fields": u_error_op_fields,
                "u_sup_error_op": u_sup_error_op,
                "u_theta_op_fields": u_theta_op_fields,
                "u_vs_Tu_fields": u_vs_Tu_fields,
                "u_vs_Tu_sup": u_vs_Tu_sup,
                "path_u_vs_Tu": path_u_vs_Tu,
                "path_u_vs_ustar": path_u_vs_ustar,
                "path_u_prev_vs_ustar": path_u_prev_vs_ustar,
                "u_star_op_fields": u_star_op_fields,
                "u_theta_field": snap_u_theta_field,
                "paths_theta": snap_paths_theta,
            })
            prev_al_inf_vals = al_inf_vals
            prev_tiled_al_inf_vals = tiled_al_inf_vals

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
                    "xs_op": xs_op_list,
                    "u_star_field": u_star_field,
                    "u_theta_field": u_theta_field,
                    "paths_star": paths_star, "paths_theta": paths_theta,
                    "target_params": target_params, "d": d,
                    "sigma": float(cfg.sigma),
                    "sigma_schedule": sigma_schedule,
                    "sigma_floor": sigma_floor,
                    "sigma_grid": _sigma_grid,
                    "tiled_sup_percentile": tiled_sup_pct,
                    "fixed_point_check": fixed_point_data},
                   output_dir)
    set_suffix_percentile(tiled_sup_pct)   # robust tiled sup in the plots too
    d_train = output_dir / "training"
    d_conv  = output_dir / "convergence"
    d_ctrl  = output_dir / "control"
    d_term  = output_dir / "terminal"
    d_err   = output_dir / "errors"
    d_sbm   = output_dir / "same_bm"
    d_op    = output_dir / "operator"

    if sanity_ratio_only:
        # Minimal output: the two §"Sanity Check" ratio curves + the pointwise
        # heatmap + the P(u*)=u* fixed-point residual (all §"Sanity Check").
        d_sbm.mkdir(parents=True, exist_ok=True)
        plot_sbm_ratio(snapshots, ts_list, d_sbm)
        plot_sbm_ratio_learned(snapshots, ts_list, d_sbm)
        if xs_op_list:
            plot_sanity_ratio_heatmap(snapshots, ts_list, xs_op_list, d_sbm)
        if fixed_point_data is not None:
            plot_fixed_point_residual(fixed_point_data, ts_list, d, d_sbm)
        log.info("sanity_ratio_only: wrote same_bm/{sbm_ratio, sbm_ratio_learned, "
                 "sanity_ratio_heatmap, u_star_fixed_point_residual}.png; all other plots skipped.")
        return

    for _d in (d_train, d_conv, d_ctrl, d_term, d_err, d_sbm, d_op):
        _d.mkdir(parents=True, exist_ok=True)

    plot_inner_steps(snapshots, d_train)
    plot_inner_convergence(snapshots, d_train)

    plot_convergence(snapshots, ts_list, d_conv)

    plot_optimal_control(ts_list, xs_list, u_star_field, d, d_ctrl,
                         u_theta_field, paths_star, paths_theta)
    plot_control_evolution(snapshots, ts_list, xs_list, d, u_star_field, d_ctrl)
    if xs_op_list:
        plot_operator_vs_next_control(snapshots, ts_list, xs_op_list, d, d_ctrl)
        plot_operator_vs_next_control(snapshots, ts_list, xs_op_list, d, d_ctrl,
                                     overlay_traj=True)

    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d, d_term)
    plot_terminal_evolution(snapshots, target_pdf_fn, d, d_term)

    plot_heatmaps(snapshots, ts_list, xs_list, d_err)
    plot_contraction_heatmaps(snapshots, ts_list, xs_list, d_err)

    if fixed_point_data is not None:
        plot_fixed_point_residual(fixed_point_data, ts_list, d, d_sbm)

    if do_op_eval and xs_op_list:
        # §"Sanity Check" plots — sourced from the operator-error data
        # (‖P(u_θ) − u*‖, analytic u*) with a both-rollout shared-BM overlay.
        plot_same_bm_lhs_curves(snapshots, ts_list, d_sbm)
        plot_same_bm_lhs_heatmaps(snapshots, ts_list, xs_op_list, d_sbm)
        plot_tiled_same_bm_lhs(snapshots, ts_list, d_sbm)
        plot_sbm_ratio(snapshots, ts_list, d_sbm)
        plot_sbm_ratio_learned(snapshots, ts_list, d_sbm)
        plot_sanity_ratio_heatmap(snapshots, ts_list, xs_op_list, d_sbm)
        plot_learned_error_heatmap(snapshots, ts_list, xs_op_list, d_conv)
        plot_learned_tiled_sup_error(snapshots, ts_list, d_conv)
        plot_learned_pointwise_over_tiled(snapshots, ts_list, xs_op_list, d_conv)
        plot_learned_pointwise_over_tiled(snapshots, ts_list, xs_op_list, d_conv,
                                         overlay_traj=True)
        plot_learned_tiled_over_tiled(snapshots, ts_list, d_conv)
        plot_operator_error_curves(snapshots, ts_list, d_op)
        plot_operator_vs_learned(snapshots, ts_list, d_op)
        plot_u_vs_Tu_prev(snapshots, ts_list, d_op)
        plot_u_vs_Tu_norm(snapshots, ts_list, d_op)
        plot_path_u_vs_Tu(snapshots, ts_list, d_op)
        plot_Tu_vs_u_next_heatmap(snapshots, ts_list, xs_op_list, d_op)
        plot_path_u_vs_ustar(snapshots, ts_list, d_conv)
        plot_path_u_vs_ustar_per_t_over_tiled(snapshots, ts_list, d_conv)
        plot_path_u_vs_ustar_tiled_over_tiled(snapshots, ts_list, d_conv)


if __name__ == "__main__":
    main()

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

Structure:  `main` builds an `EvalContext`, runs the outer loop
(`run_inner_loop` + `evaluate` per iteration), then `write_outputs`.

Run:
    python experiments/right_to_left_convergence_bimodal_same_bm/run.py \
        experiment=right_to_left_convergence_bimodal_same_bm
"""

import copy
import logging
import math
import sys
from dataclasses import dataclass
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

from adjoint_sampling import (
    DriftMLP, Sampler, ReplayBuffer, GaussianMixtureTarget,
    ram_loss, am_loss, utils,
)
from adjoint_sampling.operator import (
    operator_grid_field_all_times,
    operator_field as operator_field_at_points,
    fixed_point_residual_field,
)
from adjoint_sampling.utils import simulate_paths, rel_l2, sigma_int_from_nu
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
# Small helpers
# ---------------------------------------------------------------------------

def _scalar(z: Tensor, d: int) -> Tensor:
    """[..., d] → [...]: signed component for d=1, else the Euclidean norm."""
    return z[..., 0] if d == 1 else z.norm(dim=-1)


@torch.no_grad()
def _grid_field(fn, xs: Tensor, t_val: float, d: int) -> list[float]:
    """Evaluate `fn(states, t)` on the 1-D grid states (xs on axis 0, others 0)
    and return the scalarised values as a list."""
    states = torch.zeros(xs.shape[0], d, device=xs.device)
    states[:, 0] = xs
    t = torch.full((xs.shape[0],), t_val, device=xs.device)
    return _scalar(fn(states, t), d).tolist()


def _suffix_sup(vals, pct: float) -> list[float]:
    """Robust ‖·‖_{[t_k,T]}: for each k, the `pct`-th percentile of vals[k:]
    (== the suffix-maximum at pct=100)."""
    v = np.asarray(vals, dtype=float)
    if pct >= 100.0:
        return np.maximum.accumulate(v[::-1])[::-1].tolist()
    return [float(np.percentile(v[k:], pct)) for k in range(len(v))]


def _sup_over_x(a: Tensor, pct: float) -> Tensor:
    """Robust sup over the last axis: exact max at pct>=100, else the pct-quantile."""
    if pct >= 100.0:
        return a.max(dim=-1).values
    return torch.quantile(a, pct / 100.0, dim=-1)


def _paths_json(control_fn, n_paths: int, ts: Tensor, d: int, sigma_fn, device):
    """simulate_paths reshaped to `[n_paths][len(ts)]` (d=1) for metrics.json."""
    if n_paths <= 0:
        return None
    traj = simulate_paths(control_fn, n_paths, ts, d, sigma_fn, device)[:, :, 0]  # [K+1, n]
    return traj.T.tolist()


# ---------------------------------------------------------------------------
# Self-consistency operator  P(u)(t,x) = -σ(t) E_u[∇g(X_T^{u,x})]
# (thin wrappers over adjoint_sampling.operator binding ∇g of the target)
# ---------------------------------------------------------------------------

@torch.no_grad()
def operator_all_times(u_fn, target: GaussianMixtureTarget, xs, ts, sigma_fn,
                       nu_1, d, n_mc, device) -> Tensor:
    """P(u)(t_k, x) for every t_k and grid point in one batched EM pass → [K+1, n_grid, d]."""
    return operator_grid_field_all_times(
        u_fn, target.grad_g_fn(nu_1), xs, ts, sigma_fn, d, n_mc, device)


@torch.no_grad()
def operator_at_points(u_fn, target: GaussianMixtureTarget, x_states, t_start, ts,
                       sigma_fn, nu_1, d, n_mc, device) -> Tensor:
    """P(u)(t_start, x) at pre-built states [n_pts, d] → [n_pts, d]."""
    return operator_field_at_points(
        u_fn, target.grad_g_fn(nu_1), x_states, t_start, ts, sigma_fn, d, n_mc, device)


# ---------------------------------------------------------------------------
# Evaluation context + the per-iteration eval
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    target: GaussianMixtureTarget
    sigma_fn: object
    sigma_int_fn: object
    nu_fn: object
    nu_1: float
    d: int
    device: torch.device
    ts_eval: Tensor
    ts_list: list
    xs_ctrl: Tensor
    xs_ctrl_list: list
    xs_op: Tensor | None
    xs_op_list: list | None
    metric_paths: Tensor          # states sampled under u* (for the RelL2 log line)
    u_star_fn: object
    grad_g_fn: object
    do_op_eval: bool
    op_every: int
    n_op_mc: int
    n_path_op_samples: int
    n_sample_paths: int
    tiled_sup_pct: float
    log_every: int
    wandb: object | None

    @property
    def K(self) -> int:
        return self.ts_eval.shape[0] - 1


def _op_eval(net: nn.Module, prev_net: nn.Module | None, ctx: EvalContext,
             wandb_metrics: dict) -> dict:
    """The operator P(u_θ) block: grid-based §"Sanity Check" fields + path-based
    metrics.  Returns the snapshot fields it fills (grid fields, ratios, path
    metrics)."""
    K, d, device = ctx.K, ctx.d, ctx.device
    ts_eval, xs_op = ctx.ts_eval, ctx.xs_op
    n_op = xs_op.shape[0]

    # -- grid: P(u_θ), u_θ, u* on every (t_k, x) node -------------------------
    x_grid = torch.zeros(n_op, d, device=device)
    x_grid[:, 0] = xs_op
    x_all = x_grid[None].expand(K + 1, n_op, d).reshape((K + 1) * n_op, d)
    t_all = ts_eval[:, None].expand(K + 1, n_op).reshape(-1)

    Tu = operator_all_times(net, ctx.target, xs_op, ts_eval, ctx.sigma_fn,
                            ctx.nu_1, d, ctx.n_op_mc, device)          # [K+1, n_op, d]
    with torch.no_grad():
        u_star = ctx.u_star_fn(x_all, t_all).view(K + 1, n_op, d)
        u_theta = net(x_all, t_all).view(K + 1, n_op, d)

    Tu_s, u_star_s, u_theta_s = _scalar(Tu, d), _scalar(u_star, d), _scalar(u_theta, d)
    op_err = (Tu_s - u_star_s).abs()                                  # [K+1, n_op]
    u_err = (u_theta_s - u_star_s).abs()

    op_sup = _sup_over_x(op_err, ctx.tiled_sup_pct).tolist()
    u_sup = _sup_over_x(u_err, ctx.tiled_sup_pct).tolist()
    tiled_op_sup = _suffix_sup(op_sup, ctx.tiled_sup_pct)

    # §"Sanity Check" ratios: numerator / ‖u_n − u*‖_{[t,T]}  (no ‖σ‖ factor)
    denom = [max(v, 1e-12) for v in _suffix_sup(u_sup, ctx.tiled_sup_pct)]
    op_err_np = op_err.cpu().numpy()
    fields = {
        "T_u_fields": Tu_s.tolist(),
        "u_theta_op_fields": u_theta_s.tolist(),
        "u_star_op_fields": u_star_s.tolist(),
        "op_error_fields": op_err.tolist(),
        "u_error_op_fields": u_err.tolist(),
        "op_sup_error": op_sup,
        "u_sup_error_op": u_sup,
        "tiled_op_sup_error": tiled_op_sup,
        "sanity_ratio_pointwise": [(op_err_np[k] / denom[k]).tolist() for k in range(K + 1)],
        "sanity_ratio_tiled": [tiled_op_sup[k] / denom[k] for k in range(K + 1)],
    }
    for k in range(K + 1):
        wandb_metrics[f"op_sup_error/t{k:03d}"] = op_sup[k]
        wandb_metrics[f"tiled_op_sup_error/t{k:03d}"] = tiled_op_sup[k]
        wandb_metrics[f"sanity_ratio_tiled/t{k:03d}"] = fields["sanity_ratio_tiled"][k]

    # -- path-based metrics along u_θ^{n+1} trajectories --------------------
    n_sp = ctx.n_path_op_samples
    theta_traj = simulate_paths(net, n_sp, ts_eval, d, ctx.sigma_fn, device)

    def _mean_norm_diff(a, b):
        return (a - b).norm(dim=-1).mean().item()

    path_u_vs_ustar = []
    path_u_vs_Tu = [] if prev_net is not None else None
    path_u_prev_vs_ustar = [] if prev_net is not None else None
    for k in range(K + 1):
        xk = theta_traj[k]
        tk = torch.full((n_sp,), ts_eval[k].item(), device=device)
        with torch.no_grad():
            u_cur = net(xk, tk)
            u_star_k = ctx.u_star_fn(xk, tk)
        path_u_vs_ustar.append(_mean_norm_diff(u_cur, u_star_k))
        wandb_metrics[f"path_u_vs_ustar/t{k:03d}"] = path_u_vs_ustar[-1]
        if prev_net is not None:
            with torch.no_grad():
                u_prev = prev_net(xk, tk)
            Tu_k = operator_at_points(prev_net, ctx.target, xk, ts_eval[k].item(),
                                      ts_eval, ctx.sigma_fn, ctx.nu_1, d, ctx.n_op_mc, device)
            path_u_vs_Tu.append(_mean_norm_diff(u_cur, Tu_k))
            path_u_prev_vs_ustar.append(_mean_norm_diff(u_prev, u_star_k))
            wandb_metrics[f"path_u_vs_Tu/t{k:03d}"] = path_u_vs_Tu[-1]

    fields.update(path_u_vs_ustar=path_u_vs_ustar,
                  path_u_vs_Tu=path_u_vs_Tu,
                  path_u_prev_vs_ustar=path_u_prev_vs_ustar)
    return fields


_OP_SNAPSHOT_KEYS = (
    "T_u_fields", "u_theta_op_fields", "u_star_op_fields", "op_error_fields",
    "u_error_op_fields", "op_sup_error", "u_sup_error_op", "tiled_op_sup_error",
    "sanity_ratio_pointwise", "sanity_ratio_tiled",
    "path_u_vs_ustar", "path_u_vs_Tu", "path_u_prev_vs_ustar",
)


def evaluate(net: nn.Module, prev_net: nn.Module | None, snap_it: int,
             ctx: EvalContext) -> dict:
    """One eval checkpoint → a snapshot dict.  Op-eval fields are None unless
    `do_op_eval` and `snap_it % op_every == 0`."""
    K = ctx.K
    wandb_metrics: dict = {}

    rl2 = [rel_l2(net, ctx.metric_paths[k], ctx.ts_eval[k].item(), ctx.u_star_fn)
           for k in range(K + 1)]
    for k in range(K + 1):
        wandb_metrics[f"rel_l2/t{k:03d}"] = rl2[k]

    snap = dict.fromkeys(_OP_SNAPSHOT_KEYS)
    if ctx.do_op_eval and snap_it % ctx.op_every == 0:
        snap.update(_op_eval(net, prev_net, ctx, wandb_metrics))

    snap["outer_it"] = snap_it
    snap["rel_l2"] = rl2
    snap["paths_theta"] = _paths_json(net, ctx.n_sample_paths, ctx.ts_eval,
                                      ctx.d, ctx.sigma_fn, ctx.device)

    # stdout line
    idx = [0, K // 2, K]
    line = "  [eval] " + "  ".join(
        f"t={ctx.ts_eval[k].item():.2f} RelL2={rl2[k]:.4f}" for k in idx)
    if snap["op_error_fields"] is not None:
        means = [float(np.mean(snap["op_error_fields"][k])) for k in idx]
        line += "  ‖P(u_θ)-u*‖: " + "  ".join(
            f"t={ctx.ts_eval[k].item():.2f} {means[i]:.4f}" for i, k in enumerate(idx))
        line += f"  tiled(0)={snap['tiled_op_sup_error'][0]:.4f}"
    log.info(line)
    if ctx.wandb is not None:
        ctx.wandb.log({"outer_it": snap_it, **wandb_metrics})
    return snap


# ---------------------------------------------------------------------------
# Training — one outer iteration's inner loop (RAM or AM)
# ---------------------------------------------------------------------------

def run_inner_loop(net: nn.Module, optim, cfg: DictConfig, objective: str,
                   sampler: Sampler, buffer: ReplayBuffer, ctx: EvalContext) -> tuple[float, int]:
    """Refresh the frozen-ū data, run the inner optimisation, return (last loss, #steps)."""
    d, device = ctx.d, ctx.device
    n_outer, n_inner = cfg.algorithm.n_outer, cfg.algorithm.n_inner

    if objective == "am":
        # L_AM: regress u_θ along stopgrad controlled trajectories onto −σ∇g(X_T).
        # Lean adjoint ≡ ∇g(X_1) here (base drift 0) so am_loss's constant target
        # is exact.  am_loss carries NO λ(t)=1/σ² weight (ram_loss does) — kept
        # deliberately; reconcile before a quantitative RAM-vs-AM comparison.
        traj_xs, _ = sampler.sample_trajectory(net, n_outer, d, device)
        am_traj = torch.stack(traj_xs, dim=0)                 # [steps+1, n_outer, d]
        am_gg = ctx.grad_g_fn(am_traj[-1])
    else:
        x1 = sampler.sample(net, n_outer, d, device)
        buffer.add(x1, ctx.grad_g_fn(x1))

    # Reset Adam first moments; keep second moments (per-param step sizes)
    for state in optim.state.values():
        state.get("exp_avg", torch.tensor(0.0)).zero_()
    n_max = cfg.algorithm.n_inner_steps
    warmup = max(1, int(cfg.training.inner_warmup_frac * n_max))
    lr0, lr_min = cfg.training.lr, cfg.training.lr_min

    loss_window: list[float] = []
    for step in range(n_max):
        if step < warmup:
            lr = lr0 * (1e-4 + (1.0 - 1e-4) * step / warmup)
        else:
            p = (step - warmup) / max(1, n_max - warmup)
            lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * p))
        for group in optim.param_groups:
            group["lr"] = lr

        if objective == "am":
            idx = torch.randint(n_outer, (n_inner,), device=device)
            loss = am_loss(net, [am_traj[n][idx] for n in range(am_traj.shape[0])],
                           am_gg[idx], ctx.sigma_fn, sampler.steps)
        else:
            x1_b, gg_b = buffer.sample(n_inner, device=device)
            loss = ram_loss(net, x1_b, gg_b, ctx.sigma_fn, ctx.nu_fn, ctx.nu_1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_val = loss.item()

        if cfg.algorithm.inner_tol > 0:
            loss_window.append(loss_val)
            if len(loss_window) > cfg.algorithm.inner_patience:
                loss_window.pop(0)
            if (len(loss_window) == cfg.algorithm.inner_patience
                    and max(loss_window) - min(loss_window) < cfg.algorithm.inner_tol):
                return loss_val, step + 1
    return loss_val, n_max


# ---------------------------------------------------------------------------
# Setup, §5.0 fixed-point check, output
# ---------------------------------------------------------------------------

def build_context(cfg: DictConfig, device: torch.device) -> tuple[EvalContext, str]:
    d = cfg.target.d
    target = GaussianMixtureTarget(
        w1=float(cfg.target.w1), lambda1=float(cfg.target.lambda1), mu1=float(cfg.target.mu1),
        w2=1.0 - float(cfg.target.w1), lambda2=float(cfg.target.lambda2), mu2=float(cfg.target.mu2))

    schedule = str(cfg.get("sigma_schedule", "constant"))
    floor = float(cfg.get("sigma_floor", 0.0))
    sigma_fn, nu_fn, nu_1 = utils.make_noise_schedule(schedule, float(cfg.sigma), floor)
    sigma_int_fn = sigma_int_from_nu(nu_fn, nu_1)
    log.info(f"noise schedule: {schedule} (σ scale={cfg.sigma}, floor={floor}), ν₁={nu_1:.4f}")
    assert target.lambda1 > 1.0 / nu_1 and target.lambda2 > 1.0 / nu_1, (
        f"need λᵢ > 1/ν₁; got λ={target.lambda1},{target.lambda2}, ν₁={nu_1:.3f}")

    objective = str(cfg.algorithm.get("objective", "ram")).lower()
    assert objective in ("ram", "am"), f"algorithm.objective must be ram|am, got {objective}"
    log.info(f"training objective: {objective}")

    K = cfg.eval.n_time_slices
    ts_eval = torch.linspace(0.0, 1.0, K + 1, device=device)
    tiled_sup_pct = float(cfg.eval.get("tiled_sup_percentile", 100.0))
    if tiled_sup_pct < 100.0:
        log.info(f"tiled/sup norms clamped at the {tiled_sup_pct:g}th percentile")

    x_center = 0.5 * (target.mu1 + target.mu2)
    hw = float(cfg.eval.eval_x_range)
    xs_ctrl = torch.linspace(x_center - hw, x_center + hw, cfg.eval.n_ctrl_grid, device=device)

    do_op_eval = bool(cfg.eval.op_eval)
    if do_op_eval:
        xs_op = torch.linspace(x_center - hw, x_center + hw, int(cfg.eval.n_op_grid), device=device)
    else:
        xs_op = None

    u_star_fn = target.optimal_control_fn(sigma_fn, sigma_int_fn, nu_1, d)
    metric_paths = simulate_paths(u_star_fn, cfg.eval.n_metric_samples, ts_eval, d, sigma_fn, device)

    wandb = None
    if cfg.logging.wandb:
        import wandb as _wandb
        _wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))
        wandb = _wandb

    ctx = EvalContext(
        target=target, sigma_fn=sigma_fn, sigma_int_fn=sigma_int_fn, nu_fn=nu_fn, nu_1=nu_1,
        d=d, device=device, ts_eval=ts_eval, ts_list=ts_eval.tolist(),
        xs_ctrl=xs_ctrl, xs_ctrl_list=xs_ctrl.tolist(),
        xs_op=xs_op, xs_op_list=(xs_op.tolist() if xs_op is not None else None),
        metric_paths=metric_paths, u_star_fn=u_star_fn, grad_g_fn=target.grad_g_fn(nu_1),
        do_op_eval=do_op_eval, op_every=int(cfg.eval.op_every) if do_op_eval else 1,
        n_op_mc=int(cfg.eval.n_op_mc_samples) if do_op_eval else 0,
        n_path_op_samples=int(cfg.eval.n_path_op_samples),
        n_sample_paths=int(cfg.eval.n_sample_paths),
        tiled_sup_pct=tiled_sup_pct, log_every=int(cfg.logging.log_every), wandb=wandb)
    return ctx, objective


def fixed_point_check(cfg: DictConfig, ctx: EvalContext) -> dict | None:
    """§5.0: ‖P(u*)(t,x) − u*(t,x)‖ ≪ 1  — the analytic-u* self-consistency check."""
    if not bool(cfg.eval.get("fixed_point_check", True)):
        return None
    n_mc = int(cfg.eval.get("n_fixed_point_mc", 512))
    n_grid = int(cfg.eval.get("n_fixed_point_grid", 121))
    x_center = 0.5 * (ctx.target.mu1 + ctx.target.mu2)
    hw = float(cfg.eval.eval_x_range)
    xs = torch.linspace(x_center - hw, x_center + hw, n_grid, device=ctx.device)
    fp = fixed_point_residual_field(ctx.u_star_fn, ctx.grad_g_fn, xs, ctx.ts_eval,
                                    ctx.sigma_fn, ctx.d, n_mc, ctx.device)
    resid = fp["residual"]
    log.info(f"[fixed-point check] ‖P(u*) − u*‖ on {n_grid}×{ctx.K + 1} grid (n_mc={n_mc}): "
             f"max={resid.max().item():.4g}  mean={resid.mean().item():.4g}  "
             f"max@t<1={resid[:-1].max().item():.4g}")
    scal = lambda z: (z[..., 0] if ctx.d == 1 else z.norm(dim=-1)).tolist()
    if ctx.wandb is not None:
        ctx.wandb.log({"fixed_point/resid_max": resid.max().item(),
                       "fixed_point/resid_mean": resid.mean().item()})
    return {"xs": xs.tolist(), "residual": resid.tolist(),
            "p_u_star": scal(fp["p_u_star"]), "u_star": scal(fp["u_star"]), "n_mc": n_mc}


def write_outputs(output_dir: Path, cfg: DictConfig, ctx: EvalContext, net: nn.Module,
                  snapshots: list[dict], fp_data: dict | None, objective: str) -> None:
    d, K = ctx.d, ctx.K
    with torch.no_grad():
        u_star_field = [_grid_field(ctx.u_star_fn, ctx.xs_ctrl, ctx.ts_eval[k].item(), d)
                        for k in range(K + 1)]
        u_theta_field = [_grid_field(net, ctx.xs_ctrl, ctx.ts_eval[k].item(), d)
                         for k in range(K + 1)]
    n_sp = ctx.n_sample_paths
    paths_star = _paths_json(ctx.u_star_fn, n_sp, ctx.ts_eval, d, ctx.sigma_fn, ctx.device)
    paths_theta = _paths_json(net, n_sp, ctx.ts_eval, d, ctx.sigma_fn, ctx.device)

    save_snapshots({
        "snapshots": snapshots, "ts": ctx.ts_list, "d": d,
        "xs": ctx.xs_ctrl_list, "xs_op": ctx.xs_op_list,
        "u_star_field": u_star_field, "u_theta_field": u_theta_field,
        "paths_star": paths_star, "paths_theta": paths_theta,
        "target_params": dict(w1=ctx.target.w1, lambda1=ctx.target.lambda1, mu1=ctx.target.mu1,
                              w2=ctx.target.w2, lambda2=ctx.target.lambda2, mu2=ctx.target.mu2),
        "sigma": float(cfg.sigma), "sigma_schedule": str(cfg.get("sigma_schedule", "constant")),
        "sigma_floor": float(cfg.get("sigma_floor", 0.0)), "objective": objective,
        "tiled_sup_percentile": ctx.tiled_sup_pct, "fixed_point_check": fp_data,
    }, output_dir)

    set_suffix_percentile(ctx.tiled_sup_pct)
    d_conv, d_ctrl, d_term, d_sbm, d_op = (
        output_dir / s for s in ("convergence", "control", "terminal", "same_bm", "operator"))
    ts, xs_op = ctx.ts_list, ctx.xs_op_list

    d_sbm.mkdir(parents=True, exist_ok=True)
    plot_sbm_ratio(snapshots, ts, d_sbm)
    plot_sbm_ratio_learned(snapshots, ts, d_sbm)
    if fp_data is not None:
        plot_fixed_point_residual(fp_data, ts, d, d_sbm)
    if bool(cfg.eval.get("sanity_ratio_only", False)):
        log.info("sanity_ratio_only: wrote same_bm/{sbm_ratio, sbm_ratio_learned, "
                 "u_star_fixed_point_residual}.png; all other plots skipped.")
        return

    for _d in (d_conv, d_ctrl, d_term, d_op):
        _d.mkdir(parents=True, exist_ok=True)
    plot_optimal_control(ts, ctx.xs_ctrl_list, u_star_field, d, d_ctrl, u_theta_field)
    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, ctx.target.terminal_pdf, d, d_term)
    if ctx.do_op_eval and xs_op:
        plot_operator_vs_next_control(snapshots, ts, xs_op, d, d_ctrl)
        plot_path_u_vs_Tu(snapshots, ts, d_op)
        plot_learned_pointwise_over_tiled(snapshots, ts, xs_op, d_conv)
        plot_path_u_vs_ustar_per_t_over_tiled(snapshots, ts, d_conv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    utils.seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    ctx, objective = build_context(cfg, device)
    net = DriftMLP(d=ctx.d, hidden_dim=cfg.network.hidden_dim,
                   n_layers=cfg.network.n_layers, t_emb_dim=cfg.network.t_emb_dim,
                   time_embedding=cfg.network.get("time_embedding", "sinusoidal")).to(device)
    sampler = Sampler(ctx.sigma_fn, steps=cfg.sampler.steps)
    buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)

    fp_data = fixed_point_check(cfg, ctx)

    snapshots: list[dict] = []
    prev_net: nn.Module | None = None   # u_θ^{n-1} copy for the path-based P(u_{n-1}) eval

    # outer_it = -1 is a pre-training pass: it records the freshly-initialised
    # network as snapshot "iteration 0", so u_θ^n can be compared against
    # P(u_θ^{n-1}).  snap_it = outer_it + 1 is the recorded iteration index.
    for outer_it in range(-1, cfg.training.outer_iterations):
        snap_it = outer_it + 1
        if outer_it >= 0:
            loss_val, steps = run_inner_loop(net, optim, cfg, objective, sampler, buffer, ctx)
            if outer_it % ctx.log_every == 0:
                log.info(f"[{snap_it:4d}] {objective}_loss={loss_val:.4f}  inner_steps={steps}")
                if ctx.wandb is not None:
                    ctx.wandb.log({f"{objective}_loss": loss_val, "inner_steps": steps,
                                   "outer_it": snap_it})

        if snap_it < cfg.eval.first_k or snap_it % cfg.eval.every == 0:
            snapshots.append(evaluate(net, prev_net, snap_it, ctx))
            if ctx.do_op_eval and snap_it % ctx.op_every == 0:
                prev_net = copy.deepcopy(net).eval()

    log.info("Training complete.")
    write_outputs(Path(HydraConfig.get().runtime.output_dir), cfg, ctx, net,
                  snapshots, fp_data, objective)


if __name__ == "__main__":
    main()

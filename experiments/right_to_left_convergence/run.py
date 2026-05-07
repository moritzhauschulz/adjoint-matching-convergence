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
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Hydra changes cwd to its output dir; keep the experiment directory importable.
sys.path.insert(0, str(Path(__file__).parent))

from adjoint_sampling import DriftMLP, Sampler, ReplayBuffer, ram_loss, utils
from plotting import (plot_convergence, plot_heatmaps, plot_optimal_control,
                      plot_contraction_heatmaps, save_snapshots)


# ---------------------------------------------------------------------------
# Analytic optimal control and Riccati variance
# ---------------------------------------------------------------------------

def riccati_coefficient(t: Tensor, lambda_: float, sigma_fn) -> Tensor:
    """a(t) = λ / (1 + λ σ₀² (1 − t)).

    Scalar closed-form for constant σ.
    """
    sigma_0 = sigma_fn(t)          # [B] or scalar
    return lambda_ / (1.0 + lambda_ * sigma_0 ** 2 * (1.0 - t))


def optimal_control(x: Tensor, t: Tensor, lambda_: float, mu: float, sigma_fn) -> Tensor:
    """u*(t, x) = −σ(t) a(t) (x − μ)   (Section 4.1 Lemma)."""
    sigma_t = sigma_fn(t).unsqueeze(-1)                   # [B, 1]
    a_t = riccati_coefficient(t, lambda_, sigma_fn).unsqueeze(-1)  # [B, 1]
    return -sigma_t * a_t * (x - mu)


def riccati_mean_and_variance(ts: Tensor, lambda_: float, mu: float, sigma_fn) -> tuple[Tensor, Tensor]:
    """Integrate the mean and variance of X_t under u* on the grid ts.

    Mean ODE:     dm/dt = −σ²(t) a(t) (m − μ),   m₀ = 0
    Variance ODE: dV/dt = −2σ²(t) a(t) V + σ²(t), V₀ = 0

    Returns (Ms, Vs), each of shape [len(ts)].
    Ms[k] is the scalar mean per dimension (broadcast to all d dims via N(Ms[k]·1, Vs[k]·I)).
    Uses forward Euler on the provided grid.
    """
    n = ts.shape[0]
    Vs = torch.zeros(n, device=ts.device)
    Ms = torch.zeros(n, device=ts.device)
    for i in range(n - 1):
        dt = (ts[i + 1] - ts[i]).item()
        t_i = ts[i].unsqueeze(0)
        sigma2 = sigma_fn(t_i).pow(2).item()
        a_i = riccati_coefficient(t_i, lambda_, sigma_fn).item()
        dV = -2.0 * sigma2 * a_i * Vs[i].item() + sigma2
        dM = -sigma2 * a_i * (Ms[i].item() - mu)
        Vs[i + 1] = Vs[i] + dV * dt
        Ms[i + 1] = Ms[i] + dM * dt
    return Ms, Vs



# ---------------------------------------------------------------------------
# Evaluation metrics  (Section 4.1 / §6 of CLAUDE.md)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rel_l2(
    u_theta: nn.Module,
    t_val: float,
    Vt_val: float,
    Mt_val: float,
    lambda_: float,
    mu: float,
    sigma_fn,
    d: int,
    n_samples: int,
    device,
) -> float:
    """RelL₂(t) = ‖u_θ − u*‖_{L₂(P^{u*})} / ‖u*‖_{L₂(P^{u*})}.

    Samples X_t ~ N(Mt_val · 1_d, Vt_val · I_d), the correct marginal under u*.
    """
    t = torch.full((n_samples,), t_val, device=device)
    std = math.sqrt(max(Vt_val, 1e-8))
    x_t = Mt_val + std * torch.randn(n_samples, d, device=device)

    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, mu, sigma_fn)

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
    Mt_val: float,
    lambda_: float,
    mu: float,
    sigma_fn,
    d: int,
    n_samples: int,
    device,
) -> float:
    """AbsL₂(t) = (E_{P^{u*}} ‖u_θ(t, X_t) − u*(t, X_t)‖²)^{1/2}.

    Samples X_t ~ N(Mt_val · 1_d, Vt_val · I_d), the correct marginal under u*.
    """
    t = torch.full((n_samples,), t_val, device=device)
    std = math.sqrt(max(Vt_val, 1e-8))
    x_t = Mt_val + std * torch.randn(n_samples, d, device=device)

    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, mu, sigma_fn)

    return math.sqrt((u_hat - u_star).pow(2).sum(-1).mean().item())


@torch.no_grad()
def _error_field(
    u_theta: nn.Module,
    t_val: float,
    lambda_: float,
    mu: float,
    sigma_fn,
    d: int,
    xs: Tensor,
) -> Tensor:
    """‖u_θ(t,x) − u*(t,x)‖ for each x in xs (shape [n_grid]).

    xs must be a 1-D linspace; x_t is constructed as [n_grid, d] with xs on axis 0
    and zeros elsewhere (axis-aligned; exact for d=1).
    """
    n_grid = xs.shape[0]
    x_t = torch.zeros(n_grid, d, device=xs.device)
    x_t[:, 0] = xs
    t = torch.full((n_grid,), t_val, device=xs.device)
    u_hat = u_theta(x_t, t)
    u_star = optimal_control(x_t, t, lambda_, mu, sigma_fn)
    return (u_hat - u_star).norm(dim=-1)   # [n_grid]


@torch.no_grad()
def abs_linf(
    u_theta: nn.Module,
    t_val: float,
    lambda_: float,
    mu: float,
    sigma_fn,
    d: int,
    xs: Tensor,
) -> tuple[float, list[float]]:
    """AbsL∞(t) = max over uniform grid xs of ‖u_θ − u*‖.

    Returns (scalar_max, per_point_errors) so the caller can reuse the field
    for heatmap construction without re-evaluating the network.
    """
    field = _error_field(u_theta, t_val, lambda_, mu, sigma_fn, d, xs)
    return field.max().item(), field.tolist()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    utils.seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    d = cfg.target.d
    lambda_ = float(cfg.target.lambda_)
    mu = float(cfg.target.mu)

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

    # Precompute Riccati mean and variance on evaluation grid
    K = cfg.eval.n_time_slices
    ts_eval = torch.linspace(0.0, 1.0, K + 1, device=device)
    Ms_eval, Vs_eval = riccati_mean_and_variance(ts_eval, lambda_, mu, sigma_fn)

    # Uniform x grid for sup-norm and heatmap (precomputed once), centred at μ
    xs_linf = torch.linspace(mu - cfg.eval.linf_x_range, mu + cfg.eval.linf_x_range,
                             cfg.eval.n_linf_grid, device=device)
    xs_list = xs_linf.tolist()

    # Ground-truth optimal control field u*(t,x) — computed once, shape [K+1, n_grid]
    with torch.no_grad():
        x_col = xs_linf.unsqueeze(-1).expand(-1, d)      # [n_grid, d]
        u_star_field: list[list[float]] = []
        for k in range(K + 1):
            t_row = ts_eval[k].expand(cfg.eval.n_linf_grid)   # [n_grid]
            u_star_k = optimal_control(x_col, t_row, lambda_, mu, sigma_fn)  # [n_grid, d]
            # For d=1 store scalar; for d>1 store norm
            if d == 1:
                u_star_field.append(u_star_k[:, 0].tolist())
            else:
                u_star_field.append(u_star_k.norm(dim=-1).tolist())

    if cfg.logging.wandb:
        import wandb
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, config=dict(cfg))

    def grad_g_fn(x1: Tensor) -> Tensor:
        return lambda_ * (x1 - mu)

    snapshots: list[dict] = []
    ts_list = ts_eval.tolist()
    prev_al_inf_vals: list[float] | None = None
    prev_tiled_al_inf_vals: list[float] | None = None

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
            error_fields: list[list[float]] = []   # [K+1, n_grid] — reused for heatmap
            wandb_metrics: dict = {}

            for k in range(K + 1):
                t_val = ts_eval[k].item()
                Vt_val = Vs_eval[k].item()
                Mt_val = Ms_eval[k].item()
                rl2 = rel_l2(net, t_val, Vt_val, Mt_val, lambda_, mu, sigma_fn, d,
                             cfg.eval.n_metric_samples, device)
                al2 = abs_l2(net, t_val, Vt_val, Mt_val, lambda_, mu, sigma_fn, d,
                             cfg.eval.n_metric_samples, device)
                al_inf, field = abs_linf(net, t_val, lambda_, mu, sigma_fn, d, xs_linf)
                rl2_vals.append(rl2)
                al2_vals.append(al2)
                al_inf_vals.append(al_inf)
                error_fields.append(field)
                wandb_metrics[f"rel_l2/t{k:03d}"] = rl2
                wandb_metrics[f"abs_l2/t{k:03d}"] = al2
                wandb_metrics[f"abs_linf/t{k:03d}"] = al_inf
                if prev_al_inf_vals is not None and prev_al_inf_vals[k] > 1e-12:
                    wandb_metrics[f"contr_fact/t{k:03d}"] = al_inf_vals[k] / prev_al_inf_vals[k]

            # Contraction factor: AbsL∞(n+1) / AbsL∞(n) per time slice
            if prev_al_inf_vals is not None:
                contr_fact = [
                    (al_inf_vals[k] / prev_al_inf_vals[k])
                    if prev_al_inf_vals[k] > 1e-12 else float("nan")
                    for k in range(K + 1)
                ]
            else:
                contr_fact = [float("nan")] * (K + 1)

            # Tiled sup-norm ||u_θ - u*||_{[t_k, T]}: suffix max over time slices
            # Scalar: suffix max of al_inf_vals
            tiled_al_inf_vals: list[float] = [0.0] * (K + 1)
            running = 0.0
            for k in range(K, -1, -1):
                running = max(running, al_inf_vals[k])
                tiled_al_inf_vals[k] = running

            # Pointwise tiled error field: suffix max of error_fields over time axis
            ef_arr = np.array(error_fields)                 # [K+1, n_grid]
            tiled_ef_arr = np.maximum.accumulate(ef_arr[::-1])[::-1]  # suffix max
            tiled_error_fields = tiled_ef_arr.tolist()

            # Tiled contraction factor
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

            snapshots.append({
                "outer_it": outer_it,
                "rel_l2": rl2_vals,
                "abs_l2": al2_vals,
                "abs_linf": al_inf_vals,
                "contr_fact": contr_fact,
                "error_fields": error_fields,        # [K+1, n_grid]
                "tiled_al_inf": tiled_al_inf_vals,
                "tiled_contr_fact": tiled_contr_fact,
                "tiled_error_fields": tiled_error_fields,  # [K+1, n_grid]
            })
            prev_al_inf_vals = al_inf_vals
            prev_tiled_al_inf_vals = tiled_al_inf_vals

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
    save_snapshots({"snapshots": snapshots, "ts": ts_list, "xs": xs_list,
                    "u_star_field": u_star_field, "d": d}, output_dir)
    plot_convergence(snapshots, ts_list, output_dir)
    plot_optimal_control(ts_list, xs_list, u_star_field, d, output_dir)
    plot_heatmaps(snapshots, ts_list, xs_list, output_dir)
    plot_contraction_heatmaps(snapshots, ts_list, xs_list, output_dir)


if __name__ == "__main__":
    main()

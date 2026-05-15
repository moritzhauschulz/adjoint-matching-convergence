"""Plots for the bimodal same-BM experiment.

Extends the bimodal experiment plots with two new figures for the same-BM LHS metric:
  - same_bm_lhs_curves.png:    mean LHS(t) vs t, one curve per eval checkpoint
  - same_bm_lhs_heatmaps.png:  LHS(t, x) heatmaps over training

All other plots are identical to right_to_left_convergence_bimodal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Shared plots (identical to bimodal experiment)
# ---------------------------------------------------------------------------

def plot_convergence(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_arr = np.array(ts)
    n = len(snapshots)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    for metric_key, ylabel, fname in [
        ("rel_l2",           r"$\mathrm{RelL}_2(t)$",                 "convergence_rel_l2.png"),
        ("abs_l2",           r"$\mathrm{AbsL}_2(t)$",                 "convergence_abs_l2.png"),
        ("abs_linf",         r"$\mathrm{AbsL}_\infty(t)$",            "convergence_abs_linf.png"),
        ("tiled_al_inf",     r"$\|u_\theta-u^*\|_{[t,T]}$",          "convergence_tiled_al_inf.png"),
        ("contr_fact",       r"$\mathrm{ContrFact}(t;\,n)$",          "convergence_contr_fact.png"),
        ("tiled_contr_fact", r"$\mathrm{TiledContrFact}(t;\,n)$",    "convergence_tiled_contr_fact.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, snap in enumerate(snapshots):
            values = snap[metric_key]
            valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            if not valid:
                continue
            ax.plot(ts_arr, values, color=colours[i],
                    label=f"iter {snap['outer_it']}", linewidth=1.5)
        ax.set_xlabel("time $t$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title("Right-to-Left Convergence\n(light = early, dark = late)", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 6:
            idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
            if idxs[-1] != len(handles) - 1:
                idxs.append(len(handles) - 1)
            handles = [handles[i] for i in idxs]
            labels  = [labels[i]  for i in idxs]
        ax.legend(handles, labels, fontsize=8, loc="upper right")
        if metric_key in ("contr_fact", "tiled_contr_fact"):
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        fig.tight_layout()
        path = output_dir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def _overlay_paths(ax, ts_arr: np.ndarray, paths: list[list[float]]) -> None:
    for path in paths:
        ax.plot(ts_arr, path, color="#404040", alpha=0.6, linewidth=0.8, rasterized=True)


def plot_optimal_control(
    ts, xs, u_star_field, d, output_dir,
    u_theta_field=None, paths_star=None, paths_theta=None,
) -> None:
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs)
    Z_star = np.array(u_star_field)
    if u_theta_field is not None:
        Z_theta = np.array(u_theta_field)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        if d == 1:
            abs_max = max(np.abs(Z_star).max(), np.abs(Z_theta).max())
            kwargs = dict(cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, shading="auto")
            cb_label = r"$u(t,x)$"
        else:
            vmax = max(Z_star.max(), Z_theta.max())
            kwargs = dict(cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
            cb_label = r"$\|u(t,x)\|$"
        im = axes[0].pcolormesh(ts_arr, xs_arr, Z_star.T, **kwargs)
        axes[0].set_title(r"Ground-truth $u^*(t,x)$", fontsize=11)
        axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$x$")
        if paths_star is not None:
            _overlay_paths(axes[0], ts_arr, paths_star)
        axes[1].pcolormesh(ts_arr, xs_arr, Z_theta.T, **kwargs)
        axes[1].set_title(r"Learned $u_\theta(t,x)$ (final)", fontsize=11)
        axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$x$")
        if paths_theta is not None:
            _overlay_paths(axes[1], ts_arr, paths_theta)
        fig.colorbar(im, ax=axes.tolist(), label=cb_label)
    else:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        abs_max = np.abs(Z_star).max()
        im = ax.pcolormesh(ts_arr, xs_arr, Z_star.T, cmap="RdBu_r",
                           vmin=-abs_max, vmax=abs_max, shading="auto")
        ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
        ax.set_title(r"Ground-truth optimal control $u^*(t,x)$", fontsize=11)
        if paths_star is not None:
            _overlay_paths(ax, ts_arr, paths_star)
        fig.colorbar(im, ax=ax, label=r"$u^*(t,x)$")
    path = output_dir / "heatmap_u_star.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_control_evolution(snapshots, ts, xs, d, u_star_field, output_dir,
                           max_subplots: int = 20) -> None:
    snaps = [s for s in snapshots if "u_theta_field" in s]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    output_dir = Path(output_dir)
    ts_arr = np.array(ts); xs_arr = np.array(xs)
    Z_star = np.array(u_star_field)
    n = len(snaps); ncols = min(n, 5); nrows = math.ceil(n / ncols)
    if d == 1:
        abs_max = max(np.abs(Z_star).max(),
                      max(np.abs(np.array(s["u_theta_field"])).max() for s in snaps))
        kwargs = dict(cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, shading="auto")
        cb_label = r"$u_\theta(t,x)$"
    else:
        vmax = max(Z_star.max(), max(np.array(s["u_theta_field"]).max() for s in snaps))
        kwargs = dict(cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
        cb_label = r"$\|u_\theta(t,x)\|$"
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(ts_arr, xs_arr, np.array(snap["u_theta_field"]).T, **kwargs)
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8); ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
        if snap.get("paths_theta") is not None:
            _overlay_paths(ax, ts_arr, snap["paths_theta"])
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(r"Learned control $u_\theta(t,x)$ over training", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8, label=cb_label)
    path = output_dir / "heatmap_control_evolution.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def _error_heatmap(snapshots, field_key, ts_arr, xs_arr, title, colorbar_label,
                   path, max_subplots=20) -> None:
    snaps = [s for s in snapshots if field_key in s]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    n = len(snaps); ncols = min(n, 5); nrows = math.ceil(n / ncols)
    vmax = max(max(max(row) for row in s[field_key]) for s in snaps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        Z = np.array(snap[field_key])
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis",
                           vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8); ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(title, fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label=colorbar_label)
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_heatmaps(snapshots, ts, xs, output_dir, max_subplots=20) -> None:
    output_dir = Path(output_dir)
    ts_arr = np.array(ts); xs_arr = np.array(xs)
    _error_heatmap(snapshots, "error_fields", ts_arr, xs_arr,
                   title=r"$\|u_\theta(t,x) - u^*(t,x)\|$",
                   colorbar_label="pointwise error",
                   path=output_dir / "heatmap_error_field.png",
                   max_subplots=max_subplots)
    _error_heatmap(snapshots, "tiled_error_fields", ts_arr, xs_arr,
                   title=r"$\max_{s \geq t}\|u_\theta(s,x) - u^*(s,x)\|$",
                   colorbar_label="tiled error",
                   path=output_dir / "heatmap_tiled_error_field.png",
                   max_subplots=max_subplots)


def plot_contraction_heatmaps(snapshots, ts, xs, output_dir, max_subplots=20,
                              eps=1e-12) -> None:
    output_dir = Path(output_dir)
    snaps = [s for s in snapshots if "error_fields" in s]
    if len(snaps) < 2:
        return
    pairs = list(zip(snaps[:-1], snaps[1:]))
    if len(pairs) > max_subplots:
        idxs = np.linspace(0, len(pairs) - 1, max_subplots, dtype=int)
        pairs = [pairs[i] for i in idxs]
    n = len(pairs); ncols = min(n, 5); nrows = math.ceil(n / ncols)
    ts_arr = np.array(ts); xs_arr = np.array(xs)
    ratio_fields = []
    for s_prev, s_next in pairs:
        denom = np.array(s_prev["error_fields"]).clip(min=eps)
        numer = np.array(s_next["error_fields"])
        ratio_fields.append((numer / denom).clip(0.0, 5.0))
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=5.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, ((s_prev, s_next), ratio) in enumerate(zip(pairs, ratio_fields)):
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(ts_arr, xs_arr, ratio.T, cmap="RdBu_r",
                           norm=norm, shading="auto")
        ax.set_title(f"iter {s_prev['outer_it']}→{s_next['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8); ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        r"Contraction ratio $\|u^{n+1}_\theta - u^*\| / \|u^n_\theta - u^*\|$"
        "\n(blue < 1 = improving, red > 1 = regressing)", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8, label="ratio")
    path = output_dir / "heatmap_contr_ratio.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_inner_steps(snapshots, output_dir) -> None:
    output_dir = Path(output_dir)
    outer_its = [s["outer_it"] for s in snapshots if "inner_steps" in s]
    steps = [s["inner_steps"] for s in snapshots if "inner_steps" in s]
    if not outer_its:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(outer_its, steps, color="steelblue", linewidth=1.5)
    ax.axhline(max(steps), color="gray", linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"budget ({max(steps)})")
    ax.set_xlabel("outer iteration"); ax.set_ylabel("inner steps taken")
    ax.set_title("Inner-loop steps per outer iteration"); ax.set_ylim(bottom=0)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
    path = output_dir / "inner_steps.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_inner_convergence(snapshots, output_dir) -> None:
    snaps = [s for s in snapshots if s.get("inner_loss_curve")]
    if not snaps:
        return
    output_dir = Path(output_dir)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, snap in enumerate(snaps):
        ax.plot(range(len(snap["inner_loss_curve"])), snap["inner_loss_curve"],
                color=colours[i], label=f"outer {snap['outer_it']}", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("inner step"); ax.set_ylabel("RAM loss")
    ax.set_title("Inner-loop convergence across training\n(light = early, dark = late)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 8:
        idxs = np.linspace(0, len(handles) - 1, 8, dtype=int)
        handles = [handles[i] for i in idxs]; labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper right"); fig.tight_layout()
    path = output_dir / "inner_convergence.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d,
                                output_dir) -> None:
    output_dir = Path(output_dir)
    x1_star  = np.array([p[-1] for p in paths_star])
    x1_theta = np.array([p[-1] for p in paths_theta])
    x_lo = float(min(x1_star.min(), x1_theta.min())) - 1.0
    x_hi = float(max(x1_star.max(), x1_theta.max())) + 1.0
    x_pdf = np.linspace(x_lo, x_hi, 400)
    pdf = target_pdf_fn(x_pdf)
    bins = max(20, len(x1_star) // 5)
    x_label = r"$x_1$" if d == 1 else r"$x_1[0]$"
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(x1_star, bins=bins, density=True, alpha=0.45, color="steelblue",
            label=r"$u^*$ samples", range=(x_lo, x_hi))
    ax.hist(x1_theta, bins=bins, density=True, alpha=0.45, color="darkorange",
            label=r"$u_\theta$ samples (final)", range=(x_lo, x_hi))
    ax.plot(x_pdf, pdf, color="crimson", linewidth=1.5, linestyle="--", label="target")
    ax.set_xlabel(x_label); ax.set_ylabel("density")
    ax.set_title(r"Terminal distribution $X_1$"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    path = output_dir / "terminal_distributions.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


def plot_terminal_evolution(snapshots, target_pdf_fn, d, output_dir,
                            max_subplots=20) -> None:
    snaps = [s for s in snapshots if s.get("paths_theta") is not None]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    output_dir = Path(output_dir)
    n = len(snaps); ncols = min(n, 5); nrows = math.ceil(n / ncols)
    all_x1 = [p[-1] for s in snaps for p in s["paths_theta"]]
    x_lo = min(all_x1) - 1.0; x_hi = max(all_x1) + 1.0
    x_pdf = np.linspace(x_lo, x_hi, 400); pdf = target_pdf_fn(x_pdf)
    bins = max(10, len(snaps[0]["paths_theta"]) // 5)
    x_label = r"$x_1$" if d == 1 else r"$x_1[0]$"
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        x1 = np.array([p[-1] for p in snap["paths_theta"]])
        ax.hist(x1, bins=bins, density=True, alpha=0.5, color="darkorange",
                range=(x_lo, x_hi))
        ax.plot(x_pdf, pdf, color="crimson", linewidth=1.0, linestyle="--")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel(x_label, fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_xlim(x_lo, x_hi); ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(r"Terminal distribution $X_1$ under $u_\theta$ (dashed = target)", fontsize=11)
    path = output_dir / "terminal_evolution.png"
    fig.savefig(path, dpi=150); plt.close(fig); print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Same-BM LHS plots (new)
# ---------------------------------------------------------------------------

def plot_same_bm_lhs_curves(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Curves of mean_x LHS_SBM(t) vs t, one per eval checkpoint.

    Expected shape: right-to-left decay (small near t=T, large near t=0).
    Saves to output_dir/same_bm_lhs_curves.png.
    """
    snaps = [s for s in snapshots if s.get("same_bm_lhs_fields") is not None]
    if not snaps:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Oranges(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        fields = snap["same_bm_lhs_fields"]   # list[list[float]], shape [K+1][n_grid]
        means = [float(np.mean(f)) for f in fields]
        ax.plot(ts_arr, means, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{mean}_x\;\mathrm{LHS}_{\mathrm{SBM}}(t,x)$", fontsize=11)
    ax.set_title(
        r"Same-BM LHS: $\mathbb{E}_B[\|\sigma(t)\nabla r(X_T^{u_\theta,x})"
        r" - \sigma(t)\nabla r(X_T^{u^*,x})\|]$"
        "\n(light = early training, dark = late — should decay right-to-left)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels  = [labels[i]  for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "same_bm_lhs_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_same_bm_lhs_heatmaps(
    snapshots: list[dict],
    ts: list[float],
    xs_sbm: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    """Heatmaps of LHS_SBM(t, x) over the (t, x) grid, one per eval checkpoint.

    Saves to output_dir/same_bm_lhs_heatmaps.png.
    """
    snaps = [s for s in snapshots if s.get("same_bm_lhs_fields") is not None]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_sbm)
    n = len(snaps)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    vmax = max(
        max(max(row) for row in snap["same_bm_lhs_fields"])
        for snap in snaps
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        Z = np.array(snap["same_bm_lhs_fields"])   # [K+1, n_sbm_grid]
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis",
                           vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"Same-BM LHS $\mathbb{E}_B[\|\sigma(t)\nabla r(X_T^{u_\theta,x})"
        r" - \sigma(t)\nabla r(X_T^{u^*,x})\|]$ over $(t,x)$"
        "\n(should → 0 as $t \\to T$, right-to-left)",
        fontsize=10,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label=r"$\mathrm{LHS}_{\mathrm{SBM}}(t,x)$")
    path = output_dir / "same_bm_lhs_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tiled_same_bm_lhs(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Plot sup_{x_s ∈ ℝ, s ∈ [t,T]} E_B[‖σ(t)∇r(X_T^u) − σ(t)∇r(X_T^{u*})‖] vs t.

    This is the LHS of the §2.1 contraction bound tiled over [t,T].
    Also overlays the RHS shape: tiled_al_inf(t) · (T−t)  (up to the Lipschitz
    constant Lip(∇r) · σ₀² which is a fixed scalar, shown as a dashed reference).

    Saves to output_dir/tiled_same_bm_lhs.png.
    """
    snaps = [s for s in snapshots if s.get("tiled_same_bm_lhs") is not None]
    if not snaps:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    T = ts_arr[-1]
    n = len(snaps)
    colours = cm.Oranges(np.linspace(0.3, 1.0, max(n, 1)))
    colours_rhs = cm.Greys(np.linspace(0.3, 0.8, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, snap in enumerate(snaps):
        tiled_sbm = snap["tiled_same_bm_lhs"]           # [K+1]
        ax.plot(ts_arr, tiled_sbm, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.8)

        # RHS shape: tiled_al_inf(t) · (T − t)  [scaled to match at t=0 for visibility]
        tiled_al_inf = np.array(snap["tiled_al_inf"])    # [K+1]
        rhs_shape = tiled_al_inf * np.maximum(T - ts_arr, 0.0)
        # Scale so the two curves share the same value at t=0 (if nonzero)
        if rhs_shape[0] > 1e-12 and tiled_sbm[0] > 1e-12:
            scale = tiled_sbm[0] / rhs_shape[0]
            ax.plot(ts_arr, scale * rhs_shape,
                    color=colours_rhs[i], linewidth=1.0, linestyle="--", alpha=0.6)

    # Legend entries for the two line types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="darkorange", linewidth=1.8,
               label=r"$\sup_{x_s,\,s\in[t,T]}\mathbb{E}_B[\cdots]$ (LHS)"),
        Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--",
               label=r"$\|u-u^*\|_{[t,T]}\cdot(T-t)$ shape (RHS, rescaled)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\sup_{x_s,\,s\in[t,T]}\,\mathbb{E}_B[\|\sigma(t)\nabla r(X_T^{u,x_s})"
        r" - \sigma(t)\nabla r(X_T^{u^*,x_s})\|]$",
        fontsize=9,
    )
    ax.set_title(
        r"Tiled same-BM LHS: $\sup_{s \in [t,T], x_s}$ of pointwise SBM metric"
        "\n(orange = LHS, grey dashed = bound shape $(T-t)\cdot\|u-u^*\|_{[t,T]}$, rescaled)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # Thin legend for iteration colours
    handles, labels = [], []
    for i, snap in enumerate(snaps):
        handles.append(Line2D([0], [0], color=colours[i], linewidth=1.8))
        labels.append(f"iter {snap['outer_it']}")
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]; labels = [labels[i] for i in idxs]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles, labels, fontsize=7, loc="upper right", title="training iter")

    fig.tight_layout()
    path = output_dir / "tiled_same_bm_lhs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_same_vs_diff_bm_lhs(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Side-by-side comparison of same-BM vs different-BM tiled LHS curves.

    Left panel:  mean_x LHS(t,x) curves — same BM (solid) vs diff BM (dashed).
    Right panel: sup_{s>=t, x} LHS — same BM (solid) vs diff BM (dashed).

    If the right-to-left contraction is driven by the shared BM coupling, the
    same-BM curves should decay as t→T while the diff-BM curves stay flat or
    grow, revealing whether the coupling is essential.

    Saves to output_dir/same_vs_diff_bm_lhs.png.
    """
    snaps_sbm = [s for s in snapshots if s.get("same_bm_lhs_fields") is not None]
    snaps_dbm = [s for s in snapshots if s.get("diff_bm_lhs_fields") is not None]
    if not snaps_sbm and not snaps_dbm:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = max(len(snaps_sbm), len(snaps_dbm))
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    # ---- Left: pointwise mean over x ----
    ax = axes[0]
    for i, snap in enumerate(snaps_sbm):
        means = [float(np.mean(f)) for f in snap["same_bm_lhs_fields"]]
        ax.plot(ts_arr, means, color=colours[i], linewidth=1.6,
                label=f"SBM iter {snap['outer_it']}")
    for i, snap in enumerate(snaps_dbm):
        means = [float(np.mean(f)) for f in snap["diff_bm_lhs_fields"]]
        ax.plot(ts_arr, means, color=colours[i], linewidth=1.6, linestyle="--",
                alpha=0.75, label=f"DBM iter {snap['outer_it']}")
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{mean}_x\;\mathrm{LHS}(t,x)$", fontsize=11)
    ax.set_title(
        r"Pointwise mean: same BM (solid) vs diff BM (dashed)"
        "\n" + r"$\frac{1}{n}\sum\|\sigma(t)\nabla r(X_T^u) - \sigma(t)\nabla r(X_T^{u^*})\|$",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # ---- Right: tiled sup over [t,T] x x-grid ----
    ax = axes[1]
    for i, snap in enumerate(snaps_sbm):
        tiled = snap.get("tiled_same_bm_lhs")
        if tiled is not None:
            ax.plot(ts_arr, tiled, color=colours[i], linewidth=1.6,
                    label=f"SBM iter {snap['outer_it']}")
    for i, snap in enumerate(snaps_dbm):
        tiled = snap.get("tiled_diff_bm_lhs")
        if tiled is not None:
            ax.plot(ts_arr, tiled, color=colours[i], linewidth=1.6, linestyle="--",
                    alpha=0.75, label=f"DBM iter {snap['outer_it']}")
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\sup_{s\in[t,T],x}\;\mathrm{LHS}(s,x)$", fontsize=11)
    ax.set_title(
        r"Tiled sup: same BM (solid) vs diff BM (dashed)"
        "\n" + r"$\sup_{x_s,s\in[t,T]}\mathbb{E}[\|\sigma(t)\nabla r(X_T^u) - \sigma(t)\nabla r(X_T^{u^*})\|]$",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # Shared legend: just show line style meaning, not every iteration
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1.6,
               label="Same BM (shared noise, bound holds)"),
        Line2D([0], [0], color="steelblue", linewidth=1.6, linestyle="--", alpha=0.75,
               label="Diff BM (independent noise, no bound)"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Same-BM vs different-BM LHS comparison\n"
        "(if same-BM decays to 0 at $t\\to T$ while diff-BM does not, "
        "the shared coupling is essential for the contraction)",
        fontsize=11,
    )

    path = output_dir / "same_vs_diff_bm_lhs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sbm_ratio(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Side-by-side ratio plot: LHS_SBM / ‖u_θ − u*‖ as a function of t.

    Left panel:  mean_x [ LHS_SBM(t,x) / ‖u_θ(t,x) − u*(t,x)‖ ]  — pointwise, mean over x.
    Right panel: tiled_sbm(t) / tiled_‖u_θ−u*‖(t)                  — sup over [t,T] × x-grid.

    Both the same-BM (solid) and diff-BM (dashed) variants are shown.
    If the contraction bound holds, the same-BM ratio should approach 0 as t→T.

    Saves to output_dir/sbm_ratio.png.
    """
    snaps_sbm = [s for s in snapshots if s.get("sbm_ratio_fields") is not None]
    snaps_dbm = [s for s in snapshots if s.get("dbm_ratio_fields") is not None]
    if not snaps_sbm and not snaps_dbm:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = max(len(snaps_sbm), len(snaps_dbm))
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    # ---- Left: pointwise mean_x ratio ----
    ax = axes[0]
    for i, snap in enumerate(snaps_sbm):
        ratios = snap["sbm_ratio_fields"]       # [K+1][n_sbm_grid]
        means = [float(np.mean(r)) for r in ratios]
        ax.plot(ts_arr, means, color=colours[i], linewidth=1.6,
                label=f"SBM iter {snap['outer_it']}")
    for i, snap in enumerate(snaps_dbm):
        ratios = snap.get("dbm_ratio_fields")
        if ratios is not None:
            means = [float(np.mean(r)) for r in ratios]
            ax.plot(ts_arr, means, color=colours[i], linewidth=1.6, linestyle="--",
                    alpha=0.75, label=f"DBM iter {snap['outer_it']}")
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\mathrm{mean}_x\;\frac{\mathrm{LHS}_{\mathrm{SBM}}(t,x)}{\|u_\theta(t,x)-u^*(t,x)\|}$",
        fontsize=10,
    )
    ax.set_title(
        r"Pointwise ratio: $\mathrm{LHS}_{\mathrm{SBM}}(t,x)\;/\;\|u_\theta-u^*\|(t,x)$"
        "\n(should → 0 as $t \\to T$ for same BM)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # ---- Right: tiled ratio ----
    ax = axes[1]
    for i, snap in enumerate(snaps_sbm):
        tiled_ratio = snap.get("tiled_sbm_ratio")
        if tiled_ratio is not None:
            ax.plot(ts_arr, tiled_ratio, color=colours[i], linewidth=1.6,
                    label=f"SBM iter {snap['outer_it']}")
    for i, snap in enumerate(snaps_dbm):
        tiled_ratio = snap.get("tiled_dbm_ratio")
        if tiled_ratio is not None:
            ax.plot(ts_arr, tiled_ratio, color=colours[i], linewidth=1.6, linestyle="--",
                    alpha=0.75, label=f"DBM iter {snap['outer_it']}")
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\frac{\sup_{s\in[t,T],x}\,\mathbb{E}[\|\sigma\nabla r(X_T^{u,x})-\sigma\nabla r(X_T^{u^*,x})\|]}{\|u_\theta-u^*\|_{[t,T]}}$",
        fontsize=9,
    )
    ax.set_title(
        r"Tiled ratio: $\sup_{s\in[t,T],x}\mathrm{LHS}_{\mathrm{SBM}}\;/\;\|u_\theta-u^*\|_{[t,T]}$"
        "\n(should → 0 as $t \\to T$ for same BM if bound holds)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # Shared legend: line style meaning
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1.6,
               label="Same BM (shared noise, bound holds)"),
        Line2D([0], [0], color="steelblue", linewidth=1.6, linestyle="--", alpha=0.75,
               label="Diff BM (independent noise, no bound)"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        r"Contraction ratio: $\mathrm{LHS}_{\mathrm{SBM}}\;/\;\|u_\theta-u^*\|$"
        "\n(ratio → 0 as $t→T$ confirms the right-to-left bound)",
        fontsize=11,
    )

    path = output_dir / "sbm_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_snapshots(data: dict, output_dir: str | Path) -> None:
    path = Path(output_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


def load_and_plot(metrics_json: str | Path, output_dir: str | Path | None = None) -> None:
    """Re-generate all plots from a saved metrics.json."""
    metrics_json = Path(metrics_json)
    if output_dir is None:
        output_dir = metrics_json.parent
    with open(metrics_json) as f:
        data = json.load(f)
    snapshots = data["snapshots"]
    ts = data["ts"]
    xs = data.get("xs", [])
    xs_sbm = data.get("xs_sbm", [])
    u_star_field = data.get("u_star_field", [])
    u_theta_field = data.get("u_theta_field", None)
    paths_star = data.get("paths_star", None)
    paths_theta = data.get("paths_theta", None)
    d = data.get("d", 1)

    tp = data.get("target_params", {})
    w1 = float(tp.get("w1", 0.5)); w2 = float(tp.get("w2", 0.5))
    lambda1 = float(tp.get("lambda1", 1.0)); lambda2 = float(tp.get("lambda2", 1.0))
    mu1 = float(tp.get("mu1", -1.0)); mu2 = float(tp.get("mu2", 1.0))

    def _make_target_pdf():
        a1 = w1 / math.sqrt(lambda1); a2 = w2 / math.sqrt(lambda2)
        norm = a1 + a2; alpha1 = a1 / norm; alpha2 = a2 / norm
        v1s = 1.0 / lambda1; v2s = 1.0 / lambda2
        def pdf(x):
            x = np.asarray(x)
            p1 = alpha1 * np.exp(-0.5 * (x - mu1) ** 2 / v1s) / math.sqrt(2 * math.pi * v1s)
            p2 = alpha2 * np.exp(-0.5 * (x - mu2) ** 2 / v2s) / math.sqrt(2 * math.pi * v2s)
            return p1 + p2
        return pdf

    target_pdf_fn = _make_target_pdf()

    plot_convergence(snapshots, ts, output_dir)
    plot_inner_steps(snapshots, output_dir)
    plot_inner_convergence(snapshots, output_dir)
    if xs and u_star_field:
        plot_optimal_control(ts, xs, u_star_field, d, output_dir,
                             u_theta_field, paths_star, paths_theta)
        plot_control_evolution(snapshots, ts, xs, d, u_star_field, output_dir)
    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d, output_dir)
    plot_terminal_evolution(snapshots, target_pdf_fn, d, output_dir)
    if xs:
        plot_heatmaps(snapshots, ts, xs, output_dir)
        plot_contraction_heatmaps(snapshots, ts, xs, output_dir)
    if xs_sbm:
        plot_same_bm_lhs_curves(snapshots, ts, output_dir)
        plot_same_bm_lhs_heatmaps(snapshots, ts, xs_sbm, output_dir)
        plot_tiled_same_bm_lhs(snapshots, ts, output_dir)
        plot_same_vs_diff_bm_lhs(snapshots, ts, output_dir)
        plot_sbm_ratio(snapshots, ts, output_dir)

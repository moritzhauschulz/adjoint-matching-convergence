"""Convergence plots for the right-to-left convergence experiment.

Produces two figures from metric snapshots collected during training:
  1. RelL₂(t) vs time slice — multiple curves, one per eval checkpoint
  2. AbsL∞(t) vs time slice — same structure

Both plots share the same colour gradient (light = early training, dark = late),
making it easy to see right-to-left convergence develop over training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def plot_convergence(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Save convergence plots to output_dir/convergence_rel_l2.png and _abs_linf.png.

    Args:
        snapshots: list of dicts, one per eval checkpoint. Each dict has keys:
            "outer_it"   — training iteration
            "rel_l2"     — list of RelL₂ values, length len(ts)
            "abs_linf"   — list of AbsL∞ values, length len(ts)
        ts: list of time-slice values t_k = k/K
        output_dir: directory to write PNG files into
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_arr = np.array(ts)
    n = len(snapshots)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    for metric_key, ylabel, fname in [
        ("rel_l2",         r"$\mathrm{RelL}_2(t)$",                    "convergence_rel_l2.png"),
        ("abs_l2",         r"$\mathrm{AbsL}_2(t)$",                    "convergence_abs_l2.png"),
        ("abs_linf",       r"$\mathrm{AbsL}_\infty(t)$",               "convergence_abs_linf.png"),
        ("tiled_al_inf",   r"$\|u_\theta-u^*\|_{[t,T]}$",             "convergence_tiled_al_inf.png"),
        ("contr_fact",     r"$\mathrm{ContrFact}(t;\,n)$",             "convergence_contr_fact.png"),
        ("tiled_contr_fact", r"$\mathrm{TiledContrFact}(t;\,n)$",     "convergence_tiled_contr_fact.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))

        for i, snap in enumerate(snapshots):
            values = snap[metric_key]
            label = f"iter {snap['outer_it']}"
            valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            if not valid:
                continue
            ax.plot(ts_arr, values, color=colours[i], label=label, linewidth=1.5)

        ax.set_xlabel("time $t$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            "Right-to-Left Convergence\n"
            f"(light = early training, dark = late)",
            fontsize=11,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

        # Compact legend: only show first, middle, last checkpoint
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 6:
            idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
            if idxs[-1] != len(handles) - 1:
                idxs.append(len(handles) - 1)
            handles = [handles[i] for i in idxs]
            labels  = [labels[i]  for i in idxs]
        ax.legend(handles, labels, fontsize=8, loc="upper right")

        if metric_key in ("contr_fact", "tiled_contr_fact"):
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="ratio = 1")

        fig.tight_layout()
        path = output_dir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_optimal_control(
    ts: list[float],
    xs: list[float],
    u_star_field: list[list[float]],
    d: int,
    output_dir: str | Path,
) -> None:
    """Save heatmap of the ground-truth optimal control u*(t,x) to output_dir/heatmap_u_star.png.

    For d=1: signed value with diverging colormap (RdBu_r), centred at zero.
    For d>1: ‖u*(t,x)‖ with viridis.
    x-axis: time t, y-axis: x, colour: u*(t,x).
    """
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs)
    Z = np.array(u_star_field)   # [K+1, n_grid]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    if d == 1:
        abs_max = np.abs(Z).max()
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="RdBu_r",
                           vmin=-abs_max, vmax=abs_max, shading="auto")
        label = r"$u^*(t,x)$"
    else:
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis", shading="auto")
        label = r"$\|u^*(t,x)\|$"

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$x$", fontsize=12)
    ax.set_title(r"Ground-truth optimal control $u^*(t,x)$", fontsize=11)
    fig.colorbar(im, ax=ax, label=label)

    path = output_dir / "heatmap_u_star.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _error_heatmap(
    snapshots: list[dict],
    field_key: str,
    ts_arr: np.ndarray,
    xs_arr: np.ndarray,
    title: str,
    colorbar_label: str,
    path: Path,
    max_subplots: int = 20,
) -> None:
    """Render a grid of (t, x) heatmaps for one error field key."""
    snaps = [s for s in snapshots if field_key in s]
    if not snaps:
        return

    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]

    n = len(snaps)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    vmin = 0.0
    vmax = max(max(max(row) for row in s[field_key]) for s in snaps)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        Z = np.array(snap[field_key])   # [K+1, n_grid]
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(title, fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label=colorbar_label)

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_heatmaps(
    snapshots: list[dict],
    ts: list[float],
    xs: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    """Save error heatmap grids to output_dir.

    Produces two files:
    - heatmap_error_field.png: pointwise ‖u_θ(t,x) − u*(t,x)‖ per time slice
    - heatmap_tiled_error_field.png: max_{s≥t} ‖u_θ(s,x) − u*(s,x)‖ (tiled sup over [t,T])
    """
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs)

    _error_heatmap(
        snapshots, "error_fields", ts_arr, xs_arr,
        title=r"$\|u_\theta(t,x) - u^*(t,x)\|$ over $(t,x)$ grid",
        colorbar_label="pointwise error",
        path=output_dir / "heatmap_error_field.png",
        max_subplots=max_subplots,
    )
    _error_heatmap(
        snapshots, "tiled_error_fields", ts_arr, xs_arr,
        title=r"$\max_{s \geq t}\|u_\theta(s,x) - u^*(s,x)\|$ tiled over $(t,x)$ grid",
        colorbar_label="tiled error",
        path=output_dir / "heatmap_tiled_error_field.png",
        max_subplots=max_subplots,
    )


def plot_contraction_heatmaps(
    snapshots: list[dict],
    ts: list[float],
    xs: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
    eps: float = 1e-12,
) -> None:
    """Save pointwise contraction ratio heatmap to output_dir/heatmap_contr_ratio.png.

    For each consecutive pair of snapshots (n, n+1), plots:

        log10( ‖u^{n+1}_θ(t,x) − u*(t,x)‖ / ‖u^n_θ(t,x) − u*(t,x)‖ )

    over the (t, x) grid. Blue = improving (ratio < 1), red = regressing (ratio > 1).
    Shared symmetric colour limits across all subplots.
    """
    output_dir = Path(output_dir)

    snaps = [s for s in snapshots if "error_fields" in s]
    if len(snaps) < 2:
        return

    pairs = list(zip(snaps[:-1], snaps[1:]))
    if len(pairs) > max_subplots:
        idxs = np.linspace(0, len(pairs) - 1, max_subplots, dtype=int)
        pairs = [pairs[i] for i in idxs]

    n = len(pairs)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    ts_arr = np.array(ts)
    xs_arr = np.array(xs)

    # Compute all ratio fields; clip at [0, 5]
    ratio_fields = []
    for s_prev, s_next in pairs:
        denom = np.array(s_prev["error_fields"]).clip(min=eps)   # [K+1, n_grid]
        numer = np.array(s_next["error_fields"])                  # [K+1, n_grid]
        ratio = (numer / denom).clip(0.0, 5.0)
        ratio_fields.append(ratio)

    # Diverging norm centred at 1 over [0, 5]: blue < 1 (improving), red > 1 (regressing)
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=5.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)

    for i, ((s_prev, s_next), log_ratio) in enumerate(zip(pairs, ratio_fields)):
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(ts_arr, xs_arr, ratio.T, cmap="RdBu_r",
                           norm=norm, shading="auto")
        ax.set_title(f"iter {s_prev['outer_it']}→{s_next['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"Contraction ratio $\|u^{n+1}_\theta - u^*\| / \|u^n_\theta - u^*\|$"
        "\n(blue < 1 = improving, red > 1 = regressing)",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                 label="ratio")

    path = output_dir / "heatmap_contr_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_snapshots(snapshots: list[dict], output_dir: str | Path) -> None:
    """Write snapshots list to metrics.json for later re-plotting."""
    path = Path(output_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(snapshots, f, indent=2)
    print(f"  Saved {path}")


def load_and_plot(metrics_json: str | Path, output_dir: str | Path | None = None) -> None:
    """Re-generate plots from a previously saved metrics.json.

    Usage:
        from experiments.right_to_left_convergence.plotting import load_and_plot
        load_and_plot("results/.../metrics.json")
    """
    metrics_json = Path(metrics_json)
    if output_dir is None:
        output_dir = metrics_json.parent
    with open(metrics_json) as f:
        data = json.load(f)
    snapshots = data["snapshots"]
    ts = data["ts"]
    xs = data.get("xs", [])
    u_star_field = data.get("u_star_field", [])
    d = data.get("d", 1)
    plot_convergence(snapshots, ts, output_dir)
    if xs and u_star_field:
        plot_optimal_control(ts, xs, u_star_field, d, output_dir)
    if xs:
        plot_heatmaps(snapshots, ts, xs, output_dir)
        plot_contraction_heatmaps(snapshots, ts, xs, output_dir)

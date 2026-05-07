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
        ("rel_l2",   r"$\mathrm{RelL}_2(t)$",      "convergence_rel_l2.png"),
        ("abs_l2",   r"$\mathrm{AbsL}_2(t)$",      "convergence_abs_l2.png"),
        ("abs_linf", r"$\mathrm{AbsL}_\infty(t)$", "convergence_abs_linf.png"),
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

        fig.tight_layout()
        path = output_dir / fname
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
    plot_convergence(snapshots, ts, output_dir)

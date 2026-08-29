"""Plots for the bimodal same-BM experiment.

Extends the bimodal experiment plots with two new figures for the same-BM LHS metric:
  - same_bm_lhs_curves.png:    mean LHS(t) vs t, one curve per eval checkpoint
  - same_bm_lhs_heatmaps.png:  LHS(t, x) heatmaps over training

All other plots are identical to right_to_left_convergence_bimodal.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Log-scale helper
# ---------------------------------------------------------------------------

def _tighten_log_ylim(ax, pct: float = 2.0) -> None:
    """Stop a log y-axis from blowing up when curves touch 0 (e.g. operator
    error / LHS_SBM → 0 exactly at t=T).  Sets the lower limit to the `pct`-th
    percentile of the strictly-positive plotted y-values; leaves the top auto.
    """
    vals: list[float] = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        vals.extend(y[np.isfinite(y) & (y > 0.0)].tolist())
    if not vals:
        return
    bottom = float(np.percentile(np.array(vals), pct))
    if bottom > 0:
        ax.set_ylim(bottom=bottom)   # top left to autoscale (keeps legend headroom)


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
        log.info("Saved %s", path)


def plot_optimal_control(
    ts, xs, u_star_field, d, output_dir,
    u_theta_field=None, paths_star=None, paths_theta=None,
) -> None:
    # paths_* are accepted for call-site compatibility but no longer drawn.
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
        axes[1].pcolormesh(ts_arr, xs_arr, Z_theta.T, **kwargs)
        axes[1].set_title(r"Learned $u_\theta(t,x)$ (final)", fontsize=11)
        axes[1].set_xlabel("$t$"); axes[1].set_ylabel("$x$")
        fig.colorbar(im, ax=axes.tolist(), label=cb_label)
    else:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        abs_max = np.abs(Z_star).max()
        im = ax.pcolormesh(ts_arr, xs_arr, Z_star.T, cmap="RdBu_r",
                           vmin=-abs_max, vmax=abs_max, shading="auto")
        ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
        ax.set_title(r"Ground-truth optimal control $u^*(t,x)$", fontsize=11)
        fig.colorbar(im, ax=ax, label=r"$u^*(t,x)$")
    path = output_dir / "heatmap_u_star.png"
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(r"Learned control $u_\theta(t,x)$ over training", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8, label=cb_label)
    path = output_dir / "heatmap_control_evolution.png"
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


def _overlay_source_traj(ax, ts_arr, paths, n_show=40):
    """Overlay a subsample of source-control trajectories (thin, translucent) on a
    (t,x) heatmap.  `paths` is [n_sp, K+1] (d=1) or [n_sp, K+1, d] (component 0 used).
    """
    a = np.asarray(paths, dtype=float)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape[0] > n_show:
        idx = np.linspace(0, a.shape[0] - 1, n_show, dtype=int)
        a = a[idx]
    for row in a:
        ax.plot(ts_arr, row, color="black", linewidth=0.35, alpha=0.25)


def plot_operator_vs_next_control(snapshots, ts, xs_op, d, output_dir,
                                  max_subplots: int = 20,
                                  overlay_traj: bool = False,
                                  path_key: str = "paths_theta") -> None:
    r"""Per operator-eval pair (n, n+1), one row with four (t,x) heatmaps:

        col 0:  P(u_θ^n)                  (`T_u_fields` of snapshot n)
        col 1:  u_θ^{n+1}                 (`u_theta_op_fields` of snapshot n+1)
        col 2:  P(u_θ^n) − u_θ^{n+1}      (the outer-step-implements-P residual)
        col 3:  u_θ^{n+1} − u*            (`u_star_op_fields`; distance to the fixed point)

    ONE robust symmetric colour scale for every panel (all cols, all rows), with
    a colourbar on each row — so the residual columns are read on the same scale
    as the controls and relative error sizes are apparent.

    `overlay_traj=True` overlays a subsample of trajectories rolled out under the
    **source** control u_θ^n (snapshot n's `path_key`, default `paths_theta`) on
    every panel of that row — showing where the process the operator sees
    actually visits — and writes to `heatmap_P_vs_next_control_traj.png` instead.
    Only for d=1.

    Saves to output_dir/heatmap_P_vs_next_control{_traj}.png.
    """
    op_snaps = [s for s in snapshots
                if s.get("T_u_fields") is not None
                and s.get("u_theta_op_fields") is not None]
    if len(op_snaps) < 2:
        return
    do_traj = bool(overlay_traj) and d == 1
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)

    pairs = list(zip(op_snaps[:-1], op_snaps[1:]))
    if len(pairs) > max_subplots:
        idxs = np.linspace(0, len(pairs) - 1, max_subplots, dtype=int)
        pairs = [pairs[i] for i in idxs]
    n_pairs = len(pairs)

    P_all = [np.array(sn["T_u_fields"], dtype=float) for sn, _ in pairs]
    U_all = [np.array(sn1["u_theta_op_fields"], dtype=float) for _, sn1 in pairs]
    US_all = [np.array(sn1.get("u_star_op_fields"), dtype=float)
              if sn1.get("u_star_op_fields") is not None else None
              for _, sn1 in pairs]
    signed = (d == 1)
    cmap = "RdBu_r" if signed else "viridis"
    cb_label = r"$u(t,x)$" if signed else r"$\|u(t,x)\|$"

    d_pu_all = [P - U for P, U in zip(P_all, U_all)]
    d_us_all = [U - US if US is not None else None
                for U, US in zip(U_all, US_all)]

    # ONE robust colour limit over every panel (controls AND residuals) so the
    # residual columns are read on the same scale as the controls
    all_vals = ([P.ravel() for P in P_all]
                + [U.ravel() for U in U_all]
                + [z.ravel() for z in d_pu_all]
                + [z.ravel() for z in d_us_all if z is not None])
    lim = float(np.percentile(np.abs(np.concatenate(all_vals)), 99)) or 1.0
    kw = (dict(cmap=cmap, vmin=-lim, vmax=lim) if signed
          else dict(cmap=cmap, vmin=0.0, vmax=lim))

    def _draw(ax, Z):
        return ax.pcolormesh(ts_arr, xs_arr, Z.T, shading="auto", **kw)

    fig, axes = plt.subplots(n_pairs, 4, figsize=(19, 2.7 * n_pairs),
                             squeeze=False, constrained_layout=True)

    for r, ((sn, sn1), P, U, US, d_pu, d_us) in enumerate(
            zip(pairs, P_all, U_all, US_all, d_pu_all, d_us_all)):
        im0 = _draw(axes[r][0], P)
        _draw(axes[r][1], U)
        _draw(axes[r][2], d_pu)
        axes[r][0].set_title(rf"$P(u_\theta^{{{sn['outer_it']}}})$", fontsize=10)
        axes[r][1].set_title(rf"$u_\theta^{{{sn1['outer_it']}}}$", fontsize=10)
        axes[r][2].set_title(rf"$P(u_\theta^{{{sn['outer_it']}}}) - u_\theta^{{{sn1['outer_it']}}}$",
                             fontsize=10)
        if d_us is not None:
            _draw(axes[r][3], d_us)
            axes[r][3].set_title(rf"$u_\theta^{{{sn1['outer_it']}}} - u^*$", fontsize=10)
        else:
            axes[r][3].set_visible(False)
        if do_traj and sn.get(path_key) is not None:
            for j in range(4):
                if axes[r][j].get_visible():
                    _overlay_source_traj(axes[r][j], ts_arr, sn[path_key])
            axes[r][0].set_ylabel(rf"$x$   (paths $\sim u_\theta^{{{sn['outer_it']}}}$)", fontsize=8)
        for j in range(4):
            if axes[r][j].get_visible():
                axes[r][j].set_xlabel("$t$", fontsize=8)
                if not (do_traj and j == 0):
                    axes[r][j].set_ylabel("$x$", fontsize=8)
                axes[r][j].tick_params(labelsize=7)
                axes[r][j].set_ylim(xs_arr.min(), xs_arr.max())
        fig.colorbar(im0, ax=list(axes[r]), location="right", shrink=0.9,
                     label=cb_label, pad=0.02)

    _traj_note = (r" — black lines: trajectories under the source control $u_\theta^n$"
                  if do_traj else "")
    fig.suptitle(r"$P(u_\theta^n)$ vs the next learned control $u_\theta^{n+1}$, "
                 r"with residuals $P(u_\theta^n)-u_\theta^{n+1}$ and $u_\theta^{n+1}-u^*$"
                 "\n(one shared colour scale across all panels" + _traj_note + ")", fontsize=12)
    path = output_dir / ("heatmap_P_vs_next_control_traj.png" if do_traj
                         else "heatmap_P_vs_next_control.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


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
    fig.savefig(path, dpi=150); plt.close(fig); log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sanity-check plots (§"Sanity Check": ‖P(u_θ) − u*‖ with analytic u*, folded
# into the operator-error data; + a both-rollout shared-BM comparison)
# ---------------------------------------------------------------------------

# Robust "sup over the [t,T] suffix" — p-th percentile of the suffix instead of
# the exact max.  Set by run.py / load_and_plot from `eval.tiled_sup_percentile`.
_SUFFIX_PCT: float = 100.0


def set_suffix_percentile(pct) -> None:
    global _SUFFIX_PCT
    _SUFFIX_PCT = float(pct) if pct is not None else 100.0


def _suffix_max(a: np.ndarray) -> np.ndarray:
    """[..., K+1] -> [..., K+1] robust suffix-sup along the last axis:
    exact suffix-maximum when _SUFFIX_PCT >= 100, else the _SUFFIX_PCT-th
    percentile of each suffix a[..., k:].
    """
    a = np.asarray(a, dtype=float)
    if _SUFFIX_PCT >= 100.0:
        return np.maximum.accumulate(a[..., ::-1], axis=-1)[..., ::-1]
    K = a.shape[-1]
    return np.stack([np.percentile(a[..., k:], _SUFFIX_PCT, axis=-1) for k in range(K)],
                    axis=-1)


def plot_same_bm_lhs_curves(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """mean_x ‖P(u_θ^n)(t,x) − u*(t,x)‖ vs t, one curve per operator-eval snapshot.

    P(u_θ) is MC-estimated (rollout of u_θ only); u* is analytic (identity of the
    operator).  Should decay right-to-left; exactly 0 at t=T.
    Saves to output_dir/same_bm_lhs_curves.png.
    """
    snaps = [s for s in snapshots if s.get("op_error_fields") is not None]
    if not snaps:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Oranges(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        means = [float(np.mean(f)) for f in snap["op_error_fields"]]
        ax.plot(ts_arr, means, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{mean}_x\;\|P(u_\theta^n)(t,x) - u^*(t,x)\|$", fontsize=11)
    ax.set_title(
        r"Sanity check: $\|P(u_\theta^n)(t,x) - u^*(t,x)\|$ (analytic $u^*$)"
        "\n(light = early training, dark = late — should decay right-to-left)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "same_bm_lhs_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_same_bm_lhs_heatmaps(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    """Heatmaps of ‖P(u_θ^n)(t,x) − u*(t,x)‖ over (t, x), one per eval checkpoint.

    Saves to output_dir/same_bm_lhs_heatmaps.png.
    """
    snaps = [s for s in snapshots if s.get("op_error_fields") is not None]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)
    n = len(snaps)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    vmax = max(max(max(row) for row in s["op_error_fields"]) for s in snaps)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        Z = np.array(snap["op_error_fields"])
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis",
                           vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"Sanity check $\|P(u_\theta^n)(t,x) - u^*(t,x)\|$ over $(t,x)$"
        "\n(should → 0 as $t \\to T$, right-to-left)",
        fontsize=10,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label=r"$\|P(u_\theta^n) - u^*\|$")
    path = output_dir / "same_bm_lhs_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_tiled_same_bm_lhs(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Tiled sanity-check error  sup_{s>=t, x} ‖P(u_θ^n)(s,x) − u*(s,x)‖.

    Left  — vs t (linear y).  solid = analytic-u* tiled error (tiled_op_sup_error);
            dashed = both-rollout shared-BM tiled estimate (comparison, if present);
            grey dashed = c·‖u_θ−u*‖_{[t,T]}·√(T−t) (LS-fit amplitude, notes' bound shape).
    Right — vs τ = T−t on log–log axes; a √(T−t) law is a straight line of slope 1/2
            (dotted guide).  solid = tiled sup_{s>=t,x}; dashed = pointwise sup_x.

    Saves to output_dir/tiled_same_bm_lhs.png.
    """
    snaps = [s for s in snapshots if s.get("tiled_op_sup_error") is not None]
    if not snaps:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    T = ts_arr[-1]
    tau = T - ts_arr
    pos = tau > 1e-12
    n = len(snaps)
    colours = cm.Oranges(np.linspace(0.3, 1.0, max(n, 1)))
    colours_rhs = cm.Greys(np.linspace(0.3, 0.8, max(n, 1)))

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for i, snap in enumerate(snaps):
        tiled = np.asarray(snap["tiled_op_sup_error"], dtype=float)          # [K+1]
        u_tiled = _suffix_max(np.asarray(snap["u_sup_error_op"], dtype=float))  # ‖u_θ−u*‖_{[t,T]}
        sup_x = np.asarray(snap["op_sup_error"], dtype=float)               # pointwise sup_x

        axes[0].plot(ts_arr, tiled, color=colours[i], linewidth=1.8,
                     label=f"iter {snap['outer_it']}")

        br = snap.get("tiled_bothroll_lhs")
        if br is not None:
            axes[0].plot(ts_arr, np.asarray(br, dtype=float), color=colours[i],
                         linewidth=1.0, linestyle=":", alpha=0.8)

        shape = u_tiled * np.sqrt(np.maximum(tau, 0.0))
        m = pos & np.isfinite(shape) & np.isfinite(tiled) & (tiled > 0) & (shape > 0)
        if m.sum() >= 2:
            c = float(np.sum(shape[m] * tiled[m]) / np.sum(shape[m] ** 2))
            axes[0].plot(ts_arr[pos], (c * shape)[pos], color=colours_rhs[i],
                         linewidth=1.0, linestyle="--", alpha=0.7)

        axes[1].plot(tau[pos], tiled[pos], color=colours[i], linewidth=1.8)
        axes[1].plot(tau[pos], sup_x[pos], color=colours[i], linewidth=1.0,
                     linestyle="--", alpha=0.7)

    last = np.asarray(snaps[-1]["tiled_op_sup_error"], dtype=float)
    ml = pos & np.isfinite(last) & (last > 0)
    if ml.sum() >= 2:
        tau_m = tau[ml]; last_m = last[ml]
        tau_ref = float(np.exp(np.mean(np.log(tau_m))))
        y_ref = float(np.interp(tau_ref, tau_m[::-1], last_m[::-1]))
        a = y_ref / math.sqrt(tau_ref)
        tau_line = np.array([tau_m.min(), tau_m.max()])
        axes[1].plot(tau_line, a * np.sqrt(tau_line), color="black",
                     linewidth=1.3, linestyle=":",
                     label=r"slope $1/2$  ($\propto\sqrt{T-t}$)")

    axes[0].set_xlabel("time $t$", fontsize=12)
    axes[0].set_ylabel(r"$\sup_{s\geq t,\,x}\,\|P(u_\theta^n)(s,x) - u^*(s,x)\|$", fontsize=9)
    axes[0].set_title(
        r"Tiled sanity-check error vs $t$"
        "\n" r"solid = analytic $u^*$;  dotted = both-rollout;  grey = $c\,\|u_\theta-u^*\|_{[t,T]}\sqrt{T-t}$",
        fontsize=9)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r"$\tau = T - t$", fontsize=12)
    axes[1].set_ylabel(r"tiled (solid) / pointwise $\sup_x$ (dashed)", fontsize=9)
    axes[1].set_title(
        r"log–log vs $\tau = T-t$"
        "\n" r"a $\sqrt{T-t}$ law $\Leftrightarrow$ straight line of slope $1/2$",
        fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    _tighten_log_ylim(axes[1])
    axes[1].grid(True, alpha=0.3, which="both")
    _lg = [
        Line2D([0], [0], color="0.4", linewidth=1.8, label=r"tiled $\sup_{s\geq t,x}$"),
        Line2D([0], [0], color="0.4", linewidth=1.0, linestyle="--",
               label=r"pointwise $\sup_x$ (decays as $t\to T$)"),
        Line2D([0], [0], color="black", linewidth=1.3, linestyle=":",
               label=r"slope $1/2$  ($\propto\sqrt{T-t}$)"),
    ]
    axes[1].legend(handles=_lg, fontsize=8, loc="lower right")

    handles = [Line2D([0], [0], color=colours[i], linewidth=1.8) for i in range(n)]
    labels = [f"iter {snap['outer_it']}" for snap in snaps]
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]; labels = [labels[i] for i in idxs]
    axes[0].legend(handles, labels, fontsize=7, loc="lower left",
                   title="training iter", ncol=2)

    fig.suptitle(r"Tiled sanity-check error — is the tail shape $\propto\sqrt{T-t}$?",
                 fontsize=12)
    path = output_dir / "tiled_same_bm_lhs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_sbm_ratio(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Sanity-check ratios  ‖E[σ(t)∇g(X_T^{u_n,x})] − u*(x,t)‖ / ‖u_n − u*‖_{[t,T]}.

    Numerator = ‖P(u_n) − u*‖ (analytic-u* substitution, = op_error_fields);
    denominator = ‖u_n − u*‖_{[t,T]} only (no ‖σ‖_{[t,T]} factor).  Both → 0 as t→T.

    Left  — mean_x of the pointwise ratio (sanity_ratio_pointwise).
    Right — the tiled ratio (sanity_ratio_tiled).

    Saves to output_dir/sbm_ratio.png.
    """
    snaps = [s for s in snapshots if s.get("sanity_ratio_tiled") is not None]
    if not snaps:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    for i, snap in enumerate(snaps):
        pw = snap.get("sanity_ratio_pointwise")
        if pw is not None:
            axes[0].plot(ts_arr, [float(np.mean(r)) for r in pw],
                         color=colours[i], linewidth=1.6, label=f"iter {snap['outer_it']}")
        axes[1].plot(ts_arr, np.asarray(snap["sanity_ratio_tiled"], dtype=float),
                     color=colours[i], linewidth=1.6, label=f"iter {snap['outer_it']}")

    axes[0].set_xlabel("time $t$", fontsize=12)
    axes[0].set_ylabel(
        r"$\mathrm{mean}_x\;\frac{\|\mathbb{E}[\sigma(t)\nabla g(X_T^{u_n,x_t})]-u^*(x,t)\|}{\|u_n-u^*\|_{[t,T]}}$",
        fontsize=9)
    axes[0].set_title(r"Pointwise ratio (mean over $x$) — should $\to 0$ as $t\to T$", fontsize=10)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("time $t$", fontsize=12)
    axes[1].set_ylabel(
        r"$\frac{\sup_{x_s\in\mathbb{R},\,s\in[t,T]}\|\mathbb{E}[\sigma(s)\nabla g(X_T^{u_n,x_s})]-u^*(x,t)\|}{\|u_n-u^*\|_{[t,T]}}$",
        fontsize=9)
    axes[1].set_title(r"Tiled ratio — should $\to 0$ as $t\to T$", fontsize=10)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]; labels = [labels[i] for i in idxs]
    fig.legend(handles, labels, fontsize=8, loc="lower center",
               ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        r"Sanity-check ratio: $\|\mathbb{E}[\sigma\nabla g(X_T^{u_n})]-u^*\|\;/\;\|u_n-u^*\|_{[t,T]}$  ($\to 0$ as $t\to T$)",
        fontsize=11)
    path = output_dir / "sbm_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_sbm_ratio_learned(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""Exact analogue of ``plot_sbm_ratio`` with the learned control in the
    numerator instead of the operator image P(u_n):

        ‖u_θ^{n+1}(t,x) − u*(t,x)‖ / ‖u_θ^n − u*‖_{[t,T]}

    i.e. a per-step contraction factor for the learned outer update
    (u_θ^n → u_θ^{n+1}), with NO ‖σ‖_{[t,T]} factor in the denominator — the
    same normalisation as ``sbm_ratio``.  The denominator's tiled sup is taken
    from the PREVIOUS eval snapshot.  One curve per consecutive pair n→n+1.

    Left  — mean_x of the pointwise ratio (numerator = u_error_op_fields).
    Right — the tiled ratio  ‖u_θ^{n+1}−u*‖_{[t,T]} / ‖u_θ^n−u*‖_{[t,T]}.
    A ratio < 1 means the step contracts the error at that t; → 0 as t → T if
    the right-to-left contraction bound governs the learned update.

    Saves to output_dir/sbm_ratio_learned.png.
    """
    snaps = [s for s in snapshots
             if s.get("u_error_op_fields") is not None
             and s.get("u_sup_error_op") is not None]
    if len(snaps) < 2:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    pairs = list(zip(snaps[:-1], snaps[1:]))
    n = len(pairs)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    _EPS = 1e-12
    for i, (sn, sn1) in enumerate(pairs):
        denom = np.maximum(_suffix_max(np.asarray(sn["u_sup_error_op"], dtype=float)), _EPS)  # ‖u_θ^n−u*‖_{[t,T]}
        pw = np.asarray(sn1["u_error_op_fields"], dtype=float)          # [K+1, n_op]
        numer_sup = _suffix_max(np.asarray(sn1["u_sup_error_op"], dtype=float))

        axes[0].plot(ts_arr, np.mean(pw, axis=1) / denom,
                     color=colours[i], linewidth=1.6,
                     label=f"iter {sn['outer_it']}→{sn1['outer_it']}")
        axes[1].plot(ts_arr, numer_sup / denom,
                     color=colours[i], linewidth=1.6,
                     label=f"iter {sn['outer_it']}→{sn1['outer_it']}")

    axes[0].set_xlabel("time $t$", fontsize=12)
    axes[0].set_ylabel(
        r"$\mathrm{mean}_x\;\frac{\|u_\theta^{n+1}(t,x)-u^*(x,t)\|}{\|u_\theta^n-u^*\|_{[t,T]}}$",
        fontsize=10)
    axes[0].set_title(r"Pointwise ratio (mean over $x$) — should $\to 0$ as $t\to T$", fontsize=10)
    axes[0].axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("time $t$", fontsize=12)
    axes[1].set_ylabel(
        r"$\frac{\|u_\theta^{n+1}-u^*\|_{[t,T]}}{\|u_\theta^n-u^*\|_{[t,T]}}$",
        fontsize=10)
    axes[1].set_title(r"Tiled ratio — should $\to 0$ as $t\to T$ (< 1 = contraction per step)", fontsize=10)
    axes[1].axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]; labels = [labels[i] for i in idxs]
    fig.legend(handles, labels, fontsize=8, loc="lower center",
               ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        r"Learned-update ratio: $\|u_\theta^{n+1}-u^*\|\;/\;\|u_\theta^n-u^*\|_{[t,T]}$  ($\to 0$ as $t\to T$)",
        fontsize=11)
    path = output_dir / "sbm_ratio_learned.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_sanity_ratio_heatmap(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    r"""Heatmap of the pointwise sanity-check ratio over $(t, x)$, one panel per
    operator-eval snapshot:

        ‖P(u_n)(t,x) − u*(t,x)‖ / ‖u_n − u*‖_{[t,T]}     (sanity_ratio_pointwise)

    Numerator pointwise, denominator = tiled control error over [t,T].
    Should → 0 as t → T (right column dark).

    Saves to output_dir/sanity_ratio_heatmap.png.
    """
    snaps = [s for s in snapshots if s.get("sanity_ratio_pointwise") is not None]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)
    n = len(snaps)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    fields = [np.array(s["sanity_ratio_pointwise"], dtype=float) for s in snaps]  # [K+1, n_op]
    vmax = float(np.nanpercentile(np.concatenate([f.ravel() for f in fields]), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, (snap, Z) in enumerate(zip(snaps, fields)):
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="magma",
                           vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"Pointwise sanity-check ratio $\|P(u_n)(t,x)-u^*(t,x)\|\,/\,\|u_n-u^*\|_{[t,T]}$ over $(t,x)$"
        "\n(should → 0 as $t \\to T$; clipped at 98th pctile)",
        fontsize=10,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label="ratio")
    path = output_dir / "sanity_ratio_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_fixed_point_residual(
    fp_data: dict,
    ts: list[float],
    d: int,
    output_dir: str | Path,
) -> None:
    """Heatmap of ‖P(u*)(t,x) − u*(t,x)‖ over the (t, x) evaluation grid.

    Basic sanity check: the analytic u* should be a fixed point of the operator
    P(u)(t,x) = -σ(t) E[∇g(X_T^{u,x})], so this residual should be ≪ 1 everywhere
    — exactly 0 at t=T (boundary condition), MC-noise-limited elsewhere.
    A large residual flags a bug in the analytic u*, in ∇g, or in the operator
    estimator.  Independent of training.

    Saves to output_dir/u_star_fixed_point_residual.png.
    """
    if not fp_data or fp_data.get("residual") is None:
        return
    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(fp_data["xs"])
    R = np.array(fp_data["residual"])                     # [K+1, n_grid]
    n_mc = fp_data.get("n_mc")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    vmax = float(np.percentile(R, 99)) or float(R.max()) or 1.0
    im = axes[0].pcolormesh(ts_arr, xs_arr, R.T, cmap="magma", shading="auto",
                            vmin=0.0, vmax=vmax)
    axes[0].set_xlabel("time $t$", fontsize=12)
    axes[0].set_ylabel("state $x$", fontsize=12)
    axes[0].set_title(r"$\|P(u^*)(t,x) - u^*(t,x)\|$ over the eval grid", fontsize=11)
    fig.colorbar(im, ax=axes[0], label="residual")

    sup_x = R.max(axis=1)
    mean_x = R.mean(axis=1)
    axes[1].plot(ts_arr, sup_x, color="crimson", linewidth=1.6, label=r"$\sup_x$")
    axes[1].plot(ts_arr, mean_x, color="steelblue", linewidth=1.6, label=r"$\mathrm{mean}_x$")
    axes[1].set_xlabel("time $t$", fontsize=12)
    axes[1].set_ylabel(r"$\|P(u^*) - u^*\|$", fontsize=11)
    axes[1].set_title(r"residual vs $t$  (should stay $\ll 1$; $=0$ at $t=T$)", fontsize=10)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    mc_txt = f"  (n_mc = {n_mc})" if n_mc else ""
    fig.suptitle(
        r"Fixed-point sanity check: is the analytic $u^*$ self-consistent?"
        f"   max residual = {R.max():.3g}{mc_txt}",
        fontsize=12,
    )
    path = output_dir / "u_star_fixed_point_residual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Operator T plots
# ---------------------------------------------------------------------------

def plot_operator_error_curves(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Plot ‖P(u_θ^n)(t,x) − u*(t,x)‖_sup and ‖u_θ^n − u*‖_sup vs t.

    Both quantities are evaluated on the xs_op grid so the comparison is fair.
    At t=T the operator error is exactly 0 (boundary condition: P(u)(T,x)=u*(T,x)
    for all u), while the learned-control error need not vanish there.

    Saves to output_dir/operator_error_curves.png.
    """
    snaps = [s for s in snapshots if s.get("op_sup_error") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Oranges(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    # Left: P(u^n) error (solid) and learned-control error (dashed) on xs_op grid
    ax = axes[0]
    for i, snap in enumerate(snaps):
        c = colours[i]
        ax.plot(ts_arr, snap["op_sup_error"], color=c, linewidth=1.5)
        u_sup = snap.get("u_sup_error_op")
        if u_sup is not None:
            ax.plot(ts_arr, u_sup, color=c, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\sup_x\,\|\cdot\|(t)$", fontsize=11)
    ax.set_title(
        r"Operator error $\|P(u_\theta^n)-u^*\|_\infty$ (solid)"
        "\n" + r"vs learned-control error $\|u_\theta^n-u^*\|_\infty$ (dashed) on $x$-grid",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    # Right: tiled sup over [t, T] for both
    ax = axes[1]
    for i, snap in enumerate(snaps):
        c = colours[i]
        tiled_op = snap.get("tiled_op_sup_error")
        if tiled_op is not None:
            ax.plot(ts_arr, tiled_op, color=c, linewidth=1.5)
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\sup_{s\in[t,T],x}\,\|P(u_\theta^n)(s,x) - u^*(s,x)\|$", fontsize=11)
    ax.set_title(
        r"Tiled operator error $\|P(u_\theta^n)-u^*\|_{\infty,[t,T]}$"
        "\n(should decay right-to-left and shrink across iterations)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="sienna", linewidth=1.5,
               label=r"$\|P(u_\theta^n) - u^*\|_\infty$ (operator error)"),
        Line2D([0], [0], color="sienna", linewidth=1.2, linestyle="--", alpha=0.6,
               label=r"$\|u_\theta^n - u^*\|_\infty$ (learned-control error)"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.08))

    sm = plt.cm.ScalarMappable(cmap="Oranges",
                               norm=mcolors.Normalize(vmin=snaps[0]["outer_it"],
                                                      vmax=snaps[-1]["outer_it"]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="outer iteration", fraction=0.02, pad=0.02)

    path = output_dir / "operator_error_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_operator_vs_learned(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Compare ‖u_θ^n − u*‖_sup(t) vs ‖P(u_θ^n) − u*‖_sup(t) on the same xs_op grid.

    If each outer iteration implements one application of P, then after n steps
    ‖u_θ^{n+1} − u*‖ ≈ ‖P(u_θ^n) − u*‖.  Note: at t=T the operator error is
    always exactly 0 (P(u)(T,x) = u*(T,x) for all u by the boundary condition).

    Left: sup_x errors at each t.  Right: ratio P-error / u-error (contraction per step).

    Saves to output_dir/operator_vs_learned.png.
    """
    snaps = [s for s in snapshots if s.get("op_sup_error") is not None
             and s.get("u_sup_error_op") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    for i, snap in enumerate(snaps):
        c = colours[i]
        u_err = np.array(snap["u_sup_error_op"])
        Tu_err = np.array(snap["op_sup_error"])

        ax = axes[0]
        ax.plot(ts_arr, u_err, color=c, linewidth=1.5, linestyle="-")
        ax.plot(ts_arr, Tu_err, color=c, linewidth=1.5, linestyle="--", alpha=0.7)

        ax = axes[1]
        ratio = np.where(u_err > 1e-10, Tu_err / u_err, np.nan)
        ax.plot(ts_arr, ratio, color=c, linewidth=1.5)

    axes[0].set_xlabel("time $t$", fontsize=12)
    axes[0].set_ylabel(r"$\sup_x\,\|\cdot - u^*\|(t)$  [on $x$-grid]", fontsize=11)
    axes[0].set_title(
        r"$\|u_\theta^n - u^*\|_\infty$ (solid) vs $\|P(u_\theta^n) - u^*\|_\infty$ (dashed)"
        "\n(both on xs_op grid; at $t=T$: operator error = 0 by boundary condition)",
        fontsize=10,
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    axes[1].set_xlabel("time $t$", fontsize=12)
    axes[1].set_ylabel(
        r"$\frac{\|P(u_\theta^n)-u^*\|_\infty}{\|u_\theta^n-u^*\|_\infty}$", fontsize=11)
    axes[1].set_title(
        r"Ratio $\|P(u_\theta^n)-u^*\| / \|u_\theta^n-u^*\|$ per time slice"
        "\n(< 1: one P step reduces error; → 0 as $t→T$)",
        fontsize=10,
    )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1.5, linestyle="-",
               label=r"$\|u_\theta^n - u^*\|_\infty$ (learned control error)"),
        Line2D([0], [0], color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7,
               label=r"$\|P(u_\theta^n) - u^*\|_\infty$ (one P application)"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        r"Learned control vs operator output (same $x$-grid)"
        "\n" r"if outer iteration $\approx P$: $\|u_\theta^{n+1}-u^*\|\approx\|P(u_\theta^n)-u^*\|$",
        fontsize=11,
    )

    sm = plt.cm.ScalarMappable(cmap="Blues",
                               norm=mcolors.Normalize(vmin=snaps[0]["outer_it"],
                                                      vmax=snaps[-1]["outer_it"]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="outer iteration", fraction=0.02, pad=0.02)

    path = output_dir / "operator_vs_learned.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_u_vs_Tu_prev(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    """Plot ‖u_θ^n − P(u_θ^{n-1})‖_sup vs t, compared with ‖u_θ^n − u*‖_sup.

    Both quantities are on the xs_op grid so the comparison is fair.
    Small values of ‖u_θ^n − P(u_θ^{n-1})‖ mean the outer iteration faithfully
    implements one application of P.

    Left: sup errors vs t.  Right: ratio ‖u_θ^n − P(u_θ^{n-1})‖ / ‖u_θ^n − u*‖
    (< 1 means P-residual is smaller than the remaining distance to u*).

    Saves to output_dir/u_vs_Tu_prev.png.
    """
    snaps = [s for s in snapshots if s.get("u_vs_Tu_sup") is not None
             and s.get("u_sup_error_op") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Greens(np.linspace(0.3, 1.0, max(n, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    ax = axes[0]
    for i, snap in enumerate(snaps):
        c = colours[i]
        ax.plot(ts_arr, snap["u_sup_error_op"], color=c, linewidth=1.5, linestyle="-")
        ax.plot(ts_arr, snap["u_vs_Tu_sup"], color=c, linewidth=1.5,
                linestyle="--", alpha=0.7)
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\sup_x\,\|\cdot\|(t)$  [on $x$-grid]", fontsize=11)
    ax.set_title(
        r"$\|u_\theta^n - u^*\|_\infty$ (solid) vs $\|u_\theta^n - P(u_\theta^{n-1})\|_\infty$ (dashed)"
        "\n(dashed ≪ solid → outer step closely implements $P$)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, snap in enumerate(snaps):
        c = colours[i]
        u_err = np.array(snap["u_sup_error_op"])
        u_vs_Tu = np.array(snap["u_vs_Tu_sup"])
        ratio = np.where(u_err > 1e-10, u_vs_Tu / u_err, np.nan)
        ax.plot(ts_arr, ratio, color=c, linewidth=1.5)
    ax.axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\frac{\|u_\theta^n - P(u_\theta^{n-1})\|_\infty}{\|u_\theta^n - u^*\|_\infty}$",
        fontsize=11,
    )
    ax.set_title(
        r"P-residual / control error"
        "\n(< 1: P-implementation error smaller than remaining distance to $u^*$)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linewidth=1.5, linestyle="-",
               label=r"$\|u_\theta^n - u^*\|_\infty$ (control error)"),
        Line2D([0], [0], color="green", linewidth=1.5, linestyle="--", alpha=0.7,
               label=r"$\|u_\theta^n - P(u_\theta^{n-1})\|_\infty$ (P-implementation residual)"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.08))

    sm = plt.cm.ScalarMappable(cmap="Greens",
                               norm=mcolors.Normalize(vmin=snaps[0]["outer_it"],
                                                      vmax=snaps[-1]["outer_it"]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="outer iteration", fraction=0.02, pad=0.02)
    fig.suptitle(
        r"Does each outer iteration implement $P$?"
        "\n" r"$\|u_\theta^n - P(u_\theta^{n-1})\|$ (light = early, dark = late)",
        fontsize=11,
    )

    path = output_dir / "u_vs_Tu_prev.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_u_vs_Tu_norm(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""‖u_θ^{n+1} − P(u_θ^n)‖_{[t,T]} = sup_{s∈[t,T], x} |u^{n+1}(s,x) − P(u^n)(s,x)|.

    One curve per outer iteration.  Computed as the right-to-left suffix-max of
    sup_x |u^{n+1} − P(u^n)| (stored in u_vs_Tu_sup), so the value at each t is the
    worst-case control-implementation residual over the remaining interval [t,T].

    Saves to output_dir/u_vs_Tu_norm.png.
    """
    snaps = [s for s in snapshots if s.get("u_vs_Tu_sup") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Greens(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        sup_t = np.array(snap["u_vs_Tu_sup"])
        tiled = _suffix_max(sup_t)
        ax.plot(ts_arr, tiled, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\|u_\theta^{n+1} - P(u_\theta^n)\|_{[t,T]}$", fontsize=12)
    ax.set_title(
        r"$\sup_{s\in[t,T],\,x}\|u_\theta^{n+1}(s,x) - P(u_\theta^n)(s,x)\|$"
        "\n(light = early, dark = late training)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "u_vs_Tu_norm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_path_u_vs_Tu(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""E[‖u^{n+1}_θ(t, X^{u^{n+1}}_t) − P(u_n)(t, X^{u^{n+1}}_t)‖] vs t.

    The expectation is over paths of the CURRENT control u^{n+1}_θ, so the
    metric reflects the residual where u^{n+1}_θ is actually evaluated during
    training — not on a fixed grid.  One curve per outer iteration.

    Saves to output_dir/path_u_vs_Tu.png.
    """
    snaps = [s for s in snapshots if s.get("path_u_vs_Tu") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Greens(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        ax.plot(ts_arr, snap["path_u_vs_Tu"], color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\mathbb{E}_{X^{u^{n+1}}_t}\!\left[\|u^{n+1}_\theta - P(u_n)\|(t,\cdot)\right]$",
        fontsize=11,
    )
    ax.set_title(
        r"$\frac{1}{N}\sum_i\|u^{n+1}_\theta(t,X^{u^{n+1}}_{t,i}) - P(u_n)(t,X^{u^{n+1}}_{t,i})\|$"
        "\n(light = early, dark = late training; paths sampled under current control)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "path_u_vs_Tu.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_Tu_vs_u_next_heatmap(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    """Signed difference heatmap: P(u_θ^n)(t,x) − u_θ^{n+1}(t,x).

    For each consecutive pair of eval snapshots (n, n+1) that have operator
    data, plots the signed difference as a (t, x) heatmap with a diverging
    colour map centred at zero.  Positive (red) = P(u^n) > u^{n+1};
    negative (blue) = P(u^n) < u^{n+1}.

    If the outer iteration perfectly implements P, the heatmap is all-zero.

    Saves to output_dir/Tu_vs_u_next_heatmap.png.
    """
    op_snaps = [s for s in snapshots
                if s.get("T_u_fields") is not None
                and s.get("u_theta_op_fields") is not None]
    if len(op_snaps) < 2:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)

    # Pairs: (snap_n, snap_{n+1})
    pairs = list(zip(op_snaps[:-1], op_snaps[1:]))
    if len(pairs) > max_subplots:
        idxs = np.linspace(0, len(pairs) - 1, max_subplots, dtype=int)
        pairs = [pairs[i] for i in idxs]

    n_pairs = len(pairs)
    ncols = min(n_pairs, 5)
    nrows = math.ceil(n_pairs / ncols)

    # Compute global symmetric colour limit
    all_diffs = []
    for sn, sn1 in pairs:
        Tu = np.array(sn["T_u_fields"])       # [K+1, n_op_grid]
        u_next = np.array(sn1["u_theta_op_fields"])  # [K+1, n_op_grid]
        all_diffs.append((Tu - u_next).ravel())
    all_diffs = np.concatenate(all_diffs)
    vmax = float(np.nanpercentile(np.abs(all_diffs), 98))
    if vmax < 1e-10:
        vmax = 1.0

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, (sn, sn1) in enumerate(pairs):
        ax = axes[i // ncols][i % ncols]
        Tu = np.array(sn["T_u_fields"])
        u_next = np.array(sn1["u_theta_op_fields"])
        diff = Tu - u_next                    # [K+1, n_op_grid]
        im = ax.pcolormesh(ts_arr, xs_arr, diff.T,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.set_title(
            f"$P(u^{{{sn['outer_it']}}}) - u^{{{sn1['outer_it']}}}$",
            fontsize=9,
        )
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n_pairs, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$P(u^n)(t,x) - u^{n+1}(t,x)$",
                     fraction=0.02, pad=0.02)
    fig.suptitle(
        r"Difference heatmap: $P(u_\theta^n)(t,x) - u_\theta^{n+1}(t,x)$"
        "\n(zero = outer iteration perfectly implements $P$; "
        "red = $P$ predicts higher, blue = lower)",
        fontsize=11,
    )

    path = output_dir / "Tu_vs_u_next_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# "Assessing Convergence for the Learned Operator" plots (§3.1)
# ---------------------------------------------------------------------------

def plot_learned_error_heatmap(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    r"""Pointwise heatmap of ‖u_θ^{n+1}(t,x) − u*(t,x)‖ on the xs_op grid.

    One panel per eval checkpoint.  Uses u_error_op_fields which is already
    computed on xs_op during the operator P evaluation.

    Saves to output_dir/learned_error_heatmap.png.
    """
    snaps = [s for s in snapshots if s.get("u_error_op_fields") is not None]
    if not snaps:
        return
    if len(snaps) > max_subplots:
        idxs = np.linspace(0, len(snaps) - 1, max_subplots, dtype=int)
        snaps = [snaps[i] for i in idxs]

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)
    n = len(snaps)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    vmax = max(max(max(row) for row in s["u_error_op_fields"]) for s in snaps)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, snap in enumerate(snaps):
        ax = axes[i // ncols][i % ncols]
        Z = np.array(snap["u_error_op_fields"])   # [K+1, n_op]
        im = ax.pcolormesh(ts_arr, xs_arr, Z.T, cmap="viridis",
                           vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(f"iter {snap['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(r"Pointwise error $\|u_\theta^{n+1}(t,x) - u^*(t,x)\|$", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label=r"$\|u_\theta - u^*\|(t,x)$")
    path = output_dir / "learned_error_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_learned_tiled_sup_error(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""‖u_θ^{n+1} − u*‖_{[t,T]} = sup_{s∈[t,T], x∈xs_op} |u_θ^{n+1}(s,x) − u*(s,x)|.

    Computed as the right-to-left suffix-max of u_sup_error_op.
    One curve per eval checkpoint.

    Saves to output_dir/learned_tiled_sup_error.png.
    """
    snaps = [s for s in snapshots if s.get("u_sup_error_op") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        sup_t = np.array(snap["u_sup_error_op"])
        tiled = _suffix_max(sup_t)
        ax.plot(ts_arr, tiled, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(r"$\|u_\theta^{n+1} - u^*\|_{[t,T]}$", fontsize=12)
    ax.set_title(
        r"$\sup_{s\in[t,T],\,x}\|u_\theta^{n+1}(s,x) - u^*(s,x)\|$"
        "\n(light = early, dark = late training)",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "learned_tiled_sup_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_learned_pointwise_over_tiled(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
    overlay_traj: bool = False,
    path_key: str = "paths_theta",
) -> None:
    r"""Heatmap of the pointwise / tiled ratio from §5.3 (eq:ratio-pw-over-tiled):

        ‖(u_θ^{n+1} − u*)(x,t)‖ / ‖u_θ^n − u*‖_{[t,T]}

    No ‖σ‖ factor in the denominator (latest notes).
    Denominator uses u_sup_error_op suffix-max from the PREVIOUS eval snapshot.
    Ratio → 0 as t → T if the contraction bound governs the learned update.

    Colour scale: RdBu_r, diverging about the contraction threshold 1 — blue
    (< 1) = the step contracted the error at that (t,x), red (> 1) = it grew.

    `overlay_traj=True` overlays a subsample of trajectories rolled out under the
    **source** control u_θ^n (snapshot n's `path_key`) on each pair's panel, and
    writes to `learned_pointwise_over_tiled_traj.png`.

    Saves to output_dir/learned_pointwise_over_tiled{_traj}.png.
    """
    snaps = [s for s in snapshots
             if s.get("u_error_op_fields") is not None
             and s.get("u_sup_error_op") is not None]
    if len(snaps) < 2:
        return

    pairs = list(zip(snaps[:-1], snaps[1:]))
    if len(pairs) > max_subplots:
        idxs = np.linspace(0, len(pairs) - 1, max_subplots, dtype=int)
        pairs = [pairs[i] for i in idxs]

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    xs_arr = np.array(xs_op)
    n_pairs = len(pairs)
    ncols = min(n_pairs, 5)
    nrows = math.ceil(n_pairs / ncols)
    do_traj = bool(overlay_traj)

    # Compute all ratio fields and find global vmax (98th percentile)
    all_ratios = []
    for sn, sn1 in pairs:
        denom_sup = np.array(sn["u_sup_error_op"])
        denom_tiled = _suffix_max(denom_sup)   # [K+1]  ‖u_θ^n − u*‖_{[t,T]}
        numer = np.array(sn1["u_error_op_fields"])                    # [K+1, n_op]
        # Avoid division by zero
        denom_safe = np.where(denom_tiled > 1e-10, denom_tiled, np.nan)
        ratio = numer / denom_safe[:, np.newaxis]                      # [K+1, n_op]
        all_ratios.append(ratio)
    data_max = float(np.nanpercentile(np.concatenate([r.ravel() for r in all_ratios]), 98))
    # Linear scale symmetric about the contraction threshold 1 → white at 1,
    # blue < 1 < red, evenly-spaced colourbar ticks.  Extend past 2 if needed.
    vmax = 2.0 if data_max <= 2.0 else float(np.ceil(data_max))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cbar_ticks = np.arange(0.0, vmax + 1e-9, 0.25 if vmax <= 2.0 else 0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for i, ((sn, sn1), ratio) in enumerate(zip(pairs, all_ratios)):
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(ts_arr, xs_arr, ratio.T, cmap="RdBu_r",
                           norm=norm, shading="auto")
        if do_traj and sn.get(path_key) is not None:
            _overlay_source_traj(ax, ts_arr, sn[path_key])
            ax.set_ylim(xs_arr.min(), xs_arr.max())
        ax.set_title(f"iter {sn['outer_it']}→{sn1['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n_pairs, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    _traj_note = (r"  —  black: trajectories under the source control $u_\theta^n$"
                  if do_traj else "")
    fig.suptitle(
        r"$\frac{\|(u_\theta^{n+1}-u^*)(x,t)\|}{\|u_\theta^n-u^*\|_{[t,T]}}$"
        "\n(should → 0 as $t→T$; blue < 1 < red)" + _traj_note,
        fontsize=11,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label="ratio", ticks=cbar_ticks,
                     extend="max" if vmax < data_max else "neither")
    path = output_dir / ("learned_pointwise_over_tiled_traj.png" if do_traj
                         else "learned_pointwise_over_tiled.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_learned_tiled_over_tiled(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""‖u_θ^{n+1} − u*‖_{[t,T]} / ‖u_θ^n − u*‖_{[t,T]} vs t  (§5.3, eq:ratio-tiled-over-tiled).

    No ‖σ‖ factor in the denominator (latest notes).
    Both norms use the suffix-max of u_sup_error_op (xs_op grid).
    Denominator uses the PREVIOUS eval snapshot's tiled sup error.
    Ratio → 0 as t → T if the contraction bound governs the learned update.
    One curve per consecutive pair of eval checkpoints.

    Saves to output_dir/learned_tiled_over_tiled.png.
    """
    snaps = [s for s in snapshots if s.get("u_sup_error_op") is not None]
    if len(snaps) < 2:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps) - 1
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (sn, sn1) in enumerate(zip(snaps[:-1], snaps[1:])):
        denom_sup = np.array(sn["u_sup_error_op"])
        denom_tiled = _suffix_max(denom_sup)
        numer_sup = np.array(sn1["u_sup_error_op"])
        numer_tiled = _suffix_max(numer_sup)
        denom = np.where(denom_tiled > 1e-10, denom_tiled, np.nan)
        ratio = numer_tiled / denom
        ax.plot(ts_arr, ratio, color=colours[i],
                label=f"iter {sn['outer_it']}→{sn1['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\frac{\|u_\theta^{n+1}-u^*\|_{[t,T]}}{\|u_\theta^n-u^*\|_{[t,T]}}$",
        fontsize=11,
    )
    ax.set_title(
        r"Tiled-over-tiled contraction ratio"
        "\n" r"(→ 0 as $t→T$ if bound holds; < 1 = contraction per step)",
        fontsize=10,
    )
    ax.axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "learned_tiled_over_tiled.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Path-based u^{n+1}_θ vs u* metrics (Assessing Convergence section)
# ---------------------------------------------------------------------------

def plot_path_u_vs_ustar(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""E_{X^{u^{n+1}}_t}[‖u^{n+1}_θ(t,X) − u*(t,X)‖] vs t, one curve per eval.

    Saves to output_dir/path_u_vs_ustar.png.
    """
    snaps = [s for s in snapshots if s.get("path_u_vs_ustar") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, snap in enumerate(snaps):
        vals = np.array(snap["path_u_vs_ustar"])
        ax.plot(ts_arr, vals, color=colours[i],
                label=f"iter {snap['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\mathbb{E}_{X^{u^{n+1}}_t}[\|u^{n+1}_\theta(t,X)-u^*(t,X)\|]$",
        fontsize=10,
    )
    ax.set_title("Path-based control error vs. $u^*$ (darker = later)", fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "path_u_vs_ustar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_path_u_vs_ustar_per_t_over_tiled(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""Per-t ratio: E_{u^{n+1}}[‖u^{n+1}_θ−u*‖(t)] / E_{u^{n+1}}[‖u^n_θ−u*‖]_{[t,T]} vs t.

    Path-based analogue of §5.3 eq:ratio-pw-over-tiled; no ‖σ‖ factor (latest notes).
    Numerator: path_u_vs_ustar (per-t MC under current control).
    Denominator: suffix-max of path_u_prev_vs_ustar (per-t MC of u^n vs u* under current control).
    Both evaluated at the same trajectory sample from u^{n+1}_θ.
    One curve per snapshot with path_u_prev_vs_ustar. → 0 as t → T if bound holds.

    Saves to output_dir/path_u_vs_ustar_per_t_over_tiled.png.
    """
    snaps = [s for s in snapshots
             if s.get("path_u_vs_ustar") is not None
             and s.get("path_u_prev_vs_ustar") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, sn in enumerate(snaps):
        denom_arr = np.array(sn["path_u_prev_vs_ustar"])
        denom_tiled = _suffix_max(denom_arr)   # suffix-max of E_{u^{n+1}}[‖u^n−u*‖]
        denom = np.where(denom_tiled > 1e-10, denom_tiled, np.nan)
        numer = np.array(sn["path_u_vs_ustar"])                       # per-t path metric
        ratio = numer / denom
        ax.plot(ts_arr, ratio, color=colours[i],
                label=f"iter {sn['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\frac{\mathbb{E}_{u^{n+1}}[\|u^{n+1}_\theta-u^*\|(t)]}{\mathbb{E}_{u^{n+1}}[\|u^n_\theta-u^*\|]_{[t,T]}}$",
        fontsize=10,
    )
    ax.set_title(
        "Path-based per-$t$ / tiled ratio\n(→ 0 as $t→T$ if bound holds)",
        fontsize=10,
    )
    ax.axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "path_u_vs_ustar_per_t_over_tiled.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_path_u_vs_ustar_tiled_over_tiled(
    snapshots: list[dict],
    ts: list[float],
    output_dir: str | Path,
) -> None:
    r"""Tiled ratio: suffix_max(E_{u^{n+1}}[‖u^{n+1}_θ−u*‖])(t) / E_{u^{n+1}}[‖u^n_θ−u*‖]_{[t,T]}.

    Path-based analogue of §5.3 eq:ratio-tiled-over-tiled; no ‖σ‖ factor (latest notes).
    Numerator: suffix-max of path_u_vs_ustar.
    Denominator: suffix-max of path_u_prev_vs_ustar (MC of u^n vs u* under current control).
    Both from the same snapshot; one curve per snapshot with path_u_prev_vs_ustar.

    Saves to output_dir/path_u_vs_ustar_tiled_over_tiled.png.
    """
    snaps = [s for s in snapshots
             if s.get("path_u_vs_ustar") is not None
             and s.get("path_u_prev_vs_ustar") is not None]
    if not snaps:
        return

    output_dir = Path(output_dir)
    ts_arr = np.array(ts)
    n = len(snaps)
    colours = cm.Blues(np.linspace(0.3, 1.0, max(n, 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, sn in enumerate(snaps):
        denom_arr = np.array(sn["path_u_prev_vs_ustar"])
        denom_tiled = _suffix_max(denom_arr)   # suffix-max of E_{u^{n+1}}[‖u^n−u*‖]
        denom = np.where(denom_tiled > 1e-10, denom_tiled, np.nan)
        numer_arr = np.array(sn["path_u_vs_ustar"])
        numer_tiled = _suffix_max(numer_arr)   # suffix-max of E_{u^{n+1}}[‖u^{n+1}−u*‖]
        ratio = numer_tiled / denom
        ax.plot(ts_arr, ratio, color=colours[i],
                label=f"iter {sn['outer_it']}", linewidth=1.5)

    ax.set_xlabel("time $t$", fontsize=12)
    ax.set_ylabel(
        r"$\frac{\mathbb{E}_{u^{n+1}}[\|u^{n+1}_\theta-u^*\|]_{[t,T]}}{\mathbb{E}_{u^{n+1}}[\|u^n_\theta-u^*\|]_{[t,T]}}$",
        fontsize=10,
    )
    ax.set_title(
        "Path-based tiled / tiled ratio\n(→ 0 as $t→T$ if bound holds; < 1 = contraction)",
        fontsize=10,
    )
    ax.axhline(1.0, color="grey", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        idxs = list(range(0, len(handles), max(1, len(handles) // 5)))
        if idxs[-1] != len(handles) - 1:
            idxs.append(len(handles) - 1)
        handles = [handles[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = output_dir / "path_u_vs_ustar_tiled_over_tiled.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_snapshots(data: dict, output_dir: str | Path) -> None:
    path = Path(output_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved %s", path)


def _retile_snapshots(snapshots: list[dict], pct: float) -> None:
    """Recompute every sup / tiled-sup quantity from the stored pointwise `*_fields`
    at percentile `pct` (over x AND over the [t,T] suffix).  In-place.  `pct=100`
    reproduces the exact-max values.  Lets an existing run be re-clamped without
    retraining — see `load_and_plot(..., tiled_sup_percentile=...)`.
    """
    set_suffix_percentile(pct)          # so _suffix_max below uses `pct`

    def gsup(F):                       # [K+1, n] -> [K+1]  robust sup over x
        F = np.asarray(F, dtype=float)
        return F.max(axis=1) if pct >= 100.0 else np.percentile(F, pct, axis=1)

    for s in snapshots:
        if s.get("op_error_fields") is not None:
            ose = gsup(s["op_error_fields"])
            tose = _suffix_max(ose)
            s["op_sup_error"] = ose.tolist()
            s["tiled_op_sup_error"] = np.asarray(tose, dtype=float).tolist()
        if s.get("u_error_op_fields") is not None:
            use = gsup(s["u_error_op_fields"])
            s["u_sup_error_op"] = use.tolist()
        # sanity ratios: numerator / ‖u_n-u*‖_{[t,T]}
        if s.get("op_error_fields") is not None and s.get("u_sup_error_op") is not None:
            denom = np.maximum(np.asarray(_suffix_max(np.asarray(s["u_sup_error_op"], float)),
                                          dtype=float), 1e-12)
            oef = np.asarray(s["op_error_fields"], dtype=float)
            s["sanity_ratio_pointwise"] = (oef / denom[:, None]).tolist()
            s["sanity_ratio_tiled"] = (np.asarray(s["tiled_op_sup_error"], float) / denom).tolist()
        if s.get("u_vs_Tu_fields") is not None:
            s["u_vs_Tu_sup"] = gsup(s["u_vs_Tu_fields"]).tolist()
        if s.get("bothroll_lhs_fields") is not None:
            s["tiled_bothroll_lhs"] = np.asarray(
                _suffix_max(gsup(s["bothroll_lhs_fields"])), dtype=float).tolist()
        if s.get("error_fields") is not None:
            al = gsup(s["error_fields"])
            s["abs_linf"] = al.tolist()
            s["tiled_al_inf"] = np.asarray(_suffix_max(al), dtype=float).tolist()
            ef = np.asarray(s["error_fields"], dtype=float)
            if pct >= 100.0:
                s["tiled_error_fields"] = np.maximum.accumulate(ef[::-1])[::-1].tolist()
            else:
                s["tiled_error_fields"] = np.stack(
                    [np.percentile(ef[k:], pct, axis=0) for k in range(ef.shape[0])]).tolist()


def load_and_plot(metrics_json: str | Path, output_dir: str | Path | None = None,
                  tiled_sup_percentile: float | None = None) -> None:
    """Re-generate all plots from a saved metrics.json.

    `tiled_sup_percentile`: if given, recompute every sup / tiled-sup quantity from
    the stored pointwise fields at this percentile (over x and over [t,T]) before
    plotting — applies the `eval.tiled_sup_percentile` clamp to an existing run
    with no retraining.  None ⇒ use the values as stored.
    """
    metrics_json = Path(metrics_json)
    if output_dir is None:
        output_dir = metrics_json.parent
    with open(metrics_json) as f:
        data = json.load(f)
    if tiled_sup_percentile is not None:
        _retile_snapshots(data["snapshots"], float(tiled_sup_percentile))
        data["tiled_sup_percentile"] = float(tiled_sup_percentile)
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

    d_train = Path(output_dir) / "training"
    d_conv  = Path(output_dir) / "convergence"
    d_ctrl  = Path(output_dir) / "control"
    d_term  = Path(output_dir) / "terminal"
    d_err   = Path(output_dir) / "errors"
    d_sbm   = Path(output_dir) / "same_bm"
    d_op    = Path(output_dir) / "operator"
    for _d in (d_train, d_conv, d_ctrl, d_term, d_err, d_sbm, d_op):
        _d.mkdir(parents=True, exist_ok=True)

    plot_inner_steps(snapshots, d_train)
    plot_inner_convergence(snapshots, d_train)

    plot_convergence(snapshots, ts, d_conv)

    if xs and u_star_field:
        plot_optimal_control(ts, xs, u_star_field, d, d_ctrl,
                             u_theta_field, paths_star, paths_theta)
        plot_control_evolution(snapshots, ts, xs, d, u_star_field, d_ctrl)
    xs_op_for_ctrl = data.get("xs_op", data.get("xs_sbm", []))
    if xs_op_for_ctrl:
        plot_operator_vs_next_control(snapshots, ts, xs_op_for_ctrl, d, d_ctrl)
        plot_operator_vs_next_control(snapshots, ts, xs_op_for_ctrl, d, d_ctrl,
                                     overlay_traj=True)

    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf_fn, d, d_term)
    plot_terminal_evolution(snapshots, target_pdf_fn, d, d_term)

    if xs:
        plot_heatmaps(snapshots, ts, xs, d_err)
        plot_contraction_heatmaps(snapshots, ts, xs, d_err)

    xs_op = data.get("xs_op", data.get("xs_sbm", []))
    set_suffix_percentile(data.get("tiled_sup_percentile", 100.0))

    fp_data = data.get("fixed_point_check")
    if fp_data is not None:
        plot_fixed_point_residual(fp_data, ts, d, d_sbm)

    if xs_op:
        plot_same_bm_lhs_curves(snapshots, ts, d_sbm)
        plot_same_bm_lhs_heatmaps(snapshots, ts, xs_op, d_sbm)
        plot_tiled_same_bm_lhs(snapshots, ts, d_sbm)
        plot_sbm_ratio(snapshots, ts, d_sbm)
        plot_sbm_ratio_learned(snapshots, ts, d_sbm)
        plot_sanity_ratio_heatmap(snapshots, ts, xs_op, d_sbm)
        plot_learned_error_heatmap(snapshots, ts, xs_op, d_conv)
        plot_learned_tiled_sup_error(snapshots, ts, d_conv)
        plot_learned_pointwise_over_tiled(snapshots, ts, xs_op, d_conv)
        plot_learned_pointwise_over_tiled(snapshots, ts, xs_op, d_conv,
                                         overlay_traj=True)
        plot_learned_tiled_over_tiled(snapshots, ts, d_conv)
        plot_operator_error_curves(snapshots, ts, d_op)
        plot_operator_vs_learned(snapshots, ts, d_op)
        plot_u_vs_Tu_prev(snapshots, ts, d_op)
        plot_u_vs_Tu_norm(snapshots, ts, d_op)
        plot_path_u_vs_Tu(snapshots, ts, d_op)
        plot_Tu_vs_u_next_heatmap(snapshots, ts, xs_op, d_op)
        plot_path_u_vs_ustar(snapshots, ts, d_conv)
        plot_path_u_vs_ustar_per_t_over_tiled(snapshots, ts, d_conv)
        plot_path_u_vs_ustar_tiled_over_tiled(snapshots, ts, d_conv)

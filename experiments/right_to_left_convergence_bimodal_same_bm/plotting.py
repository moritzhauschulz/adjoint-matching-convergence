"""Plots for the right_to_left_convergence_bimodal_same_bm experiment.

Only the figures referenced in Project_Notes.tm are produced (9 files):

  control/heatmap_u_star.png                     plot_optimal_control
  control/heatmap_P_vs_next_control_traj.png     plot_operator_vs_next_control
  terminal/terminal_distributions.png            plot_terminal_distributions
  same_bm/u_star_fixed_point_residual.png        plot_fixed_point_residual
  same_bm/sbm_ratio.png                          plot_sbm_ratio
  same_bm/sbm_ratio_learned.png                  plot_sbm_ratio_learned
  operator/path_u_vs_Tu.png                      plot_path_u_vs_Tu
  convergence/learned_pointwise_over_tiled_traj.png   plot_learned_pointwise_over_tiled
  convergence/path_u_vs_ustar_per_t_over_tiled.png    plot_path_u_vs_ustar_per_t_over_tiled

`load_and_plot(metrics_json, output_dir=None, tiled_sup_percentile=None)`
regenerates them from a run's metrics.json (optionally re-clamping every
sup / tiled-sup norm at a percentile via `_retile_snapshots`).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def plot_optimal_control(ts, xs, u_star_field, d, output_dir, u_theta_field=None) -> None:
    """(t,x) heatmap of the analytic u* (and the final learned u_θ, if given).

    Saves to output_dir/heatmap_u_star.png.
    """
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
                                  max_subplots: int = 20) -> None:
    r"""Per operator-eval pair (n, n+1), one row with four (t,x) heatmaps:

        col 0:  P(u_θ^n)                  (`T_u_fields` of snapshot n)
        col 1:  u_θ^{n+1}                 (`u_theta_op_fields` of snapshot n+1)
        col 2:  P(u_θ^n) − u_θ^{n+1}      (the outer-step-implements-P residual)
        col 3:  u_θ^{n+1} − u*            (`u_star_op_fields`; distance to the fixed point)

    ONE robust symmetric colour scale for every panel (all cols, all rows), with
    a colourbar on each row — so the residual columns are read on the same scale
    as the controls and relative error sizes are apparent.  A subsample of
    trajectories rolled out under the **source** control u_θ^n (snapshot n's
    `paths_theta`) is overlaid on every panel of that row, showing where the
    process the operator sees actually visits.

    Saves to output_dir/heatmap_P_vs_next_control_traj.png.
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
        paths = sn.get("paths_theta")
        for j in range(4):
            if not axes[r][j].get_visible():
                continue
            if paths is not None:
                _overlay_source_traj(axes[r][j], ts_arr, paths)
            axes[r][j].set_xlabel("$t$", fontsize=8)
            axes[r][j].set_ylabel("$x$", fontsize=8)
            axes[r][j].tick_params(labelsize=7)
            axes[r][j].set_ylim(xs_arr.min(), xs_arr.max())
        axes[r][0].set_ylabel(rf"$x$   (paths $\sim u_\theta^{{{sn['outer_it']}}}$)", fontsize=8)
        fig.colorbar(im0, ax=list(axes[r]), location="right", shrink=0.9,
                     label=cb_label, pad=0.02)

    fig.suptitle(r"$P(u_\theta^n)$ vs the next learned control $u_\theta^{n+1}$, "
                 r"with residuals $P(u_\theta^n)-u_\theta^{n+1}$ and $u_\theta^{n+1}-u^*$"
                 "\n(one shared colour scale; black lines: trajectories under the "
                 r"source control $u_\theta^n$)", fontsize=12)
    path = output_dir / "heatmap_P_vs_next_control_traj.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


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


# Robust "sup" percentile for the tiled / suffix-sup norms; set per run by
# load_and_plot from metrics.json's `tiled_sup_percentile` (100 = exact max).
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


def plot_learned_pointwise_over_tiled(
    snapshots: list[dict],
    ts: list[float],
    xs_op: list[float],
    output_dir: str | Path,
    max_subplots: int = 20,
) -> None:
    r"""Heatmap of the pointwise / tiled ratio from §5.3 (eq:ratio-pw-over-tiled):

        ‖(u_θ^{n+1} − u*)(x,t)‖ / ‖u_θ^n − u*‖_{[t,T]}

    No ‖σ‖ factor in the denominator (latest notes).  Denominator uses
    `u_sup_error_op` suffix-max from the PREVIOUS eval snapshot.  Ratio → 0 as
    t → T if the contraction bound governs the learned update.

    Colour: RdBu_r on a linear [0, 2] scale — white at the contraction threshold
    1, blue (< 1) = the step contracted the error at that (t,x), red (> 1) = it
    grew.  Source-control (u_θ^n) trajectories overlaid on each pair's panel.

    Saves to output_dir/learned_pointwise_over_tiled_traj.png.
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
        paths = sn.get("paths_theta")
        if paths is not None:
            _overlay_source_traj(ax, ts_arr, paths)
            ax.set_ylim(xs_arr.min(), xs_arr.max())
        ax.set_title(f"iter {sn['outer_it']}→{sn1['outer_it']}", fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.set_ylabel("$x$", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n_pairs, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        r"$\frac{\|(u_\theta^{n+1}-u^*)(x,t)\|}{\|u_\theta^n-u^*\|_{[t,T]}}$"
        "\n(should → 0 as $t→T$; blue < 1 < red; "
        r"black: trajectories under the source control $u_\theta^n$)",
        fontsize=11,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[:, -1].tolist(), location="right", shrink=0.8,
                     label="ratio", ticks=cbar_ticks,
                     extend="max" if vmax < data_max else "neither")
    path = output_dir / "learned_pointwise_over_tiled_traj.png"
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
            s["op_sup_error"] = ose.tolist()
            s["tiled_op_sup_error"] = np.asarray(_suffix_max(ose), dtype=float).tolist()
        if s.get("u_error_op_fields") is not None:
            s["u_sup_error_op"] = gsup(s["u_error_op_fields"]).tolist()
        # §"Sanity Check" ratios: numerator / ‖u_n-u*‖_{[t,T]}  (no ‖σ‖ factor)
        if s.get("op_error_fields") is not None and s.get("u_sup_error_op") is not None:
            denom = np.maximum(
                np.asarray(_suffix_max(np.asarray(s["u_sup_error_op"], float)), dtype=float),
                1e-12,
            )
            oef = np.asarray(s["op_error_fields"], dtype=float)
            s["sanity_ratio_pointwise"] = (oef / denom[:, None]).tolist()
            s["sanity_ratio_tiled"] = (np.asarray(s["tiled_op_sup_error"], float) / denom).tolist()


def load_and_plot(metrics_json: str | Path, output_dir: str | Path | None = None,
                  tiled_sup_percentile: float | None = None) -> None:
    """Regenerate the 9 notes figures from a run's metrics.json.

    `tiled_sup_percentile`: if given, re-clamp every sup / tiled-sup norm at this
    percentile (over x and over the [t,T] suffix) before plotting — applies the
    `eval.tiled_sup_percentile` clamp to an existing run with no retraining.
    None ⇒ use the values as stored.
    """
    metrics_json = Path(metrics_json)
    if output_dir is None:
        output_dir = metrics_json.parent
    with open(metrics_json) as f:
        data = json.load(f)

    if tiled_sup_percentile is not None:
        _retile_snapshots(data["snapshots"], float(tiled_sup_percentile))
        data["tiled_sup_percentile"] = float(tiled_sup_percentile)
    set_suffix_percentile(data.get("tiled_sup_percentile", 100.0))

    snapshots = data["snapshots"]
    ts = data["ts"]
    d = data.get("d", 1)
    xs = data.get("xs", [])
    xs_op = data.get("xs_op", [])
    u_star_field = data.get("u_star_field", [])
    u_theta_field = data.get("u_theta_field")
    paths_star = data.get("paths_star")
    paths_theta = data.get("paths_theta")
    fp_data = data.get("fixed_point_check")

    tp = data.get("target_params", {})
    w1 = float(tp.get("w1", 0.5)); w2 = float(tp.get("w2", 1.0 - w1))
    lambda1 = float(tp.get("lambda1", 1.0)); lambda2 = float(tp.get("lambda2", 1.0))
    mu1 = float(tp.get("mu1", -1.0)); mu2 = float(tp.get("mu2", 1.0))

    def target_pdf(x):
        """p^{u*}(x) ∝ e^{r(x)} = Σ αᵢ N(μᵢ, 1/λᵢ),  αᵢ ∝ wᵢ/√λᵢ."""
        a1 = w1 / math.sqrt(lambda1); a2 = w2 / math.sqrt(lambda2)
        alpha1, alpha2 = a1 / (a1 + a2), a2 / (a1 + a2)
        x = np.asarray(x)
        p1 = alpha1 * np.exp(-0.5 * lambda1 * (x - mu1) ** 2) * math.sqrt(lambda1 / (2 * math.pi))
        p2 = alpha2 * np.exp(-0.5 * lambda2 * (x - mu2) ** 2) * math.sqrt(lambda2 / (2 * math.pi))
        return p1 + p2

    d_conv = Path(output_dir) / "convergence"
    d_ctrl = Path(output_dir) / "control"
    d_term = Path(output_dir) / "terminal"
    d_sbm = Path(output_dir) / "same_bm"
    d_op = Path(output_dir) / "operator"
    for _d in (d_conv, d_ctrl, d_term, d_sbm, d_op):
        _d.mkdir(parents=True, exist_ok=True)

    if xs and u_star_field:
        plot_optimal_control(ts, xs, u_star_field, d, d_ctrl, u_theta_field)
    if paths_star is not None and paths_theta is not None:
        plot_terminal_distributions(paths_star, paths_theta, target_pdf, d, d_term)
    if fp_data is not None:
        plot_fixed_point_residual(fp_data, ts, d, d_sbm)
    if xs_op:
        plot_operator_vs_next_control(snapshots, ts, xs_op, d, d_ctrl)
        plot_sbm_ratio(snapshots, ts, d_sbm)
        plot_sbm_ratio_learned(snapshots, ts, d_sbm)
        plot_path_u_vs_Tu(snapshots, ts, d_op)
        plot_learned_pointwise_over_tiled(snapshots, ts, xs_op, d_conv)
        plot_path_u_vs_ustar_per_t_over_tiled(snapshots, ts, d_conv)

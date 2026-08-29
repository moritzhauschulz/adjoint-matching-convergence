# Right-to-Left Convergence — Bimodal, Sanity Check

## 1. Goal

Verify the **"Sanity Check"** paragraph of the project notes (§ *Verifying Right
to Left Convergence in Wasserstein Sense*), which has two parts.

**(a) Self-consistency of the analytic $u^*$:** $P(u^*) = u^*$, checked by

$$\left\|\mathbb{E}[\sigma(t)\nabla g(X_T^{u^*,x_t})] - u^*(x_t,t)\right\| \;\ll\; 1
\qquad\text{i.e.}\qquad \lVert P(u^*)(t,x) - u^*(t,x)\rVert \ll 1$$

on the full $(x,t)$ grid — §5.0, plot `same_bm/u_star_fixed_point_residual.png`.

**(b) The right-to-left contraction bound**, stated with both processes rolled out:

$$\sup_{x_s\in\mathbb{R},\,s\in[t,T]}\left\|\mathbb{E}\!\left[\sigma(s)\nabla g(X_T^{u,x_s}) - \sigma(s)\nabla g(X_T^{u^*,x_s})\right]\right\| \leq \|\sigma\|_{[t,T]}\,\|u-u^*\|_{[t,T]}\,\|\nabla g\|_\infty\,C'e^{C(T-t)}\,2\sqrt{T-t}$$

with the two implied ratios $\to 0$ as $t\to T$. In the ratios the notes
**substitute the analytic $u^*$** for the second rollout (since $P$ is identity on
$u^*$): the numerator becomes $\lVert\mathbb{E}[\sigma(t)\nabla g(X_T^{u,x_t})] - u^*(x,t)\rVert$,
which in this codebase's sign convention **equals** $\lVert P(u_\theta)(t,x) - u^*(t,x)\rVert$
— i.e. the operator error `op_error_fields` already computed in §5.1. The
sanity-check metrics/plots are therefore **folded into §5.1**: MC noise only on the
$u_\theta$ rollout, none on $u^*$.

**Norm outside the expectation** (`‖E[·]‖`, not `E[‖·‖]`) throughout, per the
margin remark *"[MG: Bound in the numerator should be outside!]"*.

> **Notes inconsistency (flagged).** The notes' operator is written without the
> leading minus (and with a sign-flipped $u^*$); the code uses
> $P(u)(t,x) = -\sigma(t)\mathbb{E}[\nabla g(X_T^{u,x})]$ with $\nabla g = -x/\nu_1 - \nabla r$
> (the full-AS RAM target). $\lVert P(u^*)-u^*\rVert$ and $\lVert P(u_\theta)-u^*\rVert$
> are sign-convention-independent. See `src/adjoint_sampling/CLAUDE.md` §9.

---

## 2. Setting

Bimodal log-mixture reward, dimension $d=1$. **Noise schedule** via
`eval`-sibling config keys `sigma_schedule` + `sigma` (→ `utils.make_noise_schedule`):

| `sigma_schedule` | $\sigma(t)$ | $\nu_1$ | needs |
|---|---|---|---|
| `constant` (default) | $\sigma_0$ | $\sigma_0^2$ | $\lambda_i > 1/\sigma_0^2$ |
| `linear` | $\sigma_0(1-t)$ | $\sigma_0^2/3$ | $\lambda_i > 3/\sigma_0^2$ |

The Doob $h$-transform $u^*(t,x) = \sigma(t)\nabla_x\log h(t,x)$ is schedule-agnostic
for $x$-independent $\sigma(t)$; `optimal_control` takes $\Sigma_t = \nu_1 - \nu_t$
via `sigma_int_fn` (`make_sigma_int_fn`). The terminal distribution
$p^{u^*}(x_1)\propto e^{r(x_1)}$ is **schedule-independent** (`terminal_mixture_params`
unchanged). Linear runs: `sigma_schedule=linear` on the CLI (config default stays
`constant`). Note $\sigma(1)=0$ for `linear`, so $u^*(1,x)=0$ and any
$\sigma(t)$-weighted ratio denominator vanishes at $t=T$.

See `right_to_left_convergence_bimodal`'s `CLAUDE.md` for the full reward spec, Feynman-Kac optimal
control $u^*$, effective parameters $\lambda_i^*$, $\mu_i^*$, $\kappa_i$, and terminal distribution.

> **$\kappa_i$ term (fixed 2026-08-29).** $\log A_i$ carries a mode-dependent
> constant $\kappa_i = d\,\lambda_i\mu_i^2/(2\nu_1\lambda_i^*)$ from completing the
> square. It cancels in the softmax only when $\mu_1^2=\mu_2^2$ and $\lambda_1=\lambda_2$;
> for asymmetric means/precisions the earlier code (missing $\kappa_i$) gave a
> wrong analytic $u^*$ — it violated $u^*(1,x) = -\sigma(1)\nabla g(x)$ by $O(1)$.
> Guarded by `tests/test_optimal_control_boundary.py`.

---

## 3. Sanity-check metric (folded into the operator error, §5.1)

The primary metric is the **operator error with analytic $u^*$**:

$$\mathrm{Sanity}(t, x) = \left\|P(u_\theta^n)(t, x) - u^*(t, x)\right\|,
\qquad P(u_\theta^n)(t,x) = -\sigma(t)\,\mathbb{E}\!\left[\nabla g(X_T^{u_\theta^n,x})\right]$$

$P(u_\theta^n)$ is MC-estimated (`n_op_mc_samples` rollouts of $u_\theta^n$ from
$(t,x)$ to $T$); $u^*$ is analytic. This **is** `op_error_fields` from §5.1 —
computed once per operator-eval, on `xs_op`, at the `op_every` cadence.

- Pointwise field: `op_error_fields` `[K+1][n_op]`
- Sup over $x$: `op_sup_error` `[K+1]`
- Tiled: `tiled_op_sup_error` `[K+1]` $= \sup_{s\ge t,x}\mathrm{Sanity}(s,x)$

**Boundary condition:** $P(u)(T,x) = -\sigma(T)\nabla g(x) = u^*(T,x)$ for *all* $u$,
so `op_sup_error[K] = 0` and `tiled_op_sup_error[K] = 0` exactly.

### Both-rollout comparison (kept, secondary)

`operator_diff_shared_bm_field` still MC-estimates the pure both-rollout form
$\lVert\mathbb{E}_B[\sigma(\nabla g(X_T^{u_\theta}) - \nabla g(X_T^{u^*}))]\rVert$ with a
**shared** Brownian path (`n_both_rollout_samples`). By linearity it has the same
population value as $\mathrm{Sanity}$; the shared BM only lowers MC variance when
$u_\theta\approx u^*$. Stored as `bothroll_lhs_fields` / `tiled_bothroll_lhs`;
overlaid (dotted) on `tiled_same_bm_lhs.png` for reference. Disable with
`eval.both_rollout_compare=false`.

---

## 4. Ratios and the $\sqrt{T-t}$ shape (§ "Sanity Check")

The notes' bound gives $\mathrm{Tiled}\mathrm{Sanity}(t) \lesssim \|\sigma\|_{[t,T]}\,\|u_\theta-u^*\|_{[t,T]}\,\|\nabla g\|_\infty\,C'e^{C(T-t)}\,2\sqrt{T-t}$,
so the two ratios

$$\frac{\mathrm{Sanity}(t,x)}{\|u_\theta-u^*\|_{[t,T]}}
\quad(\text{pointwise, } \texttt{sanity\_ratio\_pointwise}),
\qquad
\frac{\mathrm{Tiled}\mathrm{Sanity}(t)}{\|u_\theta-u^*\|_{[t,T]}}
\quad(\text{tiled, } \texttt{sanity\_ratio\_tiled})$$

should $\to 0$ as $t\to T$. Denominator: the suffix-sup of `u_sup_error_op`
($=\|u_\theta-u^*\|_{[t_k,T]}$) — **no** $\|\sigma\|$ factor (latest notes). Both
ratios share this denominator; the pointwise numerator keeps its $x$-dependence,
the tiled one is `tiled_op_sup_error`.

Near $t\to T$ ($e^{C(T-t)}\to 1$, $\|u_\theta-u^*\|_{[t,T]}$ flat) the tiled error
shape is $\propto\sqrt{T-t}$. `tiled_same_bm_lhs.png` right panel plots it log–log
vs $\tau=T-t$ with a slope-$\tfrac12$ guide (a $\sqrt{}$ law $\Leftrightarrow$ slope $\tfrac12$).

---

## 5. Operator T evaluation (§3.1 "Comparing the Analytic Operator with the Learned Iteration")

The self-consistency operator is:

$$T(u)(t,x) = -\sigma(t)\,\mathbb{E}_u\!\left[\nabla g\!\left(X_T^{u,x}\right)\right]$$

$u^*$ is the fixed point $T(u^*) = u^*$. The outer iterations aim to make $u_\theta^{n+1} \approx T(u_\theta^n)$.

### 5.0 Fixed-point sanity check for the analytic $u^*$ ("very basic sanity check")

Before comparing the *learned* control to $T$, verify that the *analytic* $u^*$ is
actually a fixed point:

$$\left\|T(u^*)(t,x) - u^*(t,x)\right\| \;\ll\; 1 \qquad \text{for all } (t,x).$$

Computed **once** (independent of training) by
`adjoint_sampling.operator.fixed_point_residual_field(u_star_fn, grad_g_fn, xs_fp, ts_eval, ...)`
on an `n_fixed_point_grid` × $(K{+}1)$ mesh with `n_fixed_point_mc` MC rollouts per
node. Exactly $0$ at $t=T$ (boundary: $T(u)(T,x) = -\sigma(T)\nabla g(x) = u^*(T,x)$);
elsewhere the residual is Monte-Carlo-noise-limited and scales as
$n_\text{mc}^{-1/2}$ (largest near $x = (\mu_1+\mu_2)/2$, where $\nabla r(X_T)$ has
the highest variance). A residual that does **not** shrink with `n_fixed_point_mc`
would flag a bug in `optimal_control`, `grad_r`/`grad_g_fn`, or the operator.

Stored (top-level of `metrics.json`, not per-snapshot) under
`fixed_point_check = {xs, residual [K+1][n_grid], p_u_star, u_star, n_mc}`.
Plot: `same_bm/u_star_fixed_point_residual.png` (heatmap over $(t,x)$ + $\sup_x$ /
$\mathrm{mean}_x$ vs $t$).

> **Sign convention**: the notes write this operator without the leading minus
> and with a sign-flipped $u^*$; $\|T(u^*)-u^*\|$ is convention-independent. See
> `src/adjoint_sampling/CLAUDE.md` §9. Reusable across any experiment with a
> closed-form $u^*$ — pass that experiment's `u_star_fn` and `grad_g_fn`.

### 5.1 Grid-based metrics (on `xs_op` grid)

All metrics below are evaluated on the `xs_op` grid (same as `xs_sbm`, 60 points centred at $(\mu_1+\mu_2)/2$):

| Metric | Description | Stored key |
|--------|-------------|-----------|
| $T(u_\theta^n)(t,x)$ signed | Signed 1-D operator output | `T_u_fields` `[K+1][n_op]` |
| $u_\theta^n(t,x)$ signed | Signed 1-D network output | `u_theta_op_fields` `[K+1][n_op]` |
| $\|T(u_\theta^n) - u^*\|$ | Pointwise operator error | `op_error_fields` `[K+1][n_op]` |
| $\sup_x\|T(u_\theta^n) - u^*\|$ | Sup operator error at each $t$ | `op_sup_error` `[K+1]` |
| $\|T(u_\theta^n) - u^*\|_{[t,T]}$ | Tiled (suffix-max) operator error | `tiled_op_sup_error` `[K+1]` |
| $\|u_\theta^n - u^*\|$ | Pointwise control error on `xs_op` | `u_error_op_fields` `[K+1][n_op]` |
| $\sup_x\|u_\theta^n - u^*\|$ | Sup control error on `xs_op` | `u_sup_error_op` `[K+1]` |
| $\|u_\theta^n - T(u_\theta^{n-1})\|$ | T-implementation residual on `xs_op` | `u_vs_Tu_fields` `[K+1][n_op]` |
| $\sup_x\|u_\theta^n - T(u_\theta^{n-1})\|$ | Sup T-impl residual at each $t$ | `u_vs_Tu_sup` `[K+1]` |
| $\|u_\theta^{n+1} - T(u_\theta^n)\|_{[t,T]}$ | Tiled T-impl residual | computed in plot from `u_vs_Tu_sup` |
| $u^*(t,x)$ signed on `xs_op` | Analytic control on operator grid | `u_star_op_fields` `[K+1][n_op]` |

### 5.2 Path-based u^{n+1} vs T(u_n) metric (§3.1)

The key insight is that the grid-based residual $\|u_\theta^{n+1} - T(u_\theta^n)\|$ may be large in regions that $u_\theta^{n+1}$ never visits during training. The path-based metric evaluates the residual **where $u_\theta^{n+1}$ is actually used**:

$$\mathrm{PathResid}(t) = \mathbb{E}\!\left[\left\|u_\theta^{n+1}(t,\,X^{u_\theta^{n+1}}_t) - T(u_\theta^n)(t,\,X^{u_\theta^{n+1}}_t)\right\|\right] \approx \frac{1}{N}\sum_{i=1}^N \left\|u_\theta^{n+1}(t,X^{u_\theta^{n+1}}_{t,i}) - T(u_\theta^n)(t,X^{u_\theta^{n+1}}_{t,i})\right\|$$

where the paths $X^{u_\theta^{n+1}}_{t,i}$ are sampled from the current control.

| Metric | Description | Stored key |
|--------|-------------|-----------|
| $\mathrm{PathResid}(t)$ | Mean path-based T-impl residual | `path_u_vs_Tu` `[K+1]` |

**Implementation notes:**
- Requires the **previous network weights** $u_\theta^n$ (saved as `prev_net_for_op` via `copy.deepcopy`)
- Only available starting from the **second** eval checkpoint (needs a previous network)

### 5.3 Path-based u^{n+1} vs u* metrics ("Assessing Convergence for the Learned Operator")

Direct comparison of the learned control against the optimal control $u^*$ along the current control's trajectories. Analogous to §5.2 but replacing $T(u_\theta^n)$ with the analytic $u^*$:

$$\mathrm{PathUstar}(t) = \mathbb{E}_{X^{u_\theta^{n+1}}_t}\!\left[\left\|u_\theta^{n+1}(t,\,X^{u_\theta^{n+1}}_t) - u^*(t,\,X^{u_\theta^{n+1}}_t)\right\|\right]$$

The associated ratios divide by $\mathbb{E}_{t,u^{n+1}_\theta}[\|u^n_\theta - u^*\|]_{[t,T]}$ — **no** $\|\sigma\|$ factor (latest notes, eq:ratio-pw-over-tiled / eq:ratio-tiled-over-tiled) — where the denominator is also a path-based MC estimate under the **same** current control $u^{n+1}_\theta$:

$$\text{per-}t\text{ ratio} = \frac{\mathrm{PathUstar}^{n+1}(t)}{\sup_{s\in[t,T]}\mathrm{PathUstar}_\mathrm{prev}^{n+1}(s)}, \qquad \text{tiled ratio} = \frac{\sup_{s\in[t,T]}\mathrm{PathUstar}^{n+1}(s)}{\sup_{s\in[t,T]}\mathrm{PathUstar}_\mathrm{prev}^{n+1}(s)}$$

where $\mathrm{PathUstar}_\mathrm{prev}^{n+1}(t) = \mathbb{E}_{X^{u^{n+1}_\theta}_t}[\|u^n_\theta(t,X) - u^*(t,X)\|]$ is the path-based MC estimate of the **previous** iteration's error at the **current** control's states.

Both ratios should $\to 0$ as $t \to T$ if the contraction bound holds.

| Metric | Description | Stored key |
|--------|-------------|-----------|
| $\mathrm{PathUstar}(t)$ | Per-$t$ path-based control error vs $u^*$ | `path_u_vs_ustar` `[K+1]` |
| $\mathrm{PathUstar}_\mathrm{prev}(t)$ | Per-$t$ path-based error of $u^n_\theta$ vs $u^*$ at current states | `path_u_prev_vs_ustar` `[K+1]` |

**Implementation notes:**
- `theta_traj` (shared with §5.2) is simulated once per eval; `path_u_vs_ustar` is always available
- `path_u_prev_vs_ustar` requires `prev_net_for_op` (available from second eval checkpoint onward); evaluates `prev_net_for_op` and `u_star_fn` at the same `theta_traj` states
- Both ratio plots use quantities from a **single** snapshot (not consecutive pairs)

---

## 6. Implementation notes

### Training objective (`algorithm.objective`)

| value | loss | inner-step data |
|---|---|---|
| `ram` (default) | `ram_loss` — reciprocal AM, target $-\sigma(t)\nabla g(X_1)$, $X_t\sim p_{t\|1}^{\text{base}}(\cdot\|X_1)$, weight $\lambda(t)=1/\sigma(t)^2$ | minibatch of $(X_1,\nabla g)$ from the replay buffer |
| `am` | `am_loss` — reference $L_\text{AM}$: $\tfrac12\sum_n\lVert u_\theta(X_n,t_n)+\sigma(t_n)\nabla g(X_1)\rVert^2\,\Delta t$ along stopgrad **controlled** trajectories. Exact here — the base drift is $0$ so the lean adjoint $\tilde a(t)\equiv\nabla g(X_1)$. **No $\lambda(t)$ weight** (deliberate; reconcile before quantitative RAM-vs-AM comparison). | minibatch of full trajectories rolled out once per outer iteration via `sampler.sample_trajectory` (buffer bypassed) |

Both share the outer/inner cadence, frozen-$\bar u$ semantics, iteration indexing,
and eval path. `am` costs ~`sampler.steps`× the per-inner-step network evals.

### Core functions

- `operator_diff_shared_bm_field(u_fn, v_fn, xs, ...)` — both-rollout shared-BM MC
  estimate `‖E_B[σ(∇g(X_T^u) − ∇g(X_T^v))]‖` (secondary comparison only, §3)
- `operator_field(u_fn, xs, t_start, ts, ...)` — T(u)(t,x) on a 1-D grid at one t → `[n_grid, d]`
- `operator_field_all_times(u_fn, xs, ts, ...)` — T(u)(t_k,x) for **all** t_k in one batched EM
  pass → `[K+1, n_grid, d]`; wraps `adjoint_sampling.operator.operator_grid_field_all_times`.
  The op-eval block uses this (plus one batched `optimal_control` / `net` grid call) instead of
  a per-slice loop — ~1.5× on the fast-iteration config.
- `operator_field_at_points(u_fn, x_states, t_start, ts, ...)` — T(u)(t,x) at arbitrary `[n_pts, d]` states (path-based metric)
- `adjoint_sampling.operator.fixed_point_residual_field(u_star_fn, grad_g_fn, xs, ts, ...)` — `‖T(u*)−u*‖` mesh (§5.0); shared across experiments
- `simulate_paths(control_fn, n_paths, ts, d, sigma_fn, device)` — EM rollout → `[K+1, n_paths, d]`

### Config keys under `eval`

| Key | Purpose |
|-----|---------|
| `op_eval` | Enable/disable operator T eval — also hosts the sanity-check metrics (§3–§5.1) |
| `n_op_mc_samples` | MC draws for P(u)(t,x) estimation |
| `n_op_grid` | x-grid points for the operator / sanity-check grid (was `n_same_bm_grid`) |
| `op_every` | Frequency for operator evaluation |
| `sanity_ratio_only` | Minimal output: only the §"Sanity Check" plots — `same_bm/{sbm_ratio, sbm_ratio_learned, sanity_ratio_heatmap, u_star_fixed_point_residual}.png` (the last only if `fixed_point_check`); every other plot skipped |
| `tiled_sup_percentile` | Robust "sup" for **every** tiled / sup-norm: the p-th percentile instead of the exact max, applied over $x$ (grid) **and** over the $[t,T]$ suffix. `100` = exact max (default). Lower (e.g. 95) suppresses single-node / single-slice outliers. Stored in `metrics.json`; the plots (via `set_suffix_percentile`) use the same value. |
| `both_rollout_compare` | Also MC-estimate the both-rollout shared-BM LHS for comparison (§3) |
| `n_both_rollout_samples` | Shared-BM draws for that comparison (default: `n_op_mc_samples`) |
| `n_sample_paths` | Paths for terminal distribution plots |
| `n_path_op_samples` | Paths under $u^{n+1}_\theta$ for path-based residuals (§5.2 and §5.3); kept small (default 64) for cost |
| `fixed_point_check` | Enable/disable the analytic-$u^*$ fixed-point sanity check (§5.0) |
| `n_fixed_point_mc` | MC rollouts per $(t,x)$ node (default 512; residual $\propto n_\text{mc}^{-1/2}$) |
| `n_fixed_point_grid` | $x$-grid resolution for the residual heatmap (default 121) |

### Data stored per eval snapshot

- Sanity-check core: `op_error_fields`, `op_sup_error`, `tiled_op_sup_error` (§5.1 table)
- `sanity_ratio_pointwise`: `[K+1][n_op]` — `op_error_fields / ‖u_θ−u*‖_{[t,T]}` (no ‖σ‖ factor)
- `sanity_ratio_tiled`: `[K+1]` — `tiled_op_sup_error / ‖u_θ−u*‖_{[t,T]}` (no ‖σ‖ factor)
- `bothroll_lhs_fields`: `[K+1][n_op]` — both-rollout shared-BM LHS (comparison); `None` if disabled
- `tiled_bothroll_lhs`: `[K+1]` — suffix-max of the above
- Grid-based operator metrics: see §5.1 table
- `path_u_vs_Tu`: `[K+1]` — path-based T-implementation residual (§5.2); `None` on first eval
- `path_u_vs_ustar`: `[K+1]` — path-based control error vs $u^*$ (§5.3); available from first eval
- `path_u_prev_vs_ustar`: `[K+1]` — path-based error of $u^n_\theta$ vs $u^*$ at current states (§5.3); `None` on first eval

**Top-level of `metrics.json`** (not per-snapshot): `fixed_point_check =
{xs [n_grid], residual [K+1][n_grid], p_u_star [K+1][n_grid], u_star [K+1][n_grid], n_mc}`
— the §5.0 analytic-$u^*$ fixed-point residual, computed once.

### Plot outputs (in `convergence/` subfolder — §3.1 "Assessing Convergence for the Learned Operator")

| File | Content |
|------|---------|
| `learned_error_heatmap.png` | Pointwise $\|u_\theta^{n+1}(t,x)-u^*\|$ over $(t,x)$ on xs_op; one panel per eval |
| `learned_tiled_sup_error.png` | $\|u_\theta^{n+1}-u^*\|_{[t,T]} = \sup_{s\in[t,T],x}\|u_\theta^{n+1}(s,x)-u^*(s,x)\|$; one curve per eval |
| `learned_pointwise_over_tiled.png` | Ratio $\|(u_\theta^{n+1}-u^*)(x,t)\| / \|u_\theta^n-u^*\|_{[t,T]}$ over $(t,x)$; one panel per consecutive pair. Colour: `RdBu_r`, **linear** scale $[0, 2]$ (white $= 1$, the contraction threshold; blue $<1<$ red), evenly-spaced colourbar ticks; `vmax` grows past 2 (with `extend="max"`) only if the 98th-pctile ratio exceeds it |
| `learned_pointwise_over_tiled_traj.png` | Same as above with a subsample (~40) of trajectories under the **source** control $u_\theta^n$ (snapshot $n$'s `paths_theta`) overlaid on each panel (thin translucent black). `overlay_traj=True`; needs `eval.n_sample_paths > 0` |
| `learned_tiled_over_tiled.png` | Ratio $\|u_\theta^{n+1}-u^*\|_{[t,T]} / \|u_\theta^n-u^*\|_{[t,T]}$ vs $t$; → 0 as $t\to T$ if bound holds |
| `path_u_vs_ustar.png` | $\mathbb{E}_{X^{u^{n+1}}_t}[\|u_\theta^{n+1}-u^*\|]$ — path-based control error vs $u^*$; one curve per eval |
| `path_u_vs_ustar_per_t_over_tiled.png` | Per-$t$ ratio $\mathrm{PathUstar}^{n+1}(t) / \sup_{s\in[t,T]}\mathrm{PathUstar}^{n+1}_\mathrm{prev}(s)$; one curve per snapshot with `path_u_prev_vs_ustar` |
| `path_u_vs_ustar_tiled_over_tiled.png` | Tiled ratio $\sup_s\mathrm{PathUstar}^{n+1}(s) / \sup_s\mathrm{PathUstar}^{n+1}_\mathrm{prev}(s)$; one curve per snapshot with `path_u_prev_vs_ustar` |

**Notes on the ratios:**
- Denominator: $\|u_\theta^n-u^*\|_{[t,T]}$ — **no** $\|\sigma\|$ factor (latest notes; matches `sbm_ratio` / `sbm_ratio_learned`).
- Grid-based ratios use xs_op grid (suffix-max of `u_sup_error_op` for tiled quantities).
- Path-based ratios use `path_u_vs_ustar` (suffix-max for tiled); denominator reuses `u_sup_error_op` suffix-max.
- Consecutive pairs: iter $n$ supplies the denominator, iter $n+1$ the numerator.

### Plot outputs (in `same_bm/` subfolder — § "Sanity Check")

| File | Content |
|------|---------|
All sourced from `op_error_fields` (`‖P(u_θ^n) − u*‖`, analytic $u^*$) unless noted.

| `same_bm_lhs_curves.png` | $\mathrm{mean}_x\,\|P(u_\theta^n)(t,x)-u^*(t,x)\|$ vs $t$; one curve per op-eval |
| `same_bm_lhs_heatmaps.png` | $\|P(u_\theta^n)(t,x)-u^*(t,x)\|$ over $(t,x)$; one panel per op-eval |
| `tiled_same_bm_lhs.png` | 2 panels. **Left** (linear y): `tiled_op_sup_error` (solid), `tiled_bothroll_lhs` (dotted, both-rollout comparison), grey dashed $= c\,\|u_\theta-u^*\|_{[t,T]}\sqrt{T-t}$ (LS-fit). **Right**: log–log vs $\tau=T-t$ — tiled $\sup_{s\ge t,x}$ (solid), pointwise $\sup_x$ = `op_sup_error` (dashed), slope-$\tfrac12$ guide |
| `sbm_ratio.png` | Left: $\mathrm{mean}_x$ of `sanity_ratio_pointwise`; right: `sanity_ratio_tiled`. Both $\to 0$ as $t\to T$ |
| `sbm_ratio_learned.png` | **Exact analogue of `sbm_ratio` with the learned control in the numerator instead of $P(u_n)$.** Per consecutive op-eval pair $n\to n{+}1$: $\|u_\theta^{n+1}(t,x)-u^*\|\,/\,\|u_\theta^n-u^*\|_{[t,T]}$ — a per-step contraction factor for the outer update, **no** $\|\sigma\|_{[t,T]}$ factor (same normalisation as `sbm_ratio`). Left: $\mathrm{mean}_x$ pointwise (`u_error_op_fields`); right: tiled $\|u_\theta^{n+1}-u^*\|_{[t,T]}/\|u_\theta^n-u^*\|_{[t,T]}$ (`u_sup_error_op` suffix-max, prev vs current). $<1 \Rightarrow$ the step contracts the error at that $t$; grey dotted $=1$ |
| `u_star_fixed_point_residual.png` | §5.0. **Left:** heatmap $\|P(u^*)(t,x)-u^*(t,x)\|$ over the $(t,x)$ grid. **Right:** $\sup_x$ / $\mathrm{mean}_x$ vs $t$. $\ll 1$ everywhere, $=0$ at $t=T$; title shows max residual and $n_\text{mc}$ |

### Plot outputs (in `control/` subfolder)

| File | Content |
|------|---------|
| `heatmap_u_star.png` | $(t,x)$ heatmaps of $u^*$ and the final $u_\theta$ (no trajectory overlay) |
| `heatmap_control_evolution.png` | $u_\theta(t,x)$ over training, one panel per eval (no trajectory overlay) |
| `heatmap_P_vs_next_control.png` | Per consecutive op-eval pair $(n, n{+}1)$, a row of four $(t,x)$ heatmaps: **(1)** $P(u_\theta^n)$ (`T_u_fields[n]`), **(2)** $u_\theta^{n+1}$ (`u_theta_op_fields[n+1]`), **(3)** $P(u_\theta^n) - u_\theta^{n+1}$ (signed), **(4)** $u_\theta^{n+1} - u^*$ (signed, `u_star_op_fields`). **One** robust (99th-pctile) symmetric colour scale shared by *every* panel — controls and residuals alike — so relative error sizes are apparent; a colourbar on each row. All on `xs_op`. |
| `heatmap_P_vs_next_control_traj.png` | Same figure as above, with a subsample (~40) of trajectories rolled out under the **source** control $u_\theta^n$ (snapshot $n$'s `paths_theta`) overlaid on every panel of that row (thin translucent black) — shows where the process the operator $P(u_\theta^n)$ samples actually visits. `overlay_traj=True`; d=1 only; needs `eval.n_sample_paths > 0`. |

### Plot outputs (in `operator/` subfolder)

| File | Content |
|------|---------|
| `operator_error_curves.png` | $\|T(u_\theta^n)-u^*\|_\infty$ (solid) vs $\|u_\theta^n-u^*\|_\infty$ (dashed); left=pointwise, right=tiled |
| `operator_vs_learned.png` | Same quantities on xs_op, plus ratio $\|T-u^*\|/\|u-u^*\|$ |
| `u_vs_Tu_prev.png` | $\|u_\theta^n-T(u_\theta^{n-1})\|$ (dashed) vs $\|u_\theta^n-u^*\|$ (solid) |
| `u_vs_Tu_norm.png` | $\|u_\theta^{n+1}-T(u_\theta^n)\|_{[t,T]}$ only — tiled T-implementation residual |
| `path_u_vs_Tu.png` | $\mathbb{E}_{X^{u^{n+1}}_t}[\|u_\theta^{n+1}-T(u_\theta^n)\|]$ — path-based residual |
| `Tu_vs_u_next_heatmap.png` | Signed heatmap $T(u_\theta^n)(t,x) - u_\theta^{n+1}(t,x)$ |

### Notes on grids

- `xs_op`: centred at $(\mu_1+\mu_2)/2$, half-width `linf_x_range`, `n_op_grid` points —
  used by the operator T eval and all sanity-check metrics (§3–§5.1)
- `xs_linf`: same range, `n_linf_grid` points; used for older metrics (abs_l2, abs_linf, control heatmaps)
- The §5.0 fixed-point check builds its own grid (`n_fixed_point_grid` points, same range)
- The path-based metrics use states sampled from the actual controlled process — no fixed grid

### Boundary condition

$T(u)(T,x) = u^*(T,x) = -\sigma(T)\nabla g(x)$ for **all** $u$ (Feynman-Kac at $t=T$). Hence `op_sup_error[K] = 0` exactly.

---

## 7. Revision history

### 2026-08-29 — `algorithm.objective`: reference $L_\text{AM}$ option

New config key `algorithm.objective` ∈ {`ram` (default), `am`}. `am` trains with
`am_loss` along stopgrad controlled trajectories (`sampler.sample_trajectory`),
resampled per inner step from an outer-iteration cache; the replay buffer is
bypassed. Exact for this experiment (zero base drift ⇒ lean adjoint is constant
$\nabla g(X_1)$). `am_loss` carries **no** $\lambda(t)=1/\sigma(t)^2$ weight that
`ram_loss` has — left as-is on request; must be reconciled before a quantitative
RAM-vs-AM comparison. Training log / W&B key becomes `{objective}_loss`. See §6.

### 2026-08-29 — `optimal_control`: missing $\kappa_i$ term in the h-transform mixture

`log_A1`/`log_A2` were missing the mode-dependent constant
$\kappa_i = d\,\lambda_i\mu_i^2/(2\nu_1\lambda_i^*)$ that completing the square in
$-\lambda_i(x_1-\mu_i)^2/2 + x_1^2/(2\nu_1)$ produces. It shifts the mixture
weights $n_1,n_2$ and cancels only for symmetric mixtures ($\mu_1^2=\mu_2^2$,
equal $\lambda$).

- **Symptom:** for asymmetric means (e.g. $\mu_1=-3,\mu_2=0$) the analytic $u^*$
  violated the terminal identity $u^*(1,x) = -\sigma(1)\nabla g(x)$ by up to
  $\sim 1.9$; all `‖P(u_θ)-u*‖` / RelL2 metrics for such runs were invalid.
- **Unaffected:** every symmetric-mean run (all experiments before 2026-08-29
  used $\mu=\pm 3$, equal $\lambda$). `grad_r` and `terminal_mixture_params` were
  always correct — only the analytic reference $u^*$ was wrong.
- **Fix:** `+ kappa1` / `+ kappa2` in `log_A1` / `log_A2`. Same one-line fix
  applied to `experiments/right_to_left_convergence_bimodal/run.py`.
- **Test:** `tests/test_optimal_control_boundary.py` — the terminal boundary
  identity across constant/linear schedules × {asymmetric means, weights,
  precisions, coincident modes}.
- Keeper batch runs `05_const2_meanasym` / `06_linfloor025_meanasym` rerun as
  `2026-08-29_keeper_05…` / `…06…`.

### 2026-08-28 — `learned_pointwise_over_tiled`: RdBu_r colour + trajectory variant

- Colour scale changed from `viridis` to `RdBu_r` on a **linear** norm over
  $[0, 2]$ — white sits at the contraction threshold 1, blue $\Rightarrow$ ratio
  $<1$ (step contracted the error there), red $\Rightarrow >1$. Evenly-spaced
  colourbar ticks (0.25 step); `vmax` only grows past 2 (`extend="max"`) if the
  98th-pctile ratio does. (An earlier `TwoSlopeNorm` version made the colourbar
  visually nonlinear and cramped the ticks above 1 — replaced.)
- New `learned_pointwise_over_tiled_traj.png` via
  `plot_learned_pointwise_over_tiled(..., overlay_traj=True)` — overlays a
  subsample of source-control ($u_\theta^n$) trajectories on each pair's panel
  (reuses `_overlay_source_traj`). Both versions emitted (run.py + `load_and_plot`).

### 2026-08-28 — `heatmap_P_vs_next_control_traj.png`: source-control trajectory overlay

New variant of `heatmap_P_vs_next_control.png` (same 4-column grid) that overlays
a subsample of trajectories rolled out under the **source** control $u_\theta^n$
of each pair (snapshot $n$'s stored `paths_theta`) on every panel of that row.
`plot_operator_vs_next_control(..., overlay_traj=True)` → `_traj.png`; module
helper `_overlay_source_traj`. d=1 only; needs `eval.n_sample_paths > 0`. Both
the plain and `_traj` versions are emitted (run.py full mode + `load_and_plot`).

### 2026-08-28 — §5.3 ratio denominators: $\|\sigma\|$ factor dropped

The latest notes write eq:ratio-pw-over-tiled and eq:ratio-tiled-over-tiled with
denominator $\|u_\theta^n-u^*\|_{[t,T]}$ **only** — no $\|\sigma(t)\|\cdot\|\sigma\|_{[t,T]}$
factor. Removed it from all four §5.3 ratio plots:
`convergence/{learned_pointwise_over_tiled, learned_tiled_over_tiled,
path_u_vs_ustar_per_t_over_tiled, path_u_vs_ustar_tiled_over_tiled}.png`. The
`sigma_factor` parameter is gone from those functions; `run.py` no longer
computes `sigma_sup_suffix` / `sigma_factor` and no longer stores `sigma_factor`
in `metrics.json`; `load_and_plot` drops its `sigma`-based fallback. These four
now use the same normalisation as `sbm_ratio` / `sbm_ratio_learned`. `sigma_grid`
is still stored for reference. (Old `metrics.json` files still load — the stored
`sigma_factor` key is simply ignored.)

### 2026-08-28 — `heatmap_P_vs_next_control.png`: $u_\theta^0$ header row removed

The standalone $u_\theta^0$ panel above the pairs is gone; the figure is now
just the `n_pairs × 4` grid of consecutive op-eval pairs. The colour limit no
longer includes $u_\theta^0$.

### 2026-08-28 — `sbm_ratio_learned.png`: learned-control analogue of `sbm_ratio`

New `plot_sbm_ratio_learned(snapshots, ts, output_dir)` →
`same_bm/sbm_ratio_learned.png`. Same two-panel layout and normalisation as
`plot_sbm_ratio`, but the numerator uses the **learned control** $u_\theta$
rather than the operator image $P(u_n)$, comparing successive iterations:

$$\frac{\|u_\theta^{n+1}(t,x)-u^*(x,t)\|}{\|u_\theta^n-u^*\|_{[t,T]}}
\qquad\text{(no } \|\sigma\|_{[t,T]} \text{ factor)}.$$

Left = $\mathrm{mean}_x$ pointwise (from `u_error_op_fields`); right = tiled
$\|u_\theta^{n+1}-u^*\|_{[t,T]}/\|u_\theta^n-u^*\|_{[t,T]}$ (suffix-max of
`u_sup_error_op`, previous snapshot as denominator). One curve per consecutive
op-eval pair; grey dotted reference at $1$ ($<1 \Rightarrow$ per-step
contraction at that $t$). Honours `tiled_sup_percentile` via `_suffix_max`.
Wired into full mode, `sanity_ratio_only` mode, and `load_and_plot`. Differs
from the §5.3 `convergence/learned_*_over_tiled` plots only by dropping the
$\|\sigma(t)\|\cdot\|\sigma\|_{[t,T]}$ denominator factor and using the
`sbm_ratio` panel format.

### 2026-08-28 — pre-training snapshot: iteration index shifted by one

The outer loop now runs `for outer_it in range(-1, outer_iterations)`. The
`outer_it = -1` pass does **no** training — it records the freshly-initialised
network (and a full operator eval on it) as snapshot **`outer_it = 0`**. Every
training iteration's snapshot index is now `snap_it = outer_it + 1`, so:

- snapshot `0` = network at initialisation (≈ zero drift), `inner_steps = 0`;
- snapshot `n` (`n ≥ 1`) = network after the $n$-th outer loop.

This makes $u_\theta^n \approx P(u_\theta^{n-1})$ hold with the displayed indices:
the first pair of `heatmap_P_vs_next_control.png` is $P(u_\theta^0)$ (operator on
the init net) vs $u_\theta^1$ (after one outer step). W&B `outer_it` and the `[   n]` training
log line also use `snap_it`. Eval cadence gates on `snap_it` (`first_k`,
`every`, `op_every`). `plot_inner_convergence` skips snapshot 0 (empty curve);
`plot_inner_steps` shows it at `(0, 0)`.

*Existing runs predating this change are 1-indexed differently — rerun to get
the true `u_0`.*

### 2026-08-28 — `control/` plots reworked

- Trajectory overlays removed from `plot_optimal_control` and
  `plot_control_evolution` (`_overlay_paths` deleted). `plot_optimal_control`
  still accepts `paths_star` / `paths_theta` for call-site compatibility but
  draws nothing.
- New `plot_operator_vs_next_control(snapshots, ts, xs_op, d, output_dir)` →
  `control/heatmap_P_vs_next_control.png`: for each consecutive op-eval pair
  $(n, n{+}1)$, a row of four $(t,x)$ heatmaps — $P(u_\theta^n)$
  (`T_u_fields[n]`), $u_\theta^{n+1}$ (`u_theta_op_fields[n+1]`),
  $P(u_\theta^n) - u_\theta^{n+1}$ (signed), $u_\theta^{n+1} - u^*$ (signed,
  `u_star_op_fields`). One robust (99th-pctile) symmetric colour scale shared by
  every panel (controls and residuals), colourbar per row — the residual columns
  render on the control scale, so relative error size is visible. Tests
  $u_\theta^{n+1} \approx P(u_\theta^n)$ visually.
- Wired into `run.py` (`if xs_op_list:`) and `load_and_plot`.

### 2026-08-28 — `eval.tiled_sup_percentile`: robust sup for tiled norms

Every "sup" in the tiled / sup-norm computations can now be a **p-th percentile**
instead of the exact max — over $x$ (the grid) **and** over the $[t,T]$ time
suffix — controlled by `eval.tiled_sup_percentile` (default `100` = exact max).
Motivation: the true max can be dominated by a single high-variance grid node
(near $x=(\mu_1+\mu_2)/2$) or a single time slice, which then propagates left
through the suffix-max.

- **`run.py`** — `_grid_sup(field, pct)` (over $x$) and `_suffix_sup(vals, pct)`
  (over the $[t,T]$ suffix); `_sup_over_x` for torch tensors (`torch.quantile`).
  Applied to `op_sup_error`, `u_sup_error_op`, `u_vs_Tu_sup`, `abs_linf`'s sup,
  and the tiled quantities `tiled_op_sup_error`, `tiled_u_sup_error_op`,
  `tiled_al_inf_vals`, `tiled_error_fields` (per-$x$), `tiled_bothroll`. The
  sanity-check ratios inherit it through their numerator/denominator.
- **`plotting.py`** — `_suffix_max` now honours a module global `_SUFFIX_PCT`
  (percentile-of-suffix); `set_suffix_percentile(pct)` sets it. All 8 inline
  `np.maximum.accumulate(…[::-1])[::-1]` recomputes route through it. `run.py` and
  `load_and_plot` call `set_suffix_percentile` from the config / stored value.
- At `pct = 100` the result is bit-identical to before. At `pct < 100` the tiled
  quantity is no longer guaranteed monotone in $t$ (a percentile of a shrinking
  suffix), which is expected.

### 2026-08-28 — faster iteration: minimal-plot mode, ratio update, batched op-eval

- **Ratio denominator** — the two §"Sanity Check" ratios now divide by
  $\lVert u_n-u^*\rVert_{[t,T]}$ **only** (the $\lVert\sigma\rVert_{[t,T]}$ factor is
  dropped, per the notes). `sanity_ratio_pointwise` / `sanity_ratio_tiled` and
  `plot_sbm_ratio` labels updated. (The §5.3 `learned_*_over_tiled` ratios still
  carry the $\lVert\sigma(t)\rVert\lVert\sigma\rVert_{[t,T]}$ factor — different section.)
- **`eval.sanity_ratio_only`** (bool, default false) — every plot skipped except
  the §"Sanity Check" set: `same_bm/{sbm_ratio, sanity_ratio_heatmap,
  u_star_fixed_point_residual}.png` (the fixed-point one still runs if
  `fixed_point_check`); `return`s right after. ~35–70 s with the fast overrides
  (the fixed-point check is the bulk — lower `n_fixed_point_mc` / `n_fixed_point_grid`
  or set `fixed_point_check=false` to drop to ~25 s).
- **`plot_sanity_ratio_heatmap`** — new: one $(t,x)$ heatmap per iteration of the
  pointwise ratio $\lVert P(u_n)(t,x)-u^*\rVert/\lVert u_n-u^*\rVert_{[t,T]}$
  (`same_bm/sanity_ratio_heatmap.png`). Wired into full mode + `load_and_plot`.
- **Batched operator eval** — `operator_grid_field_all_times` (src) rolls out
  every time slice in one EM pass; the op-eval block now also does one batched
  `optimal_control` / `net` grid call and a vectorised `u_vs_Tu`. Per-eval network
  calls drop from $O(K^2)$ to $O(K)$; fast-config run ~37 s → ~23 s (neutral at
  keeper config, which is FLOP-bound). Results statistically identical; $t_K$ exact.
- **Fast-iteration recipe** (~30–70 s for 10 iters):
  `eval.sanity_ratio_only=true eval.n_time_slices=40 sampler.steps=40 eval.n_op_grid=40`
  `eval.n_op_mc_samples=48 eval.both_rollout_compare=false eval.n_metric_samples=256`
  `eval.n_path_op_samples=32 eval.n_linf_grid=80 logging.wandb=false`. Run several
  schedules in parallel with `OMP_NUM_THREADS=3` each.

### 2026-08-28 — linear noise schedule support

- **`utils.py`** — `sigma_linear`, `nu_linear`, `make_noise_schedule(name, sigma)`
  → `(sigma_fn, nu_fn, nu_1)`. `src/adjoint_sampling/CLAUDE.md` §6.
- **`run.py`** — reads `cfg.sigma_schedule` ("constant" | "linear"); `sigma_integral`
  → `make_sigma_int_fn(nu_fn, nu_1)` computing $\Sigma_t = \nu_1 - \nu_t$
  (schedule-agnostic). `optimal_control` + the 5 metric helpers thread a
  `sigma_int_fn` arg. `nu_1` from the schedule (not `cfg.sigma**2`).
- **§5.3 ratio denominators** — the 4 `plot_learned_*_over_tiled` /
  `plot_path_u_vs_ustar_*_over_tiled` functions now take a per-$t$ array
  `sigma_factor[k] = \|\sigma(t_k)\|\cdot\|\sigma\|_{[t_k,T]}$ (was scalar $\sigma_0^2$);
  computed in `run.py`, stored in `metrics.json` (`sigma_factor`, `sigma_grid`,
  `sigma_schedule`), falls back to $\sigma_0^2$ for old runs. The §3–§5.1
  sanity-check ratios already used `sigma_sup_suffix` (correct for any schedule).
- **`plot_sbm_ratio`** lost its now-unused `sigma` arg.
- Config: `+sigma_schedule: constant` (default; pass `sigma_schedule=linear` for
  linear runs).
- **Constraint**: `linear` with $\sigma_0=1$ needs $\lambda_i > 3$; with $\sigma_0=2$,
  $\lambda_i > 0.75$ (current $\lambda_i=1$ ok).

### 2026-08-27 — sanity check folded into §5.1 (analytic-$u^*$ substitution)

Per the notes' updated Sanity Check: the ratio numerators replace the second MC
rollout ($\sigma\mathbb{E}[\nabla g(X_T^{u^*})]$) with the analytic $u^*$ (operator
is identity on $u^*$), for accuracy. In this codebase's sign convention that
numerator **equals** $\lVert P(u_\theta) - u^*\rVert$ = the existing
`op_error_fields` (§5.1). Changes:

- **`run.py`** — `same_bm_lhs_field` → renamed `operator_diff_shared_bm_field`,
  kept only as the both-rollout shared-BM *comparison* (secondary). The standalone
  `same_bm` eval block + `do_same_bm` grid removed. The op-eval block now also
  computes `sanity_ratio_pointwise`, `sanity_ratio_tiled` (÷ $\|\sigma\|_{[t,T]}\|u_\theta-u^*\|_{[t,T]}$)
  and, if `both_rollout_compare`, `bothroll_lhs_fields` / `tiled_bothroll_lhs` on
  `xs_op` at the op-eval cadence.
- **Config** — removed `same_bm_eval`, `n_same_bm_samples`, `same_bm_every`;
  `n_same_bm_grid` → `n_op_grid`; added `both_rollout_compare`, `n_both_rollout_samples`.
- **`plotting.py`** — `plot_same_bm_lhs_curves` / `_heatmaps` / `plot_tiled_same_bm_lhs`
  / `plot_sbm_ratio` re-sourced from `op_error_fields` / `op_sup_error` /
  `tiled_op_sup_error` / `sanity_ratio_*` / `tiled_bothroll_lhs`. `plot_sbm_ratio`
  gained a `sigma` arg. `tiled_same_bm_lhs.png` left panel now overlays the
  both-rollout comparison (dotted). Filenames + `same_bm/` folder kept stable for
  the notes' figure refs.
- Removed stored keys: `same_bm_lhs_fields`, `sbm_error_fields`, `sbm_ratio_fields`,
  `tiled_same_bm_lhs`, `tiled_sbm_ratio`, `xs_sbm`.
- **§5.0 fixed-point check** (`u_star_fixed_point_residual.png`) is the notes' new
  first check $\lVert P(u^*)-u^*\rVert\ll 1$; unchanged this round.

### 2026-08-27 — norm moved outside the MC expectation (§ "Sanity Check" reformulation)

The notes' "Sanity Check" paragraph was reformulated so that the norm sits
**outside** the Monte-Carlo expectation (`‖E_B[·]‖` instead of `E_B[‖·‖]`), per the
margin remark *"[MG: Bound in the numerator should be outside!]"*. Changes:

- **`run.py` · `same_bm_lhs_field`** — return `‖mean_B[σ(t)(∇g(X_T^u) − ∇g(X_T^v))]‖`
  (was `mean_B[‖σ(t)(∇g(X_T^u) − ∇g(X_T^v))‖]`). Docstring + module docstring updated.
- **`run.py` · `diff_bm_lhs_field`** — **deleted**. With the norm outside E, the
  independent-BM estimator has the same expectation as the shared-BM one (linearity),
  so it added nothing but variance. All `dbm_*` / `tiled_diff_bm_lhs` stored keys,
  wandb metrics, and the `diff_bm_lhs_field` call site in `main()` removed.
- **`plotting.py`** — `plot_same_vs_diff_bm_lhs` **deleted** (and its call sites in
  `run.py` and `load_and_plot`). `plot_same_bm_lhs_curves`, `plot_same_bm_lhs_heatmaps`,
  `plot_tiled_same_bm_lhs`, `plot_sbm_ratio`: titles/labels updated to `‖E_B[·]‖` and to
  `∇g` (they previously wrote `∇r`); `plot_sbm_ratio` diff-BM curves + legend removed.
- **This file** — §1, §3, §4, §6 rewritten as above; §3 "Relation to the operator"
  changed from a Jensen inequality to an exact identity
  ($\mathrm{LHS}_{\mathrm{SBM}} = \|T(u_\theta) - u^*\|$).
- **Unchanged:** config keys (`same_bm_eval`, `n_same_bm_*`, `same_bm_every`), stored
  key names (`same_bm_lhs_fields`, `tiled_same_bm_lhs`, `sbm_ratio_fields`,
  `tiled_sbm_ratio`), the `same_bm/` output subfolder, and the shared-BM rollout
  itself (now a variance-reduction device).
- **Open flag:** notes write $\nabla r$, code keeps $\nabla g$ — see §1.
- **Operator-T section (§5) untouched** — it corresponds to a different notes
  section ("Comparing the Analytic Operator with the Learned Iteration").

### 2026-08-27 — ratio normalisation aligned with the notes

The sanity-check ratios (`sbm_ratio_fields`, `tiled_sbm_ratio`) now divide by
$\|\sigma\|_{[t,T]}\cdot\|u_\theta-u^*\|_{[t,T]}$, matching the notes. Changes:

- **`run.py` · `main()`** — added `sigma_sup_suffix[k] = \sup_{s\ge t_k}\sigma(s)`
  (suffix-sup over `ts_eval`; $=\sigma_0$ for the constant schedule). New
  `ratio_denom[k] = max(sigma_sup_suffix[k] * tiled_al_inf_sbm[k], eps)` is used for
  **both** ratios. Previously: pointwise ratio divided by the *pointwise* error
  $\|u_\theta-u^*\|(t,x)$ and the tiled ratio by $\|u_\theta-u^*\|_{[t,T]}$, both
  without the $\|\sigma\|_{[t,T]}$ factor.
- **`plotting.py` · `plot_sbm_ratio`** — y-labels, panel titles, suptitle and
  docstring updated to show the $\|\sigma\|_{[t,T]}\|u_\theta-u^*\|_{[t,T]}$ denominator.
- **This file** — §4, §6 updated; the earlier "Ratio normalisation (flagged)" note removed.
- **Effect:** the $\to 0$ trend is unchanged; the ratio magnitude drops by roughly
  $\sigma_0$ (pointwise ratio also changes shape since its denominator is now the
  tiled control-error norm).

### 2026-08-27 — log-scale y-axes: applied, then reverted

A blanket `ax.set_yscale("log")` was applied to every line plot and then
**reverted** at the user's request (linear y restored, `set_ylim(bottom=0)` back).
Still open: how the y-scaling of the convergence/error/ratio plots should be
handled. Pre-existing exceptions unchanged: `plot_convergence` and
`plot_inner_convergence` keep the log-y they already had.
**Retained** from that work: the `_tighten_log_ylim(ax)` helper and the log–log
right panel of `tiled_same_bm_lhs.png` (see next entry).

### 2026-08-27 — `tiled_same_bm_lhs.png`: $\sqrt{T-t}$ reference + log–log slope panel

The notes' Sanity-Check bound has a $2\sqrt{T-t}$ factor, so the expected tail
shape of $\mathrm{TiledLHS}(t)$ is $\propto\sqrt{T-t}$ (near $t\to T$, where
$e^{C(T-t)}\to 1$ and $\|u_\theta-u^*\|_{[t,T]}$ flattens).

- **Grey dashed reference corrected**: was $\|u_\theta-u^*\|_{[t,T]}\cdot(T-t)$ (linear),
  now $c\cdot\|u_\theta-u^*\|_{[t,T]}\cdot\sqrt{T-t}$ with $c$ a least-squares amplitude
  (was rescaled to match at $t=0$).
- **New right panel (retained)**: log–log vs $\tau=T-t$. This is the axis scale on
  which a $\sqrt{T-t}$ law is legible — a power law $y\propto\tau^p$ is a straight
  line of slope $p$, so $\sqrt{}$ ⇔ slope $\tfrac12$ (dotted guide shown). Plots
  both the tiled $\sup_{s\ge t}$ (solid, plateaus wherever an interior $s$
  dominates the suffix-max) and the pointwise $\sup_x$ (dashed, decays toward $0$
  as $t\to T$).
- **Left panel**: back to linear y (per the axis-scaling revert); keeps the
  corrected $\sqrt{T-t}$ grey-dashed reference.
- Complementary check not plotted: the *compensated* curve
  $\mathrm{TiledLHS}(t)/\sqrt{T-t}$ vs $t$ should be flat where the law holds.
- Function signature unchanged; figure is now 1×2.

### 2026-08-27 — fixed-point sanity check for the analytic $u^*$ (§5.0)

New "very basic sanity check" from the notes: $\|T(u^*)(t,x)-u^*(t,x)\|\ll 1$.

- **New shared module `src/adjoint_sampling/operator.py`** — `operator_field`,
  `operator_grid_field`, `fixed_point_residual_field`; exported from
  `adjoint_sampling`. Documented in `src/adjoint_sampling/CLAUDE.md` §9.
- **`run.py`** — the local `operator_field` / `operator_field_at_points` are now
  thin wrappers over the shared module (dedup, same math). New once-per-run
  block computes `fixed_point_residual_field(u_star_fn, grad_g_fn, …)` and stores
  `fixed_point_check` at the top level of `metrics.json`; logs `max` / `mean` and
  a `fixed_point/*` W&B scalar.
- **`plotting.py`** — `plot_fixed_point_residual` → `same_bm/u_star_fixed_point_residual.png`;
  wired into `load_and_plot` (skipped for older `metrics.json` without the key).
- **Config** — `eval.fixed_point_check` (true), `eval.n_fixed_point_mc` (512),
  `eval.n_fixed_point_grid` (121).
- **Verified**: residual $=0$ at $t=T$ (to $\sim10^{-6}$), and $\propto n_\text{mc}^{-1/2}$
  elsewhere → the analytic $u^*$ is self-consistent; the visible residual is MC
  noise (largest near $x=(\mu_1+\mu_2)/2$). Bump `n_fixed_point_mc` for a cleaner
  heatmap.
- **Reusable**: `fixed_point_residual_field` works for any experiment with a
  closed-form $u^*$ (`right_to_left_convergence_unimodal` / `_bimodal`,
  `gaussian_baseline`) — pass that experiment's `u_star_fn` and `grad_g_fn`; not
  yet wired into those.

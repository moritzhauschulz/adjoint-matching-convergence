# Right-to-Left Convergence — Bimodal, Same Brownian Motion

## 1. Goal

Directly verify the LHS of the right-to-left contraction bound from §2.1 of the
project notes:

$$\mathbb{E}\!\left[\left\|\sigma(t)\nabla r(X_T^{u,x_t}) - \sigma(t)\nabla r(X_T^{v,x_t})\right\|\right] \leq C(t,T)\,\|u-v\|_{[t,T]}, \quad C(t,T) \to 0 \text{ as } t \to T$$

The bound requires that $X^u$ and $X^v$ are driven by the **same Brownian motion**.
The existing bimodal experiment measures $\|u_\theta - u^*\|$ (the RHS norm) but uses
separate rollouts for each control, violating this coupling assumption.

This experiment directly estimates the LHS by simulating $X^{u_\theta,x}$ and $X^{u^*,x}$
from the same starting point $x$ at time $t$ with shared BM increments.

---

## 2. Setting

Identical to `right_to_left_convergence_bimodal`. Constant noise schedule
$\sigma(t) = \sigma_0$, bimodal log-mixture reward, dimension $d=1$.
See that experiment's `CLAUDE.md` for the full reward spec, Feynman-Kac optimal
control $u^*$, effective parameters $\lambda_i^*$, $\mu_i^*$, and terminal distribution.

---

## 3. Same-BM LHS metric

For $u = u_\theta$ (learned) and $v = u^*$ (analytic Feynman-Kac optimal control),
the **same-BM LHS** at starting point $x$ and time $t$ is:

$$\mathrm{LHS}_{\mathrm{SBM}}(t, x) = \mathbb{E}_B\!\left[\left\|\sigma(t)\nabla r\!\left(X_T^{u_\theta,x}\right) - \sigma(t)\nabla r\!\left(X_T^{u^*,x}\right)\right\|\right]$$

where both SDEs share the same Brownian path $B_{t:T}$:

$$dX_s^{u,x} = \sigma(s)\,u(s, X_s^{u,x})\,ds + \sigma(s)\,dB_s, \qquad X_t^{u,x} = x$$

The expectation is approximated by Monte Carlo over `eval.n_same_bm_samples` shared BM draws.

### Relation to the operator

The operator is $T(u)(t,x) = -\sigma(t)\,\mathbb{E}[\nabla r(X_T^{u,x})]$, and $u^*$ is its
fixed point: $T(u^*) = u^*$. Jensen's inequality gives

$$\|T(u_\theta)(t,x) - u^*(t,x)\| \leq \mathrm{LHS}_{\mathrm{SBM}}(t,x)$$

so $\mathrm{LHS}_{\mathrm{SBM}}$ upper-bounds the pointwise operator difference.

### Boundary condition

At $t = T$: both SDEs start and end at $x$, so
$\nabla r(X_T^{u_\theta,x}) = \nabla r(X_T^{u^*,x}) = \nabla r(x)$,
and $\mathrm{LHS}_{\mathrm{SBM}}(T, x) = 0$ exactly.

### Expected behaviour

$\mathrm{LHS}_{\mathrm{SBM}}(t, x)$ should decrease as $t \to T$ (right-to-left),
as the residual influence of the control vanishes near the terminal time.

---

## 4. Theoretical bound (§2.1, Lipschitz $\nabla r$ case)

$$\mathrm{LHS}_{\mathrm{SBM}}(t,x) \leq \|\sigma(t)\|\cdot\mathrm{Lip}(\nabla r)\cdot(T-t)\cdot\|\sigma\|_{[t,T]}\cdot\|u_\theta - u^*\|_{[t,T]}$$

For constant $\sigma(t) = \sigma_0$:

$$\mathrm{LHS}_{\mathrm{SBM}}(t,x) \leq \sigma_0^2\cdot\mathrm{Lip}(\nabla r)\cdot(T-t)\cdot\|u_\theta - u^*\|_{[t,T]}$$

The factor $(T-t)$ gives the right-to-left decay.

---

## 5. Implementation notes

- `same_bm_lhs_field(u_fn, v_fn, xs, t_start, ts, sigma_fn, d, ...)` — core function
  - Expands the x-grid by `n_same_bm_samples`: shape `[n_grid * n_sbm, d]`
  - Generates one shared noise tensor per time step for both SDEs
  - Returns `[n_same_bm_grid]` tensor of per-point LHS estimates

- Config keys under `eval`:
  - `same_bm_eval` — bool, enable/disable the metric (default `true` in this experiment)
  - `n_same_bm_samples` — Monte Carlo draws of $B$ per starting point
  - `n_same_bm_grid` — number of $x$ grid points
  - `same_bm_every` — compute every $N$ outer iterations (can be expensive)

- `same_bm_lhs_fields` stored in each eval snapshot: `list[list[float]]` of shape
  `[K+1][n_same_bm_grid]`; `None` for skipped outer iterations.

- $\sigma(t)$ in the LHS formula is the noise schedule at the **evaluation time** $t$,
  not at the terminal time $T$.

- The `xs_sbm` grid is centred at $(\mu_1+\mu_2)/2$ with half-width `linf_x_range`,
  independently of the `xs_linf` grid used for AbsL∞.

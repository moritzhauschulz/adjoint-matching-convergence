# Right-to-Left Convergence Experiment

**Source**: Section 4.1 of `Project_Notes.tm` ("Verifying Right to Left Convergence").

## 1. Goal

The theoretical analysis (Section 2 of the notes) establishes that the operator $T$
satisfies the bound

$$\|(T(u) - T(v))(t,x)\| \leq C(t,T) \cdot \|u - v\|_{[t,T]}$$

where $C(t,T) \to 0$ as $t \to T$. This predicts **right-to-left convergence**: the
learned control $u_\theta$ should be closest to $u^*$ at later time slices (near $T=1$)
and deviate more at earlier times (near $t=0$).

The experiment verifies this prediction in a setting where $u^*$ is available in closed form.

---

## 2. Setting

- Base drift: $b = 0$
- Noise schedule: constant $\sigma(t) = \sigma_0$, $\nu_1 = \int_0^1 \sigma(t)^2\,dt = \sigma_0^2$
- Time horizon: $T = 1$, $t \in [0, 1]$
- State dimension: $d$ (experiment uses $d = 1$ initially for clarity)
- Operator (from Section 2 of notes, lean adjoint with $b=0$):

$$T(u)(t, x) = -\sigma(t)\,\mathbb{E}\!\left[\nabla g(X_T^u) \;\middle|\; X_t = x\right]$$

---

## 3. Reward and training objective

The log-density of a Gaussian at $\mu$ with precision $\lambda$ is

$$r(x) = -\frac{\lambda}{2}\|x\|^2, \qquad \nabla r(x) = -\lambda x$$

so $r(x-\mu) = -\frac{\lambda}{2}\|x-\mu\|^2$ (maximised at $x=\mu$).

The **full adjoint sampling objective** (arXiv:2504.11713, Algorithm 1) is:

$$\min_{u}\;\mathbb{E}\!\left[\int_0^1 \tfrac{1}{2}\|u(t, X_t)\|^2\,dt + g(X_1)\right]$$

where

$$g(x_1) = \log p_1^\text{base}(x_1) - r(x_1 - \mu) = -\frac{x_1^2}{2\nu_1} - r(x_1 - \mu) + \text{const}$$

The terminal cost gradient (stored as `grad_g` in the RAM loss replay buffer) is:

$$\nabla g(x_1) = -\frac{x_1}{\nu_1} + \lambda(x_1 - \mu) = \lambda^*(x_1 - \mu^*)$$

where the equality follows from completing the square with effective parameters

$$\lambda^* = \lambda - \frac{1}{\nu_1}, \qquad \mu^* = \frac{\lambda\,\mu}{\lambda^*}$$

**Condition**: $\lambda > 1/\nu_1$ is required (enforced by an assertion in code).

Under the full AS objective, the terminal distribution is exactly $p^{u^*}(x_1) \propto e^{r(x_1 - \mu)} = \mathcal{N}(x_1;\,\mu,\,1/\lambda)$, with no $\sigma_0$ dependence.

---

## 4. Analytic optimal control (Lemma, Section 4.1 of notes)

Define the Riccati coefficient (solved via HJB with terminal condition $g(x) = \lambda^*/2\,(x-\mu^*)^2 + \text{const}$):

$$a(t) = \frac{\lambda^*}{1 + \lambda^*\,\Sigma_t}, \qquad \Sigma_t = \int_t^1 \sigma(s)^2\,ds = \sigma_0^2(1-t)$$

The optimal control is **linear in $(x - \mu^*)$**:

$$u^*(t, x) = -\sigma(t)\,a(t)\,(x - \mu^*)$$

Note: the Riccati coefficient $a(t)$ uses $\lambda^*$ and the shift is $\mu^*$, not the original $(\lambda, \mu)$.

---

## 5. Distribution of $X_t$ under $u^*$

Under $u^*$, the SDE is

$$dX_t = -\sigma^2(t)\,a(t)\,(X_t - \mu^*)\,dt + \sigma(t)\,dB_t, \qquad X_0 = 0$$

This is a linear Gaussian SDE. Decompose $X_t = m_t + Y_t$ where $m_t = \mathbb{E}[X_t]$ and
$Y_t = X_t - m_t$ is the zero-mean fluctuation. The mean and variance satisfy:

**Mean ODE** ($m_0 = 0$, attractor $\mu^*$):

$$\frac{dm_t}{dt} = -\sigma^2(t)\,a(t)\,(m_t - \mu^*)$$

**Variance ODE** ($V_0 = 0$):

$$\frac{dV_t}{dt} = -2\sigma^2(t)\,a(t)\,V_t + \sigma^2(t)$$

Both are integrated numerically via forward Euler on the evaluation grid.

As $t \to 1$: by algebraic identity $\lambda\nu_1/(1+\lambda^*\nu_1) = 1$, the mean satisfies $m_1 = \mu$;
and $V_1 = 1/\lambda$. So the marginal is $X_1 \sim \mathcal{N}(\mu,\,1/\lambda)$ as expected.

> **Grid note**: the sup-norm x-grid is `linspace(μ - x_range, μ + x_range)`, centred at $\mu$
> (the target mean). `linf_x_range` controls the half-width.

---

## 6. Convergence metrics (Section 4.1 of notes)

Metrics are evaluated at **uniform time steps** $t_k = k/K$, $k = 0, \ldots, K$.

Sampling for L₂ metrics uses $X_t \sim \mathcal{N}(m_t\,\mathbf{1}_d,\, V_t\,I_d)$ (correct marginal under $u^*$).

**Relative $L_2$** (primary metric):

$$\mathrm{RelL}_2(t;\,u_\theta, u^*) = \frac{\left(\mathbb{E}_{P^{u^*}}\|u_\theta(t, X_t) - u^*(t, X_t)\|^2\right)^{1/2}}{\left(\mathbb{E}_{P^{u^*}}\|u^*(t, X_t)\|^2\right)^{1/2}}$$

**Absolute $L_2$** (secondary metric):

$$\mathrm{AbsL}_2(t;\,u_\theta, u^*) = \left(\mathbb{E}_{P^{u^*}}\|u_\theta(t, X_t) - u^*(t, X_t)\|^2\right)^{1/2}$$

**Absolute $L_\infty$** (tertiary metric):

$$\mathrm{AbsL}_\infty(t;\,u_\theta, u^*) = \sup_x \|u_\theta(t, x) - u^*(t, x)\|$$

Estimated on a **uniform grid** $x \in [\mu - x_\text{range},\, \mu + x_\text{range}]$ with $N_\text{grid}$ equally
spaced points centred at $\mu$. Grid is deterministic and fixed across iterations for consistent comparison.

**Contraction Factor** (per time slice, between consecutive outer iterations):

$$\mathrm{ContrFact}(t;\,n) = \frac{\|u^{n+1}_\theta(t,x) - u^*(t,x)\|_\infty}{\|u^n_\theta(t,x) - u^*(t,x)\|_\infty}$$

---

**Tiled Absolute $L_\infty$** on $[t, T]$:

$$\|u_\theta - u^*\|_{[t,T]} = \sup_{s \in [t,T],\; x} \|u_\theta(s,x) - u^*(s,x)\|$$

Estimated as the suffix maximum of $\mathrm{AbsL}_\infty(t_k)$ over all $k' \geq k$.

The theoretical bound states:

$$\|(T(u) - T(v))(t,x)\| \leq |T-t|\,\|\sigma\|_{[t,T]}\,\|u-v\|_{[t,T]}\,C'e^{C(T-t)}$$

so $\|u_\theta - u^*\|_{[t,T]}$ should decrease as $t \to T$, and faster than $|T-t|$.

**Tiled Contraction Factor**:

$$\mathrm{TiledContrFact}(t;\,n) = \frac{\|u^{n+1}_\theta - u^*\|_{[t,T]}}{\|u^n_\theta - u^*\|_{[t,T]}}$$

**Tiled Error Field** (for heatmaps):

$$\mathrm{TiledError}(t_k,\,x) = \max_{j \geq k}\;\|u_\theta(t_j,x) - u^*(t_j,x)\|$$

**Expected behaviour**: all metrics decrease as $t \to 1$. The tiled metrics provide the
direct empirical counterpart to the theoretical norm $\|u-v\|_{[t,T]}$.

---

## 7. Implementation notes

- `riccati_coefficient(t, lambda_, sigma_fn)` — $a(t) = \lambda^*/(1+\lambda^*\Sigma_t)$, called with `lambda_star`
- `riccati_mean_and_variance(ts, lambda_, mu, sigma_fn)` — forward Euler on both ODEs; called with `(lambda_star, mu_star)`; returns `(Ms, Vs)` each of shape `[K+1]`
- `optimal_control(x, t, lambda_, mu, sigma_fn)` — $u^*(t,x) = -\sigma(t)a(t)(x-\mu^*)$, called with `(lambda_star, mu_star)`
- `rel_l2(...)`, `abs_l2(...)`, `abs_linf(...)` — called with `(lambda_star, mu_star)`
- `control_field(...)` — called with `(lambda_star, mu_star)`
- `grad_g_fn(x1)` — returns `lambda_star * (x1 - mu_star)` = $\lambda^*(x_1 - \mu^*)$
- Effective params computed in `main()`: `lambda_star = lambda_ - 1/nu_1`, `mu_star = lambda_ * mu / lambda_star`
- Assertion `lambda_star > 0` enforces the condition $\lambda > 1/\nu_1$

Config parameters under `target`: `d`, `lambda_`, `mu`. Config must satisfy `lambda_ > 1/sigma**2`.
With `sigma=1.0`, need `lambda_ > 1.0` (e.g. `lambda_: 2.0` gives `lambda_star = 1.0`, `mu_star = 5.0`).

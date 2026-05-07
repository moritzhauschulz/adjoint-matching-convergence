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
- Noise schedule: constant $\sigma(t) = \sigma_0$
- Time horizon: $T = 1$, $t \in [0, 1]$
- State dimension: $d$ (experiment uses $d = 1$ initially for clarity)
- Operator (from Section 2 of notes, lean adjoint with $b=0$):

$$T(u)(t, x) = -\sigma(t)\,\mathbb{E}\!\left[\nabla r(X_T^u) \;\middle|\; X_t = x\right]$$

---

## 3. Reward and training objective

$$r(x) = \frac{\lambda}{2}\|x\|^2, \qquad \nabla r(x) = \lambda x$$

The training objective (simplified SOC, **no** $\log p_1^\text{base}$ term) is:

$$\min_{u}\;\mathbb{E}\!\left[\int_0^1 \tfrac{1}{2}\|u(t, X_t)\|^2\,dt - r(X_1 - \mu)\right]$$

where $\mu \in \mathbb{R}$ is a scalar shift (broadcast to all $d$ dimensions). The effective terminal cost is therefore

$$g_\text{eff}(X_1) = -r(X_1 - \mu) = -\frac{\lambda}{2}\|X_1 - \mu\|^2$$

and its gradient (used in the RAM loss replay buffer) is:

$$\nabla g_\text{eff}(X_1) = -\lambda(X_1 - \mu)$$

> **Sign convention**: the RAM loss is written with `grad_g = λ(X₁ − μ)` (positive),
> because the adjoint sampling framework absorbs the minus sign into the loss
> (consistent with the $\mu=0$ case where `grad_g = λ X₁`).

> **Important**: this differs from the full adjoint sampling objective with the
> $\log p_1^\text{base}$ term, which is **omitted** by design.

---

## 4. Analytic optimal control (Lemma, Section 4.1 of notes)

Define the Riccati coefficient (solved via HJB with ansatz $V(t,x) = \frac{1}{2}a(t)(x-\mu)^2 + c(t)$):

$$a(t) = \frac{\lambda}{1 + \lambda \displaystyle\int_t^1 \sigma(s)^2\,ds}$$

For constant schedule $\sigma(t) = \sigma_0$:

$$a(t) = \frac{\lambda}{1 + \lambda\,\sigma_0^2\,(1 - t)}$$

The optimal control is **linear in $(x - \mu)$** (no bias term beyond the shift):

$$u^*(t, x) = -\sigma(t)\,a(t)\,(x - \mu)$$

Note: $a(t)$ depends only on $\sigma$ and $\lambda$, not on $\mu$. The Riccati coefficient is
therefore identical to the $\mu=0$ case.

---

## 5. Distribution of $X_t$ under $u^*$

Under $u^*$, the SDE is

$$dX_t = -\sigma^2(t)\,a(t)\,(X_t - \mu)\,dt + \sigma(t)\,dB_t, \qquad X_0 = 0$$

This is a linear Gaussian SDE. Decompose $X_t = m_t + Y_t$ where $m_t = \mathbb{E}[X_t]$ and
$Y_t = X_t - m_t$ is the zero-mean fluctuation. The mean and variance satisfy:

**Mean ODE** ($m_0 = 0$):

$$\frac{dm_t}{dt} = -\sigma^2(t)\,a(t)\,(m_t - \mu)$$

**Variance ODE** ($V_0 = 0$, identical to the $\mu = 0$ case):

$$\frac{dV_t}{dt} = -2\sigma^2(t)\,a(t)\,V_t + \sigma^2(t)$$

Both are integrated numerically via forward Euler on the evaluation grid.

The marginal is $X_t \sim \mathcal{N}(m_t\,\mathbf{1}_d,\; V_t\,I_d)$.

As $t \to 1$, $m_t \to \mu$ and $V_t \to V_1 < \infty$: the process concentrates near $\mu$.

> **Grid note**: the sup-norm x-grid is `linspace(μ - x_range, μ + x_range)`, always centred at $\mu$.
> `linf_x_range` controls the half-width around $\mu$.

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

- `riccati_coefficient(t, lambda_, sigma_fn)` — $a(t) = \lambda/(1+\lambda\sigma_0^2(1-t))$
- `riccati_mean_and_variance(ts, lambda_, mu, sigma_fn)` — forward Euler on both ODEs; returns `(Vs, Ms)` each of shape `[K+1]`
- `optimal_control(x, t, lambda_, mu, sigma_fn)` — $u^*(t,x) = -\sigma(t)a(t)(x-\mu)$
- `rel_l2(u_theta, t_val, Vt_val, Mt_val, lambda_, mu, sigma_fn, d, n_samples, device)` — samples $X_t \sim \mathcal{N}(m_t, V_t I)$
- `abs_l2(...)` — same signature as `rel_l2`, unnormalised numerator
- `abs_linf(u_theta, t_val, lambda_, mu, sigma_fn, d, xs)` — evaluates on fixed x-grid
- `ContrFact(t; n)` — computed as `abs_linf[n+1] / abs_linf[n]` across time slices
- `grad_g_fn(x1)` — returns `lambda_ * (x1 - mu)` for the RAM replay buffer

Config parameters under `target`: `d`, `lambda_`, `mu`.

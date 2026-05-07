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

Training objective (simplified SOC, **no** $\log p_1^\text{base}$ term):

$$\min_{u}\;\mathbb{E}\!\left[\int_0^1 \tfrac{1}{2}\|u(t, X_t)\|^2\,dt + \frac{\lambda}{2}\|X_1\|^2\right]$$

> **Important**: this differs from the full adjoint sampling objective
> $\mathcal{L}_\text{SOC}(u) = \mathbb{E}[\int \frac{1}{2}\|u\|^2 dt + g(X_1)]$
> with $g = \log p_1^\text{base} + E$.  Here $g_\text{eff}(X_1) = \frac{\lambda}{2}\|X_1\|^2$
> only; the $\log p_1^\text{base}$ term is **omitted** by design.

The RAM loss therefore uses $\nabla g_\text{eff}(X_1) = \lambda X_1$ in the replay buffer.

---

## 4. Analytic optimal control (Lemma, Section 4.1 of notes)

Define the Riccati coefficient (solved via HJB with ansatz $V(t,x) = \frac{1}{2}a(t)\|x\|^2$):

$$a(t) = \frac{\lambda}{1 + \lambda \displaystyle\int_t^1 \sigma(s)^2\,ds}$$

For constant schedule $\sigma(t) = \sigma_0$:

$$a(t) = \frac{\lambda}{1 + \lambda\,\sigma_0^2\,(1 - t)}$$

The optimal control is **linear** (no bias term):

$$u^*(t, x) = -\sigma(t)\,a(t)\,x$$

---

## 5. Distribution of $X_t$ under $u^*$

Under $u^*$, the SDE $dX_t = -\sigma^2(t)\,a(t)\,X_t\,dt + \sigma(t)\,dB_t$ with $X_0 = 0$
is a linear Gaussian SDE. The variance $V_t = \mathbb{E}[\|X_t\|^2]/d$ satisfies:

$$\frac{dV_t}{dt} = -2\sigma^2(t)\,a(t)\,V_t + \sigma^2(t), \qquad V_0 = 0$$

This ODE is integrated numerically. Samples $X_t \sim \mathcal{N}(0, V_t I)$ can then be
drawn directly without running the SDE forward — enabling cheap evaluation of metrics at
any $t$.

---

## 6. Convergence metrics (Section 4.1 of notes)

Metrics are evaluated at **uniform time steps** $t_k = k/K$, $k = 0, \ldots, K$.

**Relative $L_2$** (primary metric):

$$\mathrm{RelL}_2(t;\,u_\theta, u^*) = \frac{\left(\mathbb{E}_{P^{u^*}}\|u_\theta(t, X_t) - u^*(t, X_t)\|^2\right)^{1/2}}{\left(\mathbb{E}_{P^{u^*}}\|u^*(t, X_t)\|^2\right)^{1/2}}$$

Estimated by sampling $X_t \sim \mathcal{N}(0, V_t I)$.

**Absolute $L_2$** (secondary metric):

$$\mathrm{AbsL}_2(t;\,u_\theta, u^*) = \left(\mathbb{E}_{P^{u^*}}\|u_\theta(t, X_t) - u^*(t, X_t)\|^2\right)^{1/2}$$

The unnormalised counterpart of $\mathrm{RelL}_2$ — the numerator alone, without dividing by $\|u^*\|$. Estimated by sampling $X_t \sim \mathcal{N}(0, V_t I)$.

**Absolute $L_\infty$** (tertiary metric):

$$\mathrm{AbsL}_\infty(t;\,u_\theta, u^*) = \sup_x \|u_\theta(t, x) - u^*(t, x)\|$$

Estimated by sampling $X_t$ from the optimal process (via the SDE or $\mathcal{N}(0,V_t I)$).

**Expected behaviour**: all three metrics decrease as $t \to 1$, confirming right-to-left convergence.

---

## 7. Implementation notes

- `optimal_control(x, t, lambda_, sigma_fn)` — evaluates $u^*(t,x) = -\sigma(t)a(t)x$
- `riccati_variance(ts, lambda_, sigma_fn)` — integrates $dV/dt$ on the grid $\{t_k\}$
- `sample_optimal_marginal(t, Vt, d)` — samples $X_t \sim \mathcal{N}(0, V_t I)$
- `rel_l2(u_theta, u_star_fn, Vt, t, d, n_samples)` — estimates $\mathrm{RelL}_2(t)$
- `abs_l2(u_theta, u_star_fn, Vt, t, d, n_samples)` — estimates $\mathrm{AbsL}_2(t)$
- `abs_linf(u_theta, u_star_fn, Vt, t, d, n_samples)` — estimates $\mathrm{AbsL}_\infty(t)$

The `ram_loss` from `adjoint_sampling.losses` is reused directly; only the
`grad_g` function changes (becomes $\lambda X_1$ instead of the full adjoint sampling gradient).

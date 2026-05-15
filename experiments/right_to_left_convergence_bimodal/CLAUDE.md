# Right-to-Left Convergence Experiment (bimodal)

## 1. Goal

Verify the right-to-left contraction bound

$$\|(T(u) - T(v))(t,x)\| \leq C(t,T) \cdot \|u - v\|_{[t,T]}, \quad C(t,T) \to 0 \text{ as } t \to T$$

using the **full adjoint sampling objective** (arXiv:2504.11713, Algorithm 1) with a
log-mixture bimodal reward, where $u^*$ is available via the Feynman-Kac / Doob h-transform.

---

## 2. Setting

- Base process: $dX = \sigma(t)\,dB$, zero drift, $X_0 = 0$
- Noise schedule: constant $\sigma(t) = \sigma_0$, so $\nu_1 = \int_0^1 \sigma(t)^2\,dt = \sigma_0^2$
- $\Sigma_t = \int_t^1 \sigma(s)^2\,ds = \sigma_0^2(1-t)$
- Time horizon $T=1$, $t \in [0,1]$, state dimension $d=1$

---

## 3. Reward and objective (full adjoint sampling)

**Bimodal reward**:

$$r(x) = \log\!\left(w_1\,e^{-\frac{\lambda_1}{2}(x-\mu_1)^2} + w_2\,e^{-\frac{\lambda_2}{2}(x-\mu_2)^2}\right)$$

**Full objective** $g(x_1) = -\log p_1^\text{base}(x_1) - r(x_1)/\tau$. With $\tau=1$:

$$g(x_1) = \frac{x_1^2}{2\nu_1} - r(x_1) + \text{const}$$

$$\nabla g(x_1) = \frac{x_1}{\nu_1} - \nabla r(x_1)$$

where the gradient of the reward is:

$$\nabla r(x) = -\frac{\lambda_1(x-\mu_1)\,A_1(T,x) + \lambda_2(x-\mu_2)\,A_2(T,x)}{A_1(T,x)+A_2(T,x)}, \quad A_i(T,x) = w_i\,e^{-\lambda_i(x-\mu_i)^2/2}$$

**RAM loss** (Algorithm 1, inner loop): sample $X_{t|1}^\text{base} \sim p_{t|1}^\text{base}(\cdot|x_1)$, minimise

$$L_\text{RAM}(\theta) = \mathbb{E}\!\left[\frac{\lambda(t)}{2}\,\|u_\theta(X_t, t) + \sigma(t)\,\nabla g(X_1)\|^2\right]$$

with $\lambda(t) = 1/\sigma(t)^2$ (matching the paper).

---

## 4. Analytic optimal control (Feynman-Kac)

$u^*(t,x) = \sigma(t)\,\nabla_x \log h(t,x)$ where $h(t,x) = \mathbb{E}_0[e^{-g(X_1)}\mid X_t=x]$.

With the full objective, the effective mixture parameters are:

$$\lambda_i^* = \lambda_i - \frac{1}{\nu_1}, \qquad \mu_i^* = \frac{\lambda_i \mu_i}{\lambda_i^*}$$

**Requirement**: $\lambda_i > 1/\nu_1$ (asserted at runtime).

Define:

$$A_i^*(t,x) = \frac{w_i}{\sqrt{1+\lambda_i^*\Sigma_t}}\exp\!\left(-\frac{\lambda_i^*(x-\mu_i^*)^2}{2(1+\lambda_i^*\Sigma_t)}\right)$$

Then:

$$u^*(t,x) = -\sigma(t)\,\frac{\displaystyle\sum_{i=1}^2 \frac{\lambda_i^*(x-\mu_i^*)}{1+\lambda_i^*\Sigma_t}\,A_i^*(t,x)}{A_1^*(t,x)+A_2^*(t,x)}$$

---

## 5. Terminal distribution under $u^*$

$$p^{u^*}(x_1) \propto e^{r(x_1)} = \alpha_1\,\mathcal{N}(x_1;\,\mu_1,\,1/\lambda_1) + \alpha_2\,\mathcal{N}(x_1;\,\mu_2,\,1/\lambda_2)$$

with $\alpha_i \propto w_i/\sqrt{\lambda_i}$. No $\sigma_0$ dependence.

---

## 6. Convergence metrics

See `experiments/CLAUDE.md` for shared conventions. All metrics (RelL₂, AbsL₂, AbsL∞,
ContrFact, TiledContrFact) are evaluated at uniform time slices $t_k = k/K$.

$L_2$ metrics use empirical path samples under $u^*$ (non-Gaussian marginals at $t < T$).
$L_\infty$ uses a uniform x-grid centred at $(\mu_1+\mu_2)/2$ with half-width `linf_x_range`.

---

## 7. Implementation notes

- `sigma_integral(t)` — $\Sigma_t = \sigma_0^2(1-t)$
- `A_component(x, t, w, lam, mu_i, sigma_fn, d)` — $A_i^*(t,x)$ (called with starred params)
- `optimal_control(x, t, w1, lambda1, mu1, w2, lambda2, mu2, sigma_fn, nu_1, d)` — starred params derived internally; asserts $\lambda_i > 1/\nu_1$
- `grad_r(x1, w1, lambda1, mu1, w2, lambda2, mu2)` — $\nabla r(x_1)$
- `grad_g_fn(x1)` inside `main` — returns $x_1/\nu_1 - \nabla r(x_1)$ (passed to `ram_loss`)
- `terminal_mixture_params(w1, lambda1, mu1, w2, lambda2, mu2)` — returns $(\alpha_1, \mu_1, 1/\lambda_1, \alpha_2, \mu_2, 1/\lambda_2)$; no $\sigma_0$ parameter
- `target_params` dict written to `metrics.json`: `{w1, lambda1, mu1, w2, lambda2, mu2}` (no `sigma`)

Config key `sigma` sets $\sigma_0$ (used by `sigma_fn` and to compute $\nu_1$).
Config keys under `target`: `d`, `w1`, `lambda1`, `mu1`, `lambda2`, `mu2`.

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

## 3. Reward and objective

**Bimodal log-density** (reward):

$$r(x) = \log\!\left(w_1\,e^{-\frac{\lambda_1}{2}(x-\mu_1)^2} + w_2\,e^{-\frac{\lambda_2}{2}(x-\mu_2)^2}\right)$$

**Full adjoint sampling objective** (arXiv:2504.11713, Algorithm 1):

$$g(x_1) = \log p_1^\text{base}(x_1) - r(x_1) = -\frac{x_1^2}{2\nu_1} - r(x_1) + \text{const}$$

$$\nabla g(x_1) = -\frac{x_1}{\nu_1} - \nabla r(x_1)$$

where the score of the log-density is:

$$\nabla r(x) = -\frac{\lambda_1(x-\mu_1)\,A_1(T,x) + \lambda_2(x-\mu_2)\,A_2(T,x)}{A_1(T,x)+A_2(T,x)}, \quad A_i(T,x) = w_i\,e^{-\lambda_i(x-\mu_i)^2/2}$$

The self-consistency operator (fixed-point, from Feynman-Kac when $\exp(r)$ is integrable):

$$T(u)(t,x) = -\sigma(t)\,\mathbb{E}_u\!\left[\nabla g(X_T^{u,x})\right]$$

and $u^*$ is its fixed point: $T(u^*) = u^*$.

**RAM loss** (Algorithm 1, inner loop): sample $X_{t|1}^\text{base} \sim p_{t|1}^\text{base}(\cdot|x_1)$, minimise

$$L_\text{RAM}(\theta) = \mathbb{E}\!\left[\frac{\lambda(t)}{2}\,\|u_\theta(X_t, t) + \sigma(t)\,\nabla g(X_1)\|^2\right]$$

with $\lambda(t) = 1/\sigma(t)^2$. The target $-\sigma\nabla g(X_1)$ is stored as $\nabla g$ in the replay buffer.

---

## 4. Analytic optimal control (Feynman-Kac)

With $g = \log p_1^\text{base} - r$, the value function satisfies $h(t,x) = \mathbb{E}_\text{base}[e^{-g(X_1)}\mid X_t=x] = \mathbb{E}_\text{base}[e^{r(X_1)}/p_1^\text{base}(X_1)\mid X_t=x]$ and $u^*(t,x) = \sigma(t)\,\nabla_x \log h(t,x)$.

**Effective parameters** (requires $\lambda_i > 1/\nu_1$):

$$\lambda_i^* = \lambda_i - \frac{1}{\nu_1}, \qquad \mu_i^* = \frac{\lambda_i\,\mu_i}{\lambda_i^*}, \qquad \kappa_i = \frac{d\,\lambda_i\,\mu_i^2}{2\,\nu_1\,\lambda_i^*}$$

Completing the square in $-\lambda_i(x_1-\mu_i)^2/2 + x_1^2/(2\nu_1)$ produces the
**mode-dependent constant $\kappa_i$**, which must be carried in $A_i$. It cancels
in the $A_1/A_2$ ratio only for symmetric mixtures ($\mu_1^2=\mu_2^2$, equal
$\lambda$); omitting it (an earlier bug) silently breaks $u^*$ for asymmetric
means/precisions. Define:

$$A_i(t,x) = \frac{w_i\,e^{\kappa_i}}{\sqrt{1+\lambda_i^*\Sigma_t}}\exp\!\left(-\frac{\lambda_i^*(x-\mu_i^*)^2}{2(1+\lambda_i^*\Sigma_t)}\right), \qquad \Sigma_t = \sigma_0^2(1-t)$$

Then:

$$u^*(t,x) = -\sigma(t)\,\frac{\displaystyle\sum_{i=1}^2 \frac{\lambda_i^*(x-\mu_i^*)}{1+\lambda_i^*\Sigma_t}\,A_i(t,x)}{A_1(t,x)+A_2(t,x)}$$

---

## 5. Terminal distribution under $u^*$

With the full objective, the base measure cancels exactly:

$$p^{u^*}(x_1) \propto e^{-g(x_1)}\,p_1^{\text{base}}(x_1) = e^{r(x_1)} = \alpha_1\,\mathcal{N}(x_1;\,\mu_1,\,1/\lambda_1) + \alpha_2\,\mathcal{N}(x_1;\,\mu_2,\,1/\lambda_2)$$

with $\alpha_i \propto w_i/\sqrt{\lambda_i}$. No $\sigma_0$ dependence.

---

## 6. Convergence metrics

See `experiments/CLAUDE.md` for shared conventions. All metrics (RelL₂, AbsL₂, AbsL∞,
ContrFact, TiledContrFact) are evaluated at uniform time slices $t_k = k/K$.

$L_2$ metrics use empirical path samples under $u^*$ (non-Gaussian marginals at $t < T$).
$L_\infty$ uses a uniform x-grid centred at $(\mu_1+\mu_2)/2$ with half-width `linf_x_range`.

---

## 7. Implementation notes

- **2026-08-29:** `optimal_control` / `grad_r` / `terminal_mixture_params` are now
  thin wrappers that delegate to `adjoint_sampling.GaussianMixtureTarget` (the
  shared two-component reward — see `src/adjoint_sampling/bimodal_target.py` and
  §4). The wrappers keep the historical scalar-arg signatures used in `main`;
  constant σ ⇒ $\Sigma_t = \nu_1(1-t)$ is passed as `sigma_int_fn`.  The κ_i fix
  (missing $\kappa_i = d\lambda_i\mu_i^2/(2\nu_1\lambda_i^*)$ in `log_A_i`; only
  affected asymmetric mixtures) now lives in the shared class.  `simulate_paths`
  is imported from `adjoint_sampling.utils`.
- `grad_g_fn(x1)` inside `main` — $\nabla g = -x_1/\nu_1 - \nabla r(x_1)$ (stored in replay buffer)
- `target_params` dict written to `metrics.json`: `{w1, lambda1, mu1, w2, lambda2, mu2}` (no `sigma`)

Config key `sigma` sets $\sigma_0$ (used by `sigma_fn` and to compute $\nu_1$).
Config keys under `target`: `d`, `w1`, `lambda1`, `mu1`, `lambda2`, `mu2`.

# Adjoint Sampling — Implementation Spec

Notation follows arXiv:2504.11713 throughout.

---

## 1. Problem statement

Sample from a Boltzmann target distribution:

$$\mu(x) = \frac{\exp(-E(x)/\tau)}{Z}$$

where $E(x)$ is an energy function, $\tau > 0$ a temperature, and $Z$ an intractable partition function. We learn a control $u_\theta$ that steers a diffusion process toward $\mu$.

---

## 2. Forward SDE (controlled process)

$$dX_t = \sigma(t)\, u(X_t, t)\, dt + \sigma(t)\, dB_t, \qquad t \in [0,1], \quad X_0 = 0$$

- $X_t \in \mathbb{R}^d$ — state at time $t$
- $u(X_t, t)$ — control / policy (learned, parameterised by $\theta$)
- $\sigma(t)$ — noise schedule, scalar function $[0,1] \to \mathbb{R}$
- $B_t$ — $d$-dimensional standard Brownian motion
- $p^u(\mathbf{X})$ — path measure of the controlled process
- $p^\text{base}(\mathbf{X})$ — base (uncontrolled, $u \equiv 0$) path measure

The total drift is $b(X_t, t) = \sigma(t)\, u(X_t, t)$.

---

## 3. Objective (stochastic optimal control)

$$\mathcal{L}_\text{SOC}(u) = \mathbb{E}_{p^u}\!\left[\int_0^1 \tfrac{1}{2}\|u(X_t,t)\|^2\, dt + g(X_1)\right]$$

Terminal cost:

$$g(x) = \log p_1^\text{base}(x) + \frac{1}{\tau} E(x)$$

$p_1^\text{base}(x)$ is the marginal of the base process at $t=1$ (Gaussian, closed-form). Minimising $\mathcal{L}_\text{SOC}$ is equivalent to minimising $D_\text{KL}(p^u \| p^*)$.

---

## 4. Adjoint state equations

Given a realised trajectory $\mathbf{X} = \{X_t : 0 \le t \le 1\}$, the **lean adjoint state** $\tilde{a}(t;\mathbf{X}) \in \mathbb{R}^d$ satisfies the backward ODE:

$$\frac{d\tilde{a}(t;\mathbf{X})}{dt} = -\tilde{a}(t;\mathbf{X})^\top \nabla_x b(X_t, t), \qquad t \in [0,1]$$

with terminal condition:

$$\tilde{a}(1;\mathbf{X}) = \nabla g(X_1) = \nabla \log p_1^\text{base}(X_1) + \frac{1}{\tau}\nabla E(X_1)$$

where $\nabla_x b(X_t,t) = \sigma(t)\,\nabla_x u(X_t,t)$ is the Jacobian of the drift w.r.t. $x$.

The adjoint provides the functional derivative of $\mathcal{L}_\text{SOC}$ w.r.t. $u$ without backpropagating through the SDE solver.

---

## 5. Loss variants

| Name | Symbol | Notes |
|---|---|---|
| SOC objective | $\mathcal{L}_\text{SOC}(u)$ | Ground-truth objective; expensive |
| Adjoint Matching | $\mathcal{L}_\text{AM}(u)$ | Regress $u$ onto $\tilde{a}$; controlled process rollouts |
| Reciprocal AM | $\mathcal{L}_\text{RAM}(u)$ | Factored expectation over $p_{t\|1}^\text{base}$; more efficient |
| Geometric RAM | $\mathcal{L}_\text{GeoRAM}(\theta)$ | Adds projection operator $\mathcal{A}$ for symmetry constraints |

Practical training uses $\mathcal{L}_\text{RAM}$ or $\mathcal{L}_\text{GeoRAM}$. Time steps are sampled uniformly $t^{(j)} \sim \mathcal{U}([0,1])$ and weighted by $\lambda(t) = 1/\sigma(t)^2$.

The $\mathcal{L}_\text{RAM}$ loss (Algorithm 1 inner loop) is:

$$\mathcal{L}_\text{RAM}(u) = \int_0^1 \lambda(t)\, \mathbb{E}_{\substack{X_1 \sim p_1^{\bar{u}} \\ X_t \sim p_{t|1}^\text{base}(\cdot|X_1)}}\!\left[\tfrac{1}{2}\left\|u(X_t,t) + \sigma(t)\,\nabla g(X_1)\right\|^2\right] dt$$

The key insight: $p_{t|1}^\text{base}(\cdot | X_1)$ is a known Gaussian (see §6), so $X_t$ can be sampled without running the controlled process.

---

## 6. Base process bridge and marginal variance

The base process $dX_t = \sigma(t)\,dB_t$ with $X_0 = 0$ has marginal variance:

$$\nu_t = \int_0^t \sigma(s)^2\, ds, \qquad \nu_1 = \int_0^1 \sigma(s)^2\, ds$$

The bridge conditional is:

$$p_{t|1}^\text{base}(X_t \mid X_1) = \mathcal{N}\!\left(X_t;\; \frac{\nu_t}{\nu_1} X_1,\; \nu_{t|1} I\right), \qquad \nu_{t|1} = \frac{\nu_t(\nu_1 - \nu_t)}{\nu_1}$$

The base marginal at $t=1$: $p_1^\text{base}(x) = \mathcal{N}(x;\, 0,\, \nu_1 I)$, so $\nabla_x \log p_1^\text{base}(x) = -x/\nu_1$.

---

## 7. Discretisation

Uniform grid $0 = t_0 < t_1 < \cdots < t_N = 1$, step $\Delta t = 1/N$.

**Forward (Euler–Maruyama):**

$$X_{n+1} = X_n + \sigma(t_n)\,u(X_n,t_n)\,\Delta t + \sigma(t_n)\sqrt{\Delta t}\;\varepsilon_n, \qquad \varepsilon_n \sim \mathcal{N}(0,I)$$

**Backward adjoint (Euler, reversed in time):**

$$\tilde{a}(t_{n-1};\mathbf{X}) = \tilde{a}(t_n;\mathbf{X}) + \tilde{a}(t_n;\mathbf{X})^\top \bigl[\sigma(t_n)\,\nabla_x u(X_n,t_n)\bigr]\,\Delta t$$

---

## 8. Implementation map and key notes

| File | Responsibility |
|---|---|
| `sampler.py` | Forward SDE rollout; stores trajectory $\{X_n\}$ and noise $\{\varepsilon_n\}$ |
| `adjoint.py` | Backward adjoint solve; computes $\tilde{a}(t;\mathbf{X})$ given trajectory |
| `losses.py` | $\mathcal{L}_\text{SOC}$, $\mathcal{L}_\text{AM}$, $\mathcal{L}_\text{RAM}$; assembles gradient from adjoint |
| `utils.py` | Euler–Maruyama step, time grids, $\sigma(t)$, VJP for $\nabla_x b$ |

- **Store noise**: save $\{\varepsilon_n\}$ during the forward pass; reuse in the backward pass for consistent Brownian paths.
- **VJPs not Jacobians**: compute $\tilde{a}(t)^\top \nabla_x b$ via `torch.autograd.functional.vjp` — never materialise the full $d \times d$ Jacobian.
- **Replay buffer**: cache $(X_1^{(i)}, \nabla g^{(i)})$ pairs in buffer $\mathcal{B}$ to amortise energy evaluations (Algorithm 1 in paper).
- **Stop-gradient**: when computing $\mathcal{L}_\text{RAM}$, use $\bar{u} = \text{stopgrad}(u)$ for the rollout policy to stabilise training.

---

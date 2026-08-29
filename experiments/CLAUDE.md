# Experiments — Shared Conventions

## Index

| Experiment | Directory | Purpose |
|---|---|---|
| Gaussian baseline | `gaussian_baseline/run.py` | Sanity-check: Gaussian target with full adjoint sampling objective |
| Right-to-left convergence (unimodal) | `right_to_left_convergence_unimodal/` | Verify theoretical contraction bound using quadratic reward + analytic $u^*$ |
| Right-to-left convergence (bimodal) | `right_to_left_convergence_bimodal/` | Same contraction bound verification with log-mixture bimodal reward + Feynman-Kac $u^*$ |
| Right-to-left convergence (bimodal, sanity check) | `right_to_left_convergence_bimodal_same_bm/` | Estimates the "Sanity Check" LHS $\|\mathbb{E}_B[\sigma(t)(\nabla g(X_T^{u_\theta}) - \nabla g(X_T^{u^*}))]\|$ (norm outside the expectation); shared BM is a variance-reduction device |

## Shared conventions

- All experiments are run via Hydra: `python experiments/<name>/run.py experiment=<name>`
- All metrics are logged with W&B (opt-in) and printed to stdout
- `results/` receives all Hydra outputs (gitignored)
- Configs live in `configs/experiment/<name>.yaml`
- Each experiment directory contains its own `CLAUDE.md` with the full mathematical spec

## Interface contract across experiments

All experiment `run.py` scripts must accept the same top-level Hydra config keys:
`seed`, `device`, `logging` (wandb/project/entity/log_every).
Experiment-specific keys go under a dedicated namespace (e.g. `target`, `algorithm`, `training`).

## Shared operator / fixed-point sanity check

`src/adjoint_sampling/operator.py` provides the self-consistency operator
$P(u)(t,x) = -\sigma(t)\,\mathbb{E}[\nabla g(X_T^{u,x})]$ and
`fixed_point_residual_field(u_star_fn, grad_g_fn, xs, ts, sigma_fn, d, n_mc, device)`,
which returns $\|P(u^*)(t,x) - u^*(t,x)\|$ on an $(x\text{-grid})\times(\text{time-grid})$
mesh. Any experiment with a closed-form $u^*$ should call this as a basic sanity
check (residual $\ll 1$ everywhere, $=0$ at $t=T$). Currently wired into
`right_to_left_convergence_bimodal_same_bm` (config `eval.fixed_point_check`,
`eval.n_fixed_point_mc`, `eval.n_fixed_point_grid`; plot
`same_bm/u_star_fixed_point_residual.png`). See `src/adjoint_sampling/CLAUDE.md` §9.

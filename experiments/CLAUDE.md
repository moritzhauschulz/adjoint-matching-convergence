# Experiments — Shared Conventions

## Index

| Experiment | Directory | Purpose |
|---|---|---|
| Gaussian baseline | `gaussian_baseline/run.py` | Sanity-check: Gaussian target with full adjoint sampling objective |
| Right-to-left convergence (unimodal) | `right_to_left_convergence_unimodal/` | Verify theoretical contraction bound using quadratic reward + analytic $u^*$ |
| Right-to-left convergence (bimodal) | `right_to_left_convergence_bimodal/` | Same contraction bound verification with log-mixture bimodal reward + Feynman-Kac $u^*$ |
| Right-to-left convergence (bimodal, same BM) | `right_to_left_convergence_bimodal_same_bm/` | Directly estimates the LHS of the §2.1 bound via shared Brownian motion for $X^{u_\theta}$ and $X^{u^*}$ |

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

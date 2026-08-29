# Adjoint Matching — Convergence Experiments

## Project goal
Evaluate the convergence properties and capabilities of the adjoint matching algorithm (and its variants, like adjoint sampling) across a set of experimental benchmarks.

## Scientific Rigor
This codebase should accurately represent any underlying mathematics. If in doubt about underlying mathematics, i.e. when you cannot reliably verify their correctness or their correspondence to the user's requests, ask for clarification. Mathematical accuracy is the single most important principle in this repository. All mathematics in CLAUDE.md files should be rendered in LaTeX (or similar human readable form).

## Project Notes (project_notes.tm)
The entire purpose of the repo is to demonstrate (or refute) theoretical findings in the project_notes.tm file. This file is entirely human written, and is usually updated before any new implementation. Please always review the file, and align the CLAUDE.mds accordingly. NEVER edit the project_notes.tm. If you find a mistake, please alert the user.

## Maintenance
Update this CLAUDE.md file regularly, especially after structural changes or new implementations. Always ask the user to approve changes, never change CLAUDE.md without approval.

## Repository layout

```
src/adjoint_sampling/         # installable package: sampler, losses, network, replay buffer, utils
experiments/                  # runnable experiment scripts (one subdirectory per experiment)
  gaussian_baseline/          # Gaussian target with analytic optimal control
  right_to_left_convergence/  # Quadratic reward; verifies right-to-left contraction bound
configs/                      # Hydra config hierarchy
  config.yaml                 # root config (sets hydra output dirs → results/run/<timestamp>/)
  experiment/                 # per-experiment overrides
tests/                        # pytest unit tests
results/                      # gitignored — Hydra writes outputs here automatically
data/                         # gitignored — datasets
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running experiments

```bash
# Run a specific experiment
python experiments/<name>/run.py experiment=<name>

# Example
python experiments/right_to_left_convergence/run.py experiment=right_to_left_convergence

# Override config values on the CLI
python experiments/right_to_left_convergence/run.py experiment=right_to_left_convergence eval.every=1 training.lr=1e-4

# Hydra multirun sweep
python experiments/right_to_left_convergence/run.py --multirun experiment=right_to_left_convergence training.lr=1e-3,1e-4,1e-5
```

All outputs (logs, metrics, plots) land in `results/run/<timestamp>/`.

To re-generate plots from a completed run without retraining:
```python
from experiments.right_to_left_convergence.plotting import load_and_plot
load_and_plot("results/run/<timestamp>/metrics.json")
```

## Running tests

```bash
pytest
```

## Key conventions
- All experiments are reproducible via `seed` in config (propagated to random/numpy/torch).
- W&B logging is opt-in via `logging.wandb=true`; entity must be set in the experiment config.
- Each experiment has its own `CLAUDE.md` with the full mathematical spec.
- No notebooks — analysis is done via experiment scripts and logged metrics.
- Plots are saved as PNGs to the Hydra output directory alongside `metrics.json`.

## Before implementing any algorithm

Before writing any implementation, always ask:
"Should I search for an academic paper and/or codebase for this algorithm first?"

Then follow this decision tree strictly:

- **User says no** — implement directly from the user's spec. Do not reveal original ideas by searching, browsing, or reference any external sources. This protects potentially original ideas from leaking.
- **User says yes** — search for the paper and any official or reference codebases.
  - If found: use the paper's notation, equations, and algorithmic details as the implementation reference. Do not replicate the external codebase's structure — fit the implementation into this repo's layout.
  - If not found: ask the user for accurate implementation details before writing any code.
  - In both cases: create or update a `CLAUDE.md` in the relevant `src/` subdirectory capturing the notation, key equations, and algorithmic conventions that the implementation follows.

Never begin implementation until a `CLAUDE.md` with sufficient mathematical and algorithmic detail exists in the relevant module directory.

## Repo structure across multiple algorithms

When multiple algorithms are implemented, their module structure must stay parallel so results are directly comparable. Shared components (energy functions, samplers, metrics, time grids) live in a common module rather than being duplicated. Before adding a new algorithm, check existing modules and align interfaces — same config keys, same metric names, same logging calls.

## Git/Github usage

When starting on a new feature, always ask if a new feature branch should be opened. Never merge branches locally, unless requested by the user. Instead when a feature is ready to be merged into `develop`, make a PR on GitHub and tell the user to review it. All git actions should be approved by the user, including push/pull.

## Compute

Where possible, the code should automatically detect the available compute resources. Ideally there should always be the option for Apple Silicon and for CUDA.

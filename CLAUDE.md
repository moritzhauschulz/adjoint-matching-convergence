# Adjoint Matching — Convergence Experiments

## Project goal
Evaluate the convergence properties and capabilities of the adjoint matching algorithm (and its variants, like adjoint sampling) across a set of experimental benchmarks.

## Scientific Rigor
This codebase should accurately represent any underlying mathematics. If in doubt about underlying mathematics, i.e. when you cannot reliably verify their correctness or their correspondence to the user's requests, ask for clarification. Mathematical accuracy is the single most important principle in this repository. All mathematics in CLAUDE.md files shoudl be redndered in Latex (or similar human readable form).

## Maintenance
Update this CLAUDE.md file regularly, especially after structural changes or new implementations. Always ask the user to approve changes, never change CLAUDE.md without approval.

## Repository layout

```
src/adjoint_matching/   # installable package: algorithm, metrics, utils
experiments/            # runnable experiment scripts (entry points)
configs/                # Hydra config hierarchy
  config.yaml           # root config (sets hydra output dirs → results/)
  experiment/           # per-experiment overrides
tests/                  # pytest unit tests
results/                # gitignored — Hydra writes outputs here automatically
data/                   # gitignored — datasets
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running experiments

Single run with defaults:
```bash
python experiments/run.py
```

Override config values on the CLI:
```bash
python experiments/run.py training.lr=1e-4 algorithm.some_param=0.5
```

Hydra multirun sweep:
```bash
python experiments/run.py --multirun training.lr=1e-3,1e-4,1e-5
```

All outputs (logs, checkpoints, metrics) land in `results/<job>/<timestamp>/`.

## Running tests

```bash
pytest
```

## Key conventions
- All experiments are reproducible via `seed` in config (propagated to random/numpy/torch).
- W&B logging is opt-in via `logging.wandb=true` (default: true in `default.yaml`).
- Algorithm implementation lives in `src/adjoint_matching/algorithm.py`; metrics in `metrics.py`.
- No notebooks — analysis is done via experiment scripts and logged metrics.

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

When starting on a new feature, always ask if a new feature branch should be openend. Never merge branches locally, unless requested by the user. Instead when a featrue is ready to be merged into `develop`, make a PR on github and tell the user to review it. All git actions should be approved by the user, including push/pull.

## Compute

Where possible, the code should automatically detect the available cimpute resources. Ideally there should always be the option for apple silicon, and for CUDA.
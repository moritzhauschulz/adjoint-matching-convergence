# Adjoint Matching — Convergence Experiments

## Project goal
Evaluate the convergence properties and capabilities of the adjoint matching algorithm across a set of experimental benchmarks.

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

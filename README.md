# Adjoint Matching — Convergence Experiments

Empirical study of the convergence of the adjoint matching / adjoint sampling
algorithm (arXiv:2504.11713) and its variants. Each experiment verifies or
refutes a theoretical prediction about the algorithm's behaviour.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

```
src/adjoint_sampling/   installable package — network, sampler, losses, operator,
                        GaussianMixtureTarget, noise schedules / eval helpers
experiments/<name>/     one runnable Hydra script + plotting per experiment
configs/                Hydra config tree (configs/experiment/<name>.yaml)
tests/                  pytest
results/                gitignored — Hydra writes runs here
```

Each experiment has its own `CLAUDE.md` with the mathematical spec it implements;
`src/adjoint_sampling/CLAUDE.md` covers the shared package.

## Running an experiment

```bash
python experiments/<name>/run.py experiment=<name>
# override any config value on the CLI, e.g.
python experiments/right_to_left_convergence_bimodal_same_bm/run.py \
    experiment=right_to_left_convergence_bimodal_same_bm training.lr=1e-4 eval.every=1
```

Outputs (metrics + plots) land in `results/run/<hydra.run.dir>/`. Tests: `pytest`.

The primary experiment is **`right_to_left_convergence_bimodal_same_bm`** — the
self-consistency "sanity check": fixed-point residual `‖P(u*)−u*‖`, operator
error `‖P(u_θ)−u*‖`, and its right-to-left ratios. It trains with
`algorithm.objective` ∈ {`ram` (default), `am`} and `network.time_embedding` ∈
{`raw` (default here — feed the scalar t), `sinusoidal`}, and produces 9 figures
(see the docstring in `plotting.py`).

## Reproducing the headline runs (AM objective, raw-t, constant σ = 2, keeper eval)

```bash
KEEPER="experiment=right_to_left_convergence_bimodal_same_bm logging.wandb=false \
  algorithm.objective=am network.time_embedding=raw sigma_schedule=constant sigma=2.0 \
  training.outer_iterations=10 eval.n_op_mc_samples=256 eval.n_op_grid=121 \
  eval.n_metric_samples=2048 eval.n_sample_paths=4000 \
  eval.n_fixed_point_mc=2048 eval.n_fixed_point_grid=161"

# bimodal reward, weights 0.25 / 0.75 (modes at ±3)
python experiments/right_to_left_convergence_bimodal_same_bm/run.py $KEEPER \
  target.w1=0.25 \
  hydra.run.dir=results/run/2026-08-29_keeper_am_raw_w025_const2

# unimodal Gaussian reward at μ = 3
python experiments/right_to_left_convergence_bimodal_same_bm/run.py $KEEPER \
  target.mu1=3.0 target.mu2=3.0 \
  hydra.run.dir=results/run/2026-08-29_keeper_am_raw_gauss_mu3_const2
```

To re-clamp every sup/tiled norm at a percentile without retraining (writes a
`retiled_p90/` subfolder of the same 9 figures):

```python
from experiments.right_to_left_convergence_bimodal_same_bm.plotting import load_and_plot
load_and_plot("results/run/<dir>/metrics.json",
              output_dir="results/run/<dir>/retiled_p90", tiled_sup_percentile=90.0)
```

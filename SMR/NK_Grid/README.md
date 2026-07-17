# NK_Grid

Sweeps model performance jointly over sample size (N) and feature count (K),
writing one row per `(model, seed, draw, N, K)` combination to a CSV.
Supports regression and classification outcomes. Dataset-agnostic via
`--outcome`/`--predictor-prefix`.

Shared regression/classification model defaults live in `model_params.yaml`.
`panels.yaml` references that file once and declares one top-level `preset`
(`dev`, `medium`, or `production`) for every panel, while each panel's `models`
list still controls which models run.

The ten-model space is `ols`, `ridge`, `lasso`, `elastic_net`,
`random_forest`, `xgboost`, `lightgbm`, `shallow_neural_network`,
`extra_trees`, and `super_learner`. The Super Learner stacks four model
families using out-of-fold predictions: tuned Ridge (`RidgeCV`), Extra Trees,
a compact fixed-hyperparameter LightGBM (without nested CV), and a shallow
neural network. Regression neural networks standardize the target during
fitting. Legacy `bart` remains accepted for reproducibility.

## Data

Point `--data` at a CSV: one row per subject, an outcome column, and
predictor columns sharing a name prefix (default `Aset`/`Bset`). Cluster
users: data is already at `NK_Grid/data` (a symlink). Local-only setup: see
Notes.

## Setup

```bash
cd NK_Grid
./setup_env.sh
source ./activate_env.sh
```

Both helpers locate the environment relative to `NK_Grid/`. On a cluster,
create a fresh Linux `.venv` with `setup_env.sh`; do not copy a macOS virtual
environment. Set `PYTHON_BIN` before setup if the cluster's Python executable
has a different name, or set `VENV` to use a shared environment location.

## Quick start

```bash
python src/nk_grid.py --task regression --outcome Cm_lhourlywage \
  --models ridge --n-seeds 1 --n-draws 1 --n-sizes-n 2 --n-sizes-k 2 \
  --max-n 50 --max-k 20
```

Writes `outputs/nk_grid.csv` in seconds.

## Running sweeps

```bash
# Regression
python src/nk_grid.py --task regression --outcome Cm_lhourlywage \
  --models xgboost ridge lasso --n-seeds 2 --n-draws 2 \
  --n-sizes-n 4 --n-sizes-k 4 --max-n 100 --max-k 100

# Classification — template; fill in a real binary 0/1 column first
python src/nk_grid.py --task classification \
  --outcome "<confirmed binary 0/1 column>" \
  --models xgboost ridge lasso --out outputs/nk_grid_clf.csv
```

Run `python src/nk_grid.py --help` for all flags, or see Notes for the full
reference. See Notes for the dev/production scale presets and classification
model mapping before submitting a large run.

## Output

One row per `(model, seed, draw, N, K)`. `status` is `ok`, `skipped` (BART
below `--bart-min-n`/`--bart-min-k`, not attempted), or `failed` (raised an
exception, recorded in `error`). Re-running the same `--out` path resumes
from checkpoint. Full column reference in Notes.

## Multi-panel runs

Select the scale once at the top of `panels.yaml`; the setting applies to every
declared outcome:

```yaml
model_params: model_params.yaml
preset: medium
```

```bash
python src/run_panels.py --dry-run          # preview configs and run-size estimates
python src/run_panels.py                    # run every panel in panels.yaml
python src/run_panels.py --only smr_income  # run one named panel
```

Runs above 250,000 top-level model cells require explicit non-interactive
authorization with `--allow-large-run`. Output CSVs retain all established metrics
and add four filterable diagnostics: `K_varying`, `constant_prediction`,
`underdetermined`, and `converged`. Each CSV is paired with a minimal
`.manifest.json` and an atomic `.parts/` checkpoint directory; see
[`outputs/README.md`](outputs/README.md).

Fixed tree/LightGBM/NN parameters are selected once, before pilot/research runs,
using training-only anchor-cell CV. Preview that separate tuning budget with:

```bash
python src/tune_anchors.py \
  --outcomes Cm_lhourlywage Cm_ltotalincome \
  --dry-run
```

The tuning command writes a reviewable JSON recommendation and never edits
`model_params.yaml` automatically. After approval, copy the selected fixed values
into the YAML and increment `algorithm_version`; dev, pilot, and research runs then
reuse them without retuning.

The declared full anchor search currently exceeds its 1,000-fit safety threshold;
start it explicitly with the same non-interactive authorization convention:

```bash
python src/tune_anchors.py \
  --outcomes Cm_lhourlywage Cm_ltotalincome \
  --allow-large-run
```

Edit `panels.yaml` to fill in any placeholder outcome column before running
that panel.

## SLURM

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/NK_Grid
export VENV=/path/to/your/venv
sbatch slurm/run_nk_grid.sbatch
sbatch slurm/run_nk_grid_classification.sbatch
```

See Notes for resource sizing and per-model output layout.

## Notes

<details>
<summary>Local data setup (no cluster access)</summary>

`data/...` paths always resolve relative to `NK_Grid/` — the same on every
machine. What differs per machine is what sits at `NK_Grid/data` (cluster
symlink vs. a real local copy); the YAML/CLI never need per-machine edits.
To test locally with a real copy, replace `NK_Grid/data` with a directory
containing the same filenames, then run
`git update-index --skip-worktree NK_Grid/data` so git stops tracking that
local substitution (undo later with `--no-skip-worktree`). Never commit
real data through this path — `**/data/` is gitignored for exactly this
reason.

</details>

<details>
<summary>Dev vs. production scale, and why the two "dev" presets differ</summary>

- **Dev** (`nk_grid.py`'s own CLI defaults): `n-seeds=2 n-draws=2
  n-sizes-n=4 n-sizes-k=4 max-n=100 max-k=100` — minutes.
- **Production**: `n-seeds=100 n-draws=50 n-sizes-n=20 n-sizes-k=20
  max-n=0 max-k=0` (uncapped).

`run_panels.py`'s own `dev` preset (see Multi-panel runs) uses
`n-sizes-n/k=8`, not `4` — it's a separately tuned, independent layer, not
a typo.

Production scale is large: with ~5,000 training rows and ~4,000 predictors
(this repo's data), one model's full sweep is on the order of 10+ million
rows (`100 seeds × 50 draws × 20 × 20 grid`), multiplied by however many
models are listed. BART fits take tens of seconds per cell versus under a
second for other models, so including `bart` dominates runtime. Confirm
grid size and model list at dev scale before submitting a production run.

</details>

<details>
<summary>Classification model mapping</summary>

Under `--task classification`, model names map to classifiers, not
regressors: `ols`/`ridge`/`lasso`/`elastic_net` become logistic regression
variants (unpenalized / L2 / L1 / elastic-net); `random_forest`/`xgboost`/
`lightgbm`/`extra_trees` become their classifier counterparts, and
`shallow_neural_network` uses an MLP classifier. `super_learner` stacks
logistic regression, Extra Trees, fixed-hyperparameter LightGBM, and the
shallow neural network using out-of-fold predicted probabilities. Legacy
`bart` is not supported for classification (fails clearly). See
`model_registry.py` for the exact mapping.

</details>

<details>
<summary>Failure handling and resume behavior</summary>

`ok` and `skipped` combinations are not redone on resume; **`failed`
combinations are retried** on the next run. `skipped`/`failed` rows have
all metric columns empty.

</details>

<details>
<summary>The log grid, --batch-size, and --test-size</summary>

N values are spaced evenly in log2 space from `--min-n` (default `10`) up to
the cap. K retains the original log2 grid from 1 up to its cap. Both grids are
deduplicated to integers, so small values are sampled densely and large values
sparsely.

`--batch-size` (default `20`) is how many pending combinations are grouped
into one checkpoint-write cycle, globally across the run — not per
parallel worker (`--n-jobs` controls worker count independently).

`--test-size` (default `0.3`) is the test-set fraction; "70/30" refers to
the default, not fixed behavior — changing it changes the actual split.

</details>

<details>
<summary>Saving progress logs</summary>

Progress logs (`helpers_logging.py`) print to stderr only, not saved
automatically. Redirect if you want a copy, with `pipefail` so a real
failure isn't masked by `tee`'s own exit code:

```bash
set -o pipefail
python src/run_panels.py 2>&1 | tee run.log
```

</details>

<details>
<summary>SLURM resource sizing and output layout</summary>

`slurm/*.sbatch` submit a 10-way job array (one array task per model, 8
CPUs / 48G mem / 4-day time limit per task — edit the scripts to adjust).
Each array task writes its own CSV (`outputs/nk_grid_<model>.csv`) — this
differs from running `nk_grid.py`/`run_panels.py` directly with multiple
`--models`, which combine them into one shared CSV.

Output/error logs land in `logs/<job-name>-<job-id>_<array-index>.out/.err`
(the tracked `logs/` directory must exist before submission, which it
does). Cancel with `scancel <job-id>`; check status with `squeue --me`.

</details>

<details>
<summary>Full parameter reference</summary>

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/asample2_withlag.csv` | Path to the analysis CSV. |
| `--task` | `regression` | `regression` or `classification`. |
| `--outcome` | required | Outcome column name (both tasks). |
| `--predictor-prefix` | `Aset Bset` | Prefixes selecting predictor columns. |
| `--out` | `outputs/nk_grid.csv` / `outputs/nk_grid_clf.csv` | Output CSV path. |
| `--dataset` | `asample2_withlag` | Free-text label in the `dataset` column. |
| `--models` | `xgboost` | `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, shallow_neural_network, extra_trees, super_learner`; legacy `bart` is also accepted. |
| `--seed` | `12345` | Base seed; each of `n-seeds` runs uses `seed + offset`. |
| `--test-size` | `0.3` | Test-set fraction of the split. |
| `--n-seeds` | `2` | Independent train/test splits. |
| `--n-draws` | `2` | Repeated subsamples per seed. |
| `--n-sizes-n` / `--n-sizes-k` | `4` / `4` | Points on the log-scale N / K grid. |
| `--min-n` | `10` | Minimum N grid value; K still starts at 1. |
| `--max-n` / `--max-k` | `100` / `100` | Grid ceiling; `<=0` uncaps. |
| `--model-params` | `model_params.yaml` | Task-specific defaults used to construct models. |
| `--batch-size` | `20` | Combinations per checkpoint write. |
| `--bart-min-n` / `--bart-min-k` | `10` / `2` | BART cells below this are `skipped`. |
| `--group-split-col` | `None` | Reserved; raises `NotImplementedError` if set. |
| `--n-jobs` | `$SLURM_CPUS_PER_TASK` or `1` | Parallel worker count. |

</details>

<details>
<summary>Full output schema</summary>

Regression's 30 metrics are in `METRIC_COLUMNS`, classification's 8 are in
`CLASSIFICATION_METRIC_COLUMNS`, both in `src/nk_grid.py`.

</details>

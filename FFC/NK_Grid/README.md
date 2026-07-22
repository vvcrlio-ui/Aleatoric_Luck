# NK_Grid (FFC adaptation)

Sweeps model performance jointly over sample size (N) and feature count (K),
writing one row per `(model, seed, draw, N, K)` combination to a CSV.
Supports regression and classification outcomes. This copy adds FFC-specific
behavior the original (SMR) NK_Grid tool did not need:

- `src/prepare_ffc_analysis.py` / `src/prepare_ffc_nk_inputs.py`: clean
  `data/private/background.dta` into a numeric feature matrix (`X_`
  continuous, `C_` one-hot categorical, `M_` missing indicators) and split it
  per outcome into FFC's **official** train/test files.
- `--test-data` / `test:` (panels.yaml): fit on the full training pool,
  evaluate on a fixed external test set instead of an internal random split.

The shared model defaults live in `model_params.yaml`, referenced once at the
top of `panels.yaml`. Each panel's `models` list still selects which models run;
the YAML supplies their regression- or classification-specific parameters.

## Data

FFC's six outcomes each have their own cleaned train/test pair under
`data/intermediate_files/nk_inputs/` (gitignored, not committed — see
"Regenerating the cleaned data" below). `panels.yaml` already points each
panel at the right files, predictor prefixes (`X_ C_ M_`), and `test:` file —
you should not need to touch `--data`/`--predictor-prefix` by hand.

## Setup

```bash
cd FFC/NK_Grid
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Regenerating the cleaned data

Requires `data/private/background.dta`, `train.csv`, `test.csv` (FFC
restricted data — not in this repo; see the Office of Population Research
Data Archive, or ask whoever gave you access to this codebase):

```bash
python src/prepare_ffc_analysis.py     # background.dta -> cleaned feature matrix
python src/prepare_ffc_nk_inputs.py    # + train/test outcomes -> per-outcome nk_inputs/
```

## Quick start (one outcome, one panel)

```bash
python src/run_panels.py --manifest panels.yaml --only ffc_gpa
```

Uses whatever `preset:` `ffc_gpa` currently declares in `panels.yaml`
(`dev` for a fast local check, `production` for a full-scale run — see
Notes).

## Running a single sweep directly

```bash
python src/nk_grid.py --task regression --outcome gpa \
  --data data/intermediate_files/nk_inputs/ffc_train_gpa.csv \
  --test-data data/intermediate_files/nk_inputs/ffc_test_gpa.csv \
  --predictor-prefix X_ C_ M_ \
  --models xgboost ridge lasso --n-seeds 2 --n-draws 2 \
  --n-sizes-n 4 --n-sizes-k 4 --max-n 100 --max-k 100
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

```bash
python src/run_panels.py --dry-run       # preview all 6 panels
python src/run_panels.py                 # run every panel in panels.yaml
python src/run_panels.py --only ffc_gpa  # run one named panel
```

Panel names: `ffc_gpa`, `ffc_grit`, `ffc_materialHardship` (regression),
`ffc_eviction`, `ffc_layoff`, `ffc_jobTraining` (classification).

## SLURM

```bash
export PROJECT_DIR=/path/to/FFC/NK_Grid
export VENV=/path/to/your/venv
sbatch slurm/run_nk_grid.sbatch
```

One `#SBATCH --array` task per outcome/panel (array size 6) — see Notes for
resource sizing.

## Notes

<details>
<summary>Local data setup (no cluster access)</summary>

`data/` paths always resolve relative to `FFC/NK_Grid/` — the same on every
machine. `data/` (raw, private, and generated `intermediate_files/`) is
gitignored everywhere in this repo; it is not a symlink here, just a real
local directory each machine populates independently by running
`prepare_ffc_analysis.py` / `prepare_ffc_nk_inputs.py` against its own copy
of the restricted FFC files. Never commit real data through this path.

</details>

<details>
<summary>Dev vs. production scale</summary>

- **Dev** (`panels.yaml`'s `dev` preset): `n-seeds=3 n-draws=3
  n-sizes-n=5 n-sizes-k=5 max-n=100 max-k=100` — under a minute per panel.
- **Production**: `n-seeds=100 n-draws=50 n-sizes-n=20 n-sizes-k=20
  max-n=0 max-k=0` (uncapped) — matches the SMR project's production preset
  for cross-paper comparability.

`bart` is excluded from every FFC panel (commented out in `panels.yaml`) —
`bartpy2` is not reliable at the small N cells this grid visits. Of the
remaining 7 models, `elastic_net`/`lasso` are the cost drivers at large K:
benchmarked directly against FFC's real ~14,249-feature matrix, a single
`elastic_net` fit at full K (N=800) takes ~176s, `lasso` ~74s — both far
cheaper than they were at the equivalent SMR scale, because most FFC
features are sparse 0/1 dummies (`C_`/`M_`) rather than dense continuous
columns. No `max_k` cap is needed for production here. Still confirm grid
size and model list at dev scale before submitting a full production run.

</details>

<details>
<summary>Classification model mapping</summary>

Under `--task classification`, model names map to classifiers, not
regressors: `ols`/`ridge`/`lasso`/`elastic_net` become logistic regression
variants (unpenalized / L2 / L1 / elastic-net); `random_forest`/`xgboost`/
`lightgbm` become their classifier counterparts; `bart` is not supported
for classification (fails clearly). See `model_registry.py` for the exact
mapping.

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

`slurm/run_nk_grid.sbatch` submits a 6-way job array — one array task per
outcome/panel (8 CPUs / 48G mem / 4-day time limit per task — edit the
script to adjust). Each array task runs all 7 models for that outcome
sequentially (via `run_panels.py`) and writes one timestamped output file
per completed run (see the `preset` output-naming behavior in `nk_grid.py`
— re-running a finished config starts a new file rather than overwriting).

Output/error logs land in `logs/<job-name>-<job-id>_<array-index>.out/.err`
(the tracked `logs/` directory must exist before submission, which it
does). Cancel with `scancel <job-id>`; check status with `squeue --me`.

</details>

<details>
<summary>Full parameter reference</summary>

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/asample2_withlag.csv` (SMR default — pass FFC's `ffc_train_<outcome>.csv` explicitly) | Path to the training CSV. |
| `--test-data` | `None` | Fixed external test CSV; enables external-test mode (ignores `--test-size`). FFC panels always set this to `ffc_test_<outcome>.csv`. |
| `--task` | `regression` | `regression` or `classification`. |
| `--outcome` | required | Outcome column name (both tasks). |
| `--predictor-prefix` | `Aset Bset` (SMR default — FFC panels use `X_ C_ M_`) | Prefixes selecting predictor columns. |
| `--out` | `outputs/nk_grid.csv` / `outputs/nk_grid_clf.csv` | Output CSV path (or template path when `preset` is set — see the SLURM resource sizing note). |
| `--dataset` | `asample2_withlag` (SMR default) | Free-text label in the `dataset` column. |
| `--models` | `xgboost` | `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, bart`. |
| `--seed` | `12345` | Base seed; each of `n-seeds` runs uses `seed + offset`. |
| `--test-size` | `0.3` | Test-set fraction of the split. |
| `--n-seeds` | `2` | Independent train/test splits. |
| `--n-draws` | `2` | Repeated subsamples per seed. |
| `--n-sizes-n` / `--n-sizes-k` | `4` / `4` | Points on the log-scale N / K grid. |
| `--min-n` | `10` | Minimum N grid value; K still starts at 1. |
| `--max-n` / `--max-k` | `100` / `100` | Grid ceiling; `<=0` uncaps. |
| `--model-params` | `model_params.yaml` | Task-specific defaults for model construction. `panels.yaml` references the same file at manifest level. |
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

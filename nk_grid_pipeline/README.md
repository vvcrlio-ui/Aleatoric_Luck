# NLSY79 Zheng–Cheng Replication and Extensions

This module uses the processed NLSY79 analysis tables produced by the authors'
public [OSF source package](https://osf.io/7wjg8/). It has two distinct purposes:

1. `overall_prediction.py` reproduces the cumulative Aset1/Aset2/Bset1/Bset2
   comparison with a 70/30 split and source seed `12345`.
2. The feature-count, domain, SHAP, and sample-size scripts are extensions. They
   should not be described as analyses contained in the Zheng–Cheng paper.

The error-floor parameter from the sample-size extension is conditional on the
data, predictor space, model family, split, and power-law extrapolation. It is
not a model-independent estimate of Bayes error or literal "luck."

This is a Python structural replication, not a byte-for-byte reproduction of
the original R/Stata workflow. R's `set.seed(12345)` and Python's
`random_state=12345` do not generate identical train/test row assignments.

## Data

Data are not committed. Obtain or construct these files using the authors'
public data-management code and place them under `nk_grid_pipeline/data/`:

- `asample1_noincomelag.csv`
- `asample2_withlag.csv`
- `asample3_nosei.csv`

The default scripts use `asample2_withlag.csv` and `Cm_lhourlywage`.

## Environment

Python 3.11 is recommended:

```bash
cd nk_grid_pipeline
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Source-aligned cumulative comparison

```bash
python src/overall_prediction.py \
  --models ols ridge lasso xgboost bart
```

XGBoost uses `max_depth=2`, `eta=0.3`, and five-fold CV to choose up to 90
boosting rounds. Ridge and Lasso also select penalties using five-fold CV.
BART uses the paper's 200 trees, 1,000 recorded iterations (without thinning),
and 100 burn-in iterations by default. BART jobs use process-based parallelism because BartPy's
random-number generator is process-global.

## Extensions

```bash
python src/feature_sets.py --models xgboost ridge lasso
python src/domain_wise.py --models xgboost ridge lasso
python src/sample_size.py --models xgboost ridge lasso
python src/SHAP_vals.py
python src/SHAP_experiment.py
```

Long-running scripts checkpoint CSV results after each batch and resume
successful jobs from an existing output file. Failed model fits are recorded
with `status=failed` instead of discarding the whole run.

Each checkpoint row includes an `experiment_id`, outcome, data hash, split, and
other identifying metadata. Reusing an output path with a different experiment
does not skip work or overwrite the earlier experiment. Checkpoints created
before this metadata was introduced must be removed or passed under a new
`--out` path before resuming.

Output columns distinguish standard `r2_test_mean_baseline` from the paper's
`r2_train_mean_baseline`. In sample-size experiments, the latter uses the mean
of the actual training subset for each draw.

Power-law fits use non-negative bounds, bootstrap intervals, and stability
diagnostics. The bootstrap interval remains conditional on the fixed train/test
sample; it does not capture population sampling or holdout uncertainty.

## Joint N x K sweep (`nk_grid.py`)

`nk_grid.py` runs a single reusable sweep over sample size (N) and feature
count (K) on a shared base-2 log grid, and writes one long-format CSV where
each row is one (model, seed, draw, N, K) combination. It supports both a
continuous-outcome (`regression`) task and a binary-outcome (`classification`)
task from the same script.

```bash
# Regression (continuous outcome, e.g. log hourly wage)
python src/nk_grid.py \
  --task regression \
  --outcome Cm_lhourlywage \
  --models xgboost ridge lasso \
  --n-seeds 2 --n-draws 2 \
  --n-sizes-n 4 --n-sizes-k 4 \
  --max-n 100 --max-k 100

# Classification (binary outcome, e.g. employment)
python src/nk_grid.py \
  --task classification \
  --outcome <confirmed binary employment column> \
  --models xgboost ridge lasso \
  --out outputs/nk_grid_clf.csv
```

### Parameters

| Flag | Default | Meaning | How to set it |
|---|---|---|---|
| `--data` | `data/asample2_withlag.csv` | Path to the analysis CSV. | Point at whichever `asample*.csv` you're sweeping. |
| `--task` | `regression` | `regression` (continuous outcome, 30 continuous metrics) or `classification` (binary outcome, 8 classification metrics incl. ROC-AUC). | Pick based on the outcome column's type. |
| `--outcome` | `Cm_lhourlywage` for regression; **required** for classification | Outcome column name. | For `classification`, you must pass the confirmed binary (0/1) column explicitly — the script refuses to guess one. |
| `--out` | `outputs/nk_grid.csv` (regression) / `outputs/nk_grid_clf.csv` (classification) | Output CSV path. | Give each figure/panel/dataset its own path so runs don't overwrite each other. |
| `--dataset` | `asample2_withlag` | Free-text label written into the `dataset` column. | Set to something that identifies the source table in the output CSV. |
| `--models` | `xgboost` | One or more of the registered model names (see `MODEL_NAMES` in `model_registry.py`: `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, bart`). | Pass the models you want compared; each gets its own row per (seed, draw, N, K). |
| `--seed` | `12345` | Base seed. Each of the `n_seeds` runs uses `seed + offset` as its own `random_state` for a fresh 70/30 train/test split. | Keep the paper-aligned default unless you need a different starting point for reproducibility. |
| `--test-size` | `0.3` | Test-set fraction of the 70/30 split. | Match the paper's split unless intentionally deviating. |
| `--n-seeds` | `2` | Number of independent train/test splits (the outer "seed" loop — controls data-split variation). | Small (2) for local smoke tests; large (~100) for production error bars. |
| `--n-draws` | `2` | Number of repeated subsamples drawn *within* each seed's training set (the "draw" loop — controls sampling variation at fixed N/K). | Small (2) for smoke tests; large (~50) for production. |
| `--n-sizes-n` | `4` | Number of points on the log-scale N grid (sample size axis). | More points = smoother sample-size curve, more compute. |
| `--n-sizes-k` | `4` | Number of points on the log-scale K grid (feature-count axis). | More points = smoother feature-count curve, more compute. |
| `--max-n` | `100` | Upper cap on N. Use `0` (or any value `<=0`) to uncap and use the full training set. | `100` (or similar) for dev; `0` for the production run so the grid reaches the full training set. |
| `--max-k` | `100` | Upper cap on K. Use `0` (or any value `<=0`) to uncap and use all available predictors. | Same pattern as `--max-n`: small for dev, `0` for production. |
| `--batch-size` | `20` | Number of (model, seed, draw, N, K) jobs processed per checkpoint write. | Larger batches write the checkpoint less often (slightly faster); smaller batches checkpoint more often (safer against interruption). |
| `--bart-min-n` | `10` | Minimum N required before fitting BART. Smaller BART cells are marked `skipped`. | Keep the default unless a future BART backend handles tiny samples safely. |
| `--bart-min-k` | `2` | Minimum K required before fitting BART. Smaller BART cells are marked `skipped`. | Keep the default unless a future BART backend handles single-predictor trees safely. |
| `--group-split-col` | `None` | Reserved for a future family/sibling-grouped split (e.g. to avoid NLSY79 sibling leakage). | Leave unset; currently raises `NotImplementedError` if provided — not yet implemented. |
| `--n-jobs` | `$SLURM_CPUS_PER_TASK` or `1` | Parallel worker count for `joblib`. | Set to the number of available CPU cores; on SLURM this is picked up automatically from the job's `--cpus-per-task`. |

### Dev vs. production parameter presets

The intended workflow is to start with small, cheap values, confirm the
pipeline and CSV schema look right, and then scale up for the real run
without changing any code — only the CLI flags:

- **Dev** (fast local smoke test): `--n-seeds 2 --n-draws 2 --n-sizes-n 4
  --n-sizes-k 4 --max-n 100 --max-k 100`.
- **Production** (full run, matches `slurm/run_nk_grid.sbatch` and
  `slurm/run_nk_grid_classification.sbatch` defaults): `--n-seeds 100
  --n-draws 50 --n-sizes-n 20 --n-sizes-k 20 --max-n 0 --max-k 0`.

### Checkpointing and failure handling

Each (model, seed, draw, N, K) combination is written as its own row.
Successful fits are marked `status=ok`. BART cells below `--bart-min-n` or
`--bart-min-k` are not fitted and are marked `status=skipped` with
`error=below BART minimum N/K floor`. A model that is actually attempted and
raises an exception is marked `status=failed` with the exception recorded in
`error`, and the sweep continues rather than aborting. Re-running the same
`--out` path resumes from the checkpoint and skips combinations already
recorded as `ok` or `skipped`.

## Multi-panel runner (`run_panels.py`)

`run_panels.py` reads a declarative JSON manifest and runs one independent
`nk_grid.py` configuration per figure or panel. Presets centralize the common
grid sizes:

- `dev`: `n_seeds=2, n_draws=2, n_sizes_n=4, n_sizes_k=4, max_n=100, max_k=100`
- `medium`: `n_seeds=2, n_draws=2, n_sizes_n=4, n_sizes_k=4, max_n=100, max_k=100`
- `production`: `n_seeds=100, n_draws=50, n_sizes_n=20, n_sizes_k=20, max_n=0, max_k=0`

Each panel may override any preset value. The default manifest is
`panels.json`:

```bash
python src/run_panels.py --dry-run
python src/run_panels.py --only smr_income
python src/run_panels.py --manifest panels.json
```

Each panel writes to its own CSV and resumes through the same checkpoint
mechanism as `nk_grid.py`, so interrupted panel runs can be repeated without
duplicating completed rows.

## SLURM

The scripts in `slurm/` use job arrays, contain no user-specific cluster path,
and write one output per model. Submit them from `nk_grid_pipeline/`; the tracked
`logs/` directory lets Slurm open stdout and stderr before the script starts.
`VENV` defaults to `$PROJECT_DIR/.venv`, matching the `Environment` setup
above, so anyone who follows those steps can submit jobs with no extra
configuration. Set these variables when cluster defaults differ instead, e.g.
if your venv lives somewhere else:

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/nk_grid_pipeline
export VENV=/path/to/your/venv
export PYTHON_MODULE=Python/3.11
sbatch slurm/run_overall.sbatch
sbatch slurm/run_sample_size_light.sbatch
sbatch slurm/run_sample_size_bart.sbatch
```

`run_sample_size_light.sbatch` arrays over the non-BART models with the
same resources as the other extension scripts. `run_sample_size_bart.sbatch`
runs BART alone, with more memory and deliberately low concurrency, because
it is far heavier than the other models and was previously OOM-killed when
sharing a job's memory budget with them.

Adjust time, memory, partition, account, and module names for the target
cluster. Merge model-specific CSVs only after all array tasks finish.

## Colab

Open `colab_run.ipynb`. It clones the public repository, mounts Google Drive
for data only, installs the minimal notebook requirements, and runs smoke-sized
commands by default.

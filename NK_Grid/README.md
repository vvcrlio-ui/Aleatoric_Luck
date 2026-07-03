# NK_Grid

`NK_Grid` runs joint sample-size (N) by feature-count (K) sweeps on a shared
base-2 log grid. It writes long-format CSV output where each row is one
`(model, seed, draw, N, K)` combination. The same entry point supports
continuous-outcome regression and binary-outcome classification tasks.

## Data

Data are not committed. The tracked `data` path is a symlink to the shared
cluster data directory:

```text
/gpfs3/users/mills/tej036/aleatoric-luck/data
```

The default scripts use `data/asample2_withlag.csv` and `Cm_lhourlywage`.

## Environment

Python 3.11 is recommended:

```bash
cd NK_Grid
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Joint N x K sweep

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
| `--outcome` | `Cm_lhourlywage` for regression; **required** for classification | Outcome column name. | For `classification`, pass the confirmed binary (0/1) column explicitly; the script refuses to guess one. |
| `--out` | `outputs/nk_grid.csv` (regression) / `outputs/nk_grid_clf.csv` (classification) | Output CSV path. | Give each figure, panel, or dataset its own path so runs do not overwrite each other. |
| `--dataset` | `asample2_withlag` | Free-text label written into the `dataset` column. | Set to something that identifies the source table in the output CSV. |
| `--models` | `xgboost` | One or more registered model names: `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, bart`. | Pass the models you want compared; each gets its own row per `(seed, draw, N, K)`. |
| `--seed` | `12345` | Base seed. Each of the `n_seeds` runs uses `seed + offset` for a fresh 70/30 train/test split. | Keep the default unless you need a different starting point for reproducibility. |
| `--test-size` | `0.3` | Test-set fraction of the 70/30 split. | Match the paper's split unless intentionally deviating. |
| `--n-seeds` | `2` | Number of independent train/test splits. | Small for local smoke tests; large for production error bars. |
| `--n-draws` | `2` | Number of repeated subsamples within each seed's training set. | Small for smoke tests; larger for production. |
| `--n-sizes-n` | `4` | Number of points on the log-scale N grid. | More points give a smoother sample-size curve and more compute. |
| `--n-sizes-k` | `4` | Number of points on the log-scale K grid. | More points give a smoother feature-count curve and more compute. |
| `--max-n` | `100` | Upper cap on N. Use `0` or any value `<=0` to uncap and use the full training set. | `100` for development; `0` for production. |
| `--max-k` | `100` | Upper cap on K. Use `0` or any value `<=0` to uncap and use all available predictors. | Same pattern as `--max-n`. |
| `--batch-size` | `20` | Number of jobs processed per checkpoint write. | Larger batches write less often; smaller batches checkpoint more often. |
| `--bart-min-n` | `10` | Minimum N required before fitting BART. Smaller BART cells are marked `skipped`. | Keep the default unless a future BART backend handles tiny samples safely. |
| `--bart-min-k` | `2` | Minimum K required before fitting BART. Smaller BART cells are marked `skipped`. | Keep the default unless a future BART backend handles single-predictor trees safely. |
| `--group-split-col` | `None` | Reserved for a future grouped split. | Leave unset; currently raises `NotImplementedError` if provided. |
| `--n-jobs` | `$SLURM_CPUS_PER_TASK` or `1` | Parallel worker count for `joblib`. | Set to available CPU cores; on SLURM this is picked up automatically. |

### Dev vs. production parameter presets

- **Dev**: `--n-seeds 2 --n-draws 2 --n-sizes-n 4 --n-sizes-k 4 --max-n 100 --max-k 100`.
- **Production**: `--n-seeds 100 --n-draws 50 --n-sizes-n 20 --n-sizes-k 20 --max-n 0 --max-k 0`.

### Checkpointing and failure handling

Each `(model, seed, draw, N, K)` combination is written as its own row.
Successful fits are marked `status=ok`. BART cells below `--bart-min-n` or
`--bart-min-k` are not fitted and are marked `status=skipped` with
`error=below BART minimum N/K floor`. A model that is attempted and raises an
exception is marked `status=failed` with the exception recorded in `error`.
Re-running the same `--out` path resumes from the checkpoint and skips
combinations already recorded as `ok` or `skipped`.

## Multi-panel runner

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
and write one output per model. Submit them from `NK_Grid/`; the tracked
`logs/` directory lets Slurm open stdout and stderr before the script starts.
`VENV` defaults to `$PROJECT_DIR/.venv`, matching the environment setup above.

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/NK_Grid
export VENV=/path/to/your/venv
export PYTHON_MODULE=Python/3.11
sbatch slurm/run_nk_grid.sbatch
sbatch slurm/run_nk_grid_classification.sbatch
```

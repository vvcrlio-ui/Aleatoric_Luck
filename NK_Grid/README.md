# NK_Grid

`NK_Grid` sweeps model performance jointly over sample size (N) and feature
count (K) on a shared log grid, and writes one row per `(model, seed, draw,
N, K)` combination to a long-format CSV. The same entry point supports
continuous-outcome regression and binary-outcome classification. It is
dataset-agnostic: point `--outcome` and `--predictor-prefix` at any analysis
table's columns to run it on a different paper's data, no code changes
needed.

## Data

Point `--data` at a CSV with one row per subject: an outcome column and a
set of predictor columns sharing a name prefix (defaults below assume the
`Aset`/`Bset` prefixes used in this repo's Zheng-Cheng data).

If you have cluster access, data lives at `NK_Grid/data`, a tracked symlink
to a shared cluster directory (not part of this git repo — a separate,
pre-existing data store). If you don't, or want to test with your own
local copy, see "Using local data instead of the cluster" in Notes below.

## Setup

Python 3.11 recommended:

```bash
cd NK_Grid
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

A minimal smoke test (needs the data above; if it's missing you'll get a
clear `FileNotFoundError` pointing at the missing path, not a silent hang):

```bash
python src/nk_grid.py --task regression --outcome Cm_lhourlywage \
  --models ridge --n-seeds 1 --n-draws 1 --n-sizes-n 2 --n-sizes-k 2 \
  --max-n 50 --max-k 20
```

This runs in seconds and writes `outputs/nk_grid.csv`. Scale up from here —
see "Dev vs. production" below.

## Running sweeps

```bash
# Regression (continuous outcome)
python src/nk_grid.py \
  --task regression \
  --outcome Cm_lhourlywage \
  --models xgboost ridge lasso \
  --n-seeds 2 --n-draws 2 \
  --n-sizes-n 4 --n-sizes-k 4 \
  --max-n 100 --max-k 100

# Classification (binary outcome) — TEMPLATE, not runnable as-is: replace
# the placeholder with a confirmed binary 0/1 column first. Quotes keep
# `<`/`>` from being parsed as shell redirection if pasted literally.
python src/nk_grid.py \
  --task classification \
  --outcome "<confirmed binary 0/1 column>" \
  --models xgboost ridge lasso \
  --out outputs/nk_grid_clf.csv
```

Under `--task classification`, model names map to classifiers, not
regressors: `ols`/`ridge`/`lasso`/`elastic_net` become logistic regression
variants (unpenalized / L2 / L1 / elastic-net), `random_forest`/`xgboost`/
`lightgbm` become their classifier counterparts, and `bart` is not
supported for classification (fails clearly rather than silently). See
`model_registry.py` for the exact mapping.

Key flags: `--outcome` (required for both tasks — never guessed),
`--predictor-prefix` (default `Aset Bset`), `--models`, `--n-seeds`/
`--n-draws` (sampling repeats), `--n-sizes-n`/`--n-sizes-k` (grid
resolution), `--max-n`/`--max-k` (grid ceiling, `0` = uncapped). Run
`python src/nk_grid.py --help` for the full list, or see the Notes section
for the complete parameter table.

### Dev vs. production scale

- **Dev** (`nk_grid.py`'s own defaults): `n-seeds=2 n-draws=2 n-sizes-n=4
  n-sizes-k=4 max-n=100 max-k=100` — runs in minutes.
- **Production**: `n-seeds=100 n-draws=50 n-sizes-n=20 n-sizes-k=20 max-n=0
  max-k=0` (uncapped, uses the full dataset).

Production scale is large: with ~5,000 training rows and ~4,000 predictors
(this repo's data), one model's full sweep is on the order of **10+ million
rows** (`100 seeds × 50 draws × 20 × 20 grid`). Multiply by however many
models you list. BART fits are much slower per cell than the other models
(tens of seconds vs. under a second), so a model list that includes `bart`
dominates total runtime. Don't submit a production run without first
confirming grid size and model list at dev scale.

Note `run_panels.py`'s `dev` preset (below) uses `n-sizes-n/k=5`, not `4` —
that preset was tuned separately after this CLI's own defaults were set;
they're two independently configured layers, not a typo.

## Output

Each `(model, seed, draw, N, K)` row has identifying columns (`model`,
`seed`, `draw`, `N`, `K`, `status`, `error`, plus provenance like
`experiment_id`) and either 30 continuous metrics (`--task regression`,
e.g. `r2_test`, `rmse`, `spearman_rho`) or 8 classification metrics
(`--task classification`, e.g. `roc_auc`, `brier`, `mcfadden_pseudo_r2`).
See Notes for the full column reference.

`status` is one of:
- `ok` — fit succeeded, all metrics populated.
- `skipped` — BART cell below `--bart-min-n`/`--bart-min-k`; never
  attempted, metrics are empty.
- `failed` — attempted and raised an exception, recorded in `error`,
  metrics are empty.

Re-running the same `--out` path resumes from checkpoint: `ok` and
`skipped` combinations are not redone, but **`failed` combinations are
retried** on the next run.

## Multi-panel runs

`run_panels.py` runs one independent `nk_grid.py` configuration per named
"panel" declared in a YAML manifest (default `panels.yaml`), using shared
presets (`dev`/`medium`/`production`) so you don't repeat the same six
numbers in every panel.

```bash
python src/run_panels.py --dry-run          # preview without running
python src/run_panels.py                    # run every panel in panels.yaml
python src/run_panels.py --only smr_income  # run just one named panel
```

Each panel writes its own CSV and resumes the same way as `nk_grid.py`.
Panels whose outcome column isn't confirmed yet ship with a placeholder
(e.g. `<TBD employment column>`) — edit `panels.yaml` before running that
panel, or it fails immediately with a clear error rather than guessing.

## SLURM

`slurm/*.sbatch` submit an 8-way job array (one array task per model, `8
cpus` / `48G` mem / 4-day time limit per task — see the scripts to adjust).
**Each array task writes its own CSV** (`outputs/nk_grid_<model>.csv`) —
this differs from running `nk_grid.py` or `run_panels.py` directly with
multiple `--models`, which combine them into one shared CSV.

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/NK_Grid
export VENV=/path/to/your/venv
sbatch slurm/run_nk_grid.sbatch
sbatch slurm/run_nk_grid_classification.sbatch
```

Output/error logs land in `logs/<job-name>-<job-id>_<array-index>.out/.err`
(the tracked `logs/` directory must exist before submission, which it
does). Cancel with `scancel <job-id>`; check status with `squeue --me`.

## Notes

**Using local data instead of the cluster**: `data/...` paths always
resolve relative to `NK_Grid/`, on every machine — the YAML/CLI never needs
per-machine edits. What differs per machine is what sits at `NK_Grid/data`
(cluster symlink vs. a real local copy). To test locally with a real copy,
replace `NK_Grid/data` with a directory containing the same filenames, then
run `git update-index --skip-worktree NK_Grid/data` so git stops tracking
that local substitution (undo later with `--no-skip-worktree`). Never
commit real data through this path — `**/data/` is gitignored for exactly
this reason.

**The log grid**: N and K values are spaced evenly in log2 space from 1 up
to the cap (`--max-n`/`--max-k`, or the full dataset if uncapped),
deduplicated to integers — so small values are sampled densely and large
values sparsely, matching how prediction accuracy typically saturates.

**`--batch-size`** (default `20`): how many pending combinations are
grouped into one checkpoint-write cycle, globally across the run — not
per parallel worker (`--n-jobs` controls worker count independently).

**`--test-size`** (default `0.3`): the test-set fraction of the train/test
split; "70/30" in this doc refers to the default, not a fixed behavior —
changing `--test-size` changes the actual split ratio.

**Saving logs**: progress logs (`helpers_logging.py`) print to stderr only
and aren't saved automatically. Redirect if you want a copy, and use
`pipefail` so a real failure isn't masked by `tee`'s own exit code:

```bash
set -o pipefail
python src/run_panels.py 2>&1 | tee run.log
```

**Full parameter reference**:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/asample2_withlag.csv` | Path to the analysis CSV. |
| `--task` | `regression` | `regression` or `classification`. |
| `--outcome` | required | Outcome column name (both tasks). |
| `--predictor-prefix` | `Aset Bset` | Prefixes selecting predictor columns. |
| `--out` | `outputs/nk_grid.csv` / `outputs/nk_grid_clf.csv` | Output CSV path. |
| `--dataset` | `asample2_withlag` | Free-text label in the `dataset` column. |
| `--models` | `xgboost` | `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, bart`. |
| `--seed` | `12345` | Base seed; each of `n-seeds` runs uses `seed + offset`. |
| `--test-size` | `0.3` | Test-set fraction of the split. |
| `--n-seeds` | `2` | Independent train/test splits. |
| `--n-draws` | `2` | Repeated subsamples per seed. |
| `--n-sizes-n` / `--n-sizes-k` | `4` / `4` | Points on the log-scale N / K grid. |
| `--max-n` / `--max-k` | `100` / `100` | Grid ceiling; `<=0` uncaps. |
| `--batch-size` | `20` | Combinations per checkpoint write (see above). |
| `--bart-min-n` / `--bart-min-k` | `10` / `2` | BART cells below this are `skipped`, not attempted. |
| `--group-split-col` | `None` | Reserved; raises `NotImplementedError` if set. |
| `--n-jobs` | `$SLURM_CPUS_PER_TASK` or `1` | Parallel worker count. |

**Full output schema**: regression's 30 metrics are in `METRIC_COLUMNS`,
classification's 8 are in `CLASSIFICATION_METRIC_COLUMNS`, both in
`src/nk_grid.py`.

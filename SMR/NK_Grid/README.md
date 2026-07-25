# NK_Grid

This project evaluates model performance jointly as sample size (N) and feature
count (K) vary, writing one row to a CSV for each
`(model, seed, draw, N, K)` combination. It supports both regression and
classification tasks and can be adapted to different datasets through `--outcome`
and `--predictor-prefix`.

Shared default parameters for regression and classification models are stored in
`model_params.yaml`. `panels.yaml` references this file once and sets a top-level
`preset` (`dev`, `medium`, `pilot_full`, or `production`) that applies to all
panels; the `models` list in each panel still determines which models actually run.

The current model space contains ten models: `ols`, `ridge`, `lasso`,
`elastic_net`, `random_forest`, `xgboost`, `lightgbm`,
`shallow_neural_network`, `extra_trees`, and `super_learner`.

## Data

Use `--data` to specify a CSV file. The data should contain one row per subject,
one outcome column, and a set of predictor columns whose names share common
prefixes (the default prefixes are `Aset` and `Bset`). Cluster users can use the
data under `NK_Grid/data` directly; this path may be a symbolic link or a data
directory. See “Notes” for local data setup.

## Environment setup

```bash
cd NK_Grid
./setup_env.sh
source ./activate_env.sh
```

Both scripts locate the environment relative to `NK_Grid/`. On a cluster, use
`setup_env.sh` to create a new Linux `.venv`; do not copy a virtual environment
created on macOS. If the Python executable has a different name on the cluster,
set `PYTHON_BIN` before setup. To use a shared virtual environment in another
location, set `VENV`.

## Quick start

```bash
python src/nk_grid.py --task regression --outcome Cm_lhourlywage \
  --models ridge --n-seeds 1 --n-draws 1 --n-sizes-n 2 --n-sizes-k 2 \
  --max-n 50 --max-k 20
```

This command creates `outputs/nk_grid.csv` within seconds.

## Running experiments

```bash
# Preview every active panel and its estimated size
python src/run_panels.py --dry-run

# Run every active regression and classification panel
python src/run_panels.py
```

`run_panels.py` reads `panels.yaml` and runs only the panels declared there. If the
manifest contains both regression and classification panels, both tasks run; if
one task type is absent, no jobs are created for it. Each panel uses 4 parallel
workers by default, or `SLURM_CPUS_PER_TASK` when that variable is available.

Use `python src/nk_grid.py --help` only for lower-level single-task options, or see
“Notes” for the full parameter table.

## Output

Each `(model, seed, draw, N, K)` combination corresponds to one row in the results
CSV. `status` indicates the run status of that combination:

- `ok`: The model ran successfully and its metrics were computed normally.
- `skipped`: The combination did not meet the requirements for execution, so the
  model was not fitted; the specific reason is recorded in `error`.
- `failed`: An exception occurred during execution; the error message is recorded
  in `error`.

Each run creates the final results file `NAME.csv`, the run record
`NAME.manifest.json`, and the temporary `NAME.parts/` directory used for
checkpoint-based resumption.

When rerunning with the same output path and experiment configuration, the program
skips completed `ok` and `skipped` combinations and retries `failed` combinations.
After all results pass the integrity checks, the temporary `NAME.parts/` directory
is deleted automatically.

For detailed descriptions of the diagnostic fields and output files, see
[`outputs/README.md`](outputs/README.md).

## Panel configuration

Select the run scale once at the top of `panels.yaml`; the setting applies to all
declared outcomes:

```yaml
model_params: model_params.yaml
preset: medium
```

To run only one named panel locally, use:

```bash
python src/run_panels.py --only smr_hourlywage
```

If the number of top-level model combinations exceeds 250,000, you must explicitly
authorize the run non-interactively with `--allow-large-run`. Output CSVs retain
all existing metrics and add four filterable diagnostic fields. See
[`outputs/README.md`](outputs/README.md).

When using `run_panels.py`, `out` in `panels.yaml` serves as a filename template.
The program adds the preset name and a timestamp to the filename, for example,
`nk_grid_smr_hourlywage_dev_YYYYMMDD-HHMMSS.csv`. When resuming an incomplete run,
it continues using an existing file that matches the current experiment
configuration.

Model settings and the search grids used internally by the models are declared in
`model_params.yaml` and reused across all four scale presets. The pipeline does not
include a separate pre-run hyperparameter-tuning stage. To change these settings,
edit the YAML directly and increment `algorithm_version` at the same time so that
outputs produced under different model settings remain distinguishable.

Before running a panel, fill in any placeholder outcome columns in `panels.yaml`.

## SLURM

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/NK_Grid
export VENV=/path/to/your/venv

# Preview the panel/model array without submitting it
./slurm/submit_nk_grid.sh --dry-run

# Submit every active regression and classification panel
./slurm/submit_nk_grid.sh
```

For a production preset, explicitly authorize large array tasks:

```bash
./slurm/submit_nk_grid.sh --allow-large-run
```

The submitter reads `panels.yaml` and creates one array task for each active
`(panel, model)` pair. At submission time it writes a read-only, fully resolved job
snapshot under `logs/slurm-specs/`; queued workers use that snapshot, so later
edits to `panels.yaml` cannot change array-index assignments. Submission is
rejected if two jobs would write to the same output path. Do not submit
`run_nk_grid.sbatch` directly; it is the worker used by `submit_nk_grid.sh`. See
“Notes” for resource settings and per-model output behavior.

## Notes

<details>
<summary>Local data setup (without cluster access)</summary>

`data/...` paths are always resolved relative to `NK_Grid/`, so they are written
the same way on every machine. The only difference between machines is the actual
content of `NK_Grid/data`: it may be a symbolic link to a shared data location or
a local directory containing files with the same names. The YAML and command-line
arguments do not need to change between machines.

To test locally with a real copy of the data, place the required files in the
existing `NK_Grid/data` directory. Do not commit real data; the project ignores
the directory contents except for `data/.gitkeep`.

</details>

<details>
<summary>Command-line default scale and the four panel presets</summary>

When running `nk_grid.py` directly, the command-line defaults are:
`n-seeds=2 n-draws=2 n-sizes-n=4 n-sizes-k=4 min-n=10 max-n=100 max-k=100`.

`run_panels.py` uses a separate set of panel presets:

- **`dev`**: `n-seeds=5 n-draws=5 n-sizes-n=8 n-sizes-k=8
  min-n=10 max-n=100 max-k=100`. With all ten models, this declares 16,000 model
  combinations per outcome.
- **`medium`**: `n-seeds=8 n-draws=8 n-sizes-n=10 n-sizes-k=10
  min-n=10 max-n=100 max-k=100`. With all ten models, this declares 64,000 model
  combinations per outcome.
- **`pilot_full`**: `n-seeds=10 n-draws=5 n-sizes-n=12 n-sizes-k=12
  min-n=10 max-n=0 max-k=0 batch-size=500`. With all ten models, this declares
  72,000 model combinations per outcome.
- **`production`**: `n-seeds=100 n-draws=50 n-sizes-n=20 n-sizes-k=20
  min-n=10 max-n=0 max-k=0`. This declares 2,000,000 combinations for one model
  and 20,000,000 combinations for all ten models.

`max-n=0` and `max-k=0` mean that the full available N and K ranges are used, so
`pilot_full` is the only non-production preset without caps. `--allow-large-run`
is required only when the number of combinations exceeds 250,000. Therefore, with
all ten models, `dev`, `medium`, and `pilot_full` are below the threshold, while
`production` exceeds it.

When using the `production` preset with `run_panels.py`, the Git worktree must also
be clean, with no uncommitted changes. Before submitting a production run, use
`--dry-run` to inspect the grid size and model list, and first confirm that the
configuration works at a smaller scale. The legacy `bart` model is computationally
expensive; assess its resource requirements separately before adding it to the
model list.

</details>

<details>
<summary>Model mapping for classification tasks</summary>

With `--task classification`, model names map to classifiers rather than
regressors: `ols`, `ridge`, `lasso`, and `elastic_net` become unpenalized, L2, L1,
and elastic-net logistic regression, respectively; `random_forest`, `xgboost`,
`lightgbm`, and `extra_trees` use their corresponding classifiers; and
`shallow_neural_network` uses an MLP classifier. `super_learner` stacks logistic
regression, Extra Trees, fixed-hyperparameter LightGBM, and the shallow neural
network using out-of-fold predicted probabilities. The legacy `bart` model does
not support classification and raises a clear error if called. See
`model_registry.py` for the exact mapping.

</details>

<details>
<summary>Failure handling and checkpoint-based resumption</summary>

When resuming, the program does not recompute combinations with `ok` or `skipped`
status; **combinations with `failed` status are retried on the next run**. All
metric columns are empty in `skipped` and `failed` rows.

</details>

<details>
<summary>The logarithmic grid, --batch-size, and --test-size</summary>

N values are evenly spaced in log2 space from `--min-n` (default `10`) to the
specified upper limit. K retains the original log2 grid from 1 to its specified
upper limit. Both grids are converted to integers and deduplicated, so smaller
values are distributed more densely and larger values more sparsely.

`--batch-size` (default `20`) specifies how many pending combinations are grouped
together in each checkpoint-writing cycle. This setting applies to the entire run,
not to each parallel worker; the number of workers is controlled separately by
`--n-jobs`.

`--test-size` (default `0.3`) specifies the proportion of the data assigned to the
test set. “70/30” is only the default split, not fixed behavior; changing this
argument changes the actual data split. If an independent test set is supplied
through `--test-data`, the program uses that dataset and ignores `--test-size`.

</details>

<details>
<summary>Saving run logs</summary>

Progress logs from `helpers_logging.py` are written to standard error (stderr) and
are not saved by default. To retain a copy, use the redirection command below. It
enables `pipefail` so that a genuine run failure is not masked by `tee`'s own exit
status:

```bash
set -o pipefail
python src/run_panels.py 2>&1 | tee run.log
```

</details>

<details>
<summary>SLURM resource settings and output behavior</summary>

`slurm/submit_nk_grid.sh` reads all active panels from `panels.yaml` and submits a
dynamic job array with one task per `(panel, model)` pair. For example, two panels
with ten models each produce 20 array tasks. A missing or commented-out panel
produces no tasks, so unused classification or regression jobs are not submitted.
The submitter freezes the expanded mapping in `logs/slurm-specs/` before calling
`sbatch`; all workers in that submission therefore see the same configuration.

By default, each task uses 8 CPUs and 48 GB of memory, with a maximum runtime of 4
days; edit `slurm/run_nk_grid.sbatch` to adjust these settings. Each task creates a
separate timestamped CSV whose filename includes the panel output stem, model, and
preset, for example
`outputs/nk_grid_smr_hourlywage_ridge_dev_YYYYMMDD-HHMMSS.csv`. This differs from
running `run_panels.py` locally, which writes all models from one panel to a shared
CSV.

Standard output and error logs are stored in
`logs/<job-name>-<job-id>_<array-index>.out/.err`. The Git-tracked `logs/`
directory must exist before submission; it has already been created in this
project. Use `scancel <job-id>` to cancel a job and `squeue --me` to check status.

</details>

<details>
<summary>Complete nk_grid.py parameter reference</summary>

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `data/asample2_withlag.csv` | Path to the analysis-data CSV. |
| `--test-data` | `None` | Optional independent test-set CSV; when set, `--test-size` is ignored. |
| `--task` | `regression` | Use `regression` or `classification`. |
| `--outcome` | required | Name of the outcome column for either task. |
| `--predictor-prefix` | `Aset Bset` | Column-name prefixes used to select predictors. |
| `--out` | `outputs/nk_grid.csv` / `outputs/nk_grid_clf.csv` | Output CSV path. |
| `--dataset` | `asample2_withlag` | Custom dataset label written to the `dataset` column. |
| `--models` | `xgboost` | `ols, ridge, lasso, elastic_net, random_forest, xgboost, lightgbm, shallow_neural_network, extra_trees, super_learner`; legacy `bart` is also accepted. |
| `--seed` | `12345` | Base random seed; the `n-seeds` runs use `seed + offset` in sequence. |
| `--test-size` | `0.3` | Proportion of the data assigned to the test set. |
| `--n-seeds` | `2` | Number of independent train/test splits. |
| `--n-draws` | `2` | Number of repeated subsamples for each seed. |
| `--n-sizes-n` / `--n-sizes-k` | `4` / `4` | Number of points on the logarithmic N/K grids. |
| `--min-n` | `10` | Minimum value on the N grid; K still starts at 1. |
| `--max-n` / `--max-k` | `100` / `100` | Grid upper limits; `<=0` removes the cap. |
| `--model-params` | `model_params.yaml` | Task-specific default parameters used to construct models. |
| `--batch-size` | `20` | Number of combinations processed in each checkpoint write. |
| `--bart-min-n` / `--bart-min-k` | `10` / `2` | BART combinations below these thresholds are marked `skipped`. |
| `--group-split-col` | `None` | Reserved argument; setting it raises `NotImplementedError`. |
| `--allow-large-run` | `false` | Allow large runs with more than 250,000 top-level model combinations. |
| `--dry-run` | `false` | Print the estimated run size without fitting models. |
| `--n-jobs` | `$SLURM_CPUS_PER_TASK` or `4` | Number of parallel workers. |

</details>

<details>
<summary>Complete output schema</summary>

The 30 regression metrics are defined in `METRIC_COLUMNS`, and the 8 classification
metrics are defined in `CLASSIFICATION_METRIC_COLUMNS`; both are located in
`src/nk_grid.py`.

</details>

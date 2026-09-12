# Aleatoric Luck

This project studies how prediction quality changes with the number of training samples **N** and available variables **K**.

An FFCWS or SMR adapter first prepares the data. The engine then repeatedly samples training sets and variable subsets of different sizes, fits models, and evaluates them on test data. K counts original variable sources: a categorical variable expanded into several one-hot columns still counts as one source.

## Overall workflow

```text
Raw data → identify missing codes, screen variables, and encode categories
         → analysis data with missing values preserved
         → define train/test split and sample N rows and K sources
         → impute, standardize, and tune within training samples and CV training folds
         → refit on all N training rows → predict on the test set → save metrics
```

FFCWS retains the official train/test split and uses only the official training pool for variable screening and category vocabularies. SMR uses an existing analysis matrix and fixed feature definitions; the engine performs a random train/test split. Neither adapter imputes the full table in advance.

## Reading guide

| Topic | Documentation |
|---|---|
| Responsibilities of data preparation and model training | [Adapter principles](Adapter/README.md) |
| FFC row and column screening and missing values | [FFCWS adapter](FFCWS/adapter/README.md) |
| Differences among the three FFC encodings | [Encoding methods](FFCWS/adapter/src/ffcws_data_processor/strategies/README.md) |
| Fixed variable definitions in SMR | [SMR adapter](SMR/adapter/README.md) |
| N/K sampling, model training, and metrics | [Experiment methods](NK_Grid/README.md) |
| How the core code connects these steps | [Core code](NK_Grid/src/aleatoric_nk_grid/README.md) |
| Launching, distributing, and resuming experiments | [Launch flow](launch/README.md), [cluster execution](NK_Grid/slurm/README.md) |

## Quick start

The shared entry point is `run.sh`. Each stage starts with one command. The recommended sequence is:

```text
dry-run: preview the launch configuration
    ↓
dev: run a small trial
    or
timing_full: cover the full N/K range and check OOM errors, runtime, and other issues
    ↓
production: run the formal repeated experiment
```

Use dev for a quick trial and timing_full to assess resource requirements across the full N/K range. You can also run dev before timing_full.

### 1. dry-run: preview the configuration

The examples below run the FFC GPA panel locally on Linux/WSL. Run them from the repository root:

```bash
bash run.sh local --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset timing_full --dry-run
```

`--dry-run` displays the launch configuration. Training starts in the next stage.

### 2. dev or timing_full: run a trial

Use dev to check the path from input data through training to result output:

```bash
bash run.sh local --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset dev --checkpoints keep --output FFCWS/outputs/ffc-gpa-dev
```

Use timing_full to check memory use, runtime, convergence, and failures across the full N/K range:

```bash
bash run.sh local --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset timing_full --checkpoints keep --output FFCWS/outputs/ffc-gpa-timing
```

| Trial preset | Default size | Purpose |
|---|---|---|
| dev | 3 seeds × 3 draws; 3×3 grid; N and K each capped at 100 | Check the workflow on a small scale |
| timing_full | 1 seed × 1 draw; 20×20 grid covering full training capacity and all sources | Check OOM errors, runtime, and other issues for large training combinations |

Both use the panel's declared model list. timing_full assesses resource requirements for production.

### 3. production: run the full experiment

Once trial results meet expectations, start the repeated experiment in a new directory:

```bash
bash run.sh local --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset production --allow-large-run --checkpoints keep --output FFCWS/outputs/ffc-gpa-production
```

production defaults to 100 seeds × 50 draws on the full 20×20 grid. `--allow-large-run` enables this scale. Each stage runs independently and saves its own results.

### Choosing an execution environment

The workflow applies to local and cluster execution. Use the same environment prefix across stages, keeping the panel and preset arguments:

| Environment | Command prefix |
|---|---|
| Local Linux/WSL | `bash run.sh local` |
| BMRC | `bash run.sh slurm --profile bmrc --account YOUR_ACCOUNT` |
| Discoverer | `bash run.sh slurm --profile discoverer --account YOUR_ACCOUNT` |

Replace `YOUR_ACCOUNT` with your account. Other clusters require their own environment configuration and submission integration. Slurm submission uses a clean, committed checkout.

The launcher creates or checks the Python environment and installs locked dependencies. Local execution supports Python 3.11–3.14; use WSL on Windows.

### Data preparation and panel selection

The general entry point expects the panel schema to reference analysis data prepared by the adapter. Existing inputs can be used directly, or `--schema` can select another prepared input definition. For SMR, use `--manifest SMR/panels.yaml --panel smr_hourlywage`.

On Discoverer, `--prepare-ffc --ffc-data-dir YOUR_DATA_DIRECTORY` prepares the selected FFC panel on a compute node. The source directory contains `background.dta`, `train.csv`, `test.csv`, and labels for the selected outcome. See the reading guide above for each adapter's methods.

### Viewing results

The shared entry point writes results to `final.csv` in the selected run directory. After a trial, check process or scheduler logs for OOM errors and timeouts, and inspect the result columns `status` and `error`.

`ok` means fitting completed, `skipped` means the combination was skipped because of data conditions, and `failed` means execution failed. Completion records describe overall run status; Discoverer uses `continuation.json`.

Each row represents one model at a particular seed, draw, N, and K. `K_expanded` is the actual input column count. For regression, start with `mse`. `r2_test` uses the current training-sample mean as its baseline; R² relative to the test mean is stored separately as `r2_test_mean`.

Use a new output directory for each stage. Omitting `--output` creates `<manifest directory>/outputs/<panel>-<unique ID>/` automatically, with the validated result in `final.csv`. FFC GPA defaults to `FFCWS/outputs/ffc_median_mode_gpa-<unique ID>/`; SMR panels default to `SMR/outputs/<panel>-<unique ID>/`. These paths are inside the repository. An explicit `--output` overrides the default. To resume an existing Slurm run, use the original environment and account with `--resume PATH_TO_RUN/plan.json`.

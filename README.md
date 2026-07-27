# Aleatoric Luck

Aleatoric Luck contains one shared N×K prediction engine and article-owned data adapters.

The adapters turn private source data into validated analysis artifacts. The
shared `nk_grid` package then varies training sample size (`N`) and
sampled feature sources (`K`), fits the configured models, and writes resumable checkpoints and final results.

```text
Aleatoric_Luck/
├── Adapter/      adapter contract
├── NK_Grid/      shared engine, tests and Slurm scripts
├── SMR/          SMR adapter, schemas, panels and model parameters
└── FFCWS/        FFCWS adapter, schemas, panels and model parameters
```

Run all commands below from the repository root.

## 1. Installation

Python 3.11–3.14 is supported.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "NK_Grid[models,parquet,test]"
python -m pytest -q
```

The last command should pass before an experiment is submitted.

## 2. Private data and adapter preparation

Place the untracked source data at:

```text
SMR/data/private/asample2_withlag.csv

FFCWS/data/private/background.dta
FFCWS/data/private/train.csv
FFCWS/data/private/test.csv
```

Generate and validate the engine inputs:

```bash
python SMR/adapter/prepare.py
python FFCWS/adapter/prepare.py
```

The generated ARD tables are written below `SMR/data/ard/` and
`FFCWS/data/ard/`. Missing inputs or invalid artifacts cause preparation to
fail before an N×K run starts.

See [SMR adapter documentation](SMR/adapter/README.md),
[FFCWS adapter documentation](FFCWS/adapter/README.md), and the complete
[Adapter contract](Adapter/ADAPTER.md) for data-specific details.

## 3. Experiment presets

The top-level `preset` in each `panels.yaml` controls experiment size:

| Preset | Seeds | Draws | N levels | K levels | Configured cells per panel/model |
|---|---:|---:|---:|---:|---:|
| `dev` | 3 | 3 | 3 | 3 | 81 |
| `medium` | 8 | 8 | 10 | 10 | 6,400 |
| `timing_full` | 1 | 1 | 20 | 20 | 400 |
| `production` | 100 | 50 | 20 | 20 | 2,000,000 |

Integer grid values may be de-duplicated, so the resolved count can be lower.
The run log and manifest record the actual grid.

For timing or production, create a separate manifest in the same article
directory and change its top-level preset:

```bash
cp SMR/panels.yaml SMR/panels.timing.yaml
# Edit SMR/panels.timing.yaml:
# preset: dev  ->  preset: timing_full
```

Use the same pattern for FFCWS or for a `production` manifest. There is no
command-line `--preset` override.

## 4. Local execution

### Dry run

A dry run resolves the configuration and reports the planned size without
loading the data or fitting models:

```bash
aleatoric-nk-grid-panels --manifest SMR/panels.yaml --dry-run
aleatoric-nk-grid-panels --manifest FFCWS/panels.yaml --dry-run
```

### Run one panel

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --only smr_hourlywage
```

For a small code-path smoke test:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --only smr_hourlywage \
  --max-jobs 20
```

### Run the whole manifest

```bash
aleatoric-nk-grid-panels --manifest SMR/panels.yaml
```

Panels run sequentially locally. Use Slurm for full timing or production
experiments.

Runs above the 250,000-cell safety threshold require explicit authorization:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.production.yaml \
  --allow-large-run
```

Inspect the dry-run estimate before using `--allow-large-run`.

## 5. Slurm execution

### Prerequisites

Before submission:

1. Use a clean, reviewed, committed checkout.
2. Do not modify shared source or manifests while jobs are queued or running.
3. Ensure the checkout, `.venv`, private data, generated ARD and output
   directories are visible to every compute node at the same absolute paths.
4. Generate and validate the adapter artifacts.
5. Run both the local dry run and Slurm dry run.

The submitter uses the repository-root `.venv` by default. Override it when
needed:

```bash
export VENV=/shared/path/to/venv
export PYTHON="$VENV/bin/python"
```

If the cluster requires a Python module, set the module used to create the
virtual environment:

```bash
export PYTHON_MODULE=Python/3.11
```

### Slurm dry run

Always preview the panel/model mapping before submission:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.yaml \
  --dry-run
```

SMR expands to 20 panel/model jobs. The full FFCWS manifest expands to 180.
The Slurm launcher has no `--only` option; use a separate manifest when only a
subset of panels should be submitted.

### Submit dev or timing jobs

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.yaml \
  --max-concurrent-per-class 4 \
  --cpus-per-task 8 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

For the full-range timing run, change the manifest path:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.timing.yaml \
  --max-concurrent-per-class 4 \
  --cpus-per-task 8 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

### Submit production

Production requires an approved production manifest and explicit large-run
authorization:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.production.yaml \
  --allow-large-run \
  --max-concurrent-per-class 4 \
  --cpus-per-task 8 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

The submitter creates separate `parallel`, `serial` and optional `bart` arrays.
LightGBM and Super Learner use the serial resource class.
`--max-concurrent-per-class` limits each class separately, not the combined
number of running jobs.

Do not submit `run_nk_grid.sbatch` directly. The wrapper freezes a read-only job
snapshot and writes a receipt for each submitted Slurm job.

Production approval is separate from the existence of the preset. Complete the
[risk review](NK_Grid/RISK_REMEDIATION_REVIEW_2026-07-27.md) and cluster gates
before formal production submission.

### Monitor jobs

Use the job IDs printed by the submitter:

```bash
squeue -j JOB_ID

sacct -j JOB_ID \
  --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode

tail -f NK_Grid/logs/al-nk-grid-serial-JOB_ID_ARRAY_INDEX.out
```

Snapshots, receipts and Slurm logs are stored under `NK_Grid/logs/`.

### Requeue and partial submission recovery

Five minutes before the wall-time limit, the worker checkpoints and requests
requeue. Incomplete tasks resume from their existing checkpoint shards.

If one resource class fails to submit after another was accepted, do not repeat
the original manifest submission. Check `squeue`, then reuse the printed
snapshot for only the missing class:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --snapshot NK_Grid/logs/slurm-specs/jobs-TIMESTAMP-PID.json \
  --resource-class serial \
  --max-concurrent-per-class 4 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

Use the same resource and policy flags as the original submission. Completed
Slurm outputs are reused by default; pass `--rerun-completed` only when
recomputation is intentional.

## 6. Results and checkpoints

Results are written under `SMR/outputs/` or `FFCWS/outputs/`:

```text
<result>.csv              verified final results
<result>.manifest.json    configuration, provenance and QA
<result>.parts/           resumable checkpoint shards
<result>.run.lock         writer lease
```

Do not delete `.parts` from an incomplete or failed run. Running the same
experiment identity resumes from the last valid checkpoint.

For checkpoint internals, native-process isolation and detailed Slurm behavior,
see [NK Grid documentation](NK_Grid/README.md).

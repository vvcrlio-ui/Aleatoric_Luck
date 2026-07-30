# N-by-K analysis implementation

## Research design

This package implements the repeated sampling design used to estimate how
out-of-sample predictive performance changes with:

- training sample size (\(N\)); and
- number of available predictor variables (\(K\)).

For every outcome, model, random seed, and repeated draw, the program selects
training cases and eligible predictor variables, estimates data-dependent
preprocessing using the selected training data, fits the model, and evaluates
predictions on held-out cases.

Each empirical application defines its substantive outcomes and eligible
predictors in a data-preparation configuration, schema, and panel
specification.

## Inputs

An analysis requires:

| Input | Purpose |
|---|---|
| Analysis-ready dataset | Outcome and model input columns |
| Predictor manifest | Maps model input columns to the predictor variables counted in \(K\) |
| Schema | Defines outcomes, data split, variable types, missing-value treatment, and predictor eligibility |
| Panel specification | Defines the \(N\) and \(K\) values, repeated draws, seeds, outcomes, models, and explicit run identity |
| Model parameters | Records model-specific settings |

For a real run, declare `experiment_id`, `data_version`, and
`model_spec_version` at the manifest root or on each panel. These values identify
the experiment, analysis-ready data, and model specification carried into every
result manifest.

The common data-preparation requirements are documented in
[`../Adapter/ADAPTER.md`](../Adapter/ADAPTER.md).

## Installation and validation

Python 3.11–3.14 is supported.

```bash
python -m pip install -e "NK_Grid[models,parquet,test]"
python -m pytest -q NK_Grid/tests
```

Review a resolved analysis design while leaving the data unread and all models
unfitted:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --dry-run
```

The dry run reports the number of analysis cells and expected checkpoint
writes. Integer-valued grids can contain repeated values after rounding; the
resolved design records the actual values used.

## Analysis scale

Each panel specification selects one of four standard scales:

| Scale | Seeds | Draws per seed | \(N\) values | \(K\) values | Cells per outcome–model combination |
|---|---:|---:|---:|---:|---:|
| `dev` | 3 | 3 | 3 | 3 | 81 |
| `medium` | 8 | 8 | 10 | 10 | 6,400 |
| `timing_full` | 1 | 1 | 20 | 20 | 400 |
| `production` | 100 | 50 | 20 | 20 | 2,000,000 |

These identifiers are execution settings. Research reports should state the
resolved numbers of seeds, draws, \(N\) values, and \(K\) values.

Create a separate reviewed panel specification for timing or production runs:

```bash
cp SMR/panels.yaml SMR/panels.timing.yaml
```

Then change the top-level `preset` value in the copied file. Panel
specifications provide the authoritative preset selection.

## Local execution

Run one panel:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --only smr_hourlywage
```

Limit the number of analysis cells for a smoke test:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --only smr_hourlywage \
  --max-jobs 20
```

Run every outcome and model in a panel specification:

```bash
aleatoric-nk-grid-panels --manifest SMR/panels.yaml
```

Runs exceeding 250,000 cells require explicit authorization after the dry-run
estimate has been reviewed:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.production.yaml \
  --allow-large-run
```

## Slurm execution

Before submission:

1. use a clean, reviewed, committed checkout;
2. generate and validate the application data;
3. ensure that the checkout, environment, private data, generated data, and
   output directories are visible to every compute node at the same paths; and
4. review both the local dry run and the Slurm dry run.

Preview the seed-shard task mapping:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.yaml \
  --dry-run
```

Submit the reviewed specification:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.yaml \
  --max-concurrent-per-class 4 \
  --cpus-per-task 8 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

Production submission additionally requires `--allow-large-run`.

The repository-root `.venv` is used by default. On clusters that require
different paths or a Python module:

```bash
export VENV=/shared/path/to/venv
export PYTHON="$VENV/bin/python"
export PYTHON_MODULE=Python/3.11
```

Submit jobs through `submit_nk_grid.sh`. The wrapper records the resolved
mapping, resource settings, and a recovery command for each accepted Slurm job.

## Monitoring and recovery

Use the job identifiers printed during submission:

```bash
squeue -j JOB_ID

sacct -j JOB_ID \
  --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode

tail -f NK_Grid/logs/al-nk-grid-serial-JOB_ID_ARRAY_INDEX.out
tail -f NK_Grid/logs/al-nk-finalize-JOB_ID_ARRAY_INDEX.out
tail -f NK_Grid/logs/al-nk-publish-JOB_ID_ARRAY_INDEX.out
```

Execution records and Slurm logs are written below `NK_Grid/logs/`.

In `squeue`, seed arrays are named `al-nk-grid-parallel`,
`al-nk-grid-serial`, or `al-nk-grid-super_learner`; they are followed by
`al-nk-finalize` and `al-nk-publish`. Workers checkpoint within their seed
shard and may requeue at a checkpoint boundary. Finalizers and publishers run
after their dependencies and refuse incomplete inputs instead of publishing a
partial result.

Diagnose a frozen snapshot before recovery:

```bash
python -m aleatoric_nk_grid.seed_shards missing \
  --snapshot NK_Grid/logs/slurm-specs/jobs-TIMESTAMP-PID.json
```

The JSON reports missing and incomplete master indices plus invalid targets.
Recover a chosen resource class from that snapshot; omit `--master-indices` to
select all missing or incomplete tasks in the class, or pass the diagnosed
subset:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --snapshot NK_Grid/logs/slurm-specs/jobs-TIMESTAMP-PID.json \
  --resource-class serial \
  --master-indices 12,90 \
  --max-concurrent-per-class 4 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

Repeat per affected resource class. Recovery starts from the selected seed
tasks, then submits fresh finalizer and publish arrays. Reusing the snapshot
preserves the reviewed mapping.

## Output integrity

Every shard carries `experiment_id`, `data_version`, and
`model_spec_version`, plus its model, seed, draws, and resolved \(N\)/\(K\)
design. Finalizers compare identity and semantic contracts across seed shards;
publishers repeat those checks across model results. Duplicate publishers for
the same output are rejected through a filesystem lease, and completed results
are published atomically.

Checkpointing protects against interrupted execution. Long-term preservation
requires checksums and independent archival storage.

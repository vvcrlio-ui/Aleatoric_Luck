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
| Panel specification | Defines the \(N\) and \(K\) values, repeated draws, random seeds, outcomes, and models |
| Model parameters | Records model-specific settings |

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

Run one outcome:

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

Preview the outcome–model job mapping:

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

The launcher freezes the resolved job mapping before submission and assigns
models to the `parallel`, `serial`, or `bart` execution class. These labels
classify computing requirements.
`--max-concurrent-per-class` limits each execution class separately.

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
```

Execution records and Slurm logs are written below `NK_Grid/logs/`.

Work is saved in small, atomic checkpoint parts. Before the wall-time limit,
the worker stops after a completed checkpoint and requests requeue. A restarted
job resumes cells with pending or invalid records. Final results are
materialized only after the expected analysis-cell index passes integrity
checks.

If one execution class fails to submit after another has been accepted, first
inspect the accepted jobs with `squeue`. Then submit only the missing class from
the frozen snapshot printed by the original command:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --snapshot NK_Grid/logs/slurm-specs/jobs-TIMESTAMP-PID.json \
  --resource-class serial \
  --max-concurrent-per-class 4 \
  --serial-cpus-per-task 1 \
  --mem 48G \
  --time 4-00:00:00
```

Reusing the snapshot preserves the reviewed job mapping. Re-reading a changed
panel specification may produce a different mapping.

## Output integrity

Every result is identified by the application, outcome, model, random seed,
draw, \(N\), and \(K\). Resume logic checks this identity before accepting
prior work. Duplicate writers for the same output are rejected through a
filesystem lease, and completed results are published atomically.

Checkpointing protects against interrupted execution. Long-term preservation
requires checksums and independent archival storage.

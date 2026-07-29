# Analysis output files

This directory stores final results, run manifests, and checkpoint files used
to resume interrupted analyses.

## What files does each run create?

The panel specification declares a base output path such as
`nk_grid_smr_hourlywage.csv`. A run using a named analysis scale adds the scale
and start time to create a run-specific name:

```text
BASE.csv
└── RUN.csv = BASE_<scale>_<YYYYMMDD-HHMMSS>.csv
```

The files associated with that run are:

- `RUN.csv`: final results used for analysis and plotting;
- `RUN.manifest.json`: data identity, parameters, code version, completion
  status, and validation summary;
- `RUN.parts/`: temporary checkpoint parts used to resume an interrupted run;
  and
- `BASE.csv.run.lock`: writer lease shared by runs declared from the same base
  output path.

For the checked-in SMR panel, one observed set of names is:

```text
nk_grid_smr_hourlywage.csv.run.lock
nk_grid_smr_hourlywage_dev_20260727-155807.csv
nk_grid_smr_hourlywage_dev_20260727-155807.manifest.json
nk_grid_smr_hourlywage_dev_20260727-155807.parts/
```

The checkpoint directory is removed after successful finalization, so a
completed run normally retains the CSV, manifest, and lock file.

## Diagnostic fields in the results table

In addition to the analysis metrics, the final CSV contains four fields that flag results that may need attention:

| Field | Description |
| --- | --- |
| `K_varying` | Number of selected predictor variables containing at least one model input column with observed variation in the selected training sample |
| `constant_prediction` | Whether the model gives nearly identical predictions for all test samples. `true` indicates limited variation among predictions or an entirely invalid prediction set. |
| `underdetermined` | Used for OLS regression. `true` indicates that the number of varying encoded model input columns is at least as large as the number of training cases. |
| `converged` | Whether an estimated model completed within its configured iteration limit. Interpret this field together with `status` and `error`. |

These fields provide diagnostic warnings. Keep all successful runs in the main analysis and document every exclusion. For example, assess the sensitivity of the OLS conclusions after excluding rows where `underdetermined=true`.

## What does the manifest record?

`RUN.manifest.json` explains how a specific run was produced. It records:

- `algorithm_version`: The version of the algorithm and analysis method.
- The Git commit and whether the code had uncommitted changes when the run started.
- The relative paths and fingerprints of the input data files.
- The parameter grid and model settings actually used.
- The versions of core software packages.
- Whether the run completed successfully and the counts for each diagnostic category.

The manifest is the authoritative record for interpreting an existing result.
The `git.commit` and `git.dirty` fields identify its source-code state. The
`output` section identifies the associated CSV and checkpoint directory, while
the `completion` section records expected, completed, materialized, and failed
row counts.

## How do I resume an interrupted run?

During a run, the program saves completed results as checkpoints in
`RUN.parts/`.

If you restart the same task after an interruption, the program reads these temporary files and skips model and parameter combinations that have already finished. It then continues from where the previous run stopped.

When a run finishes normally, the program performs the following checks:

1. Merges the checkpoint files, removes duplicate rows, and creates the final CSV.
2. Reads the final CSV again.
3. Confirms that the manifest status is `complete`.
4. Confirms that the expected, written, and completed row counts match.
5. Confirms a failed-row count of zero.
6. Confirms that every model and parameter combination is unique and that every row has a valid status.

The program deletes `RUN.parts/` only after all checks pass.

> Leave the files in `RUN.parts/` unchanged. Manual edits can prevent
> successful resumption or validation.

## Common commands

### Preview the task size

Show the expected number of runs for each panel while leaving all models
unfitted:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --dry-run
```

### Run the checked-in SMR design

The checked-in `SMR/panels.yaml` uses the `dev` analysis scale:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml
```

Large analyses require a separately reviewed panel specification and the
`--allow-large-run` flag. See the
[N-by-K analysis guide](../../NK_Grid/README.md#analysis-scale) before creating
or running that specification.

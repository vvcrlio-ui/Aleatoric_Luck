# Analysis output files

This directory stores final results, run manifests, and checkpoint files used
to resume interrupted analyses.

## What files does each run create?

Each run uses one filename prefix, shown as `NAME` below, and creates the following files:

- `NAME.csv`: The final results table used for data analysis and plotting.
- `NAME.manifest.json`: A record of the data, parameters, code version, completion status, and validation results for the run.
- `NAME.parts/`: Temporary checkpoint parts used to resume an interrupted run.
- `NAME.csv.run.lock`: A writer lease that prevents concurrent processes from
  publishing the same result.

For example, if the results file is `regression.csv`, the corresponding manifest and checkpoint directory are `regression.manifest.json` and `regression.parts/`.

## Diagnostic fields in the results table

In addition to the analysis metrics, the final CSV contains four fields that flag results that may need attention:

| Field | Description |
| --- | --- |
| `K_varying` | Number of selected model input columns with observed variation. For example, 20 selected columns containing 3 constant columns produce a value of 17. |
| `constant_prediction` | Whether the model gives nearly identical predictions for all test samples. `true` indicates limited variation among predictions or an entirely invalid prediction set. |
| `underdetermined` | Used for OLS regression. `true` indicates that the number of usable model input columns is at least as large as the number of training cases, which may produce an unstable estimate. |
| `converged` | Whether the model finished fitting within the allowed number of iterations. `true` indicates convergence. `false` indicates failed convergence, a skipped fit, or another fitting failure that requires review. |

These fields provide diagnostic warnings. Keep all successful runs in the main analysis and document every exclusion. For example, assess the sensitivity of the OLS conclusions after excluding rows where `underdetermined=true`.

## What does the manifest record?

`NAME.manifest.json` explains how a specific run was produced. It records:

- `algorithm_version`: The version of the algorithm and analysis method.
- The Git commit and whether the code had uncommitted changes when the run started.
- The relative paths and fingerprints of the input data files.
- The parameter grid and model settings actually used.
- The versions of core software packages.
- Whether the run completed successfully and the counts for each diagnostic category.

## How do I resume an interrupted run?

During a run, the program saves completed results as checkpoints in `NAME.parts/`.

If you restart the same task after an interruption, the program reads these temporary files and skips model and parameter combinations that have already finished. It then continues from where the previous run stopped.

When a run finishes normally, the program performs the following checks:

1. Merges the checkpoint files, removes duplicate rows, and creates the final CSV.
2. Reads the final CSV again.
3. Confirms that the manifest status is `complete`.
4. Confirms that the expected, written, and completed row counts match.
5. Confirms a failed-row count of zero.
6. Confirms that every model and parameter combination is unique and that every row has a valid status.

The program deletes `NAME.parts/` only after all checks pass.

> Leave the files in `NAME.parts/` unchanged. Manual edits can prevent successful resumption or validation.

## Common commands

### Preview the task size

Show the expected number of runs for each panel while leaving all models
unfitted:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.yaml \
  --dry-run
```

### Allow a large run

Tasks above the safety threshold require the explicit authorization flag:

```bash
aleatoric-nk-grid-panels \
  --manifest SMR/panels.production.yaml \
  --allow-large-run
```

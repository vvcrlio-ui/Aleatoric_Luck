# NK Grid Output Files

This directory stores NK Grid results, run configurations, and checkpoint files used to resume interrupted runs.

## What files does each run create?

Each run uses one filename prefix, shown as `NAME` below, and creates the following files:

- `NAME.csv`: The final results table used for data analysis and plotting.
- `NAME.manifest.json`: A record of the data, parameters, code version, completion status, and validation results for the run.
- `NAME.parts/part-*.csv`: Temporary checkpoint files used to resume an interrupted run.

For example, if the results file is `regression.csv`, the corresponding manifest and checkpoint directory are `regression.manifest.json` and `regression.parts/`.

## Diagnostic fields in the results table

In addition to the analysis metrics, the final CSV contains four fields that flag results that may need attention:

| Field | Description |
| --- | --- |
| `K_varying` | The number of features that actually vary. For example, if 20 features are selected but 3 have the same value for every sample, this value is 17. |
| `constant_prediction` | Whether the model gives nearly identical predictions for all test samples. `true` means that the model may not have learned to distinguish between samples. It is also `true` if all predictions are invalid. |
| `underdetermined` | Used only for OLS regression. `true` means that the number of usable features is at least as large as the number of samples—in other words, there are too many features for the available data, so the result may be unstable. |
| `converged` | Whether the model finished fitting within the allowed number of iterations. `true` means that it converged normally. `false` means that it did not converge, was skipped, or failed and should be checked. |

These fields are warnings only; they do not automatically remove any results. Keep all successful runs in the main analysis. If you exclude any type of result, clearly state what was excluded. For example, you can check whether the OLS conclusions change after excluding rows where `underdetermined=true`.

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
5. Confirms that there are no failed rows.
6. Confirms that every model and parameter combination is unique and that every row has a valid status.

The program deletes `NAME.parts/` only after all checks pass.

> Do not edit the files in `NAME.parts/` manually. Doing so may prevent the run from resuming correctly or cause the validation checks to fail.

## Common commands

### Preview the task size

Show the expected number of runs for each panel without fitting any models:

```bash
python src/run_panels.py --dry-run
```

### Allow a large run

If the task exceeds the safety threshold, the program does not prompt for interactive confirmation. You must explicitly allow the run with:

```bash
python src/run_panels.py --allow-large-run
```

# BMRC: fifteen non-GPA panels in one shared pool

From a clean clone of OxfordDemSci/aleatoric_luck, branch `SMR&FFC`, run:

```bash
bash run.sh slurm --profile bmrc --suite ffc_non_gpa --preset timing_full \
  --account YOUR_PROJECT_ACCOUNT --ffc-data-dir /path/to/FFC \
  --resources FFCWS/outputs/bmrc-resources.json
```

Replace the account and raw-data directory. The directory must contain `background.dta`, `train.csv`, and `test.csv`, including the five outcome columns. These private files are not in Git. The suite combines `grit`, `materialHardship`, `eviction`, `layoff`, and `jobTraining` with `median_mode`, `median_missing_indicator`, and `tree_ordinal`. Each panel uses its nine declared models, training folds, tuning and scientific methods from the existing engine. `--suite` and `--panel` are mutually exclusive; existing single-panel commands remain supported.

Submission needs a BMRC Linux login shell, an authorized account, Slurm accounting visibility, and a shared filesystem accessible to all compute nodes. Compute nodes need package-installation access, `openssl`, and TCP connectivity to the dispatcher. The profile selects `Python/3.11.3-GCCcore-12.3.0` and `skl-compat`; set `PYTHON_MODULE` or an explicit constraint if the site requires a different compatible module/CPU combination. The launcher creates a private run environment with locked dependencies on a compute node. Bootstrap and workers use the same saved module, constraint and interpreter. Keep this checkout and its inputs unchanged while a run exists.

## Preview and presets

To prepare and run just one non-GPA panel through the same BMRC route, replace `--suite ffc_non_gpa` with, for example, `--panel ffc_median_mode_eviction`, keeping `--ffc-data-dir`. This also supports binary outcomes. Existing prepared-data single-panel entry points remain available without `--ffc-data-dir`.

Append `--dry-run` to preview without installing, reading raw data, or submitting. `--preset` is required. `production` requires `--allow-large-run` for execution.

| Preset | Experiment design per panel | Worker cap | Partition | Time per worker allocation | Automatic rounds |
|---|---|---:|---|---|---:|
| `dev` | 3 seeds × 3 draws; 3×3 grid, N/K capped at 100 | 32 | short | 1 hour | 2 |
| `timing_full` | 1 seed × 1 draw; full 20×20 grid | 600 | long | 24 hours | 2 |
| `production` | 100 seeds × 50 draws; full 20×20 grid | 600 | long | 10 days | 4 |

Grid dimensions are resolved against available samples and sources. A full 15-panel production design contains 270 million cells. Other presets retain the engine's smaller experimental defaults. `--time` and `--rounds` override the new run's time and continuation budget. Bootstrap/controller jobs default to 16 GiB and eight hours; `--plan-memory` and `--plan-time` override those settings.

## Shared resource file

The first compute-node bootstrap resolves node geometry from matching hardware and account, partition and QoS hard limits. It saves node count, constraint, workers, tasks per node, CPU settings, memory, partition and QoS in `--resources`. The worker cap excludes the dispatcher. Memory defaults to 16 GiB per worker; geometry conservatively includes dispatcher memory and two separate controller reservations. Slurm rank zero runs the dispatcher; other ranks run one worker each with one CPU per task and core binding. See Slurm's [CPU management guide](https://slurm.schedmd.com/cpu_management.html) for the allocation and binding distinction.

`timing_full` and `production` use exactly the same resource file. Actual workers can be fewer than the initial cap when whole-node geometry or hard limits require it. Geometry is never recalculated by preset or reduced because few cells remain or the cluster is busy. Temporary contention leaves the fixed job queued, or the controller waits for a submission slot. Changed hard limits produce an explicit error. When no cells remain, no worker allocation is submitted.

```bash
bash run.sh slurm --profile bmrc --suite ffc_non_gpa --preset production \
  --allow-large-run --account YOUR_PROJECT_ACCOUNT --ffc-data-dir /path/to/FFC \
  --resources FFCWS/outputs/bmrc-resources.json
```

Each run saves its own resource copy. Explicit `--workers`, `--memory`, `--plan-memory`, `--partition`, `--constraint`, or `--qos` settings must agree with an existing resource file. Use a new resource filename to change concurrency. The file is tied to its account and user. Keep the same file for the pressure test and production. Their wall times are deliberately different; changing `--time` does not change resource geometry.

## Status, interruption and continuation

Status records expected/started worker counts and hosts while the allocation runs. Workers begin claiming as they arrive; there is no wait-for-all startup barrier. These receipts document actual process startup, not a claim that every model keeps its CPU busy throughout the experiment.

The launcher prints a unique run directory, normally `FFCWS/outputs/ffc_non_gpa-<ID>/`. Generated inputs, environment and frozen configuration are inside that directory. Use the printed path:

```bash
bash run.sh status --run FFCWS/outputs/ffc_non_gpa-<ID>
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT \
  --resume FFCWS/outputs/ffc_non_gpa-<ID>
```

Resume accepts the run directory, including when bootstrap was interrupted. It uses the saved preset, inputs, resources, environment and checkpoint policy; omit new-run options. Repeating resume while a recorded job is active does not submit another job. After the automatic budget is exhausted, explicit resume grants another saved round budget. Two rounds without progress stop automatic retries; explicit resume resets that progress window. Controller recovery also has a finite budget. An ambiguous `sbatch` reply is reconciled using its journaled name and accounting receipt; it is never blindly resubmitted. If accounting cannot resolve it, the error identifies the journal for inspection.

One stopped-results scan marks valid `ok` and explicitly justified `skipped` cells in a bitmap. Pending work is the complete frozen design minus those cells. Failed, missing and truncated trailing records remain pending; duplicate completed keys count once. The full key is `(panel_id, seed, draw, N, K, model)` and the pending index contains 32-bit ordinals. Workers do not scan history. Only in-flight leases and a bounded acknowledgement cache live in dispatcher memory. Within each panel, estimated expensive cells are claimed first; a worker keeps one panel session and releases it when switching.

Preparation reuses existing fingerprints and reads an unchanged unique file once when a content hash is needed. Worker startup, panel switches, status and resume do not hash large input/result files. Resume checks the saved code version, configuration and file size/mtime metadata. This assumes immutable run inputs; metadata checks are not a cryptographic audit against deliberate replacement. Final publication streams records once more to write CSVs and verify unique coverage and metrics together, without a separate CSV checksum scan.

## Results and checkpoint retention

Each panel publishes `panels/<panel>/final.csv` and `verified.json`. Root `summary.json` and `verified.json` contain unique counts, statuses, resource configuration and result paths. `suite-state.json` tracks submitted jobs and rounds; `last-error.json`, if present, records the latest exception. Status reads these small receipts only. No plots are generated.

The CSV retains the engine's metrics, skipped reasons and convergence markers, and adds `panel_id`, `r2_holdout`, `null_mse_train_N`, and `r2_holdout_reason`:

```text
null_mse_train_N = mean_test((y_test - mean(actual N training targets))²)
r2_holdout = 1 - saved_prediction_error / null_mse_train_N
```

Saved prediction error is MSE for regression and probability-prediction Brier score for binary classification. The baseline reuses the engine's seed/draw sampling and the actual N training rows, independent of K and model. It does not train an extra model. Negative R² remains negative. Skipped cells have blank R²/baseline with an explicit reason; a zero baseline has blank R² with `zero_baseline_error`.

`--checkpoints keep` is the default. To delete intermediate checkpoints after complete validated publication, explicitly add `--checkpoints delete` to the original launch. Only that run's `rounds/` is removed. Worker logs, manifests, progress and allocation evidence are first retained under `logs/<round>/`; inputs, environment, configuration, final results and verification receipts remain. Interrupted or incomplete runs retain their checkpoints. A failure while archiving logs also leaves checkpoints in place. Checkpoint policy cannot change on resume.

## BMRC validation

Native BMRC validation is **pending**. Local validation used portable Windows queue tests, real loopback TLS clients with a deliberately lost acknowledgement, and synthetic data; it does not certify BMRC modules, cross-node networking, native numerical fits, or scheduler resource accounting. Record site validation separately from local test results.

First run a small suite with an explicit account and a separate resource file:

```bash
bash run.sh slurm --profile bmrc --suite ffc_non_gpa --preset dev \
  --account YOUR_PROJECT_ACCOUNT --ffc-data-dir /path/to/FFC \
  --workers 3 --resources FFCWS/outputs/bmrc-dev-resources.json
```

This checks installation, preparation, mixed regression/classification execution and publication; three workers may fit on one node. Then run the timing command above with the production resource file. Check `rounds/<round>/control/allocation.json`, `dispatcher-start.json`, `worker-start-*.json`, and `completed.json` (under `logs/<round>/` after deletion). Confirm that actual nodes/tasks match the saved allocation, all expected workers started, CPU affinity is recorded, and at least two distinct hosts participated before declaring cross-node acceptance. If the account's resolved geometry fits on one node, cross-node acceptance remains pending.

For scientific acceptance, compare representative saved regression and binary cells with direct engine execution using the frozen `plan.json` configurations, and independently check the paper R² baseline against the actual sampled training rows. Record command, source commit, account, job IDs, hosts, counts, failures, metric comparisons and outcome in a separate site report. Do not infer native equivalence from the portable session-routing fixture.

Local scale validation generated an actual 270-million-cell index: bitmap 33,750,000 bytes, uint32 index 1,080,000,000 bytes. With Python allocation tracing enabled, generation took 8.31 seconds and peaked at 43.48 MB of traced Python allocations. A separate 100,000-record synthetic scan took 4.83 seconds with a 42.21 MB traced peak. These are local measurements, not total process RSS or BMRC throughput estimates. No 270-million-result log was manufactured for this check.

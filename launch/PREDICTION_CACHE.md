# Saved predictions and the Super Learner

Every FFCWS and SMR panel saves, for each base-model task, the model's predictions for the test rows and its out-of-fold predictions for the training rows. The Super Learner is then fitted from these saved predictions alone, without refitting any base model, and the predictions stay in the run for later analysis. This page describes the manifest settings, the order of phases, how predictions are stored and checked, and the storage check made before they are written.

## Manifest settings

Both `FFCWS/panels.yaml` and `SMR/panels.yaml` give each panel these settings:

```yaml
prediction_cache:
  mode: holdout_oof
  base_library: standalone8-v1
  oof_folds: 5
  store_reported_sl_holdout: false
execution:
  workflow: base_then_sl
  barrier_scope: submission_plan
  protocol_version: 2
  verification_schedule: final_only
  phase_round_limits: {base: 2, sl: 1}
  sl_resources:
    worker_cap: 24
    io_concurrency: 8
    memory: 2G
    time_limit: '04:00:00'
```

| Setting | Meaning |
|---|---|
| `mode: holdout_oof` | Save test predictions and out-of-fold training predictions, as float64 without loss. |
| `base_library: standalone8-v1` | The eight base models, each fit on its own. The panel must list all eight. |
| `oof_folds` | Number of folds used for the out-of-fold predictions. |
| `store_reported_sl_holdout` | Also train a four-model Super Learner control; see [below](#which-models-the-super-learner-combines). |
| `workflow: base_then_sl` | Run all base models first, then the Super Learner. |
| `barrier_scope: submission_plan` | The Super Learner starts only after every base task of the plan has finished. |
| `protocol_version: 2` | The queue protocol described in [SCHEDULER_EFFICIENCY.md](SCHEDULER_EFFICIENCY.md). |
| `verification_schedule: final_only` | Base results are indexed but not fully checked before the Super Learner; everything is checked once at the end. Without this setting (`before_sl`), base results are fully checked before the Super Learner starts. |
| `phase_round_limits` | Most worker rounds in each phase. |
| `sl_resources` | Super Learner worker resources, used when the scheduler policy does not set `unified_compute`. |

`execution.base_round_time_limits` can give each base round its own time limit, for example `['01:00:00', '08:00:00']`. Each entry must fit within the run's time limit. A short first round frees its nodes early, and the next round continues the unfinished work with fewer workers. The CPU-hour budget and round limits apply across all phases and rounds.

The verification schedule and the other settings above are fixed when a run starts; resuming cannot change them.

## Which models the Super Learner combines

Unless the manifest sets `variants`, the Super Learner combines the seven base models other than OLS (variant `standalone8-sl7-v1`). OLS keeps its own saved predictions and its own score, but it is left out of the combination: with more columns than rows it can predict far outside the range of the outcome. To combine a different subset, set `variants` in the manifest; each subset needs its own variant ID, and results from different variants are never pooled.

For continuous outcomes, the combiner is a nonnegative least-squares fit with an intercept on centered predictions; the weights are not rescaled to sum to one. For binary outcomes, it is an unpenalized logistic regression on the positive-class probabilities, with nonnegative weights and an intercept. Test outcomes are loaded only by the scoring step, after the combiner has been fitted.

`store_reported_sl_holdout: true` adds a four-model Super Learner as a control, identified as `reported-sl4-*`. It combines Ridge, extra trees, LightGBM and the neural network. Its four models are built by the Super Learner's own constructors rather than taken from the eight base models, because a model with the same name is not necessarily the same training pipeline. The control therefore adds four more models with full and out-of-fold training, and it has to be chosen before the base phase: it cannot be added to a finished cache.

## Order of phases

1. Base models: the controller runs every base task in the plan.
2. Index: with `final_only`, the controller builds a read-only index from each task to its saved record and writes `base-input-ready.json`. This file says the index is ready, not that the data have been checked. Index workers read only record metadata, and finished parts of the index are reused after an interruption.
3. Super Learner: each task reads and checks the records it needs, checks that samples, folds and row order match exactly, fits only the combiner and scores it. A missing required prediction makes the task fail, or skip if its variant says so; it never causes base training.
4. Verification: every accepted base and Super Learner record is checked, together with the sample maps and that every expected result appears exactly once. The run then writes `base-verified.json`, `final.csv` and `verified.json`.

`final.csv` has `phase`, `pipeline_id` and `variant_id` columns that tell base-model rows from Super Learner rows, and the control pipelines from the rest when `store_reported_sl_holdout` is on.

## How predictions are stored

`prediction-cache/` in the run directory is separate from the training checkpoints. Each worker appends versioned, checksummed, length-bounded zlib records, syncs them to disk before reporting where they are, and publishes an index for each batch when it closes it. Messages to the queue service carry only references to records, never arrays. Separate sample maps hold the exact ordered row IDs, positions, labels, variable names and fold assignments.

Partly finished out-of-fold predictions go to `prediction-training-checkpoints/`, in rolling 8 MiB shards with at most 64 MiB per worker. A shard is removed only once its complete predictions are safely stored. Accepted predictions survive checkpoint cleanup. A damaged incomplete tail is repaired only after its writer has stopped and an exclusive lock is held; fully corrupted records, or two different predictions for the same task, are errors.

Modification times are used only as a quick check. Lustre can show different processes different timestamps for the same file, so when a closed data file has the expected size but a different timestamp, its bytes are compared with the recorded checksum. Changed content is never accepted.

If a task is reassigned after its worker saved predictions but never reported the score, the new worker can recover those predictions. An older worker cannot overwrite a result that has already been accepted. A disk or quota failure stops new work while keeping what has been saved. Restarting the Super Learner phase neither resets budgets nor repeats verified base training.

## Storage checks

Before a round writes predictions, the controller estimates the bytes and files still to be written, plus temporary worker space, and reserves them. The result is written to `storage-admission.json`.

When the `lfs` command is available, the check uses the Lustre project's soft byte and file quotas under `projects/PROJECT` and the free space of the filesystem. All runs of the project record their pending reservations in one shared ledger, and `quota_reserve_gb` (500 by default) and `file_reserve` (1,000,000 by default) are kept free on top of them.

Without `lfs`, the check uses the free bytes and inodes that `os.statvfs` reports for the cache directory. The ledger then lives in the run's own cache directory, so separate runs do not see each other's pending reservations.

`max_bytes`, `max_files` and `temporary_max_bytes` under `prediction_cache` replace the estimated limits and must be large enough for the estimate. After a run has been verified, the unwritten part of its reservation is released.

## Refitting a different combination

A different Super Learner can be fitted from a finished run's saved predictions without any base training:

```sh
python -m aleatoric_nk_grid.offline_sl \
  --plan RUN_DIR/plan.json --panel PANEL --seed SEED --draw DRAW --n N --k K \
  --pipelines standalone8-v1/ridge standalone8-v1/extra_trees \
              standalone8-v1/lightgbm standalone8-v1/shallow_neural_network \
  --variant-id sensitivity-sl4 --output RUN_DIR/sensitivity-sl4
```

`--store-prediction` also saves the new variant's test predictions. Custom pipeline recipes and training imported from another run are rejected.

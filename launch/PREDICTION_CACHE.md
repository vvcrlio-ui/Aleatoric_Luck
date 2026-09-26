# FFC prediction cache and the two-stage controller

`FFCWS/panels.yaml` enables lossless float64 OOF/holdout prediction persistence
for all 18 FFC panels. Select one with `--panel`; the shared Slurm launcher uses
this manifest by default. Configurations without cache options remain disabled,
and existing frozen runs must retain their code.
The scientific defaults in `model_params.yaml` are unchanged.

The batch entry pins `PYTHONPATH` to the checkout containing its scheduler entry.
Controller successors and worker subprocesses inherit that same package path;
a submitting shell left in another experiment cannot select its Python modules.
For an older frozen checkout, explicitly export its own `NK_Grid/src` when
submitting a bounded repair; do not overwrite its frozen batch entry.

Resource admission counts one worker allocation plus its control jobs against
job-count limits. CPU, memory and node requests are checked separately against
every applicable live scope; a job-count limit is not a worker-count limit.
Legacy array launchers retain their original per-job accounting.

```yaml
prediction_cache:
  mode: holdout_oof
  scope: all
  base_library: standalone8-v1
  dtype: float64
  required: true
  oof_folds: 5
  shard_target_mib: 128
  store_reported_sl_holdout: false
  quota_reserve_gb: 500
  file_reserve: 1000000
execution:
  workflow: base_then_sl
  barrier_scope: submission_plan
  protocol_version: 2
  verification_schedule: final_only
  phase_round_limits: {base: 2, sl: 1}
  sl_resources:
    worker_cap: 2
    io_concurrency: 2
    memory: 2G
    time_limit: '00:10:00'
```

An optional frozen `execution.base_round_time_limits` list assigns one walltime
to each base round, for example `['01:00:00', '08:00:00']` with an eight-hour
global maximum. A short first allocation can release its nodes and resume only
unfinished work with fewer workers. The total CPU-hour and round limits continue
to apply across every phase and round.

This is an interface example, not a production resource recommendation. Set the
normal launch worker/time bounds and `scheduler-policy.json` CPU-hour limit for
the intended small run. Optional `max_bytes`, `max_files`, and
`temporary_max_bytes` are admission limits; their values must accommodate the
frozen uncompressed estimate and temporary checkpoint bounds. Admission queries
current Lustre project soft quotas and reserves unwritten bytes in a shared
project ledger. The September 18 space snapshot is not an admission decision.

The base phase contains the eight independent model pipelines. Setting
`store_reported_sl_holdout` adds four more, reproducing the original
four-model SL - Ridge, Extra Trees, LightGBM and MLP for regression - under its
own `reported-sl4-*` identity. Those recipes come from the original SL
constructors rather than being substituted with same-name independent models,
because a same-named model is not the same training pipeline.

That control is off by default. It is four extra pipelines of full and
out-of-fold training, measured at 15.9% of base training time, not a disk
write, and it cannot be added to a sealed cache afterwards - enabling it later
means retraining the base phase. OLS remains an independent
full/OOF column and its own independent score.

The default combination is `standalone8-sl7-v1`: the seven library models except
OLS. OLS is dropped from the combination because an underdetermined fit predicts
far outside the label range - GPA labels span 1.0-4.0 against OLS predictions of
-779.8 to 760.9 - and the NNLS combiner already assigns it zero weight in 65.8%
of measured pairs. Across 20 seeds x 3 outcomes x 6 scales the seven-model
combination never lost on mean MSE, by 0.008% at the largest sizes and by 88%
where OLS diverges (`docs/sl7-no-ols-60nodes-20260918`). Set `variants`
explicitly to combine a different subset; each subset needs its own variant ID.

`reported-sl4-regression-v1` identifies the original frozen SL recipe and stays
available as a control. SL4 and SL8 remain separately named variants; results
from different variants are never pooled.

The controller completes every base task across the submission plan. With the
frozen `execution.verification_schedule: final_only` setting, it builds a
partitioned read-only task-to-reference index and writes `base-input-ready.json`.
This receipt certifies index readiness, not a completed data audit. Index workers
scan accepted journal metadata without decoding prediction arrays or copying
full score rows into the index. Completed chunks can be reused after interruption.
Stopped writers must still be sealed; repairing an interrupted writer can require
additional reads before indexing.

SL then reads and verifies each required record, checks exact sample/fold/order
alignment, fits only the combiner, and scores. Base holdout and OOF predictions
for all eight models, including OLS, remain stored. Required missing columns
cause a failure or an explicitly frozen variant skip, never base training.
After SL, distributed verification checks every accepted base and SL record,
global unique-key coverage and sample maps, then publishes `base-verified.json`,
the final CSV and `verified.json`. The CSV hash is computed while publishing.
Indexing adds no synchronous per-result index write or new RPC to the producer.

Omitting `verification_schedule` retains the historical `before_sl` behavior:
the base audit and `base-verified.json` gate SL. The schedule is part of the
frozen contract and cannot be changed by resuming an existing experiment.
`FFCWS/panels.yaml` opts future runs of all 18 panels into `final_only`; the
completed MH run retains its original plan and receipts. There is no separate
MH-only manifest. Base and SL can share their compute
policy using `unified_compute`; [launch options](README.md#dispatcher-shards-and-initial-policy)
include the explicit dispatcher count and initial policy file.

Classification predictions are positive-class probabilities with class mappings.
Regression uses centered NNLS with an intercept, without coefficient normalization.
The formal classification combiner freezes its logistic parameters and cell seed.
Holdout labels are loaded by the scoring step after fitting the combiner.

## Execution and offline analysis

Use the ordinary `launch/experiment.py slurm` command with a new FFC manifest
containing the above options. `cluster_queue.prepare_joint` accepts explicitly
prepared panel plans and creates one multi-panel barrier. The legacy suite,
flat-task-table and local full-run paths reject required cache configurations.
They do not silently fall back to mixed full SL training.

An isolated native acceptance run can be prepared without submitting jobs:

```sh
python NK_Grid/validation/prediction_acceptance.py prepare-fixture \
  --output runs/cache-native-p5 --workers 1 --memory 4G \
  --time-limit 00:15:00 --max-cpu-hours 1
python NK_Grid/validation/prediction_acceptance.py execute-local \
  --plan runs/cache-native-p5/plan.json --max-seconds 780 \
  --inject-confirmation-loss
```

This fixture uses reduced, separately identified model parameters and two
synthetic panels, including N=399/400/401. It is an engineering acceptance run,
not evidence of FFC scientific speedup. `prepare-ffc --schema ... --timing-full`
prepares a new unchanged-parameter FFC grid; it also does not submit by itself.
Start an authorized prepared plan with `python launch/cluster_scheduler.py start
runs/NEW/plan.json` after reviewing its bounds and live admission.

Refit a different subset without any base fitting:

```sh
python -m aleatoric_nk_grid.offline_sl \
  --plan runs/NEW/plan.json --panel PANEL --seed SEED --draw DRAW --n N --k K \
  --pipelines standalone8-v1/ridge standalone8-v1/extra_trees \
              standalone8-v1/lightgbm standalone8-v1/shallow_neural_network \
  --variant-id sensitivity-sl4 --output runs/NEW/sensitivity-sl4
```

Classification requires `--combiner-config` with an explicit logistic rule.
`--store-prediction` additionally saves that new variant's holdout array.
For the historical SL7/SL8 archive, use `--legacy-npz DIRECTORY --output REPORT`.
Historical metrics or holdout-only files are never treated as available OOF.

## Persistence and recovery

`prediction-cache/` is independent of training checkpoints. Exclusive worker
incarnations append versioned, checksummed, length-bounded zlib records, fsync
before returning references, and publish batch indexes on sealing. Queue/WAL
messages contain references, never arrays. Sample maps separately hold exact
ordered IDs, positions, labels, feature/source names and fold assignments.
New records use `NKPRED02` with a separately compressed, bounded JSON header;
`NKPRED01` remains readable and legacy manifests retain their original encoding.

Partial fold predictions use `prediction-training-checkpoints/`, rolling 8 MiB
shards and a 64 MiB per-worker bound. Only sealed private shards with durable
complete parent predictions can be reclaimed. Accepted prediction shards survive
checkpoint cleanup. Damaged incomplete tails are repaired only after the writer
has lost ownership and an exclusive OS lock can be held. Complete corruption or
conflicting same-identity predictions is an error.

Modification times are a quick check, not scientific identity. If a sealed
data file has the expected size but a different timestamp, validation compares
its bytes with the original sealed checksum. It never updates the receipt to
accept changed content. This handles observed cross-process Lustre timestamp
disagreement while retaining record-level integrity checks on every read.

The stopped-generation recovery index permits reassigned tasks to recover
durable predictions even if the original worker never submitted a score. Lease
authority still comes from protocol v2; an old writer cannot overwrite a new
accepted result. Disk or quota failure stops further claims/admission and retains
durable evidence. SL restart does not reset budgets or reissue verified base
training. Existing token heartbeat, byte bounds, incremental result submission,
submission journal, deadline and bounded continuation behavior remain active.
Cache-hit and partially resumed observations remain visible in cost reports but
cannot price cold training batches. Exact sample-map and fold alignment is
required before acknowledging a successful result and when SL consumes inputs.

Monitor `cluster-state.json`, round `control/latest.json`, `storage-admission.json`,
`base-input-ready.json` (for deferred audits), `base-verified.json` and final
`verified.json`. Base completion or input-index readiness alone is not final
completion. Read phase, pipeline and variant columns when analyzing `final.csv`;
this new result directory also contains auxiliary formal-SL base pipeline rows.

Current conservative limitations: custom pipeline recipes and cross-run training
imports are rejected; use the explicit offline combination command against a
verified source. Sample maps deduplicate within writers; cross-writer duplicates
are included in the conservative storage estimate. No 18-panel or 100x50
production launch follows from enabling this infrastructure.

# Operations reference

This page collects the commands and options for running experiments. [README.md](README.md) explains what the stages and phases do.

## Selecting a panel

`FFCWS/panels.yaml` holds the 18 FFCWS panels (six outcomes under three encodings). It is the default catalog, so `--manifest FFCWS/panels.yaml` can be left out. Choose one panel with `--panel`, for example `ffc_median_mode_gpa` or `ffc_tree_ordinal_materialHardship`; without `--panel` the launcher uses `ffc_median_mode_gpa`. For SMR, use `--manifest SMR/panels.yaml --panel smr_hourlywage` or `--panel smr_totalincome`. A launch runs only the selected panel.

Every panel in both catalogs saves the eight base models' test and out-of-fold predictions, fits SL7 from the seven saved columns other than OLS, and verifies all results once, at the end. [PREDICTION_CACHE.md](PREDICTION_CACHE.md) describes this workflow.

## Stages and presets

| Preset | Seeds × draws | Grid | Use |
|---|---|---|---|
| dev | 3 × 3 | 3 × 3, N and K at most 100 | Check the path from data to results on a small scale |
| timing_full | 1 × 1 | 20 × 20 over the full training sample and all sources | Check memory, run time and failures across the full range |
| production | 100 × 50 | 20 × 20 | The full experiment; requires `--allow-large-run` |

`--preset` also accepts `medium`, `pilot` and `dev-dynamic`. All presets use the panel's model list.

Start with `--dry-run`, which prints the launch configuration without installing anything, reading data or submitting jobs. Give each stage its own output directory, and submit Slurm runs from a clean, committed checkout. On Discoverer:

```bash
bash run.sh slurm --profile discoverer --account YOUR_PROJECT_ACCOUNT \
  --panel ffc_median_mode_gpa --preset timing_full \
  --scheduler-policy launch/policies/discoverer-cache.json \
  --dispatcher-shards 4 --dry-run

bash run.sh slurm --profile discoverer --account YOUR_PROJECT_ACCOUNT \
  --panel ffc_median_mode_gpa --preset dev \
  --scheduler-policy launch/policies/discoverer-cache.json --dispatcher-shards 4 \
  --checkpoints keep --output FFCWS/outputs/ffc-gpa-dev

bash run.sh slurm --profile discoverer --account YOUR_PROJECT_ACCOUNT \
  --panel ffc_median_mode_gpa --preset timing_full \
  --scheduler-policy launch/policies/discoverer-cache.json --dispatcher-shards 4 \
  --checkpoints keep --output FFCWS/outputs/ffc-gpa-timing

bash run.sh slurm --profile discoverer --account YOUR_PROJECT_ACCOUNT \
  --panel ffc_median_mode_gpa --preset production --allow-large-run \
  --scheduler-policy launch/policies/discoverer-cache.json --dispatcher-shards 4 \
  --checkpoints keep --output FFCWS/outputs/ffc-gpa-production
```

A custom policy can live outside the checkout and be passed with `--scheduler-policy /absolute/path/my-policy.json`.

## Where runs can execute

The FFCWS and SMR catalogs require the prediction cache, so they run through the shared Slurm scheduler. Local runs call the engine directly and work only for manifests without a required cache.

| Environment | Command prefix |
|---|---|
| Local Linux or WSL, manifests without a required cache | `bash run.sh local` |
| Discoverer | `bash run.sh slurm --profile discoverer --account YOUR_ACCOUNT` |
| BMRC, prepared data | `bash run.sh slurm --profile bmrc --account YOUR_ACCOUNT` |
| Other Slurm clusters | `bash run.sh slurm --account YOUR_ACCOUNT --partition YOUR_PARTITION --constraint none --qos YOUR_QOS` |

Other clusters also take `--time`, and need a compatible Python environment loaded first. The launcher creates or checks the Python environment and installs locked dependencies; it needs Python 3.11–3.14, and on Windows it runs under WSL. Site profiles in `profiles/` set Python modules, constraints, partitions and accounts; `--qos` works with any profile.

The compute nodes must share the run directory, allow TLS connections from workers to the dispatcher, and provide `srun` and `openssl`. The submitting account needs read access to its Slurm association and QoS limits and usage (`sacctmgr`, `scontrol show assoc_mgr`); submission stops if a limit cannot be read. The cache admission code expects Lustre project paths and `lfs` project-quota commands. On BMRC and other clusters the cache workflow still needs site validation of quota handling and CPU binding.

## Preparing data

The panel's schema points to analysis data prepared by the adapter. Prepared data can be reused, and `--schema` selects a different prepared input without rewriting the tracked schema. To change preprocessing, regenerate the data and schema with the adapter before starting a run.

On Discoverer, `--prepare-ffc --ffc-data-dir YOUR_DATA_DIRECTORY` prepares the selected FFCWS panel on a compute node. The directory holds `background.dta`, `train.csv`, `test.csv` and the labels for the selected outcome.

## Scheduling options

Each round checks the live account, QoS, partition, existing-job and CPU-minute limits, then submits one worker allocation and the controllers it needs. `--workers` caps the number of numerical workers, `--rounds` limits the number of worker allocations, and `--memory` sets the memory per worker; the per-node request adds a reserve for the queue services. Admission is conservative, and Slurm decides when an allocation starts. A worker allocation reserves slots for the queue services; a model task is not a separate Slurm job.

Every round uses a 600-second heartbeat and a 3,600-second task lease, both fixed in the queue manifest by `shared_queue.transport_manifest`. A failed heartbeat is retried within 30 seconds, and the tasks of a lost worker are handed out again after at most one lease interval. The queue service accepts a limited number of result submissions at once: 32 by default, never more than half its connection limit, and 256 under the Discoverer policy. A submission over the limit receives a retryable HTTP 503, and the worker keeps its result until it is accepted. Progress records show submission back-pressure, heartbeat age and expired leases.

## Dispatcher shards and initial policy

A dispatcher shard is one queue service process; it is not a compute node, a prediction file or a verification chunk. Request the number for a new run with `--dispatcher-shards` (1–8). Base and SL phases use the same setting. With more than one dispatcher, each gets two validation processes unless the policy file sets another number.

`--scheduler-policy PATH.json` supplies the initial operational policy. `--dispatcher-shards` overrides the value in that file, and fields the file leaves out keep their defaults. The launch saves both `scheduler-policy.initial.json` and the current `scheduler-policy.json`; restarting preparation keeps later operational edits. A resumed run keeps its existing policy and rejects these new-run options.

The Discoverer preset `launch/policies/discoverer-cache.json` requests four dispatchers, common resource limits for base and SL, and distributed final verification. It sizes rounds by capacity, with up to 300 compute nodes plus two controller nodes, 3 GiB per worker and ten-hour worker allocations. It is a Discoverer setting, not a resource recommendation for BMRC. The actual allocation still depends on live limits, memory, service-core reservations and the work left. If an allocation cannot fit the requested dispatchers, admission falls back to one and reports `requested_dispatcher_shards`, the actual count and `shard_admission_note`.

| Entry | `--dispatcher-shards` | Cache and SL workflow |
|---|---|---|
| Discoverer, shared single-model entry | Supported | Implemented; each release needs isolated Linux validation |
| BMRC or another Slurm cluster, prepared data | Supported, same policy code | Shared core workflow; quota and CPU binding need site validation |
| BMRC `--suite` or `--ffc-data-dir` | Rejected | Separate scheduler, not migrated |

A BMRC prepared-data preview, with its constraint selected explicitly:

```bash
bash run.sh slurm --profile bmrc --account YOUR_PROJECT_ACCOUNT \
  --constraint skl-compat --preset timing_full --dispatcher-shards 4 --dry-run
```

## Results and output locations

Without `--output`, a run is created at `<manifest directory>/outputs/<panel>-<unique ID>/`, with the validated result in `final.csv`. For the FFCWS GPA panel this is `FFCWS/outputs/ffc_median_mode_gpa-<unique ID>/`; SMR panels use `SMR/outputs/<panel>-<unique ID>/`. These paths are inside the repository. An explicit `--output`, or the path of a resumed run, takes precedence.

After a trial, check the process or scheduler logs for out-of-memory errors and timeouts, and look at the `status` and `error` columns. `cluster-state.json` records a cluster run's controller receipts, rounds and status; `complete` means validation, publication and the selected checkpoint handling have finished. `verified.json` ties the final CSV to the exact task count and the source receipts. With `--checkpoints delete`, only the run's `rounds/` checkpoint directory is removed, after publication; the plan, final CSV and verification receipts stay. Historical Discoverer runs keep their original `continuation.json` protocol.

## Resuming

Resume with the original profile and account and `--resume /absolute/run/plan.json`. The resumed run reuses the original inputs and parameters, reads the saved results and schedules only the remaining tasks; training whose results were not saved before the interruption is repeated. Single-panel plans have the format `single-model-slurm-v1`. Plans from the older grouped protocol go to their original snapshot, submission journal and continuation protocol, and their frozen code and input checks still apply, so resume such runs from their original checkout. The launcher does not change the settings of an existing run in place. Suite plans use the directory form described in [BMRC.md](BMRC.md).

## Older entry points

These remain for runs that were started with them.

- BMRC suite: `--suite ffc_non_gpa` runs fifteen panels on one dispatcher and one fixed worker allocation, with its own protocol and without the cache workflow. See [BMRC.md](BMRC.md); recover suite runs from their original checkout.
- SMR independent-SL runs of 2026-09-14/15: prepared with [fresh_single_model.py](fresh_single_model.py), which runs each model as an independent task through the direct-success dispatcher passed with `--runtime`. Its workers refuse a required cache, so new SMR runs use the shared scheduler.
- Grouped-task protocol: the scripts in [NK_Grid/slurm](../NK_Grid/slurm/README.md). `submit_flat_task_table.sh [--submit] PLAN.json` routes saved single-model plans to the shared scheduler and grouped plans to their original journaled protocol. `submit_nk_grid.sh` only prints a pointer to `run.sh`.
- GPA recovery: `python -m aleatoric_nk_grid.direct_success_queue` recovers an existing stopped GPA run. It enables lease recovery and threaded TLS handshakes; fresh experiments use `run.sh`. From the source checkout, with the intended Python environment:

```bash
PYTHONPATH="$PWD/NK_Grid/src" python -m aleatoric_nk_grid.direct_success_queue prepare \
  --base /absolute/path/to/stopped-gpa-run --repo "$PWD"
```

Without `--root`, `prepare` picks a new directory under `<repo>/FFCWS/outputs/` and prints its absolute `root` and eventual `final_csv`; without `--repo`, the imported module's checkout is used for this default. Keep the reported root. Inside an existing Slurm allocation, run:

```bash
PYTHONPATH="$PWD/NK_Grid/src" python -m aleatoric_nk_grid.direct_success_queue run \
  --root /absolute/path/reported/by/prepare --repo /absolute/path/to/frozen-scientific-checkout \
  --old /absolute/path/to/original-checkout --workers 3
```

This example needs at least four allocated tasks: three workers and a controller. The command does not submit an allocation. `--repo` must match the prepared manifest's scientific commit. `run` requires the exact prepared `--root` and publishes `<root>/final/ffc_median_mode_gpa.csv` with its `.manifest.json` only after complete validation; `--validate-only` skips the merge.

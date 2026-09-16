# Launching and resuming experiments

The launch layer turns a selected panel, model list, and experiment size into a run. It prepares the environment, checks inputs, schedules computation, and records progress. The shared engine handles preprocessing and model training.

The [BMRC suite entry](BMRC.md) adds `--suite ffc_non_gpa`: fifteen panels share one dispatcher and one fixed worker allocation. Its saved resource file, compact cell-difference recovery, directory resume, and status command are documented in that guide. The single-panel and historical recovery interfaces described here retain their own protocols.

Start with a dry-run preview, then use dev for a small trial or timing_full to check out-of-memory (OOM) errors, runtime, and other execution issues across the full N/K range. Use production for the formal repeated experiment. Each stage is started explicitly.

## From panel selection to execution

[experiment.py](experiment.py) reads the selected panel and determines the outcome, models, and N/K design. Once inputs are ready, the code checks the design against the available sample and source counts. Preview mode displays the launch request without reading analysis tables or starting training.

Local runs call the engine directly. Every new Slurm run uses the shared single-model scheduler, including BMRC, Discoverer, and explicitly configured other clusters. Each new experiment uses a separate output directory.

## Preparing the environment and data

The launcher creates or checks the Python environment for the selected execution environment. The dataset's adapter prepares the data; the schema specifies analysis-table paths, the outcome, and feature definitions. The launcher reads the panel's schema or a user-supplied schema.

Prepared analysis data can be reused. To update preprocessing, regenerate the data and schema with the adapter before starting an experiment. Some cluster integrations offer automatic data preparation before execution; see the root tutorial for details.

## Scheduling local and cluster computation

Local execution starts training processes directly. On Slurm clusters, job scripts request CPUs, memory, and wall time. Worker processes start when resources are allocated. Larger workloads can run in batches.

Each `(seed, draw, N, K, model)` is an independent task. All workers share one queue and claim another task after saving their result. Estimated expensive tasks start first. Models and their internal CV are unchanged. A worker allocation reserves one additional task for the dispatcher; a logical model task is not a separate Slurm job.

[cluster_scheduler.py](cluster_scheduler.py) handles submission and bounded continuation for every cluster. Each round checks live account, QoS, partition, existing-job and CPU-minute limits, then submits only its current worker allocation and necessary successor controllers. `--workers` caps numerical workers; `--rounds` bounds worker allocations. `--memory` is the memory allowance per worker; the generated per-node request includes a dispatcher reserve. The admission calculation is deliberately conservative and does not promise immediate allocation or maximum throughput.

The compute nodes must share the run directory, permit worker-to-dispatcher TLS connections, and provide `srun` and `openssl`. The submitting account needs read access to Slurm association/QoS limits and usage (`sacctmgr`, `scontrol show assoc_mgr`). An unresolved limit stops submission. Site profiles retain Python modules, constraints, partitions and accounts; `--qos` is accepted for any Slurm profile.

Every new queue round, fresh or resumed, uses a 600-second jittered heartbeat
and a 3,600-second lease, recorded in the immutable queue manifest. Both values
come from `shared_queue.transport_manifest`, so no launcher can set its own. Failed heartbeats
retry within 30 seconds. The TLS service admits at most eight concurrent result
submissions so journal writes cannot consume all connection slots; excess
submissions receive retryable HTTP 503 and retain their durable worker receipt.
Progress records report submission backpressure, heartbeat age and expired
leases. A lost worker can take up to one lease interval to be reclaimed.

## Resuming after interruption

A resumed run reuses the original inputs and parameters, reads saved results, and schedules only the remaining tasks.

`cluster-state.json` records controller receipts, rounds and status. `complete` means result validation, publication, and the selected checkpoint handling have finished. `verified.json` binds the final CSV, exact task count and source receipts. With `--checkpoints delete`, only that run's `rounds/` checkpoint directory is deleted, after publication; plan, final CSV and verification receipts remain. Training whose results were not saved before interruption is repeated on resume.

Use the original profile/account and `--resume /absolute/run/plan.json`. New single-panel plans have format `single-model-slurm-v1`. Historical grouped plans route to their original snapshot, submission journal and continuation protocol; their frozen code/input checks still apply, so use the original checkout for existing runs. The launcher never changes an existing experiment's execution identity in place. Suite plans use the directory form documented in [BMRC.md](BMRC.md).

Existing environment profiles include BMRC and Discoverer. Other Slurm clusters use the same scheduler with explicit `--account`, `--partition`, `--constraint` (or `none`), `--time`, and optionally `--qos`; load a compatible Python environment first. See the [root quick start](../README.md) for execution commands.

## Default outputs and the GPA recovery entry

New experiments use `run.sh`. Omitting `--output` creates
`<manifest directory>/outputs/<panel>-<unique ID>/final.csv`.
For FFC GPA this is under the repository's `FFCWS/outputs/`; SMR panels use
`SMR/outputs/`. Explicit output paths and frozen resume paths take precedence.

The new `python -m aleatoric_nk_grid.direct_success_queue` entry recovers an
existing stopped GPA run. It enables lease recovery and threaded TLS handshakes;
it does not replace `run.sh` for fresh experiments. With the intended Python
environment, from the source checkout:

```bash
PYTHONPATH="$PWD/NK_Grid/src" python -m aleatoric_nk_grid.direct_success_queue prepare \
  --base /absolute/path/to/stopped-gpa-run --repo "$PWD"
```

Without `--root`, prepare chooses a fresh directory under `<repo>/FFCWS/outputs/`
and prints its absolute `root` and eventual `final_csv`. If `--repo` is omitted,
the imported module's checkout is used for this default. An explicit `--root`
overrides the location. Keep the reported root. Inside an existing Slurm
allocation, run:

```bash
PYTHONPATH="$PWD/NK_Grid/src" python -m aleatoric_nk_grid.direct_success_queue run \
  --root /absolute/path/reported/by/prepare --repo /absolute/path/to/frozen-scientific-checkout \
  --old /absolute/path/to/original-checkout --workers 3
```

This example needs at least four allocated tasks: three workers and a controller.
The command does not submit an allocation. The run's `--repo` must match the
prepared manifest's scientific commit; preparation does not migrate that identity.
Run always requires the exact prepared `--root`, never guesses a previous run,
and publishes `<root>/final/ffc_median_mode_gpa.csv` plus `.manifest.json` only
after complete validation. `--validate-only` skips the merge. Existing runs stay
in their original directories.

# Operations reference

Start in a clean, committed checkout. Each launch selects one panel from a dataset's `panels.yaml`. The login node checks arguments, the checkout and output location with the Python standard library. It writes `launch.json` and submits one bootstrap job.

On a compute node, `bootstrap.sbatch` loads the profile's Python module when supplied, creates or reuses the shared dependency environment, prepares data when requested, freezes the plan and starts the shared Slurm scheduler.

## One command per cluster

Substitute the account, node count, manifest, panel and raw directory:

```bash
bash run.sh slurm --profile discoverer --account YOUR_ACCOUNT --nodes 4 \
  --manifest DATASET/panels.yaml --panel PANEL --preset dev \
  --prepare --data-dir /absolute/raw/directory

bash run.sh slurm --profile bmrc --account YOUR_ACCOUNT --nodes 4 \
  --manifest DATASET/panels.yaml --panel PANEL --preset dev \
  --prepare --data-dir /absolute/raw/directory

bash run.sh slurm --account YOUR_ACCOUNT --partition YOUR_PARTITION --nodes 4 \
  --qos YOUR_QOS --time 01:00:00 \
  --manifest DATASET/panels.yaml --panel PANEL --preset dev \
  --prepare --data-dir /absolute/raw/directory
```

Append `--dry-run` to preview. A preview creates no directories, reads no data, installs nothing and submits nothing. Actual submission requires Linux. The login Python must be 3.11–3.14. Dependencies are installed only inside the bootstrap allocation.

For prepared inputs, use `--schema /absolute/schema.json` instead of preparation arguments. The schema and referenced files must remain available to compute nodes.

Compute nodes need shared access to the checkout and run directory, TLS connectivity from workers to the dispatcher, `srun` and `openssl`. BMRC and the example profile require site validation. [DISCOVERER.md](DISCOVERER.md) gives Discoverer site notes.

## Profiles and defaults

`--profile NAME` sources `launch/profiles/NAME.sh`. Copy [profiles/example.sh](profiles/example.sh) to add a site. A profile exports only these values:

| Variable | Meaning |
|---|---|
| `PYTHON_MODULE` | Optional compute-node Python module |
| `NKGRID_PARTITION` | Default partition |
| `NKGRID_CONSTRAINT` | Optional node constraint |
| `NKGRID_MAX_TIME` | Maximum requested job duration |
| `NKGRID_QOS` | Optional default QoS; `account` means the explicit `--account` |
| `NKGRID_SLURM_QUERY_INTERVAL` | Optional minimum seconds between `squeue`/`sacct` calls |

CLI options override profile defaults, within the profile's maximum time. With no profile, supply partition and any constraint or QoS on the command line. `--constraint none` omits that Slurm argument. With no QoS, Slurm's account default applies. Accounts are always explicit.

## Nodes, memory and time

A new run states its resources, which are frozen in `launch.json`: `--nodes` is the total number of nodes the run may occupy, including two reserved for its controllers, so each worker allocation has at most `--nodes` minus two nodes. `--memory` is the memory of each numerical worker and `--time` the worker wall time. Each round reads the partition's node sizes with `sinfo` and places as many workers on a node as its cores and memory allow, after the queue services' cores and memory. `--workers` optionally caps the number of workers.

A round uses fewer nodes when the remaining work, priced by the run's own timings, would leave some idle, or when a `max_cpu_hours` budget in the scheduler policy would otherwise be exceeded. Account and QoS limits are not queried: Slurm admits the allocation or keeps it pending, and `squeue -j JOB` shows the reason. Each round records its actual allocation.

| Setting | Default |
|---|---|
| Nodes | Required |
| Worker memory | 2G |
| Worker rounds | 2 |
| Worker time, `timing_full` / `production` | Profile maximum; explicit `--time` required without a profile |
| Worker time, other presets | 1 hour |
| Bootstrap and controller | 1 CPU, 48G, 2 hours |

`--memory` sets base-worker memory; the FFCWS and SMR catalogs request 2G for SL workers. Use `--memory 4G` for full-grid SMR runs. A scheduler policy with an explicit `worker_memory` supplies that request; the example policies specify 3G. Bootstrap and controller memory is set separately with `--plan-memory`.

`--rounds`, `--time` and `--plan-time` override the corresponding requests.

## Preparing data

Use `--prepare --data-dir DIR` together. They cannot be combined with `--schema` or a resume. The bootstrap allocation checks that the declared files exist before running the adapter. The launcher does not download inputs or search for data directories.

Each manifest's top-level `preparation` mapping names its adapter, required files relative to `--data-dir` and generated schema. FFC also declares its YAML config, a panel-name pattern extracting strategy and outcome, and strategy arguments. SMR declares its fixed feature contract. Validation models, minimum N, split fraction and seed follow the selected panel and preset.

- FFCWS requires `background.dta`, `train.csv` and `test.csv`. Preparation selects the panel's outcome and encoding.
- SMR requires `asample2_withlag.csv`. Its adapter's `--output-root` directs schema and ARD publication into the run directory.

All generated configuration, work files, ARD and schemas stay under `<run>/prepared/`. The execution request records the generated schema path and SHA-256. Tracked schemas and original data are unchanged.

## Presets and environments

| Preset | Seeds × draws | Grid |
|---|---|---|
| dev | 3 × 3 | 3 × 3, N and K at most 100 |
| timing_full | 1 × 1 | 20 × 20 over the full input |
| production | 100 × 50 | 20 × 20; requires `--allow-large-run` |

`medium` and `pilot` are also available. Models come from the panel; `--models` selects a unique subset.

The shared environment key includes locked dependencies in `NK_Grid/requirements.txt`, the Python module and CPU type. Environments live in `nkgrid-envs/` beside the checkout, or `NKGRID_ENV_ROOT`. Bootstrap creates a missing environment under a lock; later runs reuse it unchanged. Engine code always comes from the run's checkout. Pip cache is shared there, while install temporary files stay in `<run>/tmp/bootstrap-JOBID/`.

## Storage admission

Before cache work, the controller reserves estimated unwritten bytes, temporary worker space and file count.

With an available `lfs` executable, admission uses the Lustre project ID and soft byte/file quotas under `projects/PROJECT`, plus filesystem free bytes. Pending reservations share a project ledger.

Without `lfs`, admission uses `os.statvfs` on the cache directory: `f_bavail × f_frsize` bytes and `f_favail` inodes. These values feed the same admission checks. Its ledger lives in the run's cache directory and accounts for that run's pending allocation; filesystem availability includes all data already written. It does not coordinate unwritten reservations across separate runs.

The admission receipt records the ledger location. Verified completion releases the pending reservation using that receipt. Both modes preserve data and apply the workflow's storage reserves.

## Scheduling and policy

Each round sizes its allocation as described above, then submits it with the required controllers. A controller reads the state of all the run's unfinished jobs with one `squeue` and, for jobs that have left the queue, one `sacct`. Terminal states are recorded in `cluster-state.json` and not queried again. With `NKGRID_SLURM_QUERY_INTERVAL`, every `squeue`/`sacct` call of the run, across its controller jobs, waits until that many seconds have passed since the previous one; `slurm-query.json` in the run directory holds the time of the last call. The interval is frozen in `launch.json`. Workers use one numerical thread; service cores and memory are included in the allocation. Slurm decides when jobs start. A task is not a separate Slurm job.

`--dispatcher-shards 1..8` requests queue processes for a new run. More than one defaults to two validation processes per shard. `--scheduler-policy PATH.json` supplies initial operational policy; an explicit shard count overrides it. Between rounds, `max_nodes` in `scheduler-policy.json` may be lowered, and `worker_memory`, `worker_cap` and `worker_time_limit` changed; none of them changes what a round computes. Launch saves `scheduler-policy.initial.json` and the active `scheduler-policy.json`. Resume preserves that run's policy.

The Discoverer policy `launch/policies/discoverer-cache.json` requests four dispatchers, common base/SL resource limits and distributed final verification. Its resource sizes require a suitable account and should be selected deliberately. Profiles contain site defaults; operational policy controls experiment scheduling resources.

Every round uses a 600-second heartbeat and 3,600-second task lease. Result submission has bounded concurrency; a busy service returns retryable HTTP 503 and workers retain unaccepted results. [PREDICTION_CACHE.md](PREDICTION_CACHE.md) describes base, index, SL and final verification.

## Results and recovery

A run lives at `<manifest directory>/outputs/<panel>-<ID>/`, ignored by Git.

| File | Evidence |
|---|---|
| `launch.json` | Frozen request, source commit and manifest identity |
| `launch.submission.json` | Bootstrap job ID |
| `bootstrap-journal.json` | Submission intent and recovered job identity |
| `logs/bootstrap-JOBID.out/.err` | Environment, preparation and planning logs |
| `prepared-launch.json` | Request with prepared schema and hash |
| `plan.json` | Frozen experimental plan |
| `cluster-state.json` | Controller receipts, rounds and status |
| `verified.json` | Final coverage, integrity and CSV identity |
| `final.csv` | Verified result table |

Recover a lost bootstrap submission response using the same request and journal:

```bash
python launch/experiment.py recover-bootstrap --request /absolute/run/launch.json
```

The journal matches the original unique name against both `squeue` and `sacct`; it does not resubmit an uncertain intent. For a failed bootstrap, inspect its logs and job state before launching a new run.

Resume a stopped plan from its original clean checkout, with the original profile and account:

```bash
bash run.sh slurm --profile YOUR_SITE --account YOUR_ACCOUNT --resume /absolute/run/plan.json
```

Resume keeps frozen inputs, resources, environment and checkpoint policy. Only missing tasks are scheduled. `--checkpoints keep` is the default; `delete` removes round checkpoints only after verified publication. Plans, receipts and the final CSV remain.

# Running experiments

An experiment is one panel: one outcome with one prepared input, listed in a dataset's `panels.yaml`. The launcher turns a panel and a preset into model fits, runs them on a Slurm cluster, and publishes one result table after every result has been checked.

The first sections take you from a preview to a production run and show how to follow it. The sections after [How a run executes](#how-a-run-executes) are reference material.

- [Before you start](#before-you-start)
- [Quick start](#quick-start)
- [From a small run to production](#from-a-small-run-to-production)
- [Following a run](#following-a-run)
- [Resuming a run](#resuming-a-run)
- [How a run executes](#how-a-run-executes)
- Reference: [command-line parameters](#command-line-parameters), [site profiles](#site-profiles), [nodes, memory and time](#nodes-memory-and-time), [preparing data](#preparing-data), [software environment](#software-environment), [scheduler policies](#scheduler-policies)

## Before you start

You need:

- a Linux login node on a Slurm cluster, with Python 3.11–3.14. The run installs its own dependencies on a compute node;
- compute nodes that see the checkout and the run directory on shared storage, have `srun` and `openssl`, and can reach one another over TLS;
- a Slurm project account;
- the dataset's raw files in one directory. The launcher does not download data. FFCWS needs `background.dta`, `train.csv` and `test.csv`; SMR needs `asample2_withlag.csv`;
- a checkout with every change committed. A run records its commit and refuses to continue if the checkout changes, so do not edit files or switch branches in it while a run is going. Develop in a separate clone.

## Quick start

From the repository root, preview a small FFC GPA run on BMRC. Replace `YOUR_ACCOUNT` with your project account and `/absolute/raw/directory` with the directory holding the raw files:

```bash
bash run.sh slurm --profile bmrc \
  --account YOUR_ACCOUNT --nodes 4 --memory 2G --time 01:00:00 \
  --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset dev \
  --prepare --data-dir /absolute/raw/directory --dry-run
```

`--dry-run` prints the settings the run would use. It creates no directories, reads no data, installs nothing and submits nothing. Remove `--dry-run` from the same command to submit the run. The `dev` preset does real computation on a small grid.

Each trailing `\` continues the command on the next line; leave it out when writing the command on one line.

To use another panel, change `--manifest` and `--panel` and point `--data-dir` at that dataset's raw files. FFC panels are named `ffc_<encoding>_<outcome>` and listed in [FFCWS/panels.yaml](../FFCWS/panels.yaml); the SMR panels are `smr_hourlywage` and `smr_totalincome`.

On Discoverer, use `--profile discoverer` with your Discoverer account (see [DISCOVERER.md](DISCOVERER.md)). On a cluster without a profile, give the partition and any QoS or constraint it needs:

```bash
bash run.sh slurm --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --qos YOUR_QOS --nodes 4 --memory 2G --time 01:00:00 \
  --manifest DATASET/panels.yaml --panel PANEL --preset dev \
  --prepare --data-dir /absolute/raw/directory --dry-run
```

If the data are already prepared, replace `--prepare --data-dir ...` with `--schema /absolute/schema.json`.

A preview checks only the settings. Whether the compute nodes can run the job is first tested by a dev run, and the BMRC and example profiles still need that first dev run at their sites to confirm their values.

## From a small run to production

Go through the presets in this order. Each run writes to its own new directory.

| Preset | Seeds × draws | Grid | Use |
|---|---|---|---|
| `dev` | 3 × 3 | 3 × 3, N and K at most 100 | Checks the whole path from raw data to the result table on a small grid |
| `timing_full` | 1 × 1 | 20 × 20 over the full data | Fits every grid point once, so memory use and run time over the whole N and K range are known |
| `production` | 100 × 50 | 20 × 20 over the full data | The full design; needs `--allow-large-run` |

Two more presets are available: `medium` (8 × 8 seeds and draws on a 10 × 10 grid, N and K at most 100) and `pilot` (84 seeds with one draw each, at the smallest, middle and largest point of each production axis).

For `timing_full` and `production`, `--time` defaults to the profile's maximum; without a profile it must be given. Full-grid SMR runs need `--memory 4G`. A production panel is large enough that its resources are usually set with a [scheduler policy](#scheduler-policies).

## Following a run

A run lives in `<dataset>/outputs/<panel>-<ID>/`, which Git ignores. The launcher prints the run directory and the ID of the first Slurm job when it submits.

`squeue -u "$USER"` lists the run's Slurm jobs. The first is the bootstrap job, which installs the environment, prepares the data and plans the run. It then submits the other jobs: controller jobs that decide what to run next, worker allocations that fit the models, and, at the end, a verification allocation. Each writes its log to `logs/` in the run directory:

| Log | Written by |
|---|---|
| `logs/bootstrap-JOBID.out` / `.err` | Environment setup, data preparation and planning |
| `logs/control-JOBID.out` / `.err` | Controller jobs |
| `logs/work-JOBID.out` / `.err` | Worker allocations |
| `logs/check-JOBID.out` / `.err` | Verification allocations |

`cluster-state.json` in the run directory shows where the run stands. `workflow_state` gives the phase (base models, Super Learner or final verification), `completed` the number of finished tasks, and `rounds` each worker allocation with the resources it actually received. `status` is `complete` once `final.csv` and `verified.json` have been written. Any other final status means the run stopped without a result table, for example because it used up its rounds or CPU-hour budget before every task was done; read the logs, then [resume](#resuming-a-run).

When a run is complete, `final.csv` holds one row per model fit; the [root README](../README.md#reading-the-results) explains its columns. The other files in the run directory record how the run was made:

| File | Contents |
|---|---|
| `launch.json` | The launch request, source commit and manifest |
| `prepared/` | Analysis table and schema written by the adapter, when `--prepare` was used |
| `plan.json` | The fixed experimental plan |
| `cluster-state.json` | Progress, rounds and final status |
| `verified.json` | Result of the final check of every expected result |
| `final.csv` | The result table |

## Resuming a run

To continue a run that stopped before it finished, use the same clean checkout, profile and account:

```bash
bash run.sh slurm --profile YOUR_SITE --account YOUR_ACCOUNT --resume /absolute/run/plan.json
```

The resumed run keeps its original inputs, resources, environment and checkpoint setting, and schedules only the tasks that are still missing. A completed run cannot be resumed.

If the bootstrap job was submitted but the launcher lost Slurm's reply, recover its job ID from the same request instead of submitting again:

```bash
python launch/experiment.py recover-bootstrap --request /absolute/run/launch.json
```

It looks for the job by its unique name in both `squeue` and `sacct`, and never submits a second bootstrap job. If the bootstrap job itself failed, read its logs and start a new run.

## How a run executes

The login node checks the arguments, the checkout and the output location using only the Python standard library. It writes `launch.json` and submits the bootstrap job. On a compute node, the bootstrap job loads the profile's Python module, creates or reuses the shared software environment, prepares the data when asked, fixes the plan and starts the scheduler.

### One task per fit

Each combination of seed, draw, N, K and model is one task. A production panel has 2,000,000 tasks for each of the eight base models and another 2,000,000 for the Super Learner. The tasks are independent, except that a Super Learner task needs the saved predictions of the seven models it combines. A task is not a separate Slurm job.

### Phases within a run

1. Base models: the eight models are fit on every training sample. Each task saves its test predictions and its out-of-fold training predictions as soon as it finishes.
2. Index: the saved base predictions are indexed so that the next phase can find them.
3. Super Learner: SL7 is fitted for every training sample from the saved predictions.
4. Verification: every expected result, including the saved predictions of all eight base models, is checked before `final.csv` and `verified.json` are written.

[PREDICTION_CACHE.md](PREDICTION_CACHE.md) describes how predictions are saved, checked and used by the Super Learner.

### Scheduling on the cluster

Cluster time is requested in rounds. Each round is one Slurm allocation with a time limit, in which many worker processes take tasks from a shared queue service and send back each result as soon as it is done. Tasks expected to take longest start first, and the estimates are updated from timings measured in the same run ([SCHEDULER_EFFICIENCY.md](SCHEDULER_EFFICIENCY.md)).

Before each round, the controller checks that there is enough disk space and file quota for the predictions still to be written ([storage checks](PREDICTION_CACHE.md#storage-checks)).

### Interruptions

A task counts as done only once its result has been saved and accepted. If an allocation ends or a worker fails, the next round schedules only the tasks that are still missing, so unfinished fits may be repeated. Results are only combined with results produced under the same inputs, code and settings.

## Command-line parameters

`bash run.sh slurm` selects the Slurm launcher; `bash run.sh slurm --help` prints all options. The ones used in most runs are:

| Parameter | Meaning | Default or example |
|---|---|---|
| `--profile NAME` | Load site defaults from `launch/profiles/NAME.sh`. | Optional; `bmrc` or `discoverer`. Without a profile, give `--partition`. |
| `--account ACCOUNT` | Slurm project account to charge. | Required, also when resuming. |
| `--nodes N` | Most nodes the run may occupy at once, including two kept for controller jobs. | Required for a new run; at least `3`. `--nodes 4` allows at most two nodes per worker allocation. |
| `--memory SIZE` | Memory for each base-model worker, not for the whole job. | `2G`; `4G` for full-grid SMR runs. |
| `--time TIME` | Time limit of one base-worker allocation, not of the whole experiment. | `01:00:00` for `dev`, `medium` and `pilot`; the profile's maximum for `timing_full` and `production`. |
| `--manifest PATH` | Dataset catalog with the panel definitions and data-preparation settings. | `FFCWS/panels.yaml`; or `SMR/panels.yaml`. |
| `--panel NAME` | The panel to run: its input, outcome and models. | `ffc_median_mode_gpa`; `smr_totalincome`. |
| `--preset NAME` | Number of seeds and draws and size of the N × K grid. | `dev`; see [presets](#from-a-small-run-to-production). |
| `--prepare` | Run the dataset's adapter in the bootstrap job before planning. | Off; needs `--data-dir`. |
| `--data-dir DIR` | Directory with the raw files named in the manifest. | Required with `--prepare`. |
| `--dry-run` | Print the settings without preparing data, installing or submitting. | Off. |

Other options:

| Parameter | Meaning | Default or example |
|---|---|---|
| `--partition NAME` | Slurm partition; overrides the profile. | Profile value; required without a profile. |
| `--qos NAME` | Slurm QoS; overrides the profile. | Profile value, otherwise the account's default. |
| `--constraint VALUE` | Node feature constraint; overrides the profile. | Profile value; `none` sends no constraint. |
| `--schema PATH` | Use data prepared earlier, through its schema. | The panel's schema. Cannot be combined with `--prepare`. |
| `--models NAME ...` | Run only some of the panel's models. | All models of the panel. |
| `--workers N` | Most worker processes in one allocation. | No cap; sized from nodes, memory and remaining work. |
| `--rounds N` | Most worker rounds, including the first. | `2`; the FFCWS and SMR manifests set their own round limits, which take precedence. |
| `--plan-memory SIZE` | Memory of the bootstrap and controller jobs, also reserved for the queue service. | `48G`. |
| `--plan-time TIME` | Time limit of each bootstrap and controller job. | `02:00:00`. |
| `--dispatcher-shards N` | Number of queue service processes, 1–8; overrides the policy file. | `1`. More than one adds two result-checking processes to each. |
| `--scheduler-policy PATH` | Scheduler policy file for a new run. | None; see [scheduler policies](#scheduler-policies). |
| `--allow-large-run` | Confirm a production submission. | Required for `--preset production`, except in a dry-run. |
| `--checkpoints keep` / `delete` | Keep intermediate checkpoints, or delete them once the run is verified. | `keep`. The plan, the run's records and `final.csv` are always kept. |
| `--resume PATH` | Continue the run of this `plan.json`. | Give the original profile and account only. |
| `--update` | Fast-forward the checked-out branch from `origin` before launching. | Off; needs a clean, named branch. Ignored with `--dry-run`. |

Relative paths to manifests, schemas, policies and raw data are read from the repository root. Paths inside a schema must be reachable from the compute nodes.

## Site profiles

`--profile NAME` reads `launch/profiles/NAME.sh`. To add a site, copy [profiles/example.sh](profiles/example.sh). A profile sets only these values:

| Variable | Meaning |
|---|---|
| `PYTHON_MODULE` | Optional Python module to load on compute nodes |
| `NKGRID_PARTITION` | Default partition |
| `NKGRID_CONSTRAINT` | Optional node constraint |
| `NKGRID_MAX_TIME` | Longest time a job may request |
| `NKGRID_QOS` | Optional default QoS; `account` means the value of `--account` |
| `NKGRID_SLURM_QUERY_INTERVAL` | Optional shortest gap in seconds between `squeue` or `sacct` calls, for sites that limit them |

Command-line options override the profile, within its maximum time. The account is never taken from a profile.

## Nodes, memory and time

A new run fixes its resources in `launch.json`. `--nodes` is the most nodes the run may occupy at once; two are kept for controller jobs, so a worker allocation has at most `--nodes` minus two. For each round, the controller reads the partition's node sizes with `sinfo` and places as many workers on a node as its cores and memory allow, after setting aside the queue service's share. `--workers` caps the number of workers.

A round uses fewer nodes when the remaining work, estimated from the run's own timings, would leave some idle, or when a CPU-hour budget in the scheduler policy would otherwise be exceeded. The launcher does not look up account or QoS limits: Slurm either starts the allocation or keeps it pending, and `squeue -j JOB` shows why.

`--memory` and `--time` set the base-worker defaults. A manifest can set its own time limit for each base round and separate resources for the Super Learner phase; the FFCWS and SMR manifests ask for `2G` per Super Learner worker and allow two base rounds and one Super Learner round. A scheduler policy can override worker memory, time and number of workers.

The bootstrap and controller jobs each use one CPU, `48G` and two hours unless `--plan-memory` and `--plan-time` say otherwise.

## Preparing data

With `--prepare --data-dir DIR`, the bootstrap job checks that the files named in the manifest exist in `DIR` and runs the dataset's adapter on them. The manifest's `preparation` section names the adapter, its input files and the schema it writes. The adapter checks its output with the models, smallest N, test fraction and seed of the chosen panel and preset.

The adapter's output, including the analysis table and schema, stays in `<run>/prepared/`, and `prepared-launch.json` records which schema the run uses. The tracked schemas and the raw data are not changed. `--prepare` cannot be combined with `--schema` or `--resume`.

## Software environment

The bootstrap job installs the locked dependencies in `NK_Grid/requirements.txt` into a shared environment in `nkgrid-envs/` beside the checkout, or in `NKGRID_ENV_ROOT` if set. An environment is identified by those dependencies, the Python module and the CPU type; a later run with the same three reuses it unchanged. The experiment code itself always comes from the run's own checkout.

## Scheduler policies

A scheduler policy is a JSON file that sets how a run uses the cluster: how many nodes and workers each round may take, how much memory and time each worker gets, and how many queue service processes run. It does not change what is computed. Pass it with `--scheduler-policy PATH.json`; the run keeps a copy as `scheduler-policy.initial.json` and uses `scheduler-policy.json` in its directory. Between rounds, you can lower `max_nodes` and change `worker_memory`, `worker_cap` and `worker_time_limit` in that file. A resumed run keeps its policy.

[policies/README.md](policies/README.md) lists the fields. [policies/discoverer-cache.json](policies/discoverer-cache.json) holds the values used for large runs on Discoverer.

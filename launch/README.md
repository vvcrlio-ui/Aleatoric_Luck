# Running experiments

An experiment is one panel: one outcome with one prepared input, listed in a dataset's `panels.yaml`. The launcher turns a panel and a preset into cells, computes them on a Slurm cluster, and publishes one result table once every cell has been checked.

The first sections take you from a preview to a production run and show how to follow it. The sections after [How a run executes](#how-a-run-executes) are reference material.

- [Before you start](#before-you-start)
- [Quick start](#quick-start)
- [From a small run to production](#from-a-small-run-to-production)
- [Following a run](#following-a-run)
- [Resuming a run](#resuming-a-run)
- [How a run executes](#how-a-run-executes)
- Reference: [command-line parameters](#command-line-parameters), [site profiles](#site-profiles), [nodes, memory and time](#nodes-memory-and-time), [preparing data](#preparing-data), [software environment](#software-environment)

## Before you start

You need:

- a Linux login node on a Slurm cluster, with Python 3.11–3.14. The run installs its own dependencies on a compute node;
- compute nodes that see the checkout and the run directory on shared storage and have `srun`. The shared storage must support POSIX record locks across nodes (`fcntl.lockf`), as Lustre and GPFS do;
- a Slurm project account;
- the dataset's raw files in one directory. The launcher does not download data. FFCWS needs `background.dta`, `train.csv` and `test.csv`; SMR needs `asample2_withlag.csv`;
- a checkout with every change committed. A run records its commit and refuses to continue if the checkout changes, so do not edit files or switch branches in it while a run is going. Develop in a separate clone.

## Quick start

From the repository root, preview a small FFC GPA run on BMRC. Replace `YOUR_ACCOUNT` with your project account and `/absolute/raw/directory` with the directory holding the raw files:

```bash
bash run.sh slurm --profile bmrc \
  --account YOUR_ACCOUNT --nodes 1 --memory 2G --time 01:00:00 \
  --manifest FFCWS/panels.yaml --panel ffc_median_mode_gpa --preset dev \
  --prepare --data-dir /absolute/raw/directory --dry-run
```

`--dry-run` prints the settings the run would use. It creates no directories, reads no data, installs nothing and submits nothing. Remove `--dry-run` from the same command to submit the run. The `dev` preset does real computation on a small grid.

Each trailing `\` continues the command on the next line; leave it out when writing the command on one line.

To use another panel, change `--manifest` and `--panel` and point `--data-dir` at that dataset's raw files. FFC panels are named `ffc_<encoding>_<outcome>` and listed in [FFCWS/panels.yaml](../FFCWS/panels.yaml); the SMR panels are `smr_hourlywage` and `smr_totalincome`.

On Discoverer, use `--profile discoverer` with your Discoverer account (see [DISCOVERER.md](DISCOVERER.md)). On a cluster without a profile, give the partition and any QoS or constraint it needs:

```bash
bash run.sh slurm --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --qos YOUR_QOS --nodes 1 --memory 2G --time 01:00:00 \
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

For `timing_full` and `production`, `--time` defaults to the profile's maximum; without a profile it must be given. Full-grid SMR runs need `--memory 4G`. Size a production run's `--nodes` and `--time` from the cell times of its `timing_full` run.

## Following a run

A run lives in `<dataset>/outputs/<panel>-<ID>/`, which Git ignores. The launcher prints the run directory and the ID of the first Slurm job when it submits.

`squeue -u "$USER"` lists the run's Slurm jobs. The first is the bootstrap job, which installs the environment, prepares the data, fixes the plan and submits the worker allocation. Each writes its log to `logs/` in the run directory:

| Log | Written by |
|---|---|
| `logs/bootstrap-JOBID.out` / `.err` | Environment setup, data preparation, the plan and the worker submission |
| `logs/work-JOBID.out` / `.err` | The worker allocation and, at its end, publication |

Each worker allocation works in its own `attempt-JOBID/` directory. Every worker appends one line per finished cell to its `results-rNNNNN.jsonl`, so the number of finished cells is the number of lines in all of them:

```bash
cat RUN/attempt-*/results-*.jsonl | wc -l
```

The plan's cell count is printed at the top of `logs/work-JOBID.out`. When the last cell is done, the same job publishes the results. A run is complete once `verified.json` exists.

When a run is complete, `final.csv` holds one row per model fit; the [root README](../README.md#reading-the-results) explains its columns. The other files in the run directory record how the run was made:

| File | Contents |
|---|---|
| `launch.json` | The launch request, source commit and manifest |
| `prepared/` | Analysis table and schema written by the adapter, when `--prepare` was used |
| `plan.json` | The fixed experimental plan |
| `allocation.json` | The Slurm resources of every worker allocation |
| `bootstrap-journal.json` | Every job the run submitted |
| `test-ids.npz` | Each seed's test-row IDs, in the order of every saved test prediction |
| `verified.json` | Result of the final check of every expected cell |
| `final.csv` | The result table |
| `predictions.jsonl` | Where each cell's saved predictions are; see [PREDICTIONS.md](PREDICTIONS.md) |
## Resuming a run

If a worker allocation ended before every cell was done, for example at its time limit, resume the run from the same clean checkout with the same profile and account:

```bash
bash run.sh slurm --profile YOUR_SITE --account YOUR_ACCOUNT --resume /absolute/run/plan.json
```

This submits another allocation with the same resources. It computes only the cells no earlier allocation finished and publishes the results when all are done. The launcher refuses while an earlier allocation of the run is still queued or running, and a completed run cannot be resumed. Add `--dry-run` to see the allocation without submitting it.

If the bootstrap job was submitted but the launcher lost Slurm's reply, recover its job ID from the same request instead of submitting again:

```bash
python launch/experiment.py recover-bootstrap --request /absolute/run/launch.json
```

It looks for the job by its unique name in both `squeue` and `sacct`, and never submits a second bootstrap job. If the bootstrap job itself failed, read its logs and start a new run.
## How a run executes

The login node checks the arguments, the checkout and the output location using only the Python standard library. It writes `launch.json` and submits the bootstrap job. On a compute node, the bootstrap job loads the profile's Python module, creates or reuses the shared software environment, prepares the data when asked, fixes the plan and submits one worker allocation.

### One task per cell

A cell is one seed, draw, N and K. A worker computes a whole cell: it fits the eight base models one after another, each on the N training rows and again in five out-of-fold (OOF) folds, and then combines the seven models other than OLS into SL7 from their OOF predictions, without fitting them again. It saves the cell's results and predictions before taking the next cell. Cells are independent of each other. [PREDICTIONS.md](PREDICTIONS.md) describes the saved predictions.

### Workers

One worker allocation runs one worker per physical core, as many per node as `--memory` allows. Workers take cells from a shared counter in the run directory, the cells with the largest N × K first, so the longest cells do not start last. Each worker writes only its own two files, and a cell's predictions are on disk before the line that records the cell.

### Interruptions

A cell counts as done once its line is written. If the allocation ends or a worker fails, the cells it had started are not done; a resumed run computes them again. If any model fit fails with an error, the run is not published; read the error in the results, fix its cause and start a new run.

### Publication

When the workers have finished, the allocation checks that every cell of the plan is present exactly once and that no fit failed. Only then does it write `final.csv` in the order of the plan, `predictions.jsonl` and `verified.json`.
## Command-line parameters

`bash run.sh slurm` selects the Slurm launcher; `bash run.sh slurm --help` prints all options. The ones used in most runs are:

| Parameter | Meaning | Default or example |
|---|---|---|
| `--profile NAME` | Load site defaults from `launch/profiles/NAME.sh`. | Optional; `bmrc` or `discoverer`. Without a profile, give `--partition`. |
| `--account ACCOUNT` | Slurm project account to charge. | Required, also when resuming. |
| `--nodes N` | Nodes of the worker allocation. | Required for a new run. |
| `--memory SIZE` | Memory for each worker, not for the whole job. | `2G`; `4G` for full-grid SMR runs. |
| `--time TIME` | Time limit of the worker allocation. | `01:00:00` for `dev`, `medium` and `pilot`; the profile's maximum for `timing_full` and `production`. |
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
| `--models NAME ...` | Run only some of the panel's models. | All models of the panel. `super_learner` needs the seven models it combines. |
| `--plan-memory SIZE` | Memory of the bootstrap job. | `48G`. |
| `--plan-time TIME` | Time limit of the bootstrap job. | `02:00:00`. |
| `--allow-large-run` | Confirm a production submission. | Required for `--preset production`, except in a dry-run. |
| `--resume PATH` | Continue the run of this `plan.json`. | Give the original profile and account only. |
| `--update` | Fast-forward the checked-out branch from `origin` before launching. | Off; needs a clean, named branch. Ignored with `--dry-run`. |

Relative paths to manifests, schemas and raw data are read from the repository root. Paths inside a schema must be reachable from the compute nodes.
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
| `NKGRID_OFFLINE_COMPUTE` | Set to `1` where compute nodes cannot reach PyPI; the launcher then downloads the dependency wheels on the login node |

Command-line options override the profile, within its maximum time. The account is never taken from a profile.

## Nodes, memory and time

A new run fixes its resources in `launch.json`, and the bootstrap job turns them into the worker allocation recorded in `allocation.json`. It reads the partition's node sizes with `sinfo` and gives every node the same number of workers: one per physical core, but no more than the smallest node's memory allows at `--memory` each. A resumed run reuses the same allocation.

The launcher does not look up account or QoS limits: Slurm either starts the allocation or keeps it pending, and `squeue -j JOB` shows why. Choose `--nodes` and `--time` so that the allocation can finish the plan; a `timing_full` run measures the time of every grid point.

The bootstrap job uses one CPU, `48G` and two hours unless `--plan-memory` and `--plan-time` say otherwise.
## Preparing data

With `--prepare --data-dir DIR`, the bootstrap job checks that the files named in the manifest exist in `DIR` and runs the dataset's adapter on them. The manifest's `preparation` section names the adapter, its input files and the schema it writes. The adapter checks its output with the models, smallest N, test fraction and seed of the chosen panel and preset.

The adapter's output, including the analysis table and schema, stays in `<run>/prepared/`, and `prepared-launch.json` records which schema the run uses. The tracked schemas and the raw data are not changed. `--prepare` cannot be combined with `--schema` or `--resume`.

## Software environment

The bootstrap job installs the locked dependencies in `NK_Grid/requirements.txt` into a shared environment in `nkgrid-envs/` beside the checkout, or in `NKGRID_ENV_ROOT` if set. An environment is identified by those dependencies, the Python module and the CPU type; a later run with the same three reuses it unchanged. The experiment code itself always comes from the run's own checkout. At sites whose compute nodes cannot reach PyPI, such as BMRC, the launcher first downloads the wheels on the login node, with the profile's Python module loaded, into a `wheels-` directory beside the environments, and the bootstrap job installs from that directory only.

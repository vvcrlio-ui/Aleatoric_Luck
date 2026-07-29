# Aleatoric Luck

## 1. Research question and method

Aleatoric Luck asks how out-of-sample predictive performance changes as a
researcher has more training cases (\(N\)) and more eligible predictor
variables (\(K\)). The design compares these gains across outcomes, datasets,
and statistical or machine-learning models, so that the value of collecting
more observations can be separated from the value of measuring more
variables.

For every outcome and model, the analysis repeatedly chooses \(N\) training
cases and \(K\) raw predictors, estimates all data-dependent preprocessing on
that training sample, fits the model, and evaluates it on held-out cases.
Repeated seeds and draws quantify sampling variation. Encoded columns from one
categorical source are selected as a unit; a declared missingness indicator is
its own predictor.

## 2. How to run on Slurm

Run commands from the repository root with Python 3.11 or newer. Create one
environment and install the requirements for the application you will run
(`FFCWS` adds Parquet support):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r SMR/requirements.txt
# Also run this line when using FFCWS:
python -m pip install -r FFCWS/requirements.txt
```

Put the private source data in these untracked locations:

```text
SMR/data/private/asample2_withlag.csv
FFCWS/data/private/background.dta
FFCWS/data/private/train.csv
FFCWS/data/private/test.csv
```

Build the analysis-ready data before submitting jobs:

```bash
.venv/bin/python SMR/adapter/adapter.py
.venv/bin/python FFCWS/adapter/adapter.py
```

Start with this two-cell smoke test. It uses one OLS model, two seeds, one draw,
one \(N\), and one \(K\):

```bash
cat > SMR/panels-smoke.yaml <<'YAML'
model_params: model_params.yaml
preset: dev
experiment_id: smr-smoke-v1
data_version: smr-ard-v1
model_spec_version: nkgrid-models-v1

panels:
  - name: smr_smoke
    schema: schema/asample2_withlag.json
    outcome: Cm_lhourlywage
    models: [ols]
    repeat_plan:
      - seeds: [1101, 1102]
        draws: [0]
    n_grid: [10]
    k_grid: [1]
    out: outputs/nk_grid_smr_smoke.csv
YAML

bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels-smoke.yaml
```

The smoke run should finish with two data rows in
`SMR/outputs/nk_grid_smr_smoke.csv` and a complete
`SMR/outputs/nk_grid_smr_smoke.manifest.json`. In `squeue`, the chain first
shows seed arrays named `al-nk-grid-parallel`, `al-nk-grid-serial`, or
`al-nk-grid-super_learner` (only the parallel array is present in this smoke
test), followed by `al-nk-finalize`, then `al-nk-publish`. The latter two remain
pending on dependencies until the preceding stage finishes:

```text
JOBID       NAME                    ST  NODELIST(REASON)
12345_[0-1] al-nk-grid-parallel     R   ...
12346_[0]   al-nk-finalize          PD  (Dependency)
12347_[0]   al-nk-publish           PD  (Dependency)
```

For a full run, copy the relevant `panels.yaml`, change its top-level
`preset: dev` to `preset: production`, review its output paths, and submit the
copy with explicit large-run authorization:

```bash
bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest SMR/panels.production.yaml \
  --allow-large-run

bash NK_Grid/slurm/submit_nk_grid.sh \
  --manifest FFCWS/panels.production.yaml \
  --allow-large-run
```

The finalizer/publish command exit codes mean: `0` completed successfully
(`missing` also returns `0` when its JSON reports missing work); `1` is an
invalid design, duplicate/out-of-design key, identity/semantic-contract error,
or other program failure; `3` means a shard or per-model final is
missing/incomplete and recovery can recreate it; `4` is an environment I/O
failure such as no space, permissions, or storage errors; `5` means another
process already held the publication lease for that output, so this task
published nothing. Exit code `5` is usually a duplicate submission for the same
target and is harmless: confirm the output and its manifest exist, and rerun
the command if they do not.

## 3. What to send back when something fails

Send the following files and raw command output together; do not summarize or
rewrite them:

- the affected `NK_Grid/logs/*.out` and `NK_Grid/logs/*.err` files;
- the matching job snapshot and submission receipt JSON files under
  `NK_Grid/logs/slurm-specs/`;
- the `MaxArraySize` line from `scontrol show config`;
- the unedited output of `squeue -j <job-id>` and that command's exit code;
- the mount parameters for the shared filesystem containing the configured
  output directory, for example the unedited output of
  `findmnt -T <output-directory> -o TARGET,SOURCE,FSTYPE,OPTIONS`.

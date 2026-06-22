# NLSY79 Zheng–Cheng Replication and Extensions

This module uses the processed NLSY79 analysis tables produced by the authors'
public [OSF source package](https://osf.io/7wjg8/). It has two distinct purposes:

1. `overall_prediction.py` reproduces the cumulative Aset1/Aset2/Bset1/Bset2
   comparison with a 70/30 split and source seed `12345`.
2. The feature-count, domain, SHAP, and sample-size scripts are extensions. They
   should not be described as analyses contained in the Zheng–Cheng paper.

The error-floor parameter from the sample-size extension is conditional on the
data, predictor space, model family, split, and power-law extrapolation. It is
not a model-independent estimate of Bayes error or literal "luck."

This is a Python structural replication, not a byte-for-byte reproduction of
the original R/Stata workflow. R's `set.seed(12345)` and Python's
`random_state=12345` do not generate identical train/test row assignments.

## Data

Data are not committed. Obtain or construct these files using the authors'
public data-management code and place them under `nlsy_replication/data/`:

- `asample1_noincomelag.csv`
- `asample2_withlag.csv`
- `asample3_nosei.csv`

The default scripts use `asample2_withlag.csv` and `Cm_lhourlywage`.

## Environment

Python 3.11 is recommended:

```bash
cd nlsy_replication
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Source-aligned cumulative comparison

```bash
python src/overall_prediction.py \
  --models ols ridge lasso xgboost bart
```

XGBoost uses `max_depth=2`, `eta=0.3`, and five-fold CV to choose up to 90
boosting rounds. Ridge and Lasso also select penalties using five-fold CV.

## Extensions

```bash
python src/feature_sets.py --models xgboost ridge lasso
python src/domain_wise.py --models xgboost ridge lasso
python src/sample_size.py --models xgboost ridge lasso
python src/SHAP_vals.py
python src/SHAP_experiment.py
```

Long-running scripts checkpoint CSV results after each batch and resume
successful jobs from an existing output file. Failed model fits are recorded
with `status=failed` instead of discarding the whole run.

Power-law fits use non-negative bounds, bootstrap intervals, and stability
diagnostics. The bootstrap interval remains conditional on the fixed train/test
sample; it does not capture population sampling or holdout uncertainty.

## SLURM

The scripts in `slurm/` use job arrays, contain no user-specific cluster path,
and write one output per model. Set these variables when cluster defaults differ:

```bash
export PROJECT_DIR=/path/to/aleatoric_luck-Zheng-Cheng/nlsy_replication
export VENV=$HOME/venvs/aleatoric-luck
export PYTHON_MODULE=Python/3.11
sbatch slurm/run_overall.sbatch
sbatch slurm/run_sample_size.sbatch
```

Adjust time, memory, partition, account, and module names for the target
cluster. Merge model-specific CSVs only after all array tasks finish.

## Colab

Open `colab_run.ipynb`. It clones the public repository, mounts Google Drive
for data only, installs the minimal notebook requirements, and runs smoke-sized
commands by default.

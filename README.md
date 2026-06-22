# Aleatoric Luck: Zheng–Cheng Replication and Predictive Extensions

This independent repository contains two clearly separated analyses:

1. `nlsy_replication/` works with the authors' processed NLSY79 tables and
   includes a source-aligned cumulative predictor-set comparison plus new
   feature-count, SHAP, and learning-curve extensions.
2. `ffcws_predict/` applies a separate prediction and error-floor pipeline to
   the Fragile Families Challenge Year 15 `materialHardship` outcome.

The target is treated as a continuous proportion score. The main evaluation
metric is holdout MSE, and learning curves are fit with a power-law form to
estimate a model-, feature-, split-, and extrapolation-conditional error-floor
proxy:

```text
E(n) = c * n^(-alpha) + epsilon
```

`epsilon` should not be interpreted as model-independent Bayes aleatoric
uncertainty.

## Repository Layout

```text
aleatoric_luck-Zheng-Cheng/
├── nlsy_replication/
│   ├── src/
│   ├── slurm/
│   ├── colab_run.ipynb
│   └── README.md
├── configs/
│   ├── feature_dictionary_material_hardship_v1.csv
│   └── material_hardship.yaml
├── ffcws_predict/
│   ├── cli.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── powerlaw.py
│   └── reporting.py
├── tests/
│   └── test_ffcws_predict.py
└── requirements.txt
```

## NLSY79 replication and extensions

See [`nlsy_replication/README.md`](nlsy_replication/README.md) for the data
requirements, source-aligned analysis, Colab runner, and portable SLURM job
arrays. NLSY data are not included.

The NLSY source-aligned entry point is:

```bash
cd nlsy_replication
python src/overall_prediction.py --models ols ridge lasso xgboost bart
```

## FFCWS material-hardship application

### Data

The framework expects a local ICPSR 31622 DS15 download. It does not log into
ICPSR and does not reconstruct Year 15 labels from questionnaire items.

The included portable configuration expects the following files under
`data/ds15/` (data are not included in this repository):

- `background.csv` for Challenge features
- `train.csv` for training labels
- `test.csv` for final holdout truth
- `materialHardship` as the target

Rows where the truth file has non-missing `materialHardship` define the final
holdout for this outcome.

### Workflow

Run the full local DS15 workflow with:

```bash
python -m ffcws_predict audit-features --config configs/material_hardship.yaml
python -m ffcws_predict prepare --config configs/material_hardship.yaml
python -m ffcws_predict train --config configs/material_hardship.yaml
python -m ffcws_predict learning-curve --config configs/material_hardship.yaml
python -m ffcws_predict summarize --config configs/material_hardship.yaml
```

Core outputs are written to `runs/material_hardship/` and are ignored by Git:

- `feature_dictionary_resolved.csv`
- `prepared_material_hardship.csv`
- `preprocessing_report.json`
- `model_metrics.csv`
- `holdout_predictions.csv`
- `prediction_range_diagnostics.csv`
- `imputation_comparison.json`
- `imputation_prediction_differences.csv`
- `learning_curve.csv`
- `power_law_fit.csv`
- `summary_report.md`

### Feature Dictionary

`configs/feature_dictionary_material_hardship_v1.csv` is a hand-built domain
dictionary with exact DS15 `background.csv` column names. The domains are:

- `family_economics`
- `employment`
- `housing`
- `program_participation`
- `family_structure`
- `parent_health`
- `social_support`
- `neighborhood`
- `child_early_status`

Duplicate cross-domain columns are de-duplicated for modeling and marked as
`cross_domain` in the resolved dictionary.

### Models

The training command evaluates:

- mean baseline
- OLS
- Ridge
- Lasso
- Elastic Net
- Random Forest
- XGBoost
- LightGBM

Linear models use imputation, missingness indicators, and standardization.
Tree models use imputation without standardization.

For XGBoost and LightGBM, the pipeline also runs a robustness check comparing:

- `median`: median imputation plus missingness indicators
- `native_nan`: special missing codes converted to `NaN`, no median imputation,
  no missingness indicators

## Tests

The tests use Python's standard `unittest` runner:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Syntax checking:

```bash
python -m compileall ffcws_predict nlsy_replication/src tests
```

## Reference

Zheng, H., & Cheng, S. (2025). Social Rigidity Across and Within Generations:
A Predictive Approach. *Sociological Methods & Research, 54*(4), 1683–1725.

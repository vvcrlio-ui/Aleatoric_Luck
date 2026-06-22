# Aleatoric Luck: Zheng–Cheng Replication and Extensions

This repository contains only the NLSY79 work based on Zheng and Cheng (2025).
It uses the authors' processed NLSY79 analysis tables for a Python structural
replication and a set of clearly labelled predictive extensions.

> **Author and implementation:** Zheng Cheng. This repository's Python
> replication, experimental extensions, validation framework, and computing
> workflow were developed by Zheng Cheng. The cited paper supplies the research
> design being replicated and the processed-data specification; the extensions
> below are original work in this repository.

---

## Overview

The project studies how much predictive error remains as the amount of training
data increases. It first reconstructs the cumulative predictor-set comparison
from Zheng and Cheng (2025), then extends that design to examine feature count,
predictor domains, SHAP-ranked feature orderings, and sample-size learning
curves across several model families.

For the learning-curve extension, test error is modelled as:

```text
E(n) = c * n^(-alpha) + epsilon
```

Here, `epsilon` is treated conservatively as a conditional error-floor proxy.
It depends on the observed data, included predictors, model family, split, and
extrapolation assumptions; it is not claimed to be a model-independent measure
of Bayes error or a literal measurement of luck.

## Repository layout

```text
aleatoric_luck-Zheng-Cheng/
├── nlsy_replication/
│   ├── src/                    # Replication and extension scripts
│   ├── slurm/                  # Portable SLURM job arrays
│   ├── colab_run.ipynb         # Colab runner
│   ├── requirements.txt
│   └── README.md               # Detailed data and workflow documentation
├── tests/
│   └── test_nlsy_replication.py
├── requirements.txt
└── README.md
```

NLSY data are not included. See
[`nlsy_replication/README.md`](nlsy_replication/README.md) for required input
files, model settings, checkpoint behaviour, Colab instructions, and SLURM
usage.

## Quick start

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cd nlsy_replication
python src/overall_prediction.py --models ols ridge lasso xgboost bart
```

Extension entry points are:

```bash
python src/feature_sets.py --models xgboost ridge lasso
python src/domain_wise.py --models xgboost ridge lasso
python src/sample_size.py --models xgboost ridge lasso
python src/SHAP_vals.py
python src/SHAP_experiment.py
```

## Tests

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall nlsy_replication/src tests
```

## Reference

Zheng, H., & Cheng, S. (2025). Social Rigidity Across and Within Generations:
A Predictive Approach. *Sociological Methods & Research, 54*(4), 1683–1725.

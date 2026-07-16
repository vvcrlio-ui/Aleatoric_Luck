# SMR: NK Grid and Zheng-Cheng Replication

> This is the `SMR/` subtree of the [`Aleatoric_Luck`](../README.md) repository.
> See the repository root README for how this relates to `FFC/` and the other
> predictability papers.

This directory contains two self-contained Python subprojects:

- [`NK_Grid/`](NK_Grid/README.md): joint sample-size (N) by feature-count (K)
  sweeps for regression and classification prediction metrics, applied to the
  SMR (hourly wage / total income) panels.
- [`Zheng_Cheng_Replication/`](Zheng_Cheng_Replication/README.md): the
  Zheng and Cheng replication workflow plus the early predictive extensions
  for overall, sample-size, feature-set, domain-wise, and SHAP analyses.

The expanded model space is shared by both subprojects: OLS, Ridge, Lasso,
Elastic Net, Random Forest, XGBoost, LightGBM, a one-hidden-layer neural
network, Extra Trees, and a stacked Super Learner. BART remains available as
a legacy replication model but is not part of the expanded ten-model space.

Each subproject has its own `src/`, `slurm/`, `data` directory, logs marker, and
requirements file. The shared support modules are intentionally copied into
both directories so either subproject can run independently.

## Layout (within `SMR/`)

```text
SMR/
├── NK_Grid/
│   ├── src/
│   ├── slurm/
│   ├── panels.yaml
│   ├── requirements.txt
│   └── README.md
├── Zheng_Cheng_Replication/
│   ├── src/
│   ├── slurm/
│   ├── colab_run.ipynb
│   ├── requirements.txt
│   ├── requirements-notebook.txt
│   └── README.md
├── tests/
│   ├── test_nk_grid.py
│   └── test_zheng_cheng_replication.py
├── requirements.txt
└── README.md
```

NLSY data are not committed (`**/data/` is gitignored repo-wide). Each
subproject expects its own local `data/` directory populated separately.

## Quick start

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the NK grid:

```bash
cd NK_Grid
python src/nk_grid.py --help
python src/run_panels.py --dry-run
```

Run the replication workflow:

```bash
cd Zheng_Cheng_Replication
python src/overall_prediction.py --help
python src/overall_prediction.py --models ols ridge lasso xgboost extra_trees super_learner
```

## Tests

From `SMR/`:

```bash
python -m pytest -q
python -m compileall NK_Grid/src Zheng_Cheng_Replication/src tests
```

## Reference

Zheng, H., & Cheng, S. (2025). Social Rigidity Across and Within Generations:
A Predictive Approach. *Sociological Methods & Research, 54*(4), 1683-1725.

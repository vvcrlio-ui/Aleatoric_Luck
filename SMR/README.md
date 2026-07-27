# SMR adapter

This is the article-owned `SMR/` subtree of the
[`Aleatoric_Luck`](../README.md) repository.

It contains the [`adapter/`](adapter/README.md), `schema/`, `panels.yaml`, and
`model_params.yaml` integration for the shared root
[`NK_Grid/`](../NK_Grid/README.md) engine.

The expanded model space includes OLS, Ridge, Lasso, Elastic Net, Random
Forest, XGBoost, LightGBM, a one-hidden-layer neural network, Extra Trees, and
a stacked Super Learner. BART remains available in the shared engine but is not
part of the expanded ten-model SMR panel.

## Layout (within `SMR/`)

```text
SMR/
├── adapter/
│   ├── config/asample2_withlag.json
│   ├── adapter.py
│   └── tests/
├── schema/
├── data/
│   ├── private/        raw adapter input; ignored by Git
│   └── ard/            generated adapter output; ignored by Git
├── panels.yaml
├── model_params.yaml
├── requirements.txt
└── README.md
```

NLSY data and generated outputs are not committed (`**/data/` and
`**/outputs/` are ignored repository-wide).

## Quick start

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the NK grid:

```bash
python adapter/adapter.py
aleatoric-nk-grid-panels --manifest panels.yaml --dry-run
```

## Tests

Run engine and adapter tests from the repository root:

```bash
python -m pytest -q NK_Grid/tests SMR/adapter/tests
python -m compileall NK_Grid/src SMR/adapter
```

Historical outputs produced by the removed article-local engine fork are kept
locally under `outputs/legacy_nk_grid/`; current panels write to `outputs/`.

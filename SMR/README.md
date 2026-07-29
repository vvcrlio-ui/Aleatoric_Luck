# Social Rigidity application

## Research question

How accurately can midlife socioeconomic outcomes be predicted from
earlier-life and family-background information? More specifically, how does
out-of-sample predictive performance change with the number of observed
training cases (\(N\)) and the number of available predictor variables
(\(K\))?

This application evaluates two outcomes:

| Outcome | Code in the analysis data | Type |
|---|---|---|
| Log hourly wage | `Cm_lhourlywage` | Continuous |
| Log total personal income | `Cm_ltotalincome` | Continuous |

The application follows the predictive approach developed in Zheng and Cheng,
[“Social Rigidity Across and Within Generations: A Predictive
Approach”](https://doi.org/10.1177/00491241251347984). Its purpose is to
characterize the persistence and predictability of socioeconomic outcomes.
Causal effects of individual predictors lie outside the scope of the analysis.

## Data and research design

The source is the numeric analysis matrix used for the NLSY-based replication.
It contains 4,252 model input columns representing 497 predictor variables.
Columns created from the same raw categorical variable are always made
available together. Consequently, \(K\) counts the 497 predictor variables.

The analysis repeatedly varies \(N\) and \(K\), fits each model on the selected
training data, and evaluates predictions on an internal holdout sample.
Data-dependent preprocessing is estimated within the selected training sample.

The model set includes linear regression, penalized linear models, tree-based
ensembles, gradient boosting, a shallow neural network, and a stacked ensemble.
The complete, executable model list is recorded in
[`panels.yaml`](panels.yaml).

## Reproduction

```text
SMR/
├── adapter/
│   ├── config/asample2_withlag.json
│   ├── adapter.py
│   └── tests/
├── schema/
├── data/
│   ├── private/        source data; excluded from Git
│   └── ard/            generated analysis-ready data; excluded from Git
├── panels.yaml
├── model_params.yaml
├── requirements.txt
└── README.md
```

From the repository root:

```bash
python -m pip install -r SMR/requirements.txt
python SMR/adapter/adapter.py
aleatoric-nk-grid-panels --manifest SMR/panels.yaml --dry-run
```

Run the shared implementation and SMR data-preparation tests with:

```bash
python -m pytest -q NK_Grid/tests SMR/adapter/tests
```

See the [SMR data-preparation guide](adapter/README.md) for variable
representation and validation details, and the
[N-by-K implementation guide](../NK_Grid/README.md) for execution options.

# SMR adapter

This directory implements the article-owned side of the contract in
[`../../Adapter/ADAPTER.md`](../../Adapter/ADAPTER.md).

## Standard layout

```text
SMR/
├── adapter/
│   ├── config/asample2_withlag.json     fixed feature configuration
│   ├── adapter.py                       raw analysis matrix -> engine artifacts
│   ├── build_contract.py                explicit contract-authoring utility
│   └── tests/
├── schema/                              tracked schema + canonical universe
└── data/                                ignored by Git
    ├── private/asample2_withlag.csv
    └── ard/asample2_withlag/
        ├── data.csv
        ├── feature_manifest.csv
        └── provenance.json
```

`adapter.py` uses the fixed contract rather than learning feature names,
vocabularies, or screening rules from row values. It writes artifacts in the
required order, uses the shared engine's `canonical_feature_universe()`, and
runs `validate_input()` for both outcomes before reporting success.

## Source semantics

The source is the provider-supplied, numeric analysis matrix used by the SMR
replication. It contains no `NaN`, `±inf`, or `-1` through `-9` sentinel values.
The adapter therefore has no missing-code substitutions and performs no
fitting, imputation, standardization, splitting, row deletion, or feature
screening.

Scalar columns retain the paper's supplied numeric representation. The 3,784
dummy columns are declared as 29 atomic `onehot_group` sources. The resulting
feature universe contains 4,252 predictor columns but 497 K-sampling sources.
This source-level K definition is the one required by `ADAPTER.md`; results
from the removed legacy engine, which sampled dummy columns independently, are
not directly comparable on the K dimension.

The schema sets `exchangeable=true` under this research assumption: the N×K
estimand samples from the paper's predeclared Aset/Bset predictor sources, and
these sources are treated as exchangeable units for the feature-availability
experiment. Dummy columns from one raw categorical variable remain atomic and
are sampled together. If that assumption is not appropriate for a later
analysis, the integration must stop rather than silently changing the schema.

## Run

From the repository root:

```bash
python SMR/adapter/adapter.py
```

The default input is `SMR/data/private/asample2_withlag.csv`. Validation
parameters can be aligned with a planned run:

```bash
python SMR/adapter/adapter.py \
  --validation-model ols ridge lasso elastic_net random_forest \
  --min-n 10 \
  --test-size 0.3 \
  --seed 12345
```

`build_contract.py` is an authoring utility, not part of routine adapter builds.
Run it only after deliberately changing the source header or the fixed one-hot
source declarations, then review and commit the resulting contract:

```bash
python SMR/adapter/build_contract.py
```

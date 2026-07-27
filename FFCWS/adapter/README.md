# FFCWS adapter

This directory implements the article-owned side of
[`../../Adapter/ADAPTER.md`](../../Adapter/ADAPTER.md).

## Standard layout

```text
FFCWS/
├── adapter/
│   ├── config/ffc.yaml          tracked preprocessing configuration
│   ├── adapter.py               single adapter entry point
│   ├── src/ffcws_data_processor/
│   │   ├── common/
│   │   └── strategies/
│   └── tests/
├── schema/                      tracked schemas and canonical universes
└── data/                        ignored by Git
    ├── private/
    │   ├── background.dta
    │   ├── train.csv
    │   └── test.csv
    ├── adapter_work/            generated QA and audit artifacts
    └── ard/<dataset>/           engine tables, manifest and provenance
```

`adapter.py` preserves the provider's official train/test split. Feature
eligibility, categorical vocabularies and prevalence screening are learned only
from background rows whose IDs occur in the official training pool, so the
feature-universe mode is `train_pool_screened`. The adapter writes each
dataset's ARD tables and manifest before generating its universe and schema,
then calls the shared engine's `validate_input()` before reporting success.

Rows with missing outcomes are retained in both ARD tables. The engine removes
them uniformly during outcome-specific validation and model fitting.
All three strategies use Parquet ARD tables to keep the very wide FFCWS
artifacts compact and consistent.

## Three retained strategies

- `median_mode`: numeric sources remain continuous; categorical sources become
  atomic full one-hot groups. Missing values remain `NaN`; median, mode and
  atomic group imputation are fitted by the engine within each training cell.
- `median_missing_indicator`: the same value representation plus screened,
  binary missing-code indicators. Every indicator is declared as its own
  continuous sampling source. Categorical missing codes are not category
  levels: the categorical group remains entirely `NaN` and the separate
  indicator carries the missingness signal.
- `tree_ordinal`: numeric sources remain continuous and categorical sources use
  deterministic ordinal codes whose mapping is learned from the official
  training pool. Missing and unknown states remain `NaN`.

The names describe the three historical analysis approaches; the adapter itself
does not fit an imputer. That responsibility belongs to the engine.

FFC missing codes `-9` through `-1` and blank values are normalized
deterministically. Test-only categorical states are mapped to `NaN` and never
expand the training vocabulary. One-hot columns from a raw categorical variable
are sampled atomically.

The schemas set `exchangeable=true` under the research assumption recorded in
`config/ffc.yaml`: eligible background-variable sources are treated as
exchangeable units in the N×K feature-availability estimand. The missing-
indicator strategy additionally treats each explicitly declared missingness
indicator as a sampling source. If these assumptions are unsuitable for a
planned analysis, stop rather than changing the flag merely to pass validation.

## Run

From the repository root, build and validate all three strategies:

```bash
python FFCWS/adapter/adapter.py
```

Build only selected strategies:

```bash
python FFCWS/adapter/adapter.py \
  --strategy median_mode median_missing_indicator tree_ordinal
```

Validation parameters can be aligned with a planned run:

```bash
python FFCWS/adapter/adapter.py \
  --validation-model ols ridge lasso elastic_net random_forest \
  --min-n 10 \
  --seed 12345
```

The adapter produces six outcome-specific external-test datasets for each
selected strategy, for a total of 18 schemas when all strategies are run.

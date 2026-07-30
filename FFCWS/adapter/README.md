# FFCWS data preparation

## Role in the research design

The FFCWS adapter prepares information collected from birth through age 9 for
predicting six age-15 outcomes. It preserves the predefined training and test
samples and produces three alternative representations of the same source
information.

All decisions that depend on observed values—including predictor eligibility,
categorical value sets, and prevalence-based screening—are estimated using the
predefined training sample only. The test sample is reserved exclusively for
evaluating predictive performance.

Rows with missing outcomes are retained in the prepared data. They are removed
separately for each outcome during validation and model fitting, which avoids
discarding a family merely because another outcome is unavailable.

## Predictor representations

| Representation used in research text | Configuration ID | Definition |
|---|---|---|
| One-hot representation with within-sample imputation | `median_mode` | Continuous variables remain numeric. Each categorical variable becomes a grouped set of indicator columns. Imputation values are estimated within each selected training sample. |
| One-hot representation with missingness indicators | `median_missing_indicator` | Uses the same value representation and adds screened binary indicators for missing source values. Each declared missingness indicator counts as a separate predictor variable. |
| Ordinal representation for categorical variables | `tree_ordinal` | Continuous variables remain numeric. Categorical values receive stable integer codes learned from the predefined training sample. |

The configuration IDs are used in file and command names only. Manuscripts and
research summaries should use the descriptive representation names.

Provider missing-value codes from `-9` through `-1`, together with blank values,
are converted to explicit missing values. Categorical values observed only in
the test sample are also treated as missing. Categorical value sets are derived
exclusively from the training sample.

For both one-hot representations, all columns derived from one categorical
variable enter or leave the analysis together. They therefore count as one
predictor variable in \(K\). The missingness-indicator representation counts
each separately declared indicator as one additional predictor variable.
Consequently, equal values of \(K\) may represent different source information
across the three representations.

## Inputs and generated files

```text
FFCWS/
├── adapter/
│   ├── config/ffc.yaml                  reviewed preparation specification
│   ├── adapter.py                       data-preparation entry point
│   ├── src/ffcws_data_processor/
│   └── tests/
├── schema/                              generated, versioned analysis specifications
└── data/
    ├── private/
    │   ├── background.dta               source data; excluded from Git
    │   ├── train.csv
    │   └── test.csv
    ├── adapter_work/                    generated validation records; excluded from Git
    └── ard/<representation>/            generated analysis-ready data; excluded from Git
```

For each representation, the adapter writes a training table, a test table, a
predictor manifest, provenance information, a canonical predictor-universe
definition, and one analysis schema per outcome. Schemas and predictor-universe
definitions are versioned because they determine how the untracked generated
tables are interpreted.

## Reproduction

From the repository root, prepare and validate all three representations:

```bash
python FFCWS/adapter/adapter.py
```

To prepare selected representations only:

```bash
python FFCWS/adapter/adapter.py \
  --strategy median_mode median_missing_indicator tree_ordinal
```

To align validation settings with a planned analysis:

```bash
python FFCWS/adapter/adapter.py \
  --validation-model ols ridge lasso random_forest \
  --min-n 10 \
  --seed 12345
```

Preparing all three representations produces six outcome-specific schemas for
each representation. The generated files follow the common
[adapter specification](../../Adapter/ADAPTER.md).

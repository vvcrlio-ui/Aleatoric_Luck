# Writing an Adapter for the NK Grid Engine

This document explains **how to transform the raw data from a paper into adapter
artifacts that the engine can accept, audit, and reproduce.**

This is the **complete specification** for the adapter side.

It is **recommended to use AI tools** to assist in constructing the Adapter.

## 0. Contract Summary

Adapter artifacts must satisfy all nine constraints below. If any constraint is
not met, `validate_input` will reject the artifacts before sampling begins. Later
sections provide the details.

| # | Constraint |
|---|---|
| 1 | Data tables use `.csv`, `.parquet`, or `.pq` and can be read by the engine. |
| 2 | Exactly one predictor rule is set: `predictor_columns` or `predictor_prefix`. |
| 3 | Predictors contain only finite numeric values or `NaN`: no `±inf`, text, complex-valued columns, or columns that are entirely `NaN`; categorical missing values remain `NaN`. |
| 4 | Predictors do not include any outcome column or the `id_column`. |
| 5 | Outcomes exist: regression outcomes are finite numeric values; classification outcomes contain only `{0,1}`; after rows with missing outcomes are removed, the row count meets the lower bound in Section 4. |
| 6 | Under `external_test`, both tables have matching structures; IDs are non-missing, unique, and do not overlap across tables; test-set classes are covered by the training table. |
| 7 | The feature universe has a valid origin: internal splitting requires `fixed_a_priori`; under external splitting, the vocabulary can be learned only from the training table. |
| 8 | `group_column` is `null`; `exchangeable` is `true`. |
| 9 | `schema_version=1`; with a manifest, `feature_manifest_version=1`; otherwise it is `null`. |

## 1. What an Adapter Is and Terminology

An adapter is a preprocessing program that you write for each dataset: it takes
raw data as input and produces several files in prescribed formats. It is usually
a Python script in the paper's directory.

The adapter is responsible for interpreting variables, converting missing-value
codes, deterministic encoding, defining feature vocabularies, and declaring
feature sets. It is not responsible for data splitting, imputation or
standardization fitted on training subsamples, feature or sample sampling, model
training, metric calculation, or run scheduling. The engine handles those tasks.

A valid adapter meets three requirements: identical inputs produce identical
artifacts; vocabularies and screening rules do not use data outside the range
permitted by the contract, especially test-set data and outcomes; and every
artifact passes the engine's input validation.

| Term | Meaning |
|---|---|
| **predictor** | A column in the ARD that is actually passed to the model. |
| **source** | The atomic sampling unit; one source corresponds to one or more predictor columns. |
| **feature** | A column-level record in the universe or manifest; a feature with `keep=true` corresponds to an ARD predictor. |
| **training cell** | A training subsample used for one engine fit (called a cell in the code). |

## 2. Responsibility Boundary

| Category | Owner | Description |
|---|---|---|
| Deterministic row-wise transformations | Adapter | Depend only on the current row: map yes/no to 1/0, take the log of income, or calculate age using a **reference date fixed in advance**. |
| Missing-value code conversion | Adapter | Deterministically convert sentinel values such as `-9`, `-99`, and `N/A` to `NaN`. |
| Categorical encoding and vocabularies | Adapter | A vocabulary may require multiple rows, but is **fixed once and remains unchanged throughout**; the rows it may inspect are restricted as described in Section 5. |
| Feature-set declaration | Adapter | Governed by the feature-universe contract in Section 6. |
| Training/test splitting | Engine | The adapter must not create its own random split, but it **must preserve any split predefined by the data provider exactly as supplied**. |
| Imputation and standardization | Engine | The engine refits these statistics within each training subsample. |
| Sample and feature sampling, modeling, and metrics | Engine | The adapter has no involvement. |

**The adapter must not impute in advance.** This is the rule most easily
violated. If statistics are computed from the entire table, a small training
subsample will use a median estimated from the full sample. Its performance will
therefore be artificially inflated, systematically underestimating the sample-size
effect. Under `internal_random`, this also leaks information from test rows into
training. Missing values in the ARD must therefore remain `NaN` so that the engine
can handle them within each training subsample.

## 3. Artifacts, Directories, and Paths

**The schema is the sole semantic entry point:** it references the ARD, feature
universe, and feature manifest by path. The only exception is `provenance.json`,
which the engine discovers automatically in the directory containing the training
table.

| Artifact | When required |
|---|---|
| Schema JSON | Always |
| ARD training table (`data.csv` or `data.parquet`) | Always |
| Feature-universe definition JSON | Always |
| ARD test table (`test.csv` or `test.parquet`) | When `split_mode` is `external_test` |
| `feature_manifest.csv` | When any condition in Section 7 applies |
| `provenance.json` | Optional |

```text
YourArticle/
├── adapter/            adapter code
├── schema/             tracked in Git: <dataset>.json + feature-universe JSON
└── data/               not committed to Git
    ├── private/        raw input
    └── ard/<dataset>/  data.csv, test.csv, feature_manifest.csv, provenance.json
```

The schema and feature-universe definition must be under version control; data
must not be.

### Path Resolution

The schema fields `table`, `test_table`, `feature_manifest`, and
`feature_universe.definition_file` are all resolved **relative to the directory
containing the schema file**. Absolute paths are also accepted. When the schema is
in `YourArticle/schema/`, use the following values:

| Schema field | Value |
|---|---|
| `table` | `../data/ard/my_dataset/data.csv` |
| `test_table` | `../data/ard/my_dataset/test.csv` |
| `feature_manifest` | `../data/ard/my_dataset/feature_manifest.csv` |
| `feature_universe.definition_file` | `my_dataset.feature_universe.json` |

The location of `provenance.json` is hard-coded by the engine: it must be in the
same directory as the **training table** referenced by `table`, and its filename
is fixed.

### Provenance

`provenance.json` is optional and records provenance hashes. The engine validates
only its `schema_sha256` against the schema actually loaded. Other hashes are for
auditing only, and their names and granularity may be adjusted as needed. Do not
record raw IDs or absolute paths.

## 4. ARD Requirements

An ARD is a flat table: each row corresponds to one observational unit, and
columns are not nested. It contains only the outcomes declared in the schema
(there may be more than one), predictors, and the ID column required by
`external_test`.

- **Deterministic row-wise behavior:** once the vocabulary is fixed, the output
  for each row may depend only on that row's raw values and the fixed vocabulary,
  not on other rows.
- **Normalize missing-value codes:** all domain-specific missing-value codes must
  be converted to `NaN`, using a reproducible mapping.
- **All predictors are numeric:** no text or complex-valued columns, no `±inf`,
  and no column that is entirely `NaN`.
- **Categorical missing values remain `NaN`:** do not treat “missing” itself as a
  category level. If missingness must be used as information, create a separate,
  explicitly declared missingness-indicator predictor.
- **Do not impute or remove rows with missing outcomes in advance:** the engine
  handles row removal uniformly using the thresholds in Section 8. Removing rows
  beforehand makes those thresholds meaningless.
- **Outcome:** regression tasks require finite numeric values; classification
  tasks accept only binary `{0,1}`.

**Row-count lower bound:** after rows with missing outcomes are removed, the
number of usable training rows must satisfy
`n_train ≥ max(min_n, the cross-validation lower bound of the selected models)`.
Under `internal_random`, this is calculated for the training portion after the
`test_size` split. Classification tasks additionally require the row count of
each class to be at least the number of cross-validation folds. When the data size
is close to the lower bound, confirm it during validation (Section 9) using the
actual parameters.

The predictor-selection rule is declared in the schema. Exactly one of
`predictor_columns` (an explicit list) and `predictor_prefix` (a list of prefixes)
must be set. Neither may include an outcome column or the ID column; the engine
rejects such overlaps to prevent target leakage into the feature set. When the
column set is stable, prefer an explicit list because a prefix rule automatically
includes any subsequently added column with the same prefix.

## 5. Internal / External

`split_mode` determines what the adapter must deliver and which rows may be used
to learn vocabularies.

**`internal_random`** — The adapter delivers one table, which the engine splits.
`id_column` may be `null`. If the data already contains a stable identifier,
retain it for provenance, although the engine will not use it. The feature
universe must be `fixed_a_priori`, **which means that categorical vocabularies
also must not be learned from this table.**

**`external_test`** — The data provider has already defined the training and test
sets, as with an official competition split. The adapter must preserve that split
exactly, deliver two tables, and satisfy the following requirements:

- Both `test_table` and `id_column` are required in the schema.
- Both tables contain the same predictors, outcomes, and ID column. The same raw
  category maps to the same numeric value or dummy column in both tables.
- IDs are non-missing and unique within each table, with no overlap between
  tables.
- Vocabularies and feature screening may be learned only from the training table.
  If the test table contains a category outside the training vocabulary, raise an
  error; do not expand the vocabulary using the test table.

## 6. Feature Universe

The feature universe is a normalized snapshot of the feature space. It contains
the source list, the features under each source and their order, the set of
category values and reference-category designation for one-hot features, and the
level mapping for ordinal features. It allows a run to prove that the feature
space actually resolved is exactly the one declared by the adapter.

Serialize the content with sorted keys, UTF-8, and no extraneous whitespace before
computing SHA-256. `canonical_json()` performs this step; do not assemble the JSON
manually.

The engine performs two independent checks when loading:

1. Whether the actual SHA-256 of the definition file equals the
   `definition_sha256` declared in the schema, which detects file changes.
2. Whether the universe recomputed from the actual predictors and manifest
   matches the content of the definition file, which checks that the declaration
   agrees with the data.

Generate the structure with the engine's `canonical_feature_universe()` function.
Do not write it manually, because a handwritten structure may drift from the
validation logic. See Section 9 for usage.

Values of `feature_universe.mode`:

- **`fixed_a_priori`** — The feature set, categorical vocabularies, order, ordinal
  mappings, and reference categories are all fixed in advance and do not depend
  on any row-level statistics from this dataset. Selecting columns from the
  header using names or prefix rules declared in advance is permitted; screening
  by missingness rate, variance, category frequency, or relationship with the
  outcome is not. `internal_random` requires this mode, and a perturbation test
  must demonstrate that it is unaffected by row-level data.
- **`train_pool_screened`** — Screened using an externally predefined training
  set; available only with `external_test`.

`exchangeable` declares that the sources are interchangeable for sampling
purposes: they can be treated as units randomly sampled from the same population.
If this assumption does not hold, stop the integration rather than setting it to
`true` merely to pass validation.

## 7. Feature Manifest

> Throughout this document, “manifest” always means `feature_manifest.csv`.

Without a manifest, the engine treats every predictor column as an independent
continuous source. A manifest is required if **any** of the following conditions
applies:

1. **An ordinal feature exists**, even if it occupies only one column. Otherwise,
   the engine treats it as continuous and imputes the median, potentially
   producing a nonexistent level such as 2.5.
2. **A one-hot group exists**, meaning that one source spans multiple modeling
   columns.
3. **Audit rows must be recorded**, meaning rows with `keep=false`. Such rows
   exist only in the manifest; their columns need not appear in the ARD and do
   not participate in modeling.

A `continuous` or `ordinal` source must correspond to exactly one `keep=true`
column. Therefore, one source can span multiple columns only through a
`onehot_group`.

| Column | Description |
|---|---|
| `source_column` | Name of the raw variable to which this column belongs; use the same value for multiple columns from the same variable. |
| `feature_name` | Column name in the ARD. |
| `keep` | `true` to include in modeling; `false` for auditing only. |
| `source_order` | Order among sources: an integer starting at 0 and identical within a source. |
| `feature_order` | Order within a source: starts at 0 and is contiguous. |
| `unit_type` | `continuous`, `ordinal`, or `onehot_group`. |
| `drop_first` | Used only for one-hot rows and identical within a source; use `False` for other rows. |
| `is_reference` | Only for one-hot rows with `drop_first=false`, marks the one reference-category column; use `False` for other rows. |
| `reference_level` | Raw category value corresponding to the one-hot reference category; with `drop_first=true`, declares the omitted category. |
| `level_value` | Raw category value corresponding to this dummy column. |
| `ordinal_levels` | Valid ordinal levels as a compact JSON array. |
| `source_prior` | Optional; see below. |

Leave inapplicable cells empty. Order fields need only be stable and valid; they
do not carry substantive research meaning.

`source_prior` is the fallback value used when a source is entirely missing from
a training subsample. If not declared, the default is `0` for continuous sources,
the first valid level for ordinal sources, and the reference category for one-hot
sources. For all-continuous data, schema-level `continuous_priors` may be used
instead, with no manifest required.

In this case, the engine sets the source to the same prior constant on **both the
training and test sides**; the model has not learned the source and must not
extrapolate from observed test values. For `passthrough` models, both sides are
instead forced to `NaN`. The source still counts toward K. For the models
currently registered, the exact placeholder constant does not affect prediction:
standardization centers constant columns on linear and neural-network paths, and
tree models do not split on constant columns. `source_prior` therefore serves
only to improve readability and auditability; leaving it empty is safe.

### Ordinal Constraints

`ordinal_levels` must be compact JSON (`[1,2,3]`, not `[1, 2, 3]`); the validator
explicitly rejects noncanonical forms. Levels must be nonempty, unique, finite
numeric values. If raw levels are text labels, map them to integer codes inside
the adapter.

### Valid One-Hot States

| | `drop_first=false` | `drop_first=true` |
|---|---|---|
| Manifest requirement | Exactly one row must have `is_reference=true`. | Do not mark a reference, but still use `reference_level` to declare the omitted category. |
| Valid state per row | Exactly one 1. | At most one 1. |
| All zeros | **Invalid** | **Valid; represents the reference category.** |
| Entire group is `NaN` | Valid (treated as missing). | Valid (treated as missing). |

Values in a non-missing state must be 0 or 1. Within a source, `drop_first` must
be consistent and `level_value` must not be duplicated. Partial `NaN` values
within a group (mixed missingness) are invalid in both modes.

## 8. Schema

**Except for fields marked optional, every field in the following table must be
present**, even if its value is `null`. The schema must not contain fields that
are absent from this table.

| Field | Key points |
|---|---|
| `schema_version` | Fixed at `1`. |
| `feature_manifest_version` | Must be `1` with a manifest and `null` without one. |
| `dataset` | Nonempty string. |
| `table` / `test_table` | See Section 3; under `internal_random`, `test_table` must be `null`. |
| `split_mode` | `internal_random` or `external_test`. |
| `task` | `regression` or `classification`; the latter is binary `{0,1}` only. |
| `outcome_columns` | Nonempty list of unique strings. |
| `id_column` | Required under `external_test`; must not also be an outcome. |
| `predictor_columns` / `predictor_prefix` | Both are lists of strings; set exactly one; neither may contain an outcome or ID. |
| `feature_manifest` | See Section 7. |
| `exchangeable` | Must be `true` and requires research justification. |
| `feature_universe` | `{mode, definition_file, definition_sha256}`. |
| `group_column` | Must be `null`; grouped splitting is not currently supported. |
| `imputation` | See below. |
| `max_train_outcome_missing_ratio` / `max_test_outcome_missing_ratio` | Optional, default `0.5`; document the methodological reason for overriding the default. |
| `continuous_priors` | Optional, `{source: value}`; applies only to continuous sources. |

### Imputation

This field describes engine behavior but **is part of the schema contract and
must be declared by the adapter**. Whether a particular imputation method is
meaningful for a variable is a domain judgment that only the adapter can make.
All four keys must be present, even when the data contains no feature of the
corresponding type.

```json
{
  "continuous": "median",
  "ordinal": "most_frequent",
  "onehot_group": "atomic_mode",
  "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"}
}
```

- `continuous`: `median` or `mean`.
- `ordinal`: `most_frequent` uses the mode; `median_snap` takes the median and
  snaps it to the nearest observed valid level.
- `onehot_group`: fixed at `atomic_mode`; imputes the entire group together with
  the most frequent category in the training subsample rather than imputing each
  column independently.
- `model_overrides`: may set specified models to `passthrough`, which performs no
  imputation and passes `NaN` directly. Of the currently registered models, only
  `lightgbm` and `xgboost` support `NaN` natively, so only these two are allowed.
  An empty `{}` applies the strategies above to all models.

## 9. Complete Example and Validation

This example contains three sources: `age` (continuous), `sat` (ordinal, levels
1–5), and `edu` (three categories, one-hot). The raw data contains sentinel
missing-value codes and text categories.

**The required generation order is ARD → manifest → universe → schema**: the
universe depends on the manifest, and the schema depends on the universe hash.

```python
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe
from aleatoric_nk_grid.ingest import canonical_json

# Fixed vocabularies declared in advance, not derived from this dataset (Section 6)
EDU_LEVELS = ["high_school", "college", "graduate"]
SAT_LEVELS = [1, 2, 3, 4, 5]

# --- 1. Raw data -> ARD ---
raw = pd.read_csv("raw.csv")
raw["age"] = raw["age"].replace(-9, np.nan)        # Missing-value sentinel -> NaN

ard = pd.DataFrame({"y": raw["y"], "X_age": raw["age"], "O_sat": raw["sat"]})
missing_edu = raw["edu"].isna()                    # Missing category -> entire group is NaN
for i, level in enumerate(EDU_LEVELS):
    ard[f"C_edu__{i}"] = np.where(
        missing_edu, np.nan, (raw["edu"] == level).astype(float)
    )
ard.to_csv("data.csv", index=False)

predictors = ["X_age", "O_sat"] + [f"C_edu__{i}" for i in range(len(EDU_LEVELS))]

# --- 2. Manifest: one row per modeling column ---
rows = [
    dict(source_column="age", feature_name="X_age", keep=True, source_order=0,
         feature_order=0, unit_type="continuous", drop_first=False,
         is_reference=False, reference_level=None, level_value=None,
         ordinal_levels=None, source_prior=0.0),
    dict(source_column="sat", feature_name="O_sat", keep=True, source_order=1,
         feature_order=0, unit_type="ordinal", drop_first=False,
         is_reference=False, reference_level=None, level_value=None,
         ordinal_levels=json.dumps(SAT_LEVELS, separators=(",", ":")),
         source_prior=3),
]
for i, level in enumerate(EDU_LEVELS):
    rows.append(dict(
        source_column="edu", feature_name=f"C_edu__{i}", keep=True,
        source_order=2, feature_order=i, unit_type="onehot_group",
        drop_first=False, is_reference=(i == 0),
        reference_level=EDU_LEVELS[0], level_value=level,
        ordinal_levels=None, source_prior=None,
    ))
pd.DataFrame(rows).to_csv("feature_manifest.csv", index=False)

# --- 3. Universe: read the written CSV so it exactly matches what the engine sees ---
manifest = pd.read_csv("feature_manifest.csv")
groups = source_groups(predictors, manifest, {})
Path("universe.json").write_text(canonical_json(
    canonical_feature_universe(predictors, groups, manifest)))

# --- 4. Schema: note feature_manifest_version and definition_sha256 ---
schema = {
    "schema_version": 1, "feature_manifest_version": 1,
    "dataset": "typed", "table": "data.csv", "test_table": None,
    "split_mode": "internal_random", "task": "regression",
    "outcome_columns": ["y"], "id_column": None,
    "predictor_columns": predictors, "predictor_prefix": None,
    "feature_manifest": "feature_manifest.csv", "exchangeable": True,
    "feature_universe": {
        "mode": "fixed_a_priori", "definition_file": "universe.json",
        "definition_sha256": hashlib.sha256(
            Path("universe.json").read_bytes()).hexdigest(),
    },
    "group_column": None,
    "imputation": {
        "continuous": "median", "ordinal": "median_snap",
        "onehot_group": "atomic_mode", "model_overrides": {},
    },
    "max_train_outcome_missing_ratio": 0.5,
    "max_test_outcome_missing_ratio": 0.5,
    "continuous_priors": None,
}
Path("typed.json").write_text(json.dumps(schema, indent=2))
```

If all predictors are mutually independent continuous columns, the manifest may
be omitted. [`README.md`](README.md) provides a complete template for that case.

### Validation

`models`, `min_n`, `test_size`, and `seed` are **validation parameters required
by the validation call** and are unrelated to adapter artifacts. Use any
registered model; it affects only the check of the training-row lower bound. Set
`min_n` and `test_size` to the values planned for the experiment. Under
`external_test`, `test_size` is ignored.

`validate_input` checks that data tables are readable; version numbers are
recognized; the provenance schema hash matches; exactly one predictor rule is
set and it does not overlap outcomes or the ID; predictors are finite numeric
values and are not entirely missing; outcome types are valid, missingness ratios
do not exceed their thresholds, and the row-count lower bound is met after
missing outcomes are removed; required manifest fields are present and
`keep=true` covers predictors exactly; ordinal values are valid levels; every
one-hot row has a valid state; `exchangeable` is true and `group_column` is null;
and both the feature-universe file hash and content match. Under `external_test`,
it additionally checks matching table structures, ID integrity, and class
coverage.

Degenerate cases within training subsamples—constant columns, a single class,
too few samples, or an entirely missing source—are not covered by this
validation. The engine skips them cell by cell and records diagnostics.

```python
from pathlib import Path
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.validate_input import validate_input

loaded = load_input(Path("typed.json"), "y")
_, groups = validate_input(loaded, "y", models=["ols"],
                           min_n=10, test_size=0.3, seed=1)
print([(g.name, g.unit_type, len(g.features)) for g in groups])
# [('age', 'continuous', 1), ('sat', 'ordinal', 1), ('edu', 'onehot_group', 3)]
```

### Pre-Delivery Checklist

- [ ] Repeated runs on identical input produce identical ARD, manifest, universe,
      and hash values.
- [ ] Raw special missing-value codes are deterministically converted to `NaN`.
- [ ] All predictors are numeric; missing values remain `NaN`; no imputation has
      been performed.
- [ ] Rows with missing outcomes have not been removed in advance.
- [ ] The predictor rule cannot match an outcome or ID column.
- [ ] The schema, manifest, universe, and actual predictors are synchronized
      exactly.
- [ ] Under `internal_random`, perturbing row-level data leaves the fixed universe
      and vocabularies unchanged.
- [ ] Under `external_test`, both tables have matching structures; IDs are
      non-missing and unique within each table and do not overlap across tables;
      changing test data leaves the training vocabulary unchanged.
- [ ] If provenance is provided, it contains only hashes and no raw IDs or
      absolute paths.
- [ ] `validate_input` raises no errors.

After the adapter artifacts pass validation, see
[`../NK_Grid/README.md`](../NK_Grid/README.md) for experiment configuration and
execution instructions.

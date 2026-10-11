# Writing an adapter for the NK grid engine

This document is the specification for turning a study's raw data into files the
engine accepts, audits and reproduces.

## 0. Contract summary

`validate_input` rejects the artifacts before any sampling unless all of these hold.

| # | Rule | Section |
|---|---|---|
| 1 | Data tables are `.csv`, `.parquet` or `.pq`. | 2 |
| 2 | Exactly one predictor rule is set, `predictor_columns` or `predictor_prefix`, and it matches no outcome or ID column. | 3 |
| 3 | Predictors are finite numbers or `NaN`; no column is entirely `NaN`; nothing is imputed. | 3 |
| 4 | Regression outcomes are finite numbers, classification outcomes are `{0,1}`, and enough rows remain after rows missing the outcome are removed. | 3 |
| 5 | `id_column` is set, with non-missing IDs that are unique within each table. | 4 |
| 6 | Under `external_test`, both tables have the same structure, their IDs do not overlap, and the training table covers the test classes. | 4 |
| 7 | Vocabularies come only from permitted rows: `internal_random` requires `fixed_a_priori`. | 5 |
| 8 | A manifest declares every ordinal feature, one-hot group, audit row and multi-group source. | 6 |
| 9 | `group_column` is `null` and `exchangeable` is `true`. | 7 |
| 10 | `schema_version` is `1`; `feature_manifest_version` is `1` with a manifest and `null` without one. | 7 |

## 1. What an adapter does

An adapter is a program, usually a Python script in the study's directory, that
reads a dataset's raw data and writes the files described below. Identical inputs
must produce identical files.

| Task | Owner | Notes |
|---|---|---|
| Row-wise transformations | Adapter | Depend only on the current row: yes/no to 1/0, log income, age from a reference date fixed in advance. |
| Missing-value codes | Adapter | Convert sentinels such as `-9`, `-99` and `N/A` to `NaN` with a reproducible mapping. |
| Categorical encoding and vocabularies | Adapter | Fixed once and unchanged afterwards, learned only from the rows Section 5 permits. |
| Feature-set declaration | Adapter | Section 5. |
| Train/test split | Engine | The adapter makes no random split of its own, but preserves a split the data provider defined. |
| Imputation and standardization | Engine | Refit within each training sample and cross-validation fold. |
| Sampling, models and metrics | Engine | |

The adapter must not impute. Statistics computed on the whole table would give a
small training sample a median estimated from the full sample, inflate its
performance and understate the effect of sample size; under `internal_random`
they would also carry test rows into training. Missing values therefore stay
`NaN` in everything the adapter writes.

| Term | Meaning |
|---|---|
| **ARD** | The analysis-ready data: the table of predictors, outcomes and row IDs the adapter writes. |
| **predictor** | A column of the ARD passed to the model. |
| **source** | The atomic sampling unit counted by K; one source has one or more predictor columns. |
| **feature** | A column-level record in the universe or manifest; a feature with `keep=true` is an ARD predictor. |

## 2. Artifacts, directories and paths

The schema is the single entry point: it references the ARD, feature universe and
manifest by path.

| Artifact | When required |
|---|---|
| Schema JSON | Always |
| ARD training table (`data.csv` or `data.parquet`) | Always |
| Feature-universe definition JSON | Always |
| ARD test table (`test.csv` or `test.parquet`) | Under `external_test` |
| `feature_manifest.csv` | When Section 6 requires it |
| `provenance.json` | Optional |

```text
YourArticle/
├── adapter/            adapter code
├── schema/             tracked in Git: <dataset>.json + feature-universe JSON
└── data/               not committed to Git
    ├── private/        raw input
    └── ard/<dataset>/  data.csv, test.csv, feature_manifest.csv, provenance.json
```

The schema and feature universe are under version control; data are not. When a
cluster launch prepares the data (Section 9), the adapter writes everything under
the run's `prepared/` directory instead, and the tracked directories stay unchanged.

The schema fields `table`, `test_table`, `feature_manifest` and
`feature_universe.definition_file` are resolved relative to the schema's
directory; absolute paths are also accepted. With the schema in
`YourArticle/schema/`:

| Schema field | Value |
|---|---|
| `table` | `../data/ard/my_dataset/data.csv` |
| `test_table` | `../data/ard/my_dataset/test.csv` |
| `feature_manifest` | `../data/ard/my_dataset/feature_manifest.csv` |
| `feature_universe.definition_file` | `my_dataset.feature_universe.json` |

The bundled FFCWS and SMR adapters publish each build as a versioned bundle and
switch the schema to it only once it is complete, so read their paths from the
schema rather than assuming this flat layout.

`provenance.json` is optional audit metadata, found by location: it must sit next
to the training table and keep that name. The engine rejects a run whose
provenance file is not valid JSON but does not check any digest inside it. Do not
record raw IDs or absolute paths in it.

## 3. ARD requirements

An ARD is a flat table with one row per observational unit, containing the
outcomes declared in the schema (one or more), the predictors and the row-ID
column.

- Once the vocabulary is fixed, each row's output depends only on that row's raw
  values.
- All predictors are numeric: no text or complex values, no `±inf`, and no column
  that is entirely `NaN`.
- A missing category stays `NaN`; do not make "missing" a level. To use
  missingness as information, add a separately declared indicator predictor.
- Do not remove rows with missing outcomes; the engine does so with the
  thresholds in Section 7.
- Regression outcomes are finite numbers; classification outcomes are `{0,1}`.

After rows missing the outcome are removed, the training rows must satisfy
`n_train ≥ max(min_n, the cross-validation lower bound of the selected models)`;
under `internal_random` this applies to the training part after the `test_size`
split. For classification, each class must also have at least as many rows as the
number of cross-validation folds.

Predictors are selected in the schema by exactly one of `predictor_columns` (an
explicit list) and `predictor_prefix` (a list of prefixes). Prefer the explicit
list when the column set is stable: a prefix also picks up any later column that
shares it.

## 4. Internal and external splits

`split_mode` decides what the adapter delivers.

**`internal_random`**: one table, which the engine splits. The feature universe
must be `fixed_a_priori`, so no vocabulary may be learned from this table.

**`external_test`**: the data provider has already split the data, for example
for a competition. Preserve that split and deliver two tables with the same
predictors, outcomes and ID column, where the same raw category maps to the same
value or dummy column in both. Vocabularies and screening use only the training
table. A test category outside the training vocabulary is an error; never extend
the vocabulary from the test table.

### Row IDs

Every run saves each model's per-row predictions, so `id_column` is required under
both split modes, with IDs that are non-missing and unique within each table and,
under `external_test`, do not overlap between tables. The engine never samples or
splits by ID; it records the IDs behind each saved prediction.

Use a stable identifier from the data provider when there is one, as FFCWS does
with `challengeID`. Otherwise number the records by their position in the raw
table before any filtering, so a record keeps the same ID for every outcome, as
the SMR adapter does with `smr_row_id`.

## 5. Feature universe

The feature universe is a normalized snapshot of the feature space: the sources,
the features in each and their order, the category values and reference category
of each one-hot feature, and the level mapping of each ordinal feature. Generate
it with `canonical_feature_universe()` and serialize it with `canonical_json()`
(Section 8); a handwritten structure can drift from the validation logic.

When loading, the engine recomputes the universe from the actual predictors and
manifest and requires it to match the definition file. A run's frozen plan also
records the SHA-256 of the definition file together with the data tables and
manifest, and every worker refuses to start if any of them has changed.

`feature_universe.mode` is one of:

- **`fixed_a_priori`**: the feature set, vocabularies, order, ordinal mappings and
  reference categories are fixed in advance, independent of any row of this
  dataset. Selecting columns by names or prefixes declared in advance is allowed;
  screening by missingness, variance, category frequency or relationship with the
  outcome is not. Required by `internal_random`; a perturbation test must show
  that row-level data do not change it.
- **`train_pool_screened`**: screened on the provider's training set; only with
  `external_test`.

`exchangeable` declares that the sources can be treated as units drawn at random
from one population. If that does not hold, stop the integration rather than set
it to `true` to pass validation.

## 6. Feature manifest

Without a manifest (`feature_manifest.csv`), every predictor is an independent
continuous source. A manifest is required when there is:

1. an ordinal feature, even in one column; otherwise it is imputed as continuous
   and can take a nonexistent level such as 2.5;
2. a one-hot group, one source spanning several columns;
3. an audit row with `keep=false`, which exists only in the manifest and is not
   modeled;
4. a source with several preprocessing groups, such as a value column plus
   missingness indicators that must enter and leave the model together.

A `continuous` or `ordinal` source has exactly one `keep=true` column, so only a
`onehot_group` spans several columns.

| Column | Description |
|---|---|
| `source_column` | Preprocessing group; the same value for columns preprocessed together, such as one categorical variable's one-hot columns. |
| `sampling_source` | Optional K-axis sampling unit, defaulting to `source_column`. Give several preprocessing groups the same value when they must be selected together. Every `keep=true` cell is non-empty; a blank audit cell inherits from kept rows of its `source_column`. |
| `feature_name` | Column name in the ARD. |
| `keep` | `true` to model; `false` for audit only. |
| `source_order` | Order among preprocessing groups, from 0, identical within a `source_column`. |
| `feature_order` | Order within a source, from 0 and contiguous. |
| `unit_type` | `continuous`, `ordinal` or `onehot_group`. |
| `drop_first` | One-hot rows only, identical within a source; `False` otherwise. |
| `is_reference` | Marks the one reference column of a one-hot group with `drop_first=false`; `False` otherwise. |
| `reference_level` | Raw category of the one-hot reference; with `drop_first=true`, the omitted category. |
| `level_value` | Raw category of this dummy column. |
| `ordinal_levels` | Valid ordinal levels as compact JSON. |
| `source_prior` | Optional value used when a source is entirely missing from a training sample. |

Leave inapplicable cells empty.

`sampling_source` separates sampling from preprocessing: a raw variable's value
columns can form one `onehot_group` while each of its missingness indicators is a
one-column `continuous` group, and giving them all the variable's
`sampling_source` makes the engine count them once in K and select them together.

`source_prior` can be left empty. A source entirely missing from a training sample
still counts toward K and is set to one constant on both the training and test
side (`NaN` for passthrough models), so its value does not change any prediction.
Without a manifest, schema-level `continuous_priors` serves the same purpose.

### Ordinal levels

`ordinal_levels` is compact JSON (`[1,2,3]`, not `[1, 2, 3]`) of nonempty, unique,
finite numbers. Map text labels to integer codes in the adapter.

### Valid one-hot states

| | `drop_first=false` | `drop_first=true` |
|---|---|---|
| Manifest | Exactly one row has `is_reference=true`. | No reference row; `reference_level` names the omitted category. |
| Observed row | Exactly one 1. | At most one 1; all zeros is the reference category. |
| Missing row | The whole group is `NaN`. | The whole group is `NaN`. |

Values are 0 or 1 unless the whole group is `NaN`; a partially missing group is
invalid. `drop_first` is consistent within a source and `level_value` is not
repeated.

## 7. Schema

Every field below must be present, even as `null`, except those marked optional;
no other field is allowed.

| Field | Key points |
|---|---|
| `schema_version` | `1`. |
| `feature_manifest_version` | `1` with a manifest, `null` without one. |
| `dataset` | Nonempty string. |
| `table` / `test_table` | Section 2; `test_table` is `null` under `internal_random`. |
| `split_mode` | `internal_random` or `external_test`. |
| `task` | `regression` or `classification` (binary `{0,1}`). |
| `outcome_columns` | Nonempty list of unique names. |
| `id_column` | Section 4; not also an outcome. |
| `predictor_columns` / `predictor_prefix` | Lists of strings; exactly one is set (Section 3). |
| `feature_manifest` | Section 6. |
| `exchangeable` | `true`, with a research justification (Section 5). |
| `feature_universe` | Exactly `{mode, definition_file}`. |
| `group_column` | `null`; grouped splitting is not supported. |
| `imputation` | See below. |
| `max_train_outcome_missing_ratio` / `max_test_outcome_missing_ratio` | Optional, default `0.5`; document the reason for any other value. |
| `continuous_priors` | Optional `{source: value}` for continuous sources. |

### Imputation

The engine imputes, but the adapter declares how, because whether a method suits a
variable is a judgment about the data. All four keys are required even when no
feature of that type exists.

```json
{
  "continuous": "median",
  "ordinal": "most_frequent",
  "onehot_group": "atomic_mode",
  "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"}
}
```

- `continuous`: `median` or `mean`.
- `ordinal`: `most_frequent`, or `median_snap`, the median snapped to the nearest
  observed valid level.
- `onehot_group`: `atomic_mode`, which fills the whole group with the training
  sample's most frequent category.
- `model_overrides`: `passthrough` leaves `NaN` for the model; only `lightgbm` and
  `xgboost` handle `NaN` themselves, so only they may be listed. `{}` applies the
  strategies above to every model.

## 8. Example and validation

Three sources: `age` (continuous), `sat` (ordinal, levels 1–5) and `edu` (three
categories, one-hot). The raw data have sentinel missing codes and text
categories but no identifier, so the adapter numbers the records as `row_id`.
Build in the order ARD → manifest → universe → schema: the universe depends on the
manifest and the schema references the universe.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe
from aleatoric_nk_grid.ingest import canonical_json

# Fixed vocabularies declared in advance, not derived from this dataset (Section 5)
EDU_LEVELS = ["high_school", "college", "graduate"]
SAT_LEVELS = [1, 2, 3, 4, 5]

# --- 1. Raw data -> ARD ---
raw = pd.read_csv("raw.csv")
raw["age"] = raw["age"].replace(-9, np.nan)        # Missing-value sentinel -> NaN

ard = pd.DataFrame({
    "row_id": np.arange(1, len(raw) + 1),          # Row ID fixed before any filtering
    "y": raw["y"], "X_age": raw["age"], "O_sat": raw["sat"],
})
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

# --- 4. Schema: note feature_manifest_version ---
schema = {
    "schema_version": 1, "feature_manifest_version": 1,
    "dataset": "typed", "table": "data.csv", "test_table": None,
    "split_mode": "internal_random", "task": "regression",
    "outcome_columns": ["y"], "id_column": "row_id",
    "predictor_columns": predictors, "predictor_prefix": None,
    "feature_manifest": "feature_manifest.csv", "exchangeable": True,
    "feature_universe": {
        "mode": "fixed_a_priori", "definition_file": "universe.json",
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

When all predictors are independent continuous columns, omit the manifest: set
`feature_manifest` and `feature_manifest_version` to `null` and build the universe
from `source_groups(predictors, None, priors)` with `manifest=None`, where `priors`
is the schema's `continuous_priors` or `{}`.

### Validation

`validate_input` checks every rule in this document. Its `models`, `min_n`,
`test_size` and `seed` arguments only set the row-count lower bound; use the values
planned for the experiment (`test_size` is ignored under `external_test`) and pass
`require_id=True`, as every run does.

```python
from pathlib import Path
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.validate_input import validate_input

loaded = load_input(Path("typed.json"), "y")
_, groups = validate_input(loaded, "y", models=["ols"],
                           min_n=10, test_size=0.3, seed=1, require_id=True)
print([(g.name, g.unit_type, len(g.features)) for g in groups])
# [('age', 'continuous', 1), ('sat', 'ordinal', 1), ('edu', 'onehot_group', 3)]
```

Cases that arise only inside a training sample, such as a constant column, a
single class, too few rows or an entirely missing source, are not errors; the
engine skips or handles them cell by cell and records diagnostics.

### What validation cannot check

- [ ] Repeated runs on identical input produce identical ARD, manifest and universe.
- [ ] Nothing was imputed and no row was dropped for a missing outcome.
- [ ] Under `internal_random`, perturbing row-level data leaves the universe and
      vocabularies unchanged.
- [ ] Under `external_test`, changing the test data leaves the training vocabulary
      unchanged.
- [ ] `provenance.json`, if any, contains no raw IDs or absolute paths.

## 9. Registering panels and preparing runs

Register a validated schema as a panel: add an entry whose `schema` field points
to it in a panel manifest, as `FFCWS/panels.yaml` and `SMR/panels.yaml` do, and
launch it with `--manifest` and `--panel` as the [launch guide](../launch/README.md)
describes. A launch either uses existing artifacts through `--schema` or builds
them for that run with `--prepare --data-dir DIR`.
[`../NK_Grid/README.md`](../NK_Grid/README.md) explains how the engine samples N
and K and fits models.

### The `preparation` declaration

`--prepare` requires a top-level `preparation` mapping in the manifest. Paths are
relative to the manifest's directory unless stated otherwise.

| Key | Meaning |
|---|---|
| `adapter` | Adapter script, run with the compute environment's Python. |
| `inputs` | Name → raw file, relative to `--data-dir`. Absolute paths and `..` are rejected. |
| `schema` | Generated schema, relative to the run's `prepared/` directory. |
| `config` | YAML adapter configuration that the launcher rewrites for the run (FFCWS). |
| `contract` | Fixed feature contract passed to the adapter; used when there is no `config` (SMR). |
| `panel_pattern` | Optional regular expression over the panel name; its named groups become fields. A group named `outcome` must equal the panel's outcome. |
| `arguments` | Optional extra adapter arguments. |

`inputs`, `schema` and `arguments` may contain `{panel}`, `{outcome}` and the
named groups of `panel_pattern`, filled from the selected panel. FFCWS uses
`panel_pattern` to take the strategy from the panel name and passes it as
`--strategy {strategy}`.

### How the launcher calls the adapter

Preparation runs in the bootstrap allocation, after the shared environment is
ready. The launcher checks that every declared input exists, creates
`<run>/prepared/` and runs the adapter once:

- With `config`, it copies that YAML to `prepared/adapter.yaml`, sets its `paths`
  entries for each input name, `output_root` (`prepared/work`), `ard_root`
  (`prepared/ard`) and `schema_root` (`prepared/schema`), restricts `outcomes` to
  the panel's outcome and `strategies` to the `strategy` field, and passes
  `--config prepared/adapter.yaml`.
- Without `config`, it passes `--article-root` (the manifest's directory),
  `--contract`, `--source` (the input named `source`) and
  `--output-root prepared/`.
- It then appends the declared `arguments`, followed by
  `--validation-model MODEL...`, `--min-n`, `--test-size` and `--seed` from the
  selected panel, preset and `--models`. The adapter runs `validate_input` with
  these values before it publishes anything.

The adapter writes all of its outputs under the directories it is given. The
generated schema must lie inside `prepared/`; the launcher uses it in place of the
panel's `schema` entry and records its path and SHA-256 in `prepared-launch.json`.
Raw inputs and tracked files are only read.

# Writing an adapter for the NK Grid engine

*中文版:[`writing-an-adapter.zh-CN.md`](writing-an-adapter.zh-CN.md)*

This guide walks through connecting a new dataset to the shared
`aleatoric_nk_grid` engine. It is written for someone who has a dataset and a
research question, not for someone modifying the engine.

The normative contract is [`upstream-adapter-spec.md`](upstream-adapter-spec.md);
this document is the practical path through it. Where the two disagree, the
spec wins.

## What an adapter is

The engine knows nothing about your data. It receives one **schema** file and
derives everything else from it: where the table lives, which columns are
outcomes, which are predictors, how to split, how to impute. Your adapter's
only job is to produce that schema plus an analysis-ready dataset (ARD).

The division of labour is strict, and it exists because the engine's results
must be comparable across articles:

| Yours (adapter) | The engine's |
|---|---|
| Row-wise deterministic transforms | Anything with a fitted parameter |
| Category encoding, vocabularies | Imputation (median, mode, priors) |
| Feature-universe selection | Train/test splitting |
| Writing schema + provenance | N×K subsampling, models, metrics, checkpoints |

The test for where something belongs: **if it needs to look at a subsample, or
has a parameter estimated from data, it is the engine's.** Imputing a median is
the engine's job precisely because the median must be recomputed inside every
(N, K) cell — computing it once up front would leak information across cells.

## Before you write code: three decisions

**1. What is your sampling unit?** The engine samples K *sources*, not K
columns. If one real-world variable expands into several columns (a categorical
becoming five dummies), those columns are one source and must be drawn
together. If every column is its own variable, you have no manifest to write.

**2. Internal or external split?** `internal_random` means the engine splits
your table by a seed. `external_test` means you supply an official held-out test
table (as the Fragile Families Challenge does) and the engine never resplits.
External mode requires an `id_column`.

**3. Are all your predictors continuous?** If yes, you can skip the feature
manifest entirely. If you have ordinal or one-hot variables, the manifest is
mandatory — it is how the engine learns which columns move together and how each
type should be imputed.

The two reference adapters bracket the difficulty range:

- [`SMR/adapter/prepare_smr.py`](../SMR/adapter/prepare_smr.py) — 170 lines,
  internal split, prefix-selected columns, all continuous, no manifest. Start
  here.
- [`FFCWS/adapter/`](../FFCWS/adapter/) — external test set, explicit column
  list, three encoding strategies, typed manifests with one-hot groups and
  ordinal levels.

## Directory layout

Create one folder per article at the repository root:

```text
YourArticle/
├── adapter/            your preparation code (may be as messy as you like)
├── schema/             version-controlled: <dataset>.json + feature universe
├── data/               gitignored
│   ├── private/        raw input
│   └── ard/<dataset>/  data.csv, test.csv, feature_manifest.csv, provenance.json
├── outputs/            gitignored
├── panels.yaml         run controls
└── model_params.yaml   model hyperparameters
```

Schemas are version-controlled; data never is. The schema references the ARD by
a path relative to the schema directory.

## Step 1 — Project raw data into ARD

Read your raw source, keep only outcome and predictor columns, and write a CSV
or Parquet file. Everything here must be **row-wise deterministic**: the value
written for row *i* may depend only on row *i*'s raw values, never on other rows.

Leave missing values as `NaN`. Do not impute — that is the engine's job, per
cell. Predictors must end up numeric; encode categories as dummies or integer
codes, but leave a missing category as `NaN` rather than inventing a level.

```python
header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
predictors = [c for c in header if c.startswith(("Aset", "Bset"))]
projected = pd.read_csv(source, usecols=[*outcomes, *predictors])
projected.to_csv(ard_dir / "data.csv", index=False)
```

For external-test mode, write `test.csv` the same way, with identical columns
and encoding, plus the ID column on both sides.

## Step 2 — Write the feature manifest (only if you have typed features)

One row per expanded column. The columns the engine requires:

| Column | Meaning |
|---|---|
| `source_column` | The real-world variable this column came from |
| `feature_name` | The column name as it appears in the ARD |
| `keep` | `true` for modeled columns; `false` rows are audit-only |
| `source_order` | Integer, unique per source — fixes source ordering |
| `feature_order` | Integer, unique and contiguous within a source |
| `unit_type` | `continuous`, `ordinal`, or `onehot_group` |
| `drop_first` | Only for `onehot_group`; consistent within a source |
| `is_reference` | Marks the reference dummy when `drop_first` is false |
| `reference_level` | The original category that is the reference |
| `level_value` | The original category each dummy encodes |
| `ordinal_levels` | Ordered legal levels, as a canonical JSON array |
| `source_prior` | Optional placeholder used when a source is fully missing |

Consistency rules the engine enforces: one `unit_type` per source; exactly one
kept feature for `continuous`/`ordinal` sources; a one-hot group needs either
`drop_first=true` or exactly one `is_reference` row; `keep=true` rows must cover
the resolved predictors exactly, with no orphans.

Two details that reliably trip people up. `ordinal_levels` must be *canonical*
JSON — `[1,2,3]`, not `[1, 2, 3]` — because the string is hashed into the
feature universe. And ordinal levels must be finite numbers; if your levels are
labels, map them to integer codes in the adapter and keep the labels in
`level_value` or your own documentation.

You may add extra columns for your own auditing. FFCWS carries `prevalence`,
`strategy` and `mapping_id`; the engine ignores what it does not recognize.

## Step 3 — Generate the feature universe

This is a canonical description of your sources, features, category values and
ordering, hashed into the schema. It exists so a run can prove that the feature
space it resolved is the one you declared — especially under `internal_random`,
where nothing else prevents a data-dependent feature selection from silently
changing between runs.

Generate it with the engine's own function rather than hand-writing the
structure, so it cannot drift from what validation recomputes:

```python
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.validate_input import canonical_feature_universe
from aleatoric_nk_grid.ingest import canonical_json

groups = source_groups(predictors, manifest, continuous_priors={})
universe = canonical_feature_universe(predictors, groups, manifest)
definition_path.write_text(canonical_json(universe))
```

Then record its SHA-256 in the schema. At load time the engine recomputes the
universe from your actual manifest and predictors and compares; any mismatch
fails before sampling.

## Step 4 — Write the schema

The schema is the single semantic authority. Every field is required (the three
marked optional may be omitted, but nothing unknown is tolerated):

```json
{
  "schema_version": 1,
  "feature_manifest_version": null,
  "dataset": "my_dataset",
  "table": "../data/ard/my_dataset/data.csv",
  "test_table": null,
  "split_mode": "internal_random",
  "task": "regression",
  "outcome_columns": ["y1", "y2"],
  "id_column": null,
  "predictor_columns": null,
  "predictor_prefix": ["X_"],
  "feature_manifest": null,
  "exchangeable": true,
  "feature_universe": {
    "mode": "fixed_a_priori",
    "definition_file": "my_dataset.feature_universe.json",
    "definition_sha256": "…"
  },
  "group_column": null,
  "imputation": {
    "continuous": "median",
    "ordinal": "most_frequent",
    "onehot_group": "atomic_mode",
    "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"}
  },
  "max_train_outcome_missing_ratio": 0.5,
  "max_test_outcome_missing_ratio": 0.5,
  "continuous_priors": null
}
```

Notes on the fields that carry real consequences:

**`predictor_columns` vs `predictor_prefix`** — exactly one, never both. A
prefix rule is convenient but silently picks up any new column matching it, so
prefer an explicit list when your column set is stable. Neither may include an
outcome or the ID column; the engine rejects that overlap rather than letting a
target leak into the feature set.

**`feature_universe.mode`** — `fixed_a_priori` means the feature set was decided
without looking at outcomes; `train_pool_screened` means it was derived from the
official training pool only. Internal-split schemas *must* be `fixed_a_priori`,
because with a random split there is no pool that is safely "training only".

**`imputation.model_overrides`** — only `lightgbm` and `xgboost` may be set to
`passthrough`, because only they consume `NaN` natively.

**`task: classification`** accepts binary `{0,1}` outcomes only.

The schema is hashed into the experiment identity. Changing semantics (imputation
strategy, predictor set) changes the identity and starts a fresh checkpoint;
changing only a path does not.

## Step 5 — Write provenance

Alongside the ARD, record where the data came from and which schema produced it:

```json
{
  "adapter": "smr",
  "dataset": "asample2_withlag",
  "source_sha256": "…",
  "schema_sha256": "…",
  "feature_universe_sha256": "…",
  "ard_sha256": "…"
}
```

The engine cross-checks `schema_sha256` against the schema it actually loaded, so
a stale ARD paired with an edited schema is caught. Provenance is deliberately
*not* part of the experiment identity — regenerating it does not invalidate a
running experiment. Keep raw IDs and absolute paths out of it.

## Step 6 — Declare panels

`panels.yaml` holds run controls only. Any schema-owned field appearing here is
an error, which is what keeps the semantics in exactly one place.

```yaml
model_params: model_params.yaml
preset: dev

panels:
  - name: my_outcome_run
    schema: schema/my_dataset.json
    outcome: y1
    models: [ols, ridge, random_forest, xgboost]
    out: outputs/nk_grid_my_outcome.csv
```

Panels may set `preset`, `seed`, `n_seeds`, `n_draws`, `n_sizes_n`, `n_sizes_k`,
`min_n`, `max_n`, `max_k`, `batch_size`, `n_jobs`, `test_size` (internal split
only), the failure thresholds, and `allow_large_run` / `dry_run`. Presets are
`dev`, `medium`, `pilot_full`, `production`.

## Step 7 — Smoke test

```bash
aleatoric-nk-grid-panels --manifest YourArticle/panels.yaml --dry-run
```

This resolves every panel and prints the cell-count estimate without touching
data. Then run a `dev` preset for real and inspect the output CSV and its
`.manifest.json` sidecar. Check that `K_unobserved` looks plausible and that the
failed-row count is zero — a wall of `failed` rows with the same error message
usually means a contract problem the adapter should have prevented.

## When validation rejects your input

All of these fire before any model runs, by design — a silent failed-row table
is much worse than a crash.

| Message | Cause |
|---|---|
| `schema contains unknown fields` | Typo, or a panel-owned key put in the schema |
| `must define exactly one of predictor_columns or predictor_prefix` | Both set, or neither |
| `Resolved predictors overlap protected outcome/ID columns` | A prefix rule caught your outcome or ID |
| `predictor … must be finite numeric` | An object/string column reached the ARD |
| `predictor … contains ±inf` | Division or log applied to a sentinel value |
| `predictor … is entirely missing` | A column that is `NaN` everywhere |
| `regression outcome contains non-finite values` | `inf` in the target |
| `classification outcome must contain only binary {0,1}` | Multiclass target |
| `outcome missing ratio … exceeds …` | More than half the rows lack the outcome; raise the threshold deliberately or fix upstream |
| `Resolved feature universe does not match the canonical definition` | Manifest or predictor list changed without regenerating the universe file |
| `definition_sha256 does not match definition_file` | Universe file edited without updating the schema hash |
| `provenance schema_sha256 does not match` | Stale ARD against an edited schema |
| `Kept manifest rows must exactly cover resolved predictors` | Manifest and data drifted apart |
| `ordinal_levels … is not canonical JSON` | Spaces in the JSON array |
| `contains an invalid one-hot state` | A row with two 1s, or partially missing dummies |
| `External test source … contains category states absent from train` | Test has a category the model never saw |
| `id_column … contains duplicate IDs` | Non-unique identifiers |
| `Usable training rows … below required minimum` | Too few rows after outcome deletion for the selected models' CV |

## Checklist

- [ ] Sampling unit decided; manifest written if any source expands to several columns
- [ ] Predictors numeric, missing left as `NaN`, no imputation performed
- [ ] No outcome or ID column reachable by the predictor rule
- [ ] Feature universe generated with the engine's own function and hashed into the schema
- [ ] Schema complete, with exactly one predictor rule and a correct `feature_universe.mode`
- [ ] External mode: `test.csv` structurally identical, `id_column` set, no ID overlap
- [ ] Provenance written, containing hashes rather than raw identifiers
- [ ] `panels.yaml` contains run controls only
- [ ] `--dry-run` passes, then a `dev` preset produces a clean output with no failed rows

## Where to look next

- [`upstream-adapter-spec.md`](upstream-adapter-spec.md) — the normative
  contract, including the full imputation state machine and `K_unobserved`
  semantics.
- [`../NK_Grid/README.md`](../NK_Grid/README.md) — runtime, checkpointing and
  Slurm behavior.
- `NK_Grid/tests/conftest.py` — `write_schema_bundle()` builds a complete valid
  bundle in a few lines; the fastest way to see a minimal working example.

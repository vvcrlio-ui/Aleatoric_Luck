# Data-preparation quick start

An adapter is an application-specific program that converts source data into a
validated analysis-ready dataset and its machine-readable specifications. This
short example applies only to a single-table regression dataset containing
continuous predictor variables.

Before using the example, answer three substantive questions:

1. Are all variables used for prediction (predictors) truly continuous?  
   If `1/2/3` merely represent three education levels, the answer is “no.”
2. Does each predictor variable correspond to exactly one model input column?
   For example, if one education variable is expanded into multiple columns by
   one-hot encoding, the answer is “no.”
3. Does the dataset consist of a single table with an internally generated
   training and test split?

**If any answer is “no,” read [`ADAPTER.md`](ADAPTER.md).**

## Prerequisite

Run the following command from the repository root:

```bash
python -m pip install -e NK_Grid
```

`raw.csv` must satisfy all of the following assumptions:

1. It contains a regression outcome column named `y`.
2. Predictor columns begin with `X_` and contain only continuous numeric values.
3. Missing values can be read as `NaN` by `pandas.read_csv()`, such as empty CSV
   cells or `N/A`.

If the raw data uses custom sentinel values such as `-9` or `999` to represent
missingness, read [`ADAPTER.md`](ADAPTER.md).

## Complete example

Save the following code as `adapter.py` in the same directory as `raw.csv`. It
produces the analysis-ready data table, predictor-universe definition, and
schema in that order, then validates them immediately.

```python
import json
from pathlib import Path
import pandas as pd
from aleatoric_nk_grid.preprocessing import source_groups
from aleatoric_nk_grid.ingest import canonical_json, load_input
from aleatoric_nk_grid.validate_input import canonical_feature_universe, validate_input

OUTCOME = "y"
PREFIX = "X_"

# 1. Source data -> analysis-ready data. Keep missing values as NaN; imputation
# parameters are estimated separately within each selected training sample.
raw = pd.read_csv("raw.csv")
predictors = [c for c in raw.columns if c.startswith(PREFIX)]
raw[[OUTCOME, *predictors]].to_csv("data.csv", index=False)

# 2. Predictor universe: use the shared function to match validation logic.
groups = source_groups(predictors, None, {})
Path("universe.json").write_text(canonical_json(
    canonical_feature_universe(predictors, groups, None)))

# 3. Schema: paths are resolved relative to the directory containing schema.json.
schema = {
    "schema_version": 1,
    "feature_manifest_version": None,     # All-continuous data needs no manifest
    "dataset": "my_data",
    "table": "data.csv",
    "test_table": None,                   # Internal split: no separate test table
    "split_mode": "internal_random",
    "task": "regression",
    "outcome_columns": [OUTCOME],
    "id_column": None,
    "predictor_columns": None,            # Set exactly one of this and predictor_prefix
    "predictor_prefix": [PREFIX],
    "feature_manifest": None,
    "exchangeable": True,                 # Assumes the declared predictor variables can be
                                            # sampled on equal terms; justify this assumption
    "feature_universe": {
        "mode": "fixed_a_priori",         # Fixed in advance, with no row-level screening
        "definition_file": "universe.json",
    },
    "group_column": None,
    "imputation": {
        "continuous": "median",
        "ordinal": "most_frequent",       # All four keys are required, even with no ordinal data
        "onehot_group": "atomic_mode",
        "model_overrides": {},
    },
    "max_train_outcome_missing_ratio": 0.5,
    "max_test_outcome_missing_ratio": 0.5,
    "continuous_priors": None,
}
Path("schema.json").write_text(json.dumps(schema, indent=2))

# 4. Validation: these settings leave the generated files unchanged.
loaded = load_input(Path("schema.json"), OUTCOME)
_, resolved = validate_input(loaded, OUTCOME, models=["ols"],
                             min_n=10, test_size=0.3, seed=1)
print(f"OK: {len(resolved)} predictors, all treated as independent continuous variables")
```

## Run

```bash
python adapter.py
```

On success, it prints:

```text
OK: <your predictor count> predictors, all treated as independent continuous variables
```

The current directory should now contain `data.csv`, `universe.json`, and
`schema.json`. Validation raises an error when their structure or
interpretation falls outside the common analysis requirements.

## When You Must Read the Full Guide

Data containing ordinal variables, one-hot representations, predictor variables
expanded into multiple model input columns, or predefined training and test
samples require the “Internal / External,” “Feature Universe,” and “Feature
Manifest” sections of [`ADAPTER.md`](ADAPTER.md).

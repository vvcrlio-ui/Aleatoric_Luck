# How adapters prepare data

Adapters convert dataset-specific inputs into analysis tables that the shared engine can read. They also describe each column and identify columns that must be sampled together. FFCWS and SMR have separate adapters but use the same experiment logic during training.

## Define variables before modeling

Preparation includes identifying missing-value codes, converting values using explicit rules, encoding categorical variables, and excluding identifiers and outcomes from predictors. When one original variable expands into several columns, the adapter preserves their relationship.

For example, all one-hot columns for an occupation variable enter the model together. Selecting this source increases K by one. Missingness indicators for that variable can be bound to the same source.

## Information used during preparation

FFCWS uses the official training pool to screen variables and determine categories, then applies those definitions to the test data.

SMR uses a predefined feature list and category groups. The engine performs the train/test split.

See the [FFCWS](../FFCWS/adapter/README.md) and [SMR](../SMR/adapter/README.md) descriptions for dataset-specific details.

## Why imputation happens during training

Imputing with the median of the entire training pool before sampling a small N would give the model information from a larger sample. For internal train/test splits, imputing the full table could also introduce test information into training.

Adapters therefore preserve missing values as NaN. After sampling N rows and K sources, the engine learns imputation parameters from that training sample. During cross-validation, each training fold learns its own parameters. Standardization follows the same rule.

Rows with missing outcomes are initially retained. The engine checks and filters missingness for the selected outcome, preserving the original missingness rate for that check.

## Passing data to the engine

The analysis table contains numeric values, NaN, outcomes, and any required IDs. The feature manifest specifies column types and groups. The schema links these files and declares the task type and split method. Training starts from this schema.

Once data and feature definitions pass validation, the adapter saves a new version and updates the schema. Existing runs continue to use their original version.

See the [root quick start](../README.md) for execution steps.

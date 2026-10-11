# How adapters prepare data

Adapters convert dataset-specific inputs into analysis tables that the shared engine can read. They also describe each column and identify columns that must be sampled together. FFCWS and SMR have separate adapters but use the same experiment logic during training.

## Define variables before modeling

Preparation includes identifying missing-value codes, converting values using explicit rules, encoding categorical variables, and excluding identifiers and outcomes from predictors. When one original variable expands into several columns, the adapter preserves their relationship.

For example, all one-hot columns for an occupation variable enter the model together, and selecting the variable increases K by one. Missingness indicators for that variable can be attached to it so that they are selected with it. In the adapter files this unit is called a source.

## Information used during preparation

FFCWS uses the official training pool to screen variables and determine categories, then applies those definitions to the test data.

SMR uses a predefined feature list and category groups. The engine performs the train/test split.

See the [FFCWS](../FFCWS/adapter/README.md) and [SMR](../SMR/adapter/README.md) descriptions for dataset-specific details.

## Why imputation happens during training

Imputing with the median of the entire training pool before sampling a small N would give the model information from a larger sample. For internal train/test splits, imputing the full table could also introduce test information into training.

Adapters therefore preserve missing values as NaN. After sampling N rows and K variables, the engine learns imputation parameters from that training sample. During cross-validation, each training fold learns its own parameters. Standardization follows the same rule.

Rows with missing outcomes are initially retained. The engine checks and filters missingness for the selected outcome, preserving the original missingness rate for that check.

## Passing data to the engine

The analysis table contains numeric values, NaN, outcomes, and a row ID. The ID plays no part in sampling or splitting; it lets predictions from separately trained models be matched to the same person. The feature manifest specifies column types and groups. The schema links these files and declares the task type and split method. Training starts from this schema.

The adapter writes the analysis table and schema only after the data and feature definitions pass validation. When the launcher prepares the data, it runs the adapter inside the run and keeps the output in that run's own directory, so later changes to the data or the preparation rules do not affect a run already started.

[ADAPTER.md](ADAPTER.md) specifies the files, schema fields, and checks for writing a new adapter. See the [launch guide](../launch/README.md) for running an experiment.

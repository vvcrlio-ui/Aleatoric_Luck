# FFCWS analysis specifications

Run `python FFCWS/adapter/adapter.py` from the repository root to regenerate
this directory.

For each predictor representation, the adapter generates:

- one canonical definition of the predictor variables eligible for the
  \(K\)-dimension; and
- one machine-readable analysis schema for each of the six outcomes.

The schemas record the predefined training and test samples, variable
representations, missing-value treatment, and links to the generated
analysis-ready tables. Schemas and predictor-universe definitions are
versioned because they determine how those untracked tables are interpreted.
The source data and generated data tables remain excluded from version
control.

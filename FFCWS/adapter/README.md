# FFCWS adapter

The copied data processor remains article-owned under the collision-free import
name `ffcws_data_processor` and now emits v5 typed feature
manifests, ARD datasets, canonical feature-universe definitions, authoritative
engine schemas and provenance. Raw/private files remain under `FFCWS/data/` and
are ignored.

Category vocabularies are learned from the official training pool. After
outcome-missing rows are removed, any category state that is absent from that
outcome's training rows is atomically masked in test and recorded in
`outcome_category_coverage.csv`.

Numeric missing-code indicators from the legacy
`median_missing_indicator` strategy are retained as `keep=false` audit rows.
The modeled numeric source remains a single continuous feature so one raw
source has one legal v5 `unit_type` and one K unit.

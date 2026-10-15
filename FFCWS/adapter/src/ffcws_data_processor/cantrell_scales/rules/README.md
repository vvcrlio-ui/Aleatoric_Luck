# Files read by the Cantrell et al. preprocessing

All files come from `data/dataset/fragile-families/` of [predicting-life-outcomes-ffcws](https://github.com/emilycantrell/predicting-life-outcomes-ffcws) at commit 995c845 (Apache License 2.0).

- `Metadata_Edit_Tracker.csv`, `VariablesToRemove.json` and `parsable_scales_instructions.csv` are copied unchanged.
- `FFMetadata_v09_rules.csv` keeps the columns of `FFMetadata_v09.csv` that the predictor rules read: `new_name`, `old_name`, `type`, `warning`, `wave`, `n_cities_asked` and `subtopics`.
- `FFMetadata_v09_scale_input_labels.csv` lists the value labels in `FFMetadata_v09.csv` for every variable named in the scale instructions, one row per value. The unit conversions read them where the Challenge file has no label.

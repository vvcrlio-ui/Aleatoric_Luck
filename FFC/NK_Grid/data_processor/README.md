# FFC data processor

Build the three approved, un-imputed encoding strategies for NK Grid:

```bash
.venv/bin/python data_processor/scripts/build_ffc_strategies.py \
  --config data_processor/configs/ffc.yaml
```

Run one strategy with `--strategy median_mode`,
`median_missing_indicator`, or `tree_ordinal`. Generated data is written under
the gitignored `data/intermediate_files/preprocessing/` directory.

# FFC data processor

Build the three approved, un-imputed encoding strategies for NK Grid:

```bash
PYTHONPATH=adapter/data_processor/src python \
  adapter/data_processor/scripts/build_ffc_strategies.py \
  --config adapter/data_processor/configs/ffc.yaml
```

Run one strategy with `--strategy median_mode`,
`median_missing_indicator`, or `tree_ordinal`. The supplied configuration writes
intermediate QA artifacts to `data/adapter_work/`, engine ARD tables to
`data/ard/`, and authoritative contracts to `schema/`. Raw and derived tables
remain ignored; schemas and canonical universe definitions are versioned.

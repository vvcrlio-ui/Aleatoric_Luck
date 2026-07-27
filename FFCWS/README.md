# FFCWS adapter

This is the article-owned FFCWS integration for the shared root
[`NK_Grid/`](../NK_Grid/README.md) engine.

```text
FFCWS/
├── adapter/
│   ├── contracts/ffc.yaml
│   ├── prepare.py
│   ├── src/
│   └── tests/
├── schema/
├── data/
│   ├── private/
│   ├── adapter_work/
│   └── ard/
├── panels.yaml
├── model_params.yaml
└── requirements.txt
```

The adapter retains the `median_mode`, `median_missing_indicator`, and
`tree_ordinal` processing strategies. Their precise feature and missing-value
semantics are documented in [`adapter/README.md`](adapter/README.md).

From this directory:

```bash
python -m pip install -r requirements.txt
python adapter/prepare.py
aleatoric-nk-grid-panels --manifest panels.yaml --dry-run
```

Raw/private data, generated ARD tables and model outputs are ignored by Git.

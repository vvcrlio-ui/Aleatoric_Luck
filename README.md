# Aleatoric Luck

Predictability research infrastructure organized around one shared N×K engine
and article-owned upstream adapters.

```text
Aleatoric_Luck/
├── NK_Grid/   shared package (import: aleatoric_nk_grid)
├── SMR/       SMR adapter, schema, panels and model parameters
├── FFCWS/     FFCWS adapter, typed manifests, schemas and panels
├── FFC/       legacy FFC fork retained for rollback
└── docs/      adapter contract and implementation plan
```

The old `SMR/NK_Grid/` and `FFC/NK_Grid/` forks remain intact as rollback
baselines. New runs use only the root package and its locked Python 3.11–3.14
environment:

```bash
python -m pip install -e NK_Grid
aleatoric-nk-grid-panels --manifest SMR/panels.yaml --dry-run
```

## Boundary

- Article adapters own deterministic raw-to-ARD projection, vocabularies,
  feature-universe declarations, schemas and provenance.
- `aleatoric_nk_grid` owns validation, splitting, N/K sampling, per-cell typed
  preprocessing, models, metrics, experiment identity, checkpoints and failure
  governance.
- Schemas are the only semantic authority; panel files contain run controls
  only.

See [`docs/upstream-adapter-spec.md`](docs/upstream-adapter-spec.md) for the
contract and
[`docs/split-nk-grid-engine-adapter.md`](docs/split-nk-grid-engine-adapter.md)
for the migration design.

## Data

Raw and derived data, outputs and logs are ignored repository-wide. FFCWS
provenance contains hashes and aggregate metadata only; it does not contain raw
IDs or absolute source paths.

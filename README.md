# Aleatoric Luck

Predictability research infrastructure organized around one shared N×K engine
and article-owned upstream adapters.

```text
Aleatoric_Luck/
├── NK_Grid/                shared engine package (import: aleatoric_nk_grid)
│   ├── src/ tests/ slurm/  engine code, engine tests, cluster scripts
│   └── README.md           runtime, checkpoint and Slurm semantics
├── SMR/                    SMR adapter, schema, panels and model parameters
│   ├── NK_Grid/            legacy SMR fork (rollback baseline, frozen)
│   └── Zheng_Cheng_Replication/
├── FFCWS/                  FFCWS adapter, typed manifests, schemas and panels
├── FFC/                    legacy FFC fork + ffc_replication (rollback, frozen)
└── docs/                   adapter contract and implementation plan
```

The old `SMR/NK_Grid/` and `FFC/NK_Grid/` forks remain intact as rollback
baselines. New runs use only the root package and its locked Python 3.11–3.14
environment.

## Quick start

```bash
python -m pip install -e "NK_Grid[models,parquet,test]"
python -m pytest                                            # repo-wide suite
aleatoric-nk-grid-panels --manifest SMR/panels.yaml --dry-run
```

A dry run prints the resolved configuration and cell-count estimate for every
panel without touching data. Drop `--dry-run` to execute locally; presets
(`dev`, `medium`, `pilot_full`, `production`) are declared per panel in each
article's `panels.yaml`, and anything above the large-run threshold requires
`--allow-large-run`.

## Cluster runs

Slurm submission goes through the engine's submit script, which freezes a job
snapshot, splits the array into `parallel` / `serial` / `bart` resource
classes, and records a submission receipt:

```bash
NK_Grid/slurm/submit_nk_grid.sh --manifest SMR/panels.yaml --dry-run
NK_Grid/slurm/submit_nk_grid.sh --manifest FFCWS/panels.yaml --allow-large-run
```

Workers checkpoint every batch, stop at a checkpoint boundary on the Slurm
advance signal, and requeue themselves; completed tasks are reused instead of
recomputed unless `--rerun-completed` is passed. See
[`NK_Grid/README.md`](NK_Grid/README.md) for runtime, checkpoint-compaction and
recovery semantics.

## Boundary

- Article adapters own deterministic raw-to-ARD projection, vocabularies,
  feature-universe declarations, schemas and provenance.
- `aleatoric_nk_grid` owns validation, splitting, N/K sampling, per-cell typed
  preprocessing, models, metrics, experiment identity, checkpoints and failure
  governance.
- Schemas are the only semantic authority; panel files contain run controls
  only.

See [`docs/upstream-adapter-spec.md`](docs/upstream-adapter-spec.md) for the
contract (including the checklist for wiring a new article) and
[`docs/split-nk-grid-engine-adapter.md`](docs/split-nk-grid-engine-adapter.md)
for the migration design.

## Data

Raw and derived data, outputs and logs are ignored repository-wide. FFCWS
provenance contains hashes and aggregate metadata only; it does not contain raw
IDs or absolute source paths. Committed schema and feature-universe JSON files
carry codebook metadata (column names, category levels), never row-level data.

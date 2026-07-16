# NK Grid output files

Each run uses one filename stem and produces three related artifacts:

- `NAME.csv`: the materialized result table used for analysis and figures.
- `NAME.manifest.json`: the minimal provenance, configuration, completion, and QA record.
- `NAME.parts/part-*.csv`: immutable checkpoint shards used for crash-safe resume.

The final CSV preserves all established metric columns. Four cell-level diagnostic
columns are appended because they must remain filterable by model, seed, draw, N,
and K:

| Column | Meaning |
| --- | --- |
| `K_varying` | Number of selected training features with more than one observed value. |
| `constant_prediction` | `true` when there are fewer than two exactly distinct finite test predictions. This uses the same exact-uniqueness rule as the correlation metrics; an all-nonfinite prediction vector is also flagged. |
| `underdetermined` | `true` for regression OLS when `K_varying >= N` in that fitted cell. The row is retained. |
| `converged` | For fitted iterative estimators, `false` when `n_iter_` reaches `max_iter`; wrapper and SuperLearner components are checked recursively. Closed-form, tree, and boosting estimators without this contract default to `true`. Skipped and failed rows use `false`. |

Flags never delete rows automatically. Primary summaries should retain every
successful row. Any filtered sensitivity analysis must state its rule explicitly,
for example excluding `underdetermined=true` for an OLS robustness curve.

The manifest records the `algorithm_version`, Git commit and dirty state, relative
data paths and fingerprints, realized grids, resolved model parameters, core package
versions, output completeness, and compact diagnostic counts. `model_params.yaml`
is the reusable instruction file; the manifest is the immutable receipt of the
values that actually applied to one run.

`algorithm_version` is a manually maintained methodological version. Increment it
when a change affects the statistical meaning of results, including model libraries,
hyperparameters, CV folds, feature sampling, data splitting, or metric definitions.
Logging and checkpoint performance changes alone do not require a new version.

Checkpoint shards are authoritative while a run is incomplete. On resume, existing
shards are scanned and completed cells are skipped. On normal exit, shards are
deduplicated and atomically merged into the final CSV. Do not edit individual shard
files.

Large runs never prompt for interactive input. Above the safety threshold they must
be explicitly authorized:

```bash
python src/run_panels.py --allow-large-run
```

Preview declared panel sizes without fitting models:

```bash
python src/run_panels.py --dry-run
```

"""Build a dynamic-plan request JSON for the `production` preset.

`production` is declared in run_panels.PRESETS but not in
chunk_planning.DYNAMIC_PRESETS, so `--preset production` is rejected.  This
script writes the equivalent request payload directly, so the run needs no
engine change.  Both grids are derived from the panel's own schema and data
(never hardcoded), using the engine's own log2_size_grid.

Usage:
  python make_production_request.py MANIFEST PANEL ROOT ACCOUNT PARTITION \
      CONSTRAINT WORKERS ROUNDS TIME_LIMIT OUT_JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aleatoric_nk_grid import run_panels
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.nk_grid import log2_size_grid


def resolved_grids(config_json: dict, schema_path: Path, outcome: str) -> tuple[list[int], list[int]]:
    """Mirror the engine's own grid resolution for this panel."""

    loaded = load_input(schema_path, outcome)
    schema = loaded.schema
    if schema.split_mode != "external_test":
        raise SystemExit(
            f"this script only handles split_mode=external_test; schema says {schema.split_mode!r}"
        )
    # external_test_split drops rows whose outcome is missing; the surviving
    # training rows are what log2_size_grid is given inside the engine.
    n_train = int(loaded.train[outcome].notna().sum())

    definition = Path(str(schema.feature_universe["definition_file"]))
    if not definition.is_absolute():
        definition = (schema.path.parent / definition).resolve()
    n_units = len(json.loads(definition.read_text(encoding="utf-8"))["sources"])

    n_grid = [int(v) for v in log2_size_grid(
        n_train, config_json["n_sizes_n"], config_json["max_n"], min_size=config_json["min_n"],
    )]
    k_grid = [int(v) for v in log2_size_grid(
        n_units, config_json["n_sizes_k"], config_json["max_k"],
    )]
    print(
        f"[grid] train_rows={n_train} feature_units={n_units} "
        f"N={len(n_grid)} points K={len(k_grid)} points",
        file=sys.stderr,
    )
    return n_grid, k_grid


def main(argv: list[str]) -> None:
    if len(argv) != 10:
        raise SystemExit(__doc__)
    (manifest, panel, root, account, partition,
     constraint, workers, rounds, time_limit, out_json) = argv
    manifest_path = Path(manifest)
    root_path = Path(root)

    resolved = run_panels.resolved_panels(manifest_path, only={panel}, preset="production")
    if not resolved:
        raise SystemExit(f"panel {panel!r} not found in {manifest}")
    _, config = resolved[0]
    config_json = run_panels.config_to_json(config)
    config_json["out"] = str(root_path / "final.csv")
    config_json["n_jobs"] = 1
    # The production design is ~17.1M model cells against a 250,000 guard
    # (nk_grid.LARGE_RUN_THRESHOLD).  The guard fires only after the task table
    # has already been written, so leaving it unset wastes the whole planning
    # job before failing.  This script exists to authorize exactly that size.
    config_json["allow_large_run"] = True

    schema_path = Path(str(config_json["schema"]))
    if not schema_path.is_absolute():
        schema_path = (manifest_path.parent / schema_path).resolve()
    n_grid, k_grid = resolved_grids(config_json, schema_path, str(config_json["outcome"]))

    panel_family = run_panels.load_manifest(manifest_path).get("panel_family")
    if not isinstance(panel_family, str) or not panel_family:
        panel_family = manifest_path.parent.name.lower()

    payload = {
        "config": config_json,
        "n_grid": n_grid,
        "k_grid": k_grid,
        "cluster": {
            "workers": int(workers),
            "rounds": int(rounds),
            "partition": partition,
            "time_limit": time_limit,
            "account": account,
            "constraint": constraint,
        },
        "task_table": str(root_path / "tasks.parquet"),
        "snapshot": str(root_path / "snapshot.json"),
        "output_dir": str(root_path / "out"),
        "panel": panel_family,
    }
    Path(out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cells = len(n_grid) * len(k_grid) * config_json["n_seeds"] * config_json["n_draws"] * len(config_json["models"])
    print(f"[request] wrote {out_json}", file=sys.stderr)
    print(f"[request] N={n_grid}", file=sys.stderr)
    print(f"[request] K={k_grid}", file=sys.stderr)
    print(f"[request] cells={cells:,}  models={len(config_json['models'])} "
          f"seeds={config_json['n_seeds']} draws={config_json['n_draws']}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])

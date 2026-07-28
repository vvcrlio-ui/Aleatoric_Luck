"""Resolve article panel manifests without duplicating schema-owned semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .helpers_logging import log_progress
from .ingest import SCHEMA_FIELDS, load_schema
from .model_registry import DEFAULT_MODEL_PARAMS_PATH
from .nk_grid import NKGridConfig, estimate_run_size, run_nk_grid


ROOT = Path(__file__).resolve().parents[2]
PRESETS: dict[str, dict[str, int]] = {
    "dev": {
        "n_seeds": 3,
        "n_draws": 3,
        "n_sizes_n": 3,
        "n_sizes_k": 3,
        "min_n": 10,
        "max_n": 100,
        "max_k": 100,
    },
    "medium": {
        "n_seeds": 8,
        "n_draws": 8,
        "n_sizes_n": 10,
        "n_sizes_k": 10,
        "min_n": 10,
        "max_n": 100,
        "max_k": 100,
    },
    "timing_full": {
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 20,
        "n_sizes_k": 20,
        "min_n": 10,
        "max_n": 0,
        "max_k": 0,
        "batch_size": 20,
    },
    "production": {
        "n_seeds": 100,
        "n_draws": 50,
        "n_sizes_n": 20,
        "n_sizes_k": 20,
        "min_n": 10,
        "max_n": 0,
        "max_k": 0,
        # Keep the checkpoint/signal boundary small. Automatic WAL compaction
        # controls the physical part count without making slow models redo
        # 1,000 cells after a forced requeue.
        "batch_size": 20,
    },
}
DEFAULTS: dict[str, Any] = {
    "seed": 12345,
    "test_size": 0.3,
    "batch_size": 20,
    # Panel resolution must not depend on the submit host's environment.
    # Slurm workers replace this scheduler-only value from their allocation.
    "n_jobs": 4,
    "model_params": DEFAULT_MODEL_PARAMS_PATH,
    "failed_abs_threshold": 50,
    "failed_ratio_threshold": 0.05,
    "native_process_max_attempts": 2,
    "native_process_timeout_seconds": 21_600,
    "allow_large_run": False,
    "dry_run": False,
    "rerun_completed": True,
    "experiment_id": "nkgrid-dev-v1",
    "data_version": "dev-data-v1",
    "model_spec_version": "nkgrid-models-v1",
}
PANEL_FIELDS = frozenset(
    {
        "name",
        "schema",
        "preset",
        "models",
        "model_params",
        "seed",
        "n_seeds",
        "n_draws",
        "n_sizes_n",
        "n_sizes_k",
        "min_n",
        "max_n",
        "max_k",
        "batch_size",
        "n_jobs",
        "test_size",
        "allow_large_run",
        "dry_run",
        "rerun_completed",
        "failed_abs_threshold",
        "failed_ratio_threshold",
        "native_process_max_attempts",
        "native_process_timeout_seconds",
        "outcome",
        "out",
        "experiment_id",
        "data_version",
        "model_spec_version",
        "resume_group",
        "repeat_plan",
        "n_grid",
        "k_grid",
    }
)
CONFIG_FIELDS = set(NKGridConfig.__dataclass_fields__)


def _resolve_path(value: str | Path, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if (
        not isinstance(manifest, dict)
        or "panels" not in manifest
        or not isinstance(manifest["panels"], list)
    ):
        raise ValueError("Manifest must be a YAML object with a 'panels' list.")
    return manifest


def resolve_panel(panel: dict[str, Any], manifest_dir: Path) -> tuple[str, NKGridConfig]:
    if not isinstance(panel, dict):
        raise ValueError("Each panel must be a mapping")
    name = panel.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each panel requires a non-empty 'name'.")
    conflicts = sorted(set(panel) & SCHEMA_FIELDS)
    if conflicts:
        raise ValueError(
            f"Panel {name} contains schema-owned fields: {', '.join(conflicts)}"
        )
    unknown = sorted(set(panel) - PANEL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown panel keys for {name}: {', '.join(unknown)}")
    for required in ("schema", "outcome", "out", "models"):
        if required not in panel:
            raise ValueError(f"Panel {name} requires '{required}'.")
    preset_name = panel.get("preset", "dev")
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset for panel {name}: {preset_name}")
    schema_path = _resolve_path(panel["schema"], manifest_dir)
    schema = load_schema(schema_path)
    outcome = str(panel["outcome"])
    if outcome not in schema.outcome_columns:
        raise ValueError(
            f"Panel {name} outcome {outcome!r} is not declared by its schema"
        )
    if schema.split_mode == "external_test" and "test_size" in panel:
        raise ValueError(
            f"Panel {name} must not set test_size for external_test mode"
        )
    values = {
        **DEFAULTS,
        **PRESETS[preset_name],
        **{key: value for key, value in panel.items() if key != "name"},
    }
    values["schema"] = schema_path
    values["out"] = _resolve_path(values["out"], manifest_dir)
    values["model_params"] = _resolve_path(values["model_params"], manifest_dir)
    values["models"] = tuple(str(model) for model in values["models"])
    values["outcome"] = outcome
    values["preset"] = preset_name
    if "repeat_plan" in panel:
        if "n_seeds" in panel or "n_draws" in panel:
            raise ValueError(f"Panel {name} cannot combine repeat_plan with n_seeds/n_draws")
        pairs: list[tuple[int, int]] = []
        plan = panel["repeat_plan"]
        if not isinstance(plan, list):
            raise ValueError(f"Panel {name} repeat_plan must be a list")
        for block in plan:
            if not isinstance(block, dict) or set(block) != {"seeds", "draws"}:
                raise ValueError(f"Panel {name} repeat_plan blocks require seeds and draws")
            pairs.extend((seed, draw) for seed in block["seeds"] for draw in block["draws"])
        values["repeat_plan"] = tuple(pairs)
        values["n_seeds"] = values["n_draws"] = 1
    for grid_name in ("n_grid", "k_grid"):
        if grid_name in values and values[grid_name] is not None:
            if not isinstance(values[grid_name], list) or not values[grid_name]:
                raise ValueError(f"Panel {name} {grid_name} must be a non-empty list")
            values[grid_name] = tuple(int(value) for value in values[grid_name])
    extra = sorted(set(values) - CONFIG_FIELDS)
    if extra:
        raise ValueError(f"Panel {name} did not resolve cleanly: {extra}")
    return name, NKGridConfig(**values)


def resolved_panels(
    manifest_path: Path, only: set[str] | None = None
) -> list[tuple[str, NKGridConfig]]:
    manifest = load_manifest(manifest_path)
    allowed_root = {
        "panels", "model_params", "preset", "experiment_id", "data_version",
        "model_spec_version", "resume_group", "repeat_plan", "n_grid", "k_grid",
    }
    root_unknown = sorted(set(manifest) - allowed_root)
    if root_unknown:
        raise ValueError(f"Unknown panel-manifest root keys: {root_unknown}")
    shared = {
        key: manifest[key]
        for key in (
            "model_params", "preset", "experiment_id", "data_version",
            "model_spec_version", "resume_group", "repeat_plan", "n_grid", "k_grid",
        )
        if key in manifest
    }
    panels = [
        resolve_panel({**shared, **panel}, Path(manifest_path).parent)
        for panel in manifest["panels"]
        if only is None or panel.get("name") in only
    ]
    if only is not None:
        missing = sorted(only - {name for name, _ in panels})
        if missing:
            raise ValueError(f"Unknown panel(s): {', '.join(missing)}")
    return panels


def config_to_json(config: NKGridConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in CONFIG_FIELDS:
        value = getattr(config, field)
        if isinstance(value, Path):
            result[field] = str(value)
        elif isinstance(value, tuple):
            result[field] = list(value)
        else:
            result[field] = value
    return {key: result[key] for key in sorted(result)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run declared N×K grid panels.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    # ``default=None`` keeps an absent flag from overriding a panel that already
    # declares ``allow_large_run``; passing the flag still authorizes every panel.
    parser.add_argument("--allow-large-run", action="store_true", default=None)
    args = parser.parse_args(argv)
    panels = resolved_panels(
        args.manifest, only=set(args.only) if args.only else None
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "panels": [
                        {
                            "name": name,
                            "config": config_to_json(config),
                            "estimate": estimate_run_size(config),
                        }
                        for name, config in panels
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    for name, config in panels:
        log_progress(f"panel {name} starting out={config.out}")
        run_nk_grid(
            config,
            max_jobs=args.max_jobs,
            allow_large_run=args.allow_large_run,
        )
        log_progress(f"panel {name} finished out={config.out}")


if __name__ == "__main__":
    main()

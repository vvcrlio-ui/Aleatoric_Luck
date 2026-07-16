"""Run declared N x K grid panels with shared presets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .helpers_logging import log_progress
    from .nk_grid import NKGridConfig, run_nk_grid
except ImportError:
    from helpers_logging import log_progress
    from nk_grid import NKGridConfig, run_nk_grid


PRESETS: dict[str, dict[str, int]] = {
    "dev": {
        "n_seeds": 5,
        "n_draws": 5,
        "n_sizes_n": 8,
        "n_sizes_k": 8,
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
    "production": {
        "n_seeds": 100,
        "n_draws": 50,
        "n_sizes_n": 20,
        "n_sizes_k": 20,
        "min_n": 10,
        "max_n": 0,
        "max_k": 0,
    },
}

DEFAULTS: dict[str, Any] = {
    "seed": 12345,
    "test_size": 0.3,
    "batch_size": 20,
    "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    "group_split_col": None,
    "task": "regression",
    "bart_min_n": 10,
    "bart_min_k": 2,
    "model_params": ROOT / "model_params.yaml",
}

CONFIG_FIELDS = set(NKGridConfig.__dataclass_fields__)


def _resolve_path(value: str | Path, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        manifest = yaml.safe_load(handle)
    if (
        not isinstance(manifest, dict)
        or "panels" not in manifest
        or not isinstance(manifest["panels"], list)
    ):
        raise ValueError("Manifest must be a YAML object with a 'panels' list.")
    return manifest


def resolve_panel(panel: dict[str, Any], manifest_dir: Path) -> tuple[str, NKGridConfig]:
    name = panel.get("name")
    if not name:
        raise ValueError("Each panel requires a non-empty 'name'.")
    preset_name = panel.get("preset", "dev")
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset for panel {name}: {preset_name}")

    if "test" in panel and "test_data" in panel:
        raise ValueError(f"Panel {name} cannot define both test and test_data.")
    panel_values = dict(panel)
    if "test" in panel_values:
        panel_values["test_data"] = panel_values.pop("test")

    values = {**DEFAULTS, **PRESETS[preset_name], **panel_values}
    values.pop("name", None)
    values["preset"] = preset_name
    unknown = sorted(set(values) - CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown config keys for panel {name}: {', '.join(unknown)}")

    for required in ("data", "out", "dataset", "outcome", "models"):
        if required not in values:
            raise ValueError(f"Panel {name} requires '{required}'.")

    values["data"] = _resolve_path(values["data"], manifest_dir)
    if values.get("test_data") is not None:
        values["test_data"] = _resolve_path(values["test_data"], manifest_dir)
    values["model_params"] = _resolve_path(values["model_params"], manifest_dir)
    values["out"] = _resolve_path(values["out"], manifest_dir)
    values["models"] = tuple(values["models"])
    return str(name), NKGridConfig(**values)


def resolved_panels(manifest_path: Path, only: set[str] | None = None) -> list[tuple[str, NKGridConfig]]:
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    shared_values = {}
    if "model_params" in manifest:
        shared_values["model_params"] = manifest["model_params"]
    panels = [
        resolve_panel({**shared_values, **panel}, manifest_dir)
        for panel in manifest["panels"]
        if only is None or panel.get("name") in only
    ]
    if only is not None:
        found = {name for name, _ in panels}
        missing = sorted(only - found)
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
    parser = argparse.ArgumentParser(description="Run declared N x K grid panels.")
    parser.add_argument("--manifest", default=str(ROOT / "panels.yaml"))
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    panels = resolved_panels(
        manifest_path,
        only=set(args.only) if args.only else None,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "panels": [
                        {"name": name, "config": config_to_json(config)}
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
        run_nk_grid(config, max_jobs=args.max_jobs)
        log_progress(f"panel {name} finished out={config.out}")


if __name__ == "__main__":
    main()

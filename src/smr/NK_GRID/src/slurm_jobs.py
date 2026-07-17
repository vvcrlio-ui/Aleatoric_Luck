"""Expand configured panels into one Slurm array task per panel and model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    from .helpers_logging import log_progress
    from .nk_grid import LARGE_RUN_THRESHOLD, NKGridConfig, estimate_run_size, run_nk_grid
    from .run_panels import ROOT, config_to_json, resolved_panels
except ImportError:
    from helpers_logging import log_progress
    from nk_grid import LARGE_RUN_THRESHOLD, NKGridConfig, estimate_run_size, run_nk_grid
    from run_panels import ROOT, config_to_json, resolved_panels


@dataclass(frozen=True)
class SlurmJob:
    panel: str
    model: str
    config: NKGridConfig


def _model_output_path(path: Path, model: str) -> Path:
    return path.with_name(f"{path.stem}_{model}{path.suffix}")


def build_slurm_jobs(manifest_path: Path) -> list[SlurmJob]:
    """Return the active panel/model jobs declared by one manifest."""

    jobs: list[SlurmJob] = []
    for panel_name, config in resolved_panels(manifest_path):
        for model_name in config.models:
            jobs.append(
                SlurmJob(
                    panel=panel_name,
                    model=model_name,
                    config=replace(
                        config,
                        models=(model_name,),
                        out=_model_output_path(Path(config.out), model_name),
                    ),
                )
            )
    if not jobs:
        raise ValueError(f"No active panel/model jobs found in {manifest_path}")
    _require_unique_output_paths(jobs)
    return jobs


def _require_unique_output_paths(jobs: list[SlurmJob]) -> None:
    owners: dict[Path, list[str]] = {}
    for job in jobs:
        output_path = Path(job.config.out).resolve()
        owners.setdefault(output_path, []).append(f"{job.panel}/{job.model}")
    duplicates = {
        path: labels for path, labels in owners.items() if len(labels) > 1
    }
    if duplicates:
        examples = "; ".join(
            f"{path}: {', '.join(labels)}"
            for path, labels in list(duplicates.items())[:3]
        )
        raise ValueError(
            "Slurm panel/model jobs must have unique output paths; duplicate "
            f"targets found ({examples}). Use a distinct 'out' for each panel."
        )


def _config_from_json(payload: dict[str, Any]) -> NKGridConfig:
    values = dict(payload)
    for key in ("data", "out", "model_params", "test_data"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    for key in ("models", "predictor_prefix"):
        values[key] = tuple(values[key])
    return NKGridConfig(**values)


def write_job_snapshot(
    manifest_path: Path,
    snapshot_path: Path,
    *,
    allow_large_run: bool,
) -> list[SlurmJob]:
    """Freeze a resolved array mapping for workers that may start much later."""

    manifest_path = manifest_path.resolve()
    jobs = build_slurm_jobs(manifest_path)
    require_large_run_authorization(jobs, allow_large_run=allow_large_run)
    payload = {
        "format_version": 1,
        "source_manifest": str(manifest_path),
        "jobs": [
            {
                "panel": job.panel,
                "model": job.model,
                "config": config_to_json(job.config),
            }
            for job in jobs
        ],
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(snapshot_path)
    os.chmod(snapshot_path, 0o444)
    return jobs


def load_job_snapshot(snapshot_path: Path) -> list[SlurmJob]:
    """Load the immutable array mapping produced at submission time."""

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("jobs"), list):
        raise ValueError(f"Unsupported Slurm job snapshot: {snapshot_path}")
    jobs = [
        SlurmJob(
            panel=str(item["panel"]),
            model=str(item["model"]),
            config=_config_from_json(item["config"]),
        )
        for item in payload["jobs"]
    ]
    if not jobs:
        raise ValueError(f"No jobs found in Slurm job snapshot: {snapshot_path}")
    _require_unique_output_paths(jobs)
    return jobs


def job_summary(job: SlurmJob, index: int) -> dict:
    return {
        "index": index,
        "panel": job.panel,
        "model": job.model,
        "config": config_to_json(job.config),
        "estimate": estimate_run_size(job.config),
    }


def require_large_run_authorization(
    jobs: list[SlurmJob], *, allow_large_run: bool
) -> None:
    oversized = [
        (job.panel, job.model, estimate_run_size(job.config)["top_level_model_cells"])
        for job in jobs
        if estimate_run_size(job.config)["top_level_model_cells"]
        > LARGE_RUN_THRESHOLD
    ]
    if oversized and not allow_large_run:
        examples = ", ".join(
            f"{panel}/{model}={cells:,}" for panel, model, cells in oversized[:3]
        )
        raise ValueError(
            "Large Slurm submission requires --allow-large-run because one or more "
            f"array tasks exceed {LARGE_RUN_THRESHOLD:,} model cells ({examples})."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze, list, or run panel/model jobs for the NK Grid array."
    )
    parser.add_argument("command", choices=("count", "list", "snapshot", "run"))
    parser.add_argument("--manifest", default=str(ROOT / "panels.yaml"))
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else None
    if args.command == "snapshot":
        if snapshot_path is None:
            parser.error("snapshot requires --snapshot")
        jobs = write_job_snapshot(
            manifest_path,
            snapshot_path,
            allow_large_run=args.allow_large_run,
        )
        print(len(jobs))
        return

    jobs = (
        load_job_snapshot(snapshot_path)
        if snapshot_path is not None
        else build_slurm_jobs(manifest_path)
    )

    if args.command == "list":
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path) if snapshot_path is None else None,
                    "snapshot": str(snapshot_path) if snapshot_path is not None else None,
                    "job_count": len(jobs),
                    "jobs": [job_summary(job, index) for index, job in enumerate(jobs)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_large_run_authorization(jobs, allow_large_run=args.allow_large_run)
    if args.command == "count":
        print(len(jobs))
        return

    if args.index is None:
        parser.error("run requires --index")
    if not 0 <= args.index < len(jobs):
        parser.error(f"--index must be between 0 and {len(jobs) - 1}")
    job = jobs[args.index]
    log_progress(
        f"slurm job index={args.index} panel={job.panel} model={job.model} "
        f"out={job.config.out}"
    )
    run_nk_grid(job.config, allow_large_run=args.allow_large_run)


if __name__ == "__main__":
    main()

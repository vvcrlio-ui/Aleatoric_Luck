"""Expand configured panels into one Slurm array task per panel and model."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment import (
    MODEL_ENV_KEYS,
    SERIAL_OUTER_MODELS,
    manifest_path as result_manifest_path,
    utc_now,
    write_json_atomic,
)
from .helpers_logging import log_progress
from .nk_grid import (
    LARGE_RUN_THRESHOLD, NKGridConfig, estimate_run_size, run_nk_grid,
    group_repeat_pairs_by_seed, resolve_repeat_pairs,
)
from .run_panels import ROOT, config_to_json, resolved_panels


def chunk_master_indices(indices: tuple[int, ...], max_array_size: int) -> tuple[tuple[int, ...], ...]:
    """Partition master indices while each Slurm array uses local 0..N-1 IDs."""
    if max_array_size < 1:
        raise ValueError("max_array_size must be positive")
    return tuple(tuple(indices[start : start + max_array_size]) for start in range(0, len(indices), max_array_size))


def write_chunk_map(snapshot_path: Path, resource_class: str, chunk: tuple[int, ...], ordinal: int) -> Path:
    """Persist immutable local-array-index to master-job-index mapping."""
    if not chunk:
        raise ValueError("chunk must not be empty")
    path = snapshot_path.parent / f"{snapshot_path.stem}.{resource_class}.chunk-{ordinal:04d}.json"
    if resource_class not in RESOURCE_CLASSES:
        raise ValueError(f"Unknown Slurm resource class: {resource_class}")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in chunk):
        raise ValueError("chunk indices must be non-negative integers")
    if len(set(chunk)) != len(chunk):
        raise ValueError("chunk indices must not contain duplicates")
    write_json_atomic(path, {"format_version": 1, "snapshot": str(snapshot_path.resolve()), "resource_class": resource_class, "master_indices": list(chunk)})
    os.chmod(path, 0o444)
    return path


def load_chunk_map(
    path: Path,
    *,
    snapshot_path: Path | None = None,
    resource_class: str | None = None,
) -> tuple[int, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("master_indices")
    if payload.get("format_version") != 1 or not isinstance(values, list) or not values:
        raise ValueError(f"Invalid Slurm chunk map: {path}")
    if not isinstance(payload.get("snapshot"), str) or payload.get("resource_class") not in RESOURCE_CLASSES:
        raise ValueError(f"Invalid Slurm chunk map: {path}")
    if snapshot_path is not None and Path(payload["snapshot"]).resolve() != snapshot_path.resolve():
        raise ValueError("Slurm chunk map belongs to a different snapshot")
    if resource_class is not None and payload["resource_class"] != resource_class:
        raise ValueError("Slurm chunk map belongs to a different resource class")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"Invalid Slurm chunk map: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Invalid Slurm chunk map: {path}")
    return tuple(values)


@dataclass(frozen=True)
class SlurmJob:
    panel: str
    model: str
    seed: int
    draws: tuple[int, ...]
    config: NKGridConfig
    final_out: Path


RESOURCE_CLASSES = ("parallel", "serial", "super_learner")
SUBMISSION_RECEIPT_FORMAT_VERSION = 1
OPTIONAL_SLURM_ENV_KEYS = (
    "ALEATORIC_NK_GRID_SOURCE_FALLBACK",
    "PYTHON_MODULE",
    "PYTHONPATH",
    *MODEL_ENV_KEYS,
)
SLURM_ADVANCE_SIGNAL_SECONDS = 300
SLURM_REQUEUE_MARGIN_SECONDS = 60
MAX_REQUEUE_WATCHDOG_SECONDS = (
    SLURM_ADVANCE_SIGNAL_SECONDS - SLURM_REQUEUE_MARGIN_SECONDS
)


def resource_class_for_model(model: str) -> str:
    if model == "super_learner":
        return "super_learner"
    if model == "lightgbm":
        return "serial"
    return "parallel"


def resource_class_indices(
    jobs: list[SlurmJob], resource_class: str
) -> tuple[int, ...]:
    if resource_class not in RESOURCE_CLASSES:
        raise ValueError(f"Unknown Slurm resource class: {resource_class}")
    return resource_class_partition(jobs)[resource_class]


def resource_class_partition(
    jobs: list[SlurmJob],
) -> dict[str, tuple[int, ...]]:
    """Return a defensive, mutually exclusive partition of snapshot indices."""

    partition = {
        resource_class: tuple(
            index
            for index, job in enumerate(jobs)
            if resource_class_for_model(job.model) == resource_class
        )
        for resource_class in RESOURCE_CLASSES
    }
    flattened = [
        index
        for resource_class in RESOURCE_CLASSES
        for index in partition[resource_class]
    ]
    expected = set(range(len(jobs)))
    if len(flattened) != len(expected) or set(flattened) != expected:
        raise RuntimeError(
            "Slurm resource classes must partition all snapshot indices "
            "without duplicates"
        )
    return partition


def _model_output_path(path: Path, panel: str, model: str, seed: int) -> Path:
    return path.parent / ".seed-shards" / panel / model / f"seed-{seed}{path.suffix}"


def _model_final_output_path(path: Path, panel: str, model: str) -> Path:
    return path.parent / ".seed-final" / panel / f"{model}{path.suffix}"


def build_slurm_jobs(manifest_path: Path) -> list[SlurmJob]:
    """Return the active panel/model jobs declared by one manifest."""

    jobs: list[SlurmJob] = []
    for panel_name, config in resolved_panels(manifest_path):
        for model_name in config.models:
            for seed, draws in group_repeat_pairs_by_seed(resolve_repeat_pairs(config)).items():
                jobs.append(SlurmJob(
                    panel=panel_name,
                    model=model_name,
                    seed=seed,
                    draws=draws,
                    config=replace(
                        config, models=(model_name,),
                        out=_model_output_path(Path(config.out), panel_name, model_name, seed),
                    ),
                    final_out=_model_final_output_path(
                        Path(config.out), panel_name, model_name
                    ),
                ))
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
    final_owners: dict[Path, set[tuple[str, str]]] = {}
    for job in jobs:
        final_owners.setdefault(Path(job.final_out).resolve(), set()).add(
            (job.panel, job.model)
        )
    duplicate_finals = {
        path: owners for path, owners in final_owners.items() if len(owners) > 1
    }
    if duplicate_finals:
        raise ValueError(
            "Slurm per-model final outputs must be unique within a panel"
        )


def _config_from_json(payload: dict[str, Any]) -> NKGridConfig:
    values = dict(payload)
    for key in ("schema", "out", "model_params"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values["models"] = tuple(values["models"])
    if values.get("repeat_plan") is not None:
        values["repeat_plan"] = tuple(tuple(pair) for pair in values["repeat_plan"])
    for field in ("n_grid", "k_grid"):
        if values.get(field) is not None:
            values[field] = tuple(values[field])
    return NKGridConfig(**values)


def _with_rerun_policy(
    jobs: list[SlurmJob], rerun_completed: bool
) -> list[SlurmJob]:
    return [
        replace(
            job,
            config=replace(job.config, rerun_completed=bool(rerun_completed)),
        )
        for job in jobs
    ]


def write_job_snapshot(
    manifest_path: Path,
    snapshot_path: Path,
    *,
    allow_large_run: bool | None,
    rerun_completed: bool = False,
) -> list[SlurmJob]:
    """Freeze a resolved array mapping for workers that may start much later."""

    manifest_path = manifest_path.resolve()
    jobs = _with_rerun_policy(
        build_slurm_jobs(manifest_path), rerun_completed
    )
    require_large_run_authorization(jobs, allow_large_run=allow_large_run)
    payload = {
        "format_version": 1,
        "source_manifest": str(manifest_path),
        "panel_outputs": {
            panel_name: str(config.out)
            for panel_name, config in resolved_panels(manifest_path)
        },
        "execution_policy": {
            "rerun_completed": bool(rerun_completed),
        },
        "jobs": [
            {
                "panel": job.panel,
                "model": job.model,
                "seed": job.seed,
                "draws": list(job.draws),
                "config": config_to_json(job.config),
                "final_out": str(job.final_out),
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


def _read_job_snapshot(snapshot_path: Path) -> tuple[dict[str, Any], list[SlurmJob]]:
    """Read and validate one frozen snapshot with a single file read."""

    raw = snapshot_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("format_version") != 1 or not isinstance(payload.get("jobs"), list):
        raise ValueError(f"Unsupported Slurm job snapshot: {snapshot_path}")
    jobs = [
        SlurmJob(
            panel=str(item["panel"]),
            model=str(item["model"]),
            seed=int(item["seed"]),
            draws=tuple(int(draw) for draw in item["draws"]),
            config=_config_from_json(item["config"]),
            final_out=Path(item["final_out"]),
        )
        for item in payload["jobs"]
    ]
    if not jobs:
        raise ValueError(f"No jobs found in Slurm job snapshot: {snapshot_path}")
    _require_unique_output_paths(jobs)
    return payload, jobs


def load_job_snapshot(snapshot_path: Path) -> list[SlurmJob]:
    """Load the read-only array mapping produced at submission time."""

    _, jobs = _read_job_snapshot(snapshot_path)
    return jobs


def apply_worker_overrides(
    config: NKGridConfig,
    *,
    rerun_completed: bool,
    environ: Mapping[str, str] | None = None,
) -> NKGridConfig:
    """Apply scheduler-only values without changing experiment identity."""

    environment = os.environ if environ is None else environ
    raw_cpus = environment.get("SLURM_CPUS_PER_TASK")
    n_jobs = config.n_jobs
    if raw_cpus is not None:
        try:
            n_jobs = int(raw_cpus)
        except ValueError as exc:
            raise ValueError(
                "SLURM_CPUS_PER_TASK must be a positive integer"
            ) from exc
        if n_jobs < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
    return replace(
        config,
        n_jobs=n_jobs,
        rerun_completed=bool(rerun_completed),
    )


def _array_parts(array_spec: str) -> tuple[str, int | None]:
    base_spec = array_spec
    raw_limit = None
    if "%" in array_spec:
        base_spec, raw_limit = array_spec.rsplit("%", 1)
    if not base_spec:
        raise ValueError("Slurm array specification must not be empty")
    if raw_limit is None:
        return base_spec, None
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(
            "Slurm array concurrency suffix must be a positive integer"
        ) from exc
    if limit < 1:
        raise ValueError(
            "Slurm array concurrency suffix must be a positive integer"
        )
    return base_spec, limit


def _expand_array_indices(array_spec: str, job_count: int) -> tuple[int, ...]:
    """Expand the subset of Slurm array syntax emitted by the submitter."""

    if job_count < 1:
        raise ValueError("job_count must be positive")
    base_spec, _ = _array_parts(array_spec)
    indices: list[int] = []
    for token in base_spec.split(","):
        if not token:
            raise ValueError("Slurm array specification contains an empty token")
        step = 1
        range_token = token
        if ":" in token:
            range_token, raw_step = token.rsplit(":", 1)
            try:
                step = int(raw_step)
            except ValueError as exc:
                raise ValueError("Slurm array step must be a positive integer") from exc
            if step < 1:
                raise ValueError("Slurm array step must be a positive integer")
        if "-" in range_token:
            raw_start, raw_end = range_token.split("-", 1)
            try:
                start, end = int(raw_start), int(raw_end)
            except ValueError as exc:
                raise ValueError("Slurm array range must contain integers") from exc
            if start > end:
                raise ValueError("Slurm array range start must not exceed its end")
            indices.extend(range(start, end + 1, step))
        else:
            if step != 1:
                raise ValueError("Slurm array step requires a range")
            try:
                indices.append(int(range_token))
            except ValueError as exc:
                raise ValueError("Slurm array index must be an integer") from exc
    if len(indices) != len(set(indices)):
        raise ValueError("Slurm array specification contains duplicate indices")
    if any(index < 0 or index >= job_count for index in indices):
        raise ValueError(
            f"Slurm array indices must be between 0 and {job_count - 1}"
        )
    return tuple(indices)


def write_submission_receipt(
    snapshot_path: Path,
    *,
    slurm_job_id: str,
    array_spec: str,
    worker_script: Path,
    allow_large_run: bool,
    rerun_completed: bool = False,
    cpus_per_task: int = 8,
    memory: str = "48G",
    time_limit: str = "4-00:00:00",
    max_restarts: int = 5,
    requeue_watchdog_seconds: int = 240,
    resource_class: str | None = None,
    chunk_map: Path | None = None,
    dependency_job_id: str | None = None,
    receipt_path: Path | None = None,
) -> Path:
    """Persist an unambiguous Slurm job-ID to snapshot mapping."""

    if not slurm_job_id.isdigit():
        raise ValueError(f"Invalid Slurm job ID: {slurm_job_id!r}")
    if not array_spec.strip():
        raise ValueError("Slurm array specification must not be empty")
    _, max_concurrent = _array_parts(array_spec)
    if cpus_per_task < 1:
        raise ValueError("cpus_per_task must be positive")
    if not memory.strip():
        raise ValueError("memory must not be empty")
    if not time_limit.strip():
        raise ValueError("time_limit must not be empty")
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")
    if not 0 <= requeue_watchdog_seconds <= MAX_REQUEUE_WATCHDOG_SECONDS:
        raise ValueError(
            "requeue_watchdog_seconds must be between 0 and "
            f"{MAX_REQUEUE_WATCHDOG_SECONDS}"
        )
    snapshot_path = snapshot_path.resolve()
    worker_script = worker_script.resolve()
    engine_dir = Path(
        os.environ.get("ENGINE_DIR", worker_script.parent.parent)
    ).resolve()
    venv = Path(os.environ.get("VENV", engine_dir.parent / ".venv")).resolve()
    python = Path(os.environ.get("PYTHON", venv / "bin" / "python")).resolve()
    environment_controls = {
        key: os.environ.get(key) for key in OPTIONAL_SLURM_ENV_KEYS
    }
    snapshot_payload, jobs = _read_job_snapshot(snapshot_path)
    if chunk_map is None:
        selected_indices = _expand_array_indices(array_spec, len(jobs))
    else:
        selected_indices = load_chunk_map(chunk_map, snapshot_path=snapshot_path, resource_class=resource_class)
        local_indices = _expand_array_indices(array_spec, len(selected_indices))
        if local_indices != tuple(range(len(selected_indices))):
            raise ValueError("Chunk receipt array must cover all local chunk indices")
    if resource_class is not None:
        if resource_class not in RESOURCE_CLASSES:
            raise ValueError(f"Unknown Slurm resource class: {resource_class}")
        wrong_class = [
            index
            for index in selected_indices
            if resource_class_for_model(jobs[index].model) != resource_class
        ]
        if wrong_class:
            raise ValueError(
                f"Array indices do not belong to resource class "
                f"{resource_class!r}: {wrong_class[:3]}"
            )
    if receipt_path is None:
        receipt_path = snapshot_path.parent / f"slurm-{slurm_job_id}.json"
    else:
        receipt_path = receipt_path.resolve()
    if receipt_path.exists():
        raise FileExistsError(f"Slurm submission receipt already exists: {receipt_path}")

    job_name = (
        f"al-nk-grid-{resource_class}"
        if resource_class is not None
        else "al-nk-grid"
    )
    log_output = engine_dir / "logs" / "%x-%A_%a.out"
    log_error = engine_dir / "logs" / "%x-%A_%a.err"

    def rerun_command(task_index: str) -> str:
        environment_prefix = ["env"]
        for key, value in environment_controls.items():
            if value is None:
                environment_prefix.extend(["-u", key])
            else:
                environment_prefix.append(f"{key}={value}")
        environment_prefix.extend(
            [
                f"ENGINE_DIR={engine_dir}",
                f"VENV={venv}",
                f"PYTHON={python}",
                f"MAX_SLURM_RESTARTS={max_restarts}",
                f"REQUEUE_WATCHDOG_SECONDS={requeue_watchdog_seconds}",
            ]
        )
        return shlex.join(
            [
                *environment_prefix,
                "sbatch",
                "--requeue",
                f"--signal=B:USR1@{SLURM_ADVANCE_SIGNAL_SECONDS}",
                "--open-mode=append",
                f"--job-name={job_name}",
                f"--output={log_output}",
                f"--error={log_error}",
                f"--chdir={engine_dir}",
                "--export=ALL",
                f"--cpus-per-task={cpus_per_task}",
                f"--mem={memory}",
                f"--time={time_limit}",
                f"--array={task_index}",
                str(worker_script),
                str(snapshot_path),
                "1" if allow_large_run else "0",
                "1" if rerun_completed else "0",
                *( [str(chunk_map)] if chunk_map is not None else [] ),
            ]
        )

    payload = {
        "format_version": SUBMISSION_RECEIPT_FORMAT_VERSION,
        "created_at": utc_now(),
        "slurm_job_id": slurm_job_id,
        "array": array_spec,
        "job_count": len(selected_indices),
        "snapshot_job_count": len(jobs),
        "snapshot": str(snapshot_path),
        "source_manifest": snapshot_payload.get("source_manifest"),
        "worker_script": str(worker_script),
        "execution_paths": {
            "engine_dir": str(engine_dir),
            "venv": str(venv),
            "python": str(python),
        },
        "scheduler": {
            "job_name": job_name,
            "output": str(log_output),
            "error": str(log_error),
            "requeue": True,
            "open_mode": "append",
            "signal": f"B:USR1@{SLURM_ADVANCE_SIGNAL_SECONDS}",
        },
        # Large-run authorization has two independent sources. Record the CLI
        # level under an unambiguous name so a panel-authorized task is not
        # audited as "unauthorized"; the effective value is per job below.
        "cli_allow_large_run": bool(allow_large_run),
        "execution_policy": {
            "rerun_completed": bool(rerun_completed),
            "max_restarts": int(max_restarts),
            "max_concurrent_per_class": max_concurrent,
            "advance_signal_seconds": SLURM_ADVANCE_SIGNAL_SECONDS,
            "requeue_watchdog_seconds": int(requeue_watchdog_seconds),
        },
        "submission_environment": environment_controls,
        "resources": {
            "class": resource_class,
            "cpus_per_task": int(cpus_per_task),
            "memory": memory,
            "time_limit": time_limit,
        },
        "chunk": {
            "map": str(chunk_map.resolve()) if chunk_map is not None else None,
            "master_indices": list(selected_indices),
            "dependency_job_id": dependency_job_id,
        },
        "rerun_command_template": rerun_command("<task-index>"),
        "jobs": [
            {
                "index": index,
                "panel": job.panel,
                "model": job.model,
                "configured_out": str(Path(job.config.out).resolve()),
                "panel_allow_large_run": bool(job.config.allow_large_run),
                "effective_allow_large_run": bool(
                    allow_large_run or job.config.allow_large_run
                ),
                "rerun_command": rerun_command(
                    str(local_index if chunk_map is not None else index)
                ),
            }
            for local_index, index in enumerate(selected_indices)
            for job in (jobs[index],)
        ],
    }
    write_json_atomic(receipt_path, payload)
    os.chmod(receipt_path, 0o444)
    return receipt_path


def dependency_never_satisfied_diagnostics(
    snapshot_path: Path,
    *,
    scheduler_output: str | None = None,
    cleanup: bool = False,
) -> dict[str, Any]:
    """Inspect only receipt-owned jobs for DependencyNeverSatisfied."""

    snapshot_path = snapshot_path.resolve()
    receipt_job_ids: set[str] = set()
    for receipt_path in snapshot_path.parent.glob("slurm-*.json"):
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if receipt_path.exists():
                log_progress(
                    f"warning: could not read existing Slurm submission receipt "
                    f"{receipt_path}: {type(exc).__name__}: {exc}"
                )
            continue
        job_id = payload.get("slurm_job_id")
        if (
            payload.get("format_version") == SUBMISSION_RECEIPT_FORMAT_VERSION
            and isinstance(payload.get("snapshot"), str)
            and Path(payload["snapshot"]).resolve() == snapshot_path
            and isinstance(job_id, str)
            and job_id.isdigit()
        ):
            receipt_job_ids.add(job_id)

    ordered_ids = sorted(receipt_job_ids, key=int)
    if scheduler_output is None and ordered_ids:
        completed = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(ordered_ids),
                "--format",
                "%A|%r|%T",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        scheduler_output = completed.stdout
    scheduler_output = scheduler_output or ""
    never_satisfied: set[str] = set()
    for line in scheduler_output.splitlines():
        fields = line.strip().split("|", 2)
        if len(fields) != 3:
            continue
        job_id, reason, _ = fields
        if (
            job_id in receipt_job_ids
            and reason == "DependencyNeverSatisfied"
        ):
            never_satisfied.add(job_id)
    ordered_never_satisfied = sorted(never_satisfied, key=int)
    cleaned: list[str] = []
    if cleanup and ordered_never_satisfied:
        subprocess.run(
            ["scancel", *ordered_never_satisfied],
            check=True,
        )
        cleaned = ordered_never_satisfied
    return {
        "snapshot": str(snapshot_path),
        "receipt_job_ids": ordered_ids,
        "dependency_never_satisfied_job_ids": ordered_never_satisfied,
        "cleaned_job_ids": cleaned,
    }


def job_summary(job: SlurmJob, index: int) -> dict:
    return {
        "index": index,
        "panel": job.panel,
        "model": job.model,
        "seed": job.seed,
        "draws": list(job.draws),
        "config": config_to_json(job.config),
        "estimate": estimate_run_size(job.config),
    }


def require_large_run_authorization(
    jobs: list[SlurmJob], *, allow_large_run: bool | None
) -> None:
    """Reject oversized tasks that neither the CLI nor their panel authorized."""

    oversized = []
    for job in jobs:
        if allow_large_run or job.config.allow_large_run:
            continue
        cells = estimate_run_size(job.config)["top_level_model_cells"]
        if cells > LARGE_RUN_THRESHOLD:
            oversized.append((job.panel, job.model, cells))
    if oversized:
        examples = ", ".join(
            f"{panel}/{model}={cells:,}" for panel, model, cells in oversized[:3]
        )
        raise ValueError(
            "Large Slurm submission requires --allow-large-run because one or more "
            f"array tasks exceed {LARGE_RUN_THRESHOLD:,} model cells ({examples})."
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze, list, or run panel/model jobs for the NK Grid array."
    )
    parser.add_argument(
        "command",
        choices=(
            "chunk-map",
            "count",
            "dependency-diagnostics",
            "indices",
            "list",
            "snapshot",
            "receipt",
            "run",
        ),
    )
    parser.add_argument("--manifest", default=str(ROOT / "panels.yaml"))
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--array-spec", default=None)
    parser.add_argument("--worker-script", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--chunk-map", type=Path, default=None)
    # ``default=None`` keeps an absent flag from overriding a panel that already
    # declares ``allow_large_run``; passing the flag still authorizes every task.
    parser.add_argument("--allow-large-run", action="store_true", default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--memory", default="48G")
    parser.add_argument("--time-limit", default="4-00:00:00")
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--requeue-watchdog-seconds", type=int, default=240)
    parser.add_argument("--resource-class", choices=RESOURCE_CLASSES, default=None)
    parser.add_argument("--chunk-map-path", type=Path, default=None)
    parser.add_argument("--dependency-job-id", default=None)
    parser.add_argument("--max-array-size", type=int, default=None)
    parser.add_argument("--chunk-ordinal", type=int, default=None)
    parser.add_argument("--master-indices", default=None)
    parser.add_argument(
        "--cleanup-never-satisfied",
        action="store_true",
    )
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
            rerun_completed=args.rerun_completed,
        )
        print(len(jobs))
        return

    if args.command == "receipt":
        if snapshot_path is None:
            parser.error("receipt requires --snapshot")
        if args.job_id is None:
            parser.error("receipt requires --job-id")
        if args.array_spec is None:
            parser.error("receipt requires --array-spec")
        if args.worker_script is None:
            parser.error("receipt requires --worker-script")
        receipt = write_submission_receipt(
            snapshot_path,
            slurm_job_id=args.job_id,
            array_spec=args.array_spec,
            worker_script=Path(args.worker_script),
            # The receipt records the CLI-level authorization handed to workers;
            # panel-level authorization travels inside the frozen snapshot.
            allow_large_run=bool(args.allow_large_run),
            rerun_completed=args.rerun_completed,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            time_limit=args.time_limit,
            max_restarts=args.max_restarts,
            requeue_watchdog_seconds=args.requeue_watchdog_seconds,
            resource_class=args.resource_class,
            chunk_map=args.chunk_map_path,
            dependency_job_id=args.dependency_job_id,
            receipt_path=Path(args.receipt) if args.receipt else None,
        )
        print(receipt)
        return

    if args.command == "dependency-diagnostics":
        if snapshot_path is None:
            parser.error("dependency-diagnostics requires --snapshot")
        print(
            json.dumps(
                dependency_never_satisfied_diagnostics(
                    snapshot_path,
                    cleanup=args.cleanup_never_satisfied,
                ),
                sort_keys=True,
            )
        )
        return

    jobs = (
        load_job_snapshot(snapshot_path)
        if snapshot_path is not None
        else _with_rerun_policy(
            build_slurm_jobs(manifest_path), args.rerun_completed
        )
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

    if args.command == "indices":
        if args.resource_class is None:
            parser.error("indices requires --resource-class")
        print(
            ",".join(
                str(index)
                for index in resource_class_indices(jobs, args.resource_class)
            )
        )
        return

    if args.command == "chunk-map":
        if snapshot_path is None:
            parser.error("chunk-map requires --snapshot")
        if args.resource_class is None:
            parser.error("chunk-map requires --resource-class")
        if args.max_array_size is None or args.max_array_size < 1:
            parser.error("chunk-map requires a positive --max-array-size")
        jobs = load_job_snapshot(snapshot_path)
        class_indices = resource_class_indices(jobs, args.resource_class)
        selected_indices = class_indices
        if args.master_indices is not None:
            try:
                selected_indices = tuple(
                    int(value) for value in args.master_indices.split(",")
                    if value
                )
            except ValueError as exc:
                parser.error("--master-indices must be comma-separated integers")
            if (
                len(selected_indices) != len(set(selected_indices))
                or any(index not in class_indices for index in selected_indices)
            ):
                parser.error(
                    "--master-indices must be unique members of the "
                    "declared resource class"
                )
        chunks = chunk_master_indices(selected_indices, args.max_array_size)
        if args.chunk_ordinal is None:
            print(json.dumps([str(write_chunk_map(snapshot_path, args.resource_class, chunk, ordinal)) for ordinal, chunk in enumerate(chunks)]))
            return
        if not 0 <= args.chunk_ordinal < len(chunks):
            parser.error("--chunk-ordinal is outside the generated chunk range")
        print(write_chunk_map(snapshot_path, args.resource_class, chunks[args.chunk_ordinal], args.chunk_ordinal))
        return

    require_large_run_authorization(jobs, allow_large_run=args.allow_large_run)
    if args.command == "count":
        print(len(jobs))
        return

    if args.index is None:
        parser.error("run requires --index")
    if args.chunk_map is not None:
        local_indices = load_chunk_map(args.chunk_map, snapshot_path=snapshot_path)
        if not 0 <= args.index < len(local_indices):
            parser.error(f"--index must be between 0 and {len(local_indices) - 1} for chunk map")
        args.index = local_indices[args.index]
    if not 0 <= args.index < len(jobs):
        parser.error(f"--index must be between 0 and {len(jobs) - 1}")
    job = jobs[args.index]
    runtime_config = apply_worker_overrides(
        job.config,
        rerun_completed=args.rerun_completed,
    )
    stop_file_value = os.environ.get("NK_GRID_STOP_REQUEST_FILE")
    stop_file = Path(stop_file_value) if stop_file_value else None
    log_progress(
        f"slurm job snapshot={snapshot_path} "
        f"array_job_id={os.environ.get('SLURM_ARRAY_JOB_ID', 'unknown')} "
        f"index={args.index} panel={job.panel} model={job.model} "
        f"n_jobs={runtime_config.n_jobs} "
        f"rerun_completed={runtime_config.rerun_completed} "
        f"out={runtime_config.out}"
    )
    result = run_nk_grid(
        runtime_config,
        allow_large_run=args.allow_large_run,
        stop_after_batch=(
            (lambda: stop_file.exists()) if stop_file is not None else None
        ),
        defer_materialization_on_stop=stop_file is not None,
        execution_pairs=tuple((job.seed, draw) for draw in job.draws),
        defer_failure_policy=True,
        exact_output_path=True,
    )
    if (
        stop_file is not None
        and stop_file.exists()
        and isinstance(result, Path)
    ):
        result_manifest = json.loads(
            result_manifest_path(result).read_text(encoding="utf-8")
        )
        if result_manifest.get("completion", {}).get("status") == "complete":
            stop_file.unlink()
            log_progress(
                "stop request arrived after completion; requeue is unnecessary"
            )


if __name__ == "__main__":
    main()

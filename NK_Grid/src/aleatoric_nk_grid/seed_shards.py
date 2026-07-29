"""Validation, diagnosis, and publication of deterministic seed shards."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment import manifest_path, output_run_lock, utc_now, write_json_atomic
from .nk_grid import group_repeat_pairs_by_seed, resolve_repeat_pairs
from .slurm_jobs import _config_from_json


KEY_COLUMNS = ("model", "seed", "draw", "N", "K")
VALID_STATUSES = frozenset({"ok", "skipped", "failed"})
_IDENTITY_FIELDS = ("experiment_id", "data_version", "model_spec_version")
_LEGACY_DIGEST_FIELDS = frozenset(
    {
        "file_" + "sha256",
        "semantic_" + "sha256",
        "snapshot_" + "sha256",
        "worker_script_" + "sha256",
    }
)


class SeedShardIncompleteError(ValueError):
    """A missing/incomplete artifact can be recreated by a worker/finalizer."""


class SeedShardValidationError(ValueError):
    """An artifact violates the explicit identity or complete-design contract."""


def _difference(left: Any, right: Any, path: str) -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return (
                    f"{path}.{key}: left={left.get(key)!r} "
                    f"right={right.get(key)!r}"
                )
            difference = _difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif left != right:
        return f"{path}: left={left!r} right={right!r}"
    return None


def _legacy_digest_field(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            current = f"{path}.{key}" if path else str(key)
            if key in _LEGACY_DIGEST_FIELDS:
                return current
            found = _legacy_digest_field(nested, current)
            if found:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _legacy_digest_field(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeedShardValidationError(f"Cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SeedShardValidationError(f"{label} must be a JSON object: {path}")
    legacy = _legacy_digest_field(payload)
    if legacy:
        raise SeedShardValidationError(
            f"Legacy digest field is not accepted: {legacy}"
        )
    return payload


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    snapshot = _load_json(Path(snapshot_path), "master snapshot")
    if (
        snapshot.get("format_version") != 1
        or not isinstance(snapshot.get("jobs"), list)
        or not snapshot["jobs"]
    ):
        raise SeedShardValidationError("Unsupported or empty master snapshot")
    return snapshot


def _target_jobs(
    snapshot: Mapping[str, Any], panel: str, model: str
) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (index, item)
        for index, item in enumerate(snapshot["jobs"])
        if isinstance(item, dict)
        and item.get("panel") == panel
        and item.get("model") == model
    ]
    if not selected:
        raise SeedShardValidationError(f"No seed jobs for {panel}/{model}")
    return selected


def _validate_master_target(
    snapshot: Mapping[str, Any], panel: str, model: str
) -> tuple[Any, Path, dict[int, dict[str, Any]]]:
    selected = _target_jobs(snapshot, panel, model)
    try:
        config = _config_from_json(selected[0][1]["config"])
        expected = group_repeat_pairs_by_seed(resolve_repeat_pairs(config))
        declared_seeds = [int(item["seed"]) for _, item in selected]
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedShardValidationError(
            f"Invalid master config for {panel}/{model}"
        ) from exc
    duplicates = sorted(
        seed for seed, count in Counter(declared_seeds).items() if count > 1
    )
    if duplicates:
        raise SeedShardValidationError(
            f"Duplicate master seed jobs: {duplicates[:20]}"
        )

    final_outputs: set[Path] = set()
    shard_outputs: set[Path] = set()
    by_seed: dict[int, dict[str, Any]] = {}
    for _, item in selected:
        try:
            seed = int(item["seed"])
            draws = tuple(int(draw) for draw in item["draws"])
            shard_out = Path(item["config"]["out"]).resolve()
            final_out = Path(item["final_out"]).resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedShardValidationError(
                "Master seed job lacks explicit seed/draw/output fields"
            ) from exc
        if shard_out in shard_outputs:
            raise SeedShardValidationError(
                f"Duplicate shard output path: {shard_out}"
            )
        shard_outputs.add(shard_out)
        final_outputs.add(final_out)
        if seed not in expected:
            raise SeedShardValidationError(f"Unexpected master seed: {seed}")
        if draws != tuple(expected[seed]):
            raise SeedShardValidationError(
                f"seed={seed}: draws do not match complete repeat plan"
            )
        by_seed[seed] = item
    missing = sorted(set(expected) - set(by_seed))
    if missing:
        raise SeedShardValidationError(
            f"Missing master seed jobs: {missing[:20]}"
        )
    if len(final_outputs) != 1:
        raise SeedShardValidationError(
            "Master seed jobs disagree on per-model final_out"
        )
    return config, next(iter(final_outputs)), by_seed


def _integer(value: Any, field: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SeedShardValidationError(
            f"{field} must be a finite integer"
        ) from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise SeedShardValidationError(f"{field} must be a finite integer")
    return int(parsed)


def _resolved_grids(manifest: Mapping[str, Any], label: str) -> tuple[list[int], list[int]]:
    design = manifest.get("design")
    if not isinstance(design, Mapping):
        raise SeedShardValidationError(f"{label}: design is missing")
    try:
        n_grid = [_integer(value, "N") for value in design["n_grid"]]
        k_grid = [_integer(value, "K") for value in design["k_grid"]]
    except (KeyError, TypeError) as exc:
        raise SeedShardValidationError(
            f"{label}: resolved N/K grid is invalid"
        ) from exc
    if not n_grid or not k_grid:
        raise SeedShardValidationError(
            f"{label}: resolved N/K grid must not be empty"
        )
    return n_grid, k_grid


def _expected_keys(
    config: Any,
    models: Sequence[str],
    grids: tuple[Sequence[int], Sequence[int]],
) -> set[tuple[str, int, int, int, int]]:
    return {
        (model, int(seed), int(draw), int(n_value), int(k_value))
        for model in models
        for seed, draw in resolve_repeat_pairs(config)
        for n_value in grids[0]
        for k_value in grids[1]
    }


def _validate_identity(identity: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(identity, Mapping) or identity.get("mode") != "explicit-v1":
        raise SeedShardValidationError(f"{label}: identity is not explicit-v1")
    for field in _IDENTITY_FIELDS:
        if field not in identity:
            raise SeedShardValidationError(f"{label}: identity.{field} missing")
    return identity


def _validate_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    draws: tuple[int, ...],
    identity: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    grids: tuple[list[int], list[int]] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[list[int], list[int]]]:
    label = f"seed={seed}"
    current_identity = _validate_identity(manifest.get("identity"), label)
    current_contract = manifest.get("semantic_contract")
    if not isinstance(current_contract, Mapping):
        raise SeedShardValidationError(f"{label}: semantic contract missing")
    current_grids = _resolved_grids(manifest, label)
    if identity is not None:
        for field in _IDENTITY_FIELDS:
            if current_identity[field] != identity[field]:
                raise SeedShardValidationError(
                    f"{label}: identity.{field} differs"
                )
    if contract is not None and (
        difference := _difference(
            contract, current_contract, "semantic_contract"
        )
    ):
        raise SeedShardValidationError(f"{label}: {difference}")
    if grids is not None and current_grids != grids:
        raise SeedShardValidationError(
            f"{label}: resolved N/K grid differs"
        )

    execution = manifest.get("execution")
    completion = manifest.get("completion")
    if not isinstance(execution, Mapping) or not isinstance(completion, Mapping):
        raise SeedShardValidationError(
            f"{label}: execution/completion is missing"
        )
    expected_rows = len(draws) * len(current_grids[0]) * len(current_grids[1])
    if (
        execution.get("mode") != "seed-shard"
        or execution.get("seed") != seed
        or execution.get("draws") != list(draws)
    ):
        raise SeedShardValidationError(
            f"{label}: execution declaration does not match shard"
        )
    if (
        execution.get("expected_rows") != expected_rows
        or completion.get("expected_rows") != expected_rows
    ):
        raise SeedShardValidationError(
            f"{label}: expected row count differs"
        )
    if (
        completion.get("materialized_rows") != expected_rows
        or completion.get("status")
        not in {"complete", "complete_with_failures"}
    ):
        raise SeedShardIncompleteError(f"{label}: shard is incomplete")
    return current_identity, current_contract, current_grids


def _read_csv_header(path: Path, label: str) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            header = csv.DictReader(source).fieldnames
    except FileNotFoundError as exc:
        raise SeedShardIncompleteError(f"{label}: output is missing") from exc
    if not header or any(column not in header for column in (*KEY_COLUMNS, "status")):
        raise SeedShardValidationError(
            f"{label}: CSV key/status columns are missing"
        )
    return header


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row.get("model")),
        _integer(row.get("seed"), "seed"),
        _integer(row.get("draw"), "draw"),
        _integer(row.get("N"), "N"),
        _integer(row.get("K"), "K"),
    )


def _read_artifact_keys(
    path: Path,
    *,
    expected_keys: set[tuple[str, int, int, int, int]],
) -> tuple[set[tuple[str, int, int, int, int]], Counter[str], list[str]]:
    header = _read_csv_header(path, str(path))
    keys: set[tuple[str, int, int, int, int]] = set()
    statuses: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = _row_key(row)
            status = str(row.get("status"))
            if status not in VALID_STATUSES:
                raise SeedShardValidationError(
                    f"{path}: invalid status {status!r}"
                )
            if key not in expected_keys:
                raise SeedShardValidationError(
                    f"{path}: out-of-design cell key {key}"
                )
            if key in keys:
                raise SeedShardValidationError(
                    f"{path}: duplicate cell key {key}"
                )
            keys.add(key)
            statuses[status] += 1
    return keys, statuses, header


def _publish_marker(target: Path) -> Path:
    return target.with_name(f".{target.name}.publishing")


def _artifact_provenance(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    rows: int,
    **identity: Any,
) -> dict[str, Any]:
    updated_at = manifest.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise SeedShardValidationError(
            f"{manifest_path(path)}: updated_at is missing"
        )
    return {
        **identity,
        "path": str(path.resolve()),
        "updated_at": updated_at,
        "rows": int(rows),
    }


def _existing_artifact_equivalent(
    target: Path,
    *,
    expected_keys: set[tuple[str, int, int, int, int]],
    identity: Mapping[str, Any],
    contract: Mapping[str, Any],
    grids: tuple[list[int], list[int]],
    execution_mode: str,
    input_artifacts: Sequence[Mapping[str, Any]],
) -> bool:
    target_manifest = manifest_path(target)
    if (
        _publish_marker(target).exists()
        or not target.exists()
        or not target_manifest.exists()
    ):
        return False
    manifest = _load_json(target_manifest, "existing final manifest")
    current_identity = _validate_identity(
        manifest.get("identity"), str(target_manifest)
    )
    for field in _IDENTITY_FIELDS:
        if current_identity[field] != identity[field]:
            raise SeedShardValidationError(
                f"{target_manifest}: identity.{field} differs"
            )
    if difference := _difference(
        manifest.get("semantic_contract"),
        contract,
        "semantic_contract",
    ):
        raise SeedShardValidationError(
            f"{target_manifest}: {difference}"
        )
    if _resolved_grids(manifest, str(target_manifest)) != grids:
        raise SeedShardValidationError(
            f"{target_manifest}: resolved N/K grid differs"
        )
    if manifest.get("input_artifacts") != list(input_artifacts):
        return False
    execution = manifest.get("execution")
    completion = manifest.get("completion")
    if (
        not isinstance(execution, Mapping)
        or execution.get("mode") != execution_mode
        or not isinstance(completion, Mapping)
        or completion.get("materialized_rows") != len(expected_keys)
        or completion.get("status")
        not in {"complete", "complete_with_failures"}
    ):
        return False
    keys, _, _ = _read_artifact_keys(
        target, expected_keys=expected_keys
    )
    return keys == expected_keys


def _status_summary(
    statuses: Counter[str],
    *,
    failed_abs_threshold: int,
    failed_ratio_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failed = int(statuses["failed"])
    ok = int(statuses["ok"])
    skipped = int(statuses["skipped"])
    denominator = ok + failed
    ratio = float(failed / denominator) if denominator else None
    violation: str | None
    if denominator == 0:
        violation = "failure-policy denominator is zero (no ok/failed cells)"
    elif failed > failed_abs_threshold:
        violation = (
            f"failed_count={failed} exceeds "
            f"failed_abs_threshold={failed_abs_threshold}"
        )
    elif ratio is not None and ratio > failed_ratio_threshold:
        violation = (
            f"failed_ratio={ratio:.6f} exceeds "
            f"failed_ratio_threshold={failed_ratio_threshold:.6f}"
        )
    else:
        violation = None
    completion = {
        "expected_rows": sum(statuses.values()),
        "materialized_rows": sum(statuses.values()),
        "completed_rows": ok + skipped,
        "failed_rows": failed,
        "status": "complete_with_failures" if failed else "complete",
    }
    policy = {
        "failed_abs_threshold": int(failed_abs_threshold),
        "failed_ratio_threshold": float(failed_ratio_threshold),
        "failed_count": failed,
        "ok_count": ok,
        "skipped_count": skipped,
        "denominator": denominator,
        "failed_ratio": ratio,
        "passed": violation is None,
        "violation": violation,
    }
    return completion, policy


def _write_temporary_manifest(target: Path, payload: Mapping[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".manifest.tmp", dir=target.parent
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _reserve_temporary_csv(target: Path, operation: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.{operation}.",
        suffix=".csv.tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _publish_pair(
    temporary_csv: Path,
    temporary_manifest: Path,
    target: Path,
) -> None:
    marker = _publish_marker(target)
    target_manifest = manifest_path(target)
    csv_backup = target.with_name(f".{target.name}.previous")
    manifest_backup = target_manifest.with_name(
        f".{target_manifest.name}.previous"
    )
    csv_backup.unlink(missing_ok=True)
    manifest_backup.unlink(missing_ok=True)
    marker.touch(exist_ok=True)
    try:
        if target.exists():
            os.replace(target, csv_backup)
        if target_manifest.exists():
            os.replace(target_manifest, manifest_backup)
        os.replace(temporary_csv, target)
        os.replace(temporary_manifest, target_manifest)
    except BaseException:
        target.unlink(missing_ok=True)
        target_manifest.unlink(missing_ok=True)
        if csv_backup.exists():
            os.replace(csv_backup, target)
        if manifest_backup.exists():
            os.replace(manifest_backup, target_manifest)
        raise
    finally:
        csv_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


def _shard_state(item: Mapping[str, Any], config: Any) -> str:
    try:
        seed = int(item["seed"])
        draws = tuple(int(draw) for draw in item["draws"])
        output = Path(item["config"]["out"])
        shard_manifest = manifest_path(output)
        if not output.exists() or not shard_manifest.exists():
            return "missing"
        manifest = _load_json(shard_manifest, f"seed={seed} manifest")
        _validate_shard_manifest(
            manifest,
            seed=seed,
            draws=draws,
            identity=None,
            contract=None,
            grids=None,
        )
        return "complete"
    except SeedShardIncompleteError:
        return "incomplete"
    except (KeyError, TypeError, ValueError):
        return "invalid"


def _publication_residuals(snapshot: Mapping[str, Any]) -> list[str]:
    targets: set[Path] = set()
    for job in snapshot["jobs"]:
        if isinstance(job, Mapping) and isinstance(job.get("final_out"), str):
            targets.add(Path(job["final_out"]).resolve())
    panel_outputs = snapshot.get("panel_outputs")
    if isinstance(panel_outputs, Mapping):
        targets.update(
            Path(output).resolve()
            for output in panel_outputs.values()
            if isinstance(output, str)
        )

    residuals: set[Path] = set()
    for target in targets:
        target_manifest = manifest_path(target)
        candidates = (
            _publish_marker(target),
            target.with_name(f".{target.name}.previous"),
            target_manifest.with_name(f".{target_manifest.name}.previous"),
        )
        residuals.update(path for path in candidates if path.exists())
    return [str(path) for path in sorted(residuals)]


def diagnose_missing(master_snapshot: Path) -> dict[str, list[int]]:
    snapshot = _load_snapshot(master_snapshot)
    missing: list[int] = []
    incomplete: list[int] = []
    invalid: list[int] = []
    configs: dict[tuple[str, str], Any] = {}
    for index, job in enumerate(snapshot["jobs"]):
        try:
            target = (str(job["panel"]), str(job["model"]))
            config = configs.setdefault(
                target, _config_from_json(job["config"])
            )
            state = _shard_state(job, config)
        except (KeyError, TypeError, ValueError):
            invalid.append(index)
            continue
        if state == "missing":
            missing.append(index)
        elif state == "incomplete":
            incomplete.append(index)
        elif state == "invalid":
            invalid.append(index)
    return {
        "missing_master_indices": missing,
        "incomplete_master_indices": incomplete,
        "invalid_targets": invalid,
        "publication_residuals": _publication_residuals(snapshot),
    }


def recovery_indices(
    master_snapshot: Path,
    *,
    resource_class: str,
    requested_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    from .slurm_jobs import RESOURCE_CLASSES, resource_class_for_model

    if resource_class not in RESOURCE_CLASSES:
        raise SeedShardValidationError(
            f"Unknown Slurm resource class: {resource_class}"
        )
    snapshot = _load_snapshot(master_snapshot)
    diagnosis = diagnose_missing(master_snapshot)
    recoverable = set(
        diagnosis["missing_master_indices"]
        + diagnosis["incomplete_master_indices"]
    )
    if requested_indices is None:
        candidates = sorted(recoverable)
    else:
        candidates = list(requested_indices)
        if (
            len(candidates) != len(set(candidates))
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(snapshot["jobs"])
                for index in candidates
            )
        ):
            raise SeedShardValidationError(
                "Requested recovery indices must be unique valid master indices"
            )
    return tuple(
        index
        for index in candidates
        if index in recoverable
        and resource_class_for_model(
            str(snapshot["jobs"][index]["model"])
        )
        == resource_class
    )


def build_finalizer_map(master_snapshot: Path, path: Path | None = None) -> Path:
    snapshot = _load_snapshot(master_snapshot)
    targets = sorted(
        {
            (str(job["panel"]), str(job["model"]))
            for job in snapshot["jobs"]
        }
    )
    target_path = path or (
        master_snapshot.parent / f"{master_snapshot.stem}.finalizers.json"
    )
    write_json_atomic(
        target_path,
        {
            "format_version": 1,
            "snapshot": str(master_snapshot.resolve()),
            "targets": [
                {"panel": panel, "model": model}
                for panel, model in targets
            ],
        },
    )
    os.chmod(target_path, 0o444)
    return target_path


def build_publish_map(master_snapshot: Path, path: Path | None = None) -> Path:
    snapshot = _load_snapshot(master_snapshot)
    models_by_panel: dict[str, set[str]] = {}
    for job in snapshot["jobs"]:
        models_by_panel.setdefault(str(job["panel"]), set()).add(
            str(job["model"])
        )
    target_path = path or (
        master_snapshot.parent / f"{master_snapshot.stem}.publish.json"
    )
    write_json_atomic(
        target_path,
        {
            "format_version": 1,
            "snapshot": str(master_snapshot.resolve()),
            "targets": [
                {"panel": panel, "models": sorted(models)}
                for panel, models in sorted(models_by_panel.items())
            ],
        },
    )
    os.chmod(target_path, 0o444)
    return target_path


def _load_map(
    master_snapshot: Path, map_path: Path, index: int
) -> Mapping[str, Any]:
    payload = _load_json(map_path, "finalization map")
    if (
        payload.get("format_version") != 1
        or Path(payload.get("snapshot", "")).resolve()
        != master_snapshot.resolve()
    ):
        raise SeedShardValidationError(
            "Finalization map does not belong to this snapshot"
        )
    targets = payload.get("targets")
    if not isinstance(targets, list) or not 0 <= index < len(targets):
        raise SeedShardValidationError("Finalization map index is out of bounds")
    target = targets[index]
    if not isinstance(target, Mapping):
        raise SeedShardValidationError("Invalid finalization map target")
    return target


def map_task_count(map_path: Path) -> int:
    payload = _load_json(map_path, "finalization map")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise SeedShardValidationError("Finalization map has no targets")
    return len(targets)


def finalize_seed_shards(
    master_snapshot: Path, *, panel: str, model: str
) -> Path:
    """Validate and merge every seed shard for one panel/model."""
    snapshot = _load_snapshot(master_snapshot)
    config, target, by_seed = _validate_master_target(
        snapshot, panel, model
    )
    expected_by_seed = group_repeat_pairs_by_seed(resolve_repeat_pairs(config))
    target.parent.mkdir(parents=True, exist_ok=True)

    manifests: list[tuple[dict[str, Any], Path, int, tuple[int, ...]]] = []
    input_artifacts: list[dict[str, Any]] = []
    identity: Mapping[str, Any] | None = None
    contract: Mapping[str, Any] | None = None
    grids: tuple[list[int], list[int]] | None = None
    header: list[str] | None = None
    for seed, draws in expected_by_seed.items():
        job = by_seed[seed]
        shard = Path(job["config"]["out"])
        shard_manifest = manifest_path(shard)
        if not shard.exists() or not shard_manifest.exists():
            raise SeedShardIncompleteError(
                f"seed={seed}: missing output or manifest"
            )
        manifest = _load_json(shard_manifest, f"seed={seed} manifest")
        identity, contract, grids = _validate_shard_manifest(
            manifest,
            seed=seed,
            draws=tuple(draws),
            identity=identity,
            contract=contract,
            grids=grids,
        )
        current_header = _read_csv_header(shard, f"seed={seed}")
        if header is None:
            header = current_header
        elif current_header != header:
            raise SeedShardValidationError(f"seed={seed}: CSV header differs")
        manifests.append((manifest, shard, seed, tuple(draws)))
        input_artifacts.append(
            _artifact_provenance(
                shard,
                manifest,
                rows=len(draws) * len(grids[0]) * len(grids[1]),
                seed=seed,
            )
        )

    assert identity is not None and contract is not None and grids is not None
    expected_keys = _expected_keys(config, (model,), grids)

    descriptor, raw_database = tempfile.mkstemp(
        prefix="nk-grid-seed-", suffix=".sqlite", dir=target.parent
    )
    os.close(descriptor)
    database = Path(raw_database)
    temporary_csv = _reserve_temporary_csv(target, "seed-finalize")
    temporary_manifest: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database)
        connection.execute(
            'CREATE TABLE rows ("model", "seed", "draw", "N", "K", '
            'payload TEXT NOT NULL, PRIMARY KEY '
            '("model", "seed", "draw", "N", "K"))'
        )
        statuses: Counter[str] = Counter()
        for _, shard, seed, draws in manifests:
            with shard.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    key = _row_key(row)
                    status = str(row.get("status"))
                    if status not in VALID_STATUSES:
                        raise SeedShardValidationError(
                            f"seed={seed}: invalid status {status!r}"
                        )
                    if (
                        key[0] != model
                        or key[1] != seed
                        or key[2] not in draws
                        or key not in expected_keys
                    ):
                        raise SeedShardValidationError(
                            f"seed={seed}: out-of-design cell key {key}"
                        )
                    try:
                        connection.execute(
                            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)",
                            (*key, json.dumps(row, sort_keys=True)),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise SeedShardValidationError(
                            f"Duplicate cell key: {key}"
                        ) from exc
                    statuses[status] += 1
        actual_keys = {
            tuple(row)
            for row in connection.execute(
                'SELECT "model", "seed", "draw", "N", "K" FROM rows'
            )
        }
        if actual_keys != expected_keys:
            raise SeedShardValidationError(
                "Merged cell keys differ "
                f"(missing={sorted(expected_keys - actual_keys)[:3]}, "
                f"extra={sorted(actual_keys - expected_keys)[:3]})"
            )
        with temporary_csv.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=header or [])
            writer.writeheader()
            for (payload,) in connection.execute(
                'SELECT payload FROM rows ORDER BY '
                '"model", "seed", "draw", "N", "K"'
            ):
                writer.writerow(json.loads(payload))
            destination.flush()
            os.fsync(destination.fileno())

        completion, policy = _status_summary(
            statuses,
            failed_abs_threshold=int(config.failed_abs_threshold),
            failed_ratio_threshold=float(config.failed_ratio_threshold),
        )
        final_manifest = copy.deepcopy(manifests[0][0])
        final_manifest["updated_at"] = utc_now()
        final_manifest["execution"] = {
            "mode": "finalized-seed-shards",
            "expected_rows": len(expected_keys),
        }
        final_manifest["completion"] = completion
        final_manifest["failure_policy"] = policy
        final_manifest["input_artifacts"] = input_artifacts
        final_manifest.pop("diagnostics", None)
        temporary_manifest = _write_temporary_manifest(
            target, final_manifest
        )
        with output_run_lock(target):
            if _existing_artifact_equivalent(
                target,
                expected_keys=expected_keys,
                identity=identity,
                contract=contract,
                grids=grids,
                execution_mode="finalized-seed-shards",
                input_artifacts=input_artifacts,
            ):
                return target
            _publish_pair(temporary_csv, temporary_manifest, target)
        return target
    finally:
        if connection is not None:
            connection.close()
        temporary_csv.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        database.unlink(missing_ok=True)


def _panel_output(snapshot: Mapping[str, Any], panel: str) -> Path:
    panel_outputs = snapshot.get("panel_outputs")
    if not isinstance(panel_outputs, Mapping) or not isinstance(
        panel_outputs.get(panel), str
    ):
        raise SeedShardValidationError(
            f"Master snapshot lacks declared output for panel {panel!r}"
        )
    return Path(panel_outputs[panel]).resolve()


def _panel_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    shared = copy.deepcopy(dict(contract))
    for field in ("model", "resolved_model_params", "environment_overrides"):
        shared.pop(field, None)
    return shared


def publish_panel(
    master_snapshot: Path, *, panel: str, models: Sequence[str]
) -> Path:
    """Publish all complete per-model finals into one complete panel result."""
    snapshot = _load_snapshot(master_snapshot)
    expected_models = sorted(
        {
            str(job["model"])
            for job in snapshot["jobs"]
            if job.get("panel") == panel
        }
    )
    if list(models) != expected_models:
        raise SeedShardValidationError(
            f"Publish map models differ for panel {panel!r}"
        )
    target = _panel_output(snapshot, panel)
    target.parent.mkdir(parents=True, exist_ok=True)

    artifacts: list[
        tuple[str, Path, dict[str, Any], Any, set[tuple[str, int, int, int, int]]]
    ] = []
    input_artifacts: list[dict[str, Any]] = []
    identity: Mapping[str, Any] | None = None
    shared_contract: dict[str, Any] | None = None
    grids: tuple[list[int], list[int]] | None = None
    header: list[str] | None = None
    all_expected_keys: set[tuple[str, int, int, int, int]] = set()
    for model in expected_models:
        config, model_final, _ = _validate_master_target(
            snapshot, panel, model
        )
        model_manifest_path = manifest_path(model_final)
        if not model_final.exists() or not model_manifest_path.exists():
            raise SeedShardIncompleteError(
                f"{panel}/{model}: per-model final is missing"
            )
        manifest = _load_json(
            model_manifest_path, f"{panel}/{model} final manifest"
        )
        current_identity = _validate_identity(
            manifest.get("identity"), f"{panel}/{model}"
        )
        current_contract = manifest.get("semantic_contract")
        if not isinstance(current_contract, Mapping):
            raise SeedShardValidationError(
                f"{panel}/{model}: semantic contract missing"
            )
        current_shared_contract = _panel_contract(current_contract)
        current_grids = _resolved_grids(manifest, f"{panel}/{model}")
        if identity is None:
            identity = current_identity
            shared_contract = current_shared_contract
            grids = current_grids
        else:
            for field in _IDENTITY_FIELDS:
                if current_identity[field] != identity[field]:
                    raise SeedShardValidationError(
                        f"{panel}/{model}: identity.{field} differs"
                    )
            if (
                difference := _difference(
                    shared_contract,
                    current_shared_contract,
                    "semantic_contract",
                )
            ):
                raise SeedShardValidationError(
                    f"{panel}/{model}: {difference}"
                )
            if current_grids != grids:
                raise SeedShardValidationError(
                    f"{panel}/{model}: resolved N/K grid differs"
                )
        execution = manifest.get("execution")
        if (
            not isinstance(execution, Mapping)
            or execution.get("mode") != "finalized-seed-shards"
        ):
            raise SeedShardIncompleteError(
                f"{panel}/{model}: per-model final is incomplete"
            )
        model_expected_keys = _expected_keys(
            config, (model,), current_grids
        )
        keys, _, current_header = _read_artifact_keys(
            model_final, expected_keys=model_expected_keys
        )
        if keys != model_expected_keys:
            raise SeedShardValidationError(
                f"{panel}/{model}: per-model key set is incomplete"
            )
        if header is None:
            header = current_header
        elif current_header != header:
            raise SeedShardValidationError(
                f"{panel}/{model}: CSV header differs"
            )
        if all_expected_keys & model_expected_keys:
            raise SeedShardValidationError(
                f"{panel}/{model}: cross-model cell keys overlap"
            )
        all_expected_keys |= model_expected_keys
        artifacts.append(
            (model, model_final, manifest, config, model_expected_keys)
        )
        input_artifacts.append(
            _artifact_provenance(
                model_final,
                manifest,
                rows=len(keys),
                model=model,
            )
        )

    assert identity is not None and shared_contract is not None and grids is not None
    panel_contract = copy.deepcopy(shared_contract)
    panel_contract["model"] = expected_models
    with output_run_lock(target):
        if _existing_artifact_equivalent(
            target,
            expected_keys=all_expected_keys,
            identity=identity,
            contract=panel_contract,
            grids=grids,
            execution_mode="published-model-finals",
            input_artifacts=input_artifacts,
        ):
            return target

    descriptor, raw_database = tempfile.mkstemp(
        prefix="nk-grid-panel-", suffix=".sqlite", dir=target.parent
    )
    os.close(descriptor)
    database = Path(raw_database)
    temporary_csv = _reserve_temporary_csv(target, "panel-publish")
    temporary_manifest: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database)
        connection.execute(
            'CREATE TABLE rows ("model", "seed", "draw", "N", "K", '
            'payload TEXT NOT NULL, PRIMARY KEY '
            '("model", "seed", "draw", "N", "K"))'
        )
        statuses: Counter[str] = Counter()
        model_policies: dict[str, Any] = {}
        for model, model_final, model_manifest, _, model_expected in artifacts:
            model_policies[model] = copy.deepcopy(
                model_manifest.get("failure_policy")
            )
            with model_final.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    key = _row_key(row)
                    status = str(row.get("status"))
                    if status not in VALID_STATUSES or key not in model_expected:
                        raise SeedShardValidationError(
                            f"{panel}/{model}: invalid row {key}"
                        )
                    try:
                        connection.execute(
                            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)",
                            (*key, json.dumps(row, sort_keys=True)),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise SeedShardValidationError(
                            f"Duplicate cross-model cell key: {key}"
                        ) from exc
                    statuses[status] += 1
        actual_keys = {
            tuple(row)
            for row in connection.execute(
                'SELECT "model", "seed", "draw", "N", "K" FROM rows'
            )
        }
        if actual_keys != all_expected_keys:
            raise SeedShardValidationError(
                "Published panel key set is incomplete"
            )
        with temporary_csv.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=header or [])
            writer.writeheader()
            for (payload,) in connection.execute(
                'SELECT payload FROM rows ORDER BY '
                '"model", "seed", "draw", "N", "K"'
            ):
                writer.writerow(json.loads(payload))
            destination.flush()
            os.fsync(destination.fileno())

        failed = int(statuses["failed"])
        ok = int(statuses["ok"])
        skipped = int(statuses["skipped"])
        panel_passed = all(
            isinstance(policy, Mapping) and policy.get("passed") is True
            for policy in model_policies.values()
        )
        first_manifest = artifacts[0][2]
        panel_manifest = copy.deepcopy(first_manifest)
        panel_manifest["updated_at"] = utc_now()
        panel_manifest["semantic_contract"] = panel_contract
        panel_manifest["design"]["models"] = expected_models
        panel_manifest["execution"] = {
            "mode": "published-model-finals",
            "expected_rows": len(all_expected_keys),
        }
        panel_manifest["completion"] = {
            "expected_rows": len(all_expected_keys),
            "materialized_rows": len(all_expected_keys),
            "completed_rows": ok + skipped,
            "failed_rows": failed,
            "status": "complete_with_failures" if failed else "complete",
        }
        panel_manifest["failure_policy"] = {
            "models": model_policies,
            "failed_count": failed,
            "ok_count": ok,
            "skipped_count": skipped,
            "passed": panel_passed,
        }
        panel_manifest["input_artifacts"] = input_artifacts
        panel_manifest.pop("diagnostics", None)
        temporary_manifest = _write_temporary_manifest(
            target, panel_manifest
        )
        with output_run_lock(target):
            if _existing_artifact_equivalent(
                target,
                expected_keys=all_expected_keys,
                identity=identity,
                contract=panel_contract,
                grids=grids,
                execution_mode="published-model-finals",
                input_artifacts=input_artifacts,
            ):
                return target
            _publish_pair(temporary_csv, temporary_manifest, target)
        return target
    finally:
        if connection is not None:
            connection.close()
        temporary_csv.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        database.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and publish NK Grid seed shards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--snapshot", type=Path, required=True)
    finalize.add_argument("--map", dest="finalization_map", type=Path, required=True)
    finalize.add_argument("--index", type=int, required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--snapshot", type=Path, required=True)
    publish.add_argument("--map", dest="finalization_map", type=Path, required=True)
    publish.add_argument("--index", type=int, required=True)
    missing = subparsers.add_parser("missing")
    missing.add_argument("--snapshot", type=Path, required=True)
    build = subparsers.add_parser("build-map")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--kind", choices=("finalizer", "publish"), required=True)
    build.add_argument("--map", dest="finalization_map", type=Path)
    count = subparsers.add_parser("map-count")
    count.add_argument("--map", dest="finalization_map", type=Path, required=True)
    recovery = subparsers.add_parser("recovery-indices")
    recovery.add_argument("--snapshot", type=Path, required=True)
    recovery.add_argument(
        "--resource-class",
        choices=("parallel", "serial", "super_learner"),
        required=True,
    )
    recovery.add_argument("--master-indices")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "missing":
            print(json.dumps(diagnose_missing(args.snapshot), sort_keys=True))
            return 0
        if args.command == "build-map":
            builder = (
                build_finalizer_map
                if args.kind == "finalizer"
                else build_publish_map
            )
            print(builder(args.snapshot, args.finalization_map))
            return 0
        if args.command == "map-count":
            print(map_task_count(args.finalization_map))
            return 0
        if args.command == "recovery-indices":
            requested_indices = None
            if args.master_indices is not None:
                try:
                    requested_indices = tuple(
                        int(value)
                        for value in args.master_indices.split(",")
                        if value
                    )
                except ValueError as exc:
                    raise SeedShardValidationError(
                        "--master-indices must be comma-separated integers"
                    ) from exc
            print(
                ",".join(
                    str(index)
                    for index in recovery_indices(
                        args.snapshot,
                        resource_class=args.resource_class,
                        requested_indices=requested_indices,
                    )
                )
            )
            return 0
        target = _load_map(
            args.snapshot, args.finalization_map, args.index
        )
        if args.command == "finalize":
            if not isinstance(target.get("panel"), str) or not isinstance(
                target.get("model"), str
            ):
                raise SeedShardValidationError("Invalid finalizer map target")
            finalize_seed_shards(
                args.snapshot,
                panel=target["panel"],
                model=target["model"],
            )
        else:
            models = target.get("models")
            if (
                not isinstance(target.get("panel"), str)
                or not isinstance(models, list)
                or any(not isinstance(model, str) for model in models)
            ):
                raise SeedShardValidationError("Invalid publish map target")
            publish_panel(
                args.snapshot, panel=target["panel"], models=models
            )
        return 0
    except SeedShardIncompleteError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 3
    except OSError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 4
    except (SeedShardValidationError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

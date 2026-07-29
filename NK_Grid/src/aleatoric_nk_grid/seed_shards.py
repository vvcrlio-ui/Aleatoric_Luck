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
_LEGACY_DIGEST_FIELDS = frozenset(
    {
        "file_" + "sha256",
        "semantic_" + "sha256",
        "snapshot_" + "sha256",
        "worker_script_" + "sha256",
    }
)


class SeedShardIncompleteError(ValueError):
    """A shard is missing or incomplete and can be recreated by a worker."""


class SeedShardValidationError(ValueError):
    """A shard or snapshot violates the immutable explicit identity contract."""


def _difference(left: Any, right: Any, path: str) -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}: left={left.get(key)!r} right={right.get(key)!r}"
            result = _difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
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


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    try:
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedShardValidationError(f"Cannot parse master snapshot: {snapshot_path}") from exc
    if snapshot.get("format_version") != 1 or not isinstance(snapshot.get("jobs"), list):
        raise SeedShardValidationError("Unsupported master snapshot")
    legacy = _legacy_digest_field(snapshot)
    if legacy:
        raise SeedShardValidationError(f"Legacy digest field is not accepted: {legacy}")
    return snapshot


def _target_jobs(snapshot: Mapping[str, Any], panel: str, model: str) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (index, item) for index, item in enumerate(snapshot["jobs"])
        if item.get("panel") == panel and item.get("model") == model
    ]
    if not selected:
        raise SeedShardValidationError(f"No seed jobs for {panel}/{model}")
    return selected


def _validate_master_target(snapshot: Mapping[str, Any], panel: str, model: str) -> tuple[Any, Path, dict[int, dict[str, Any]]]:
    selected = _target_jobs(snapshot, panel, model)
    try:
        config = _config_from_json(selected[0][1]["config"])
        expected = group_repeat_pairs_by_seed(resolve_repeat_pairs(config))
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedShardValidationError(f"Invalid master config for {panel}/{model}") from exc
    try:
        declared_seeds = [int(item["seed"]) for _, item in selected]
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedShardValidationError("Master seed job lacks an explicit seed") from exc
    duplicates = sorted(seed for seed, count in Counter(declared_seeds).items() if count > 1)
    if duplicates:
        raise SeedShardValidationError(f"Duplicate master seed jobs: {duplicates[:20]}")
    seeds: list[int] = []
    final_outputs: set[Path] = set()
    outputs: set[Path] = set()
    by_seed: dict[int, dict[str, Any]] = {}
    for _, item in selected:
        try:
            seed = int(item["seed"])
            draws = tuple(int(draw) for draw in item["draws"])
            out = Path(item["config"]["out"])
            final_out = Path(item["final_out"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedShardValidationError("Master seed job lacks explicit seed/draw/output fields") from exc
        seeds.append(seed)
        final_outputs.add(final_out.resolve())
        if out.resolve() in outputs:
            raise SeedShardValidationError(f"Duplicate shard output path: {out}")
        outputs.add(out.resolve())
        if seed not in expected:
            raise SeedShardValidationError(f"Unexpected master seed: {seed}")
        if draws != tuple(expected[seed]):
            raise SeedShardValidationError(f"seed={seed}: draws do not match complete repeat plan")
        by_seed[seed] = item
    missing = sorted(set(expected) - set(by_seed))
    if missing:
        raise SeedShardValidationError(f"Missing master seed jobs: {missing[:20]}")
    if len(final_outputs) != 1:
        raise SeedShardValidationError("Master seed jobs disagree on final_out")
    return config, next(iter(final_outputs)), by_seed


def _integer(value: Any, field: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SeedShardValidationError(f"{field} must be a finite integer") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise SeedShardValidationError(f"{field} must be a finite integer")
    return int(parsed)


def _expected_keys(
    config: Any, model: str, grids: tuple[Any, Any]
) -> set[tuple[str, int, int, int, int]]:
    n_grid = tuple(_integer(value, "N") for value in grids[0] or ())
    k_grid = tuple(_integer(value, "K") for value in grids[1] or ())
    if not n_grid or not k_grid:
        raise SeedShardValidationError("Seed finalization requires resolved N/K grids in snapshot")
    return {
        (model, int(seed), int(draw), n, k)
        for seed, draw in resolve_repeat_pairs(config)
        for n in n_grid
        for k in k_grid
    }


def _validate_shard_manifest(
    manifest: Mapping[str, Any], *, seed: int, draws: tuple[int, ...], config: Any,
    identity: Mapping[str, Any] | None, contract: Mapping[str, Any] | None,
    grids: tuple[Any, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[Any, Any]]:
    legacy = _legacy_digest_field(manifest)
    if legacy:
        raise SeedShardValidationError(f"Legacy digest field is not accepted: {legacy}")
    current_identity = manifest.get("identity")
    if not isinstance(current_identity, Mapping) or current_identity.get("mode") != "explicit-v1":
        raise SeedShardValidationError(f"seed={seed}: manifest is not explicit-v1")
    for field in ("experiment_id", "data_version", "model_spec_version"):
        if field not in current_identity:
            raise SeedShardValidationError(f"seed={seed}: identity.{field} missing")
    current_contract = manifest.get("semantic_contract")
    if not isinstance(current_contract, Mapping):
        raise SeedShardValidationError(f"seed={seed}: semantic contract missing")
    if identity is not None and (difference := _difference(identity, current_identity, "identity")):
        raise SeedShardValidationError(f"seed={seed}: {difference}")
    if contract is not None and (difference := _difference(contract, current_contract, "semantic_contract")):
        raise SeedShardValidationError(f"seed={seed}: {difference}")
    design = manifest.get("design")
    execution = manifest.get("execution")
    completion = manifest.get("completion")
    if not isinstance(design, Mapping) or not isinstance(execution, Mapping) or not isinstance(completion, Mapping):
        raise SeedShardValidationError(f"seed={seed}: manifest design/execution/completion missing")
    current_grids = (design.get("n_grid"), design.get("k_grid"))
    if grids is not None and current_grids != grids:
        raise SeedShardValidationError(f"seed={seed}: resolved N/K grid differs")
    if execution.get("mode") != "seed-shard" or execution.get("seed") != seed or execution.get("draws") != list(draws):
        raise SeedShardValidationError(f"seed={seed}: execution declaration does not match shard")
    try:
        expected_rows = len(draws) * len(current_grids[0]) * len(current_grids[1])
    except TypeError as exc:
        raise SeedShardValidationError(f"seed={seed}: resolved N/K grid is invalid") from exc
    if execution.get("expected_rows") != expected_rows or completion.get("expected_rows") != expected_rows:
        raise SeedShardValidationError(f"seed={seed}: expected row count differs")
    if completion.get("materialized_rows") != expected_rows or completion.get("status") not in {"complete", "complete_with_failures"}:
        raise SeedShardIncompleteError(f"seed={seed}: shard is incomplete")
    return current_identity, current_contract, current_grids


def _shard_state(item: Mapping[str, Any], config: Any) -> str:
    """Return complete, incomplete, or invalid without raising for absence."""
    try:
        seed = int(item["seed"])
        draws = tuple(int(draw) for draw in item["draws"])
        out = Path(item["config"]["out"])
        path = manifest_path(out)
        if not out.exists() or not path.exists():
            return "missing"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _validate_shard_manifest(manifest, seed=seed, draws=draws, config=config, identity=None, contract=None, grids=None)
        return "complete"
    except SeedShardIncompleteError:
        return "incomplete"
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "invalid"


def build_finalizer_map(master_snapshot: Path, path: Path | None = None) -> Path:
    snapshot = _load_snapshot(master_snapshot)
    targets = sorted({(str(job.get("panel")), str(job.get("model"))) for job in snapshot["jobs"]})
    if not targets:
        raise SeedShardValidationError("Master snapshot has no jobs")
    target_path = path or master_snapshot.parent / f"{master_snapshot.stem}.finalizers.json"
    write_json_atomic(target_path, {"format_version": 1, "snapshot": str(master_snapshot.resolve()), "targets": [{"panel": panel, "model": model} for panel, model in targets]})
    os.chmod(target_path, 0o444)
    return target_path


def _load_finalizer_target(master_snapshot: Path, finalizer_map: Path, index: int) -> tuple[str, str]:
    payload = json.loads(finalizer_map.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or Path(payload.get("snapshot", "")).resolve() != master_snapshot.resolve():
        raise SeedShardValidationError("Finalizer map does not belong to this snapshot")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not 0 <= index < len(targets):
        raise SeedShardValidationError("Finalizer map index is out of bounds")
    target = targets[index]
    if not isinstance(target, Mapping) or not isinstance(target.get("panel"), str) or not isinstance(target.get("model"), str):
        raise SeedShardValidationError("Invalid finalizer map target")
    return target["panel"], target["model"]


def diagnose_missing(master_snapshot: Path) -> dict[str, list[int]]:
    snapshot = _load_snapshot(master_snapshot)
    missing: list[int] = []
    incomplete: list[int] = []
    invalid: list[int] = []
    configs: dict[tuple[str, str], Any] = {}
    for index, job in enumerate(snapshot["jobs"]):
        try:
            target = (str(job["panel"]), str(job["model"]))
            config = configs.setdefault(target, _config_from_json(job["config"]))
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
    return {"missing_master_indices": missing, "incomplete_master_indices": incomplete, "invalid_targets": invalid}


def finalize_seed_shards(master_snapshot: Path, *, panel: str, model: str) -> Path:
    """Stream strictly validated shards into one atomically published CSV."""
    snapshot = _load_snapshot(master_snapshot)
    config, target, by_seed = _validate_master_target(snapshot, panel, model)
    expected = group_repeat_pairs_by_seed(resolve_repeat_pairs(config))
    target.parent.mkdir(parents=True, exist_ok=True)
    with output_run_lock(target):
        manifests: list[tuple[dict[str, Any], Path]] = []
        identity: Mapping[str, Any] | None = None
        contract: Mapping[str, Any] | None = None
        grids: tuple[Any, Any] | None = None
        header: list[str] | None = None
        for seed, draws in expected.items():
            job = by_seed[seed]
            shard = Path(job["config"]["out"])
            shard_manifest_path = manifest_path(shard)
            if not shard.exists() or not shard_manifest_path.exists():
                raise SeedShardIncompleteError(f"seed={seed}: missing output or manifest")
            try:
                manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SeedShardValidationError(f"seed={seed}: manifest cannot be parsed") from exc
            identity, contract, grids = _validate_shard_manifest(manifest, seed=seed, draws=tuple(draws), config=config, identity=identity, contract=contract, grids=grids)
            with shard.open(newline="", encoding="utf-8") as source:
                current_header = csv.DictReader(source).fieldnames
            if not current_header or any(column not in current_header for column in KEY_COLUMNS):
                raise SeedShardValidationError(f"seed={seed}: CSV key columns are missing")
            if header is None:
                header = current_header
            elif header != current_header:
                raise SeedShardValidationError(f"seed={seed}: CSV header differs")
            manifests.append((manifest, shard))

        assert grids is not None
        expected_keys = _expected_keys(config, model, grids)

        with tempfile.NamedTemporaryFile(prefix="nk-grid-seed-", suffix=".sqlite", dir=target.parent, delete=False) as handle:
            database = Path(handle.name)
        temporary = target.with_name(f".{target.name}.seed-finalize.tmp")
        try:
            con = sqlite3.connect(database)
            con.execute('CREATE TABLE rows ("model", "seed", "draw", "N", "K", payload TEXT NOT NULL, PRIMARY KEY ("model", "seed", "draw", "N", "K"))')
            for (_, shard), (seed, draws) in zip(manifests, expected.items()):
                with shard.open(newline="", encoding="utf-8") as source:
                    for row in csv.DictReader(source):
                        key = (str(row.get("model")), *(_integer(row.get(field), field) for field in KEY_COLUMNS[1:]))
                        if key[0] != model or key[1] != seed or key[2] not in draws or key not in expected_keys:
                            raise SeedShardValidationError(f"seed={seed}: out-of-design cell key {key}")
                        try:
                            con.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)", (*key, json.dumps(row, sort_keys=True)))
                        except sqlite3.IntegrityError as exc:
                            raise SeedShardValidationError(f"Duplicate cell key: {key}") from exc
            actual_keys = {tuple(row) for row in con.execute('SELECT "model", "seed", "draw", "N", "K" FROM rows')}
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise SeedShardValidationError(f"Merged cell keys differ (missing={missing[:3]}, extra={extra[:3]})")
            statuses = Counter(row[0] for row in con.execute("SELECT json_extract(payload, '$.status') FROM rows"))
            if set(statuses) - {"ok", "skipped", "failed"}:
                raise SeedShardValidationError(f"Invalid cell status values: {sorted(set(statuses) - {'ok', 'skipped', 'failed'})}")
            with temporary.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=header or [])
                writer.writeheader()
                for (payload,) in con.execute('SELECT payload FROM rows ORDER BY "model", "seed", "draw", "N", "K"'):
                    writer.writerow(json.loads(payload))
                destination.flush()
                os.fsync(destination.fileno())
            failed = int(statuses["failed"])
            ok = int(statuses["ok"])
            skipped = int(statuses["skipped"])
            denominator = ok + failed
            ratio = float(failed / denominator) if denominator else None
            policy = {
                "failed_abs_threshold": int(config.failed_abs_threshold),
                "failed_ratio_threshold": float(config.failed_ratio_threshold),
                "failed_count": failed,
                "ok_count": ok,
                "skipped_count": skipped,
                "denominator": denominator,
                "failed_ratio": ratio,
            }
            violation = ("failure-policy denominator is zero (no ok/failed cells)" if denominator == 0 else (f"failed_count={failed} exceeds failed_abs_threshold={policy['failed_abs_threshold']}" if failed > policy["failed_abs_threshold"] else (f"failed_ratio={ratio:.6f} exceeds failed_ratio_threshold={policy['failed_ratio_threshold']:.6f}" if ratio > policy["failed_ratio_threshold"] else None)))
            final_manifest = copy.deepcopy(manifests[0][0])
            final_manifest["updated_at"] = utc_now()
            final_manifest["execution"] = {"mode": "finalized-seed-shards", "expected_rows": len(expected_keys)}
            final_manifest["completion"] = {"expected_rows": len(expected_keys), "materialized_rows": len(expected_keys), "completed_rows": ok + skipped, "failed_rows": failed, "status": "complete_with_failures" if failed else "complete"}
            final_manifest["failure_policy"] = {**policy, "passed": violation is None, "violation": violation}
            os.replace(temporary, target)
            write_json_atomic(manifest_path(target), final_manifest)
            return target
        finally:
            if "con" in locals():
                con.close()
            temporary.unlink(missing_ok=True)
            database.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and merge NK Grid seed shards.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--snapshot", type=Path, required=True)
    finalize.add_argument("--map", dest="finalizer_map", type=Path, required=True)
    finalize.add_argument("--index", type=int, required=True)
    missing = subparsers.add_parser("missing")
    missing.add_argument("--snapshot", type=Path, required=True)
    build = subparsers.add_parser("build-map")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--map", dest="finalizer_map", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "missing":
            print(json.dumps(diagnose_missing(args.snapshot), sort_keys=True))
            return 0
        if args.command == "build-map":
            print(build_finalizer_map(args.snapshot, args.finalizer_map))
            return 0
        panel, model = _load_finalizer_target(args.snapshot, args.finalizer_map, args.index)
        finalize_seed_shards(args.snapshot, panel=panel, model=model)
        return 0
    except SeedShardIncompleteError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 3
    except (SeedShardValidationError, OSError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

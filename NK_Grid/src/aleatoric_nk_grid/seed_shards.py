"""Validation and streaming publication of seed-sharded NK Grid results."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .experiment import manifest_path, utc_now, write_json_atomic
from .nk_grid import group_repeat_pairs_by_seed, resolve_repeat_pairs
from .slurm_jobs import _config_from_json


KEY_COLUMNS = ("model", "seed", "draw", "N", "K")


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


def finalize_seed_shards(master_snapshot: Path, *, panel: str, model: str) -> Path:
    """Strictly validate shard contracts and atomically merge them via SQLite."""

    snapshot = json.loads(Path(master_snapshot).read_text(encoding="utf-8"))
    jobs = [item for item in snapshot["jobs"] if item["panel"] == panel and item["model"] == model]
    if not jobs:
        raise ValueError(f"No seed jobs for {panel}/{model}")
    config = _config_from_json(jobs[0]["config"])
    final_out = Path(jobs[0]["final_out"])
    expected = group_repeat_pairs_by_seed(resolve_repeat_pairs(config))
    by_seed = {int(item["seed"]): item for item in jobs}
    missing = sorted(set(expected) - set(by_seed))
    if missing:
        raise ValueError(f"Missing master seed jobs: {missing[:20]} (total={len(missing)})")
    manifests: list[tuple[dict[str, Any], Path]] = []
    issues: list[str] = []
    identity: Any = None
    contract: Any = None
    grids: Any = None
    for seed, draws in expected.items():
        job = by_seed[seed]
        out = Path(job["config"]["out"])
        path = manifest_path(out)
        if not out.exists() or not path.exists():
            issues.append(f"seed={seed}: missing output or manifest")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        execution = manifest.get("execution", {})
        if execution.get("seed") != seed or execution.get("draws") != list(draws):
            issues.append(f"seed={seed}: execution declaration does not match shard")
        if manifest.get("completion", {}).get("materialized_rows") != execution.get("expected_rows"):
            issues.append(f"seed={seed}: shard is incomplete")
        for label, value in (("identity", manifest.get("identity")), ("semantic_contract", manifest.get("semantic_contract"))):
            baseline = identity if label == "identity" else contract
            if baseline is None:
                if label == "identity": identity = value
                else: contract = value
            elif (difference := _difference(baseline, value, label)):
                issues.append(f"seed={seed}: {difference}")
        grid = (manifest.get("design", {}).get("n_grid"), manifest.get("design", {}).get("k_grid"))
        if grids is None: grids = grid
        elif grids != grid: issues.append(f"seed={seed}: resolved N/K grid differs")
        manifests.append((manifest, out))
    if issues:
        raise ValueError("Seed shard validation failed: " + "; ".join(issues[:20]) + f" (total={len(issues)})")
    target = final_out
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="nk-grid-seed-", suffix=".sqlite", delete=False) as handle:
        database = Path(handle.name)
    try:
        con = sqlite3.connect(database)
        con.execute('CREATE TABLE rows ("model", "seed", "draw", "N", "K", payload TEXT NOT NULL, PRIMARY KEY ("model", "seed", "draw", "N", "K"))')
        columns: list[str] | None = None
        for _, shard in manifests:
            with shard.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if columns is None: columns = reader.fieldnames or []
                for row in reader:
                    key = tuple(row[column] for column in KEY_COLUMNS)
                    try:
                        con.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)", (*key, json.dumps(row, sort_keys=True)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"Duplicate cell key: {key}") from exc
        expected_rows = len(config.models) * len(resolve_repeat_pairs(config)) * len(grids[0]) * len(grids[1])
        actual_rows = con.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        if actual_rows != expected_rows:
            raise ValueError(f"Merged rows={actual_rows}, expected={expected_rows}")
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns or [])
            writer.writeheader()
            for (payload,) in con.execute('SELECT payload FROM rows ORDER BY "model", CAST("seed" AS INTEGER), CAST("draw" AS INTEGER), CAST("N" AS INTEGER), CAST("K" AS INTEGER)'):
                writer.writerow(json.loads(payload))
            destination.flush(); os.fsync(destination.fileno())
        os.replace(temporary, target)
        manifest = manifests[0][0]
        manifest["updated_at"] = utc_now()
        manifest["execution"] = {"mode": "finalized-seed-shards", "expected_rows": expected_rows}
        manifest["failure_policy"]["passed"] = manifest["failure_policy"].get("failed_count", 0) == 0
        write_json_atomic(manifest_path(target), manifest)
        return target
    finally:
        if 'con' in locals(): con.close()
        database.unlink(missing_ok=True)

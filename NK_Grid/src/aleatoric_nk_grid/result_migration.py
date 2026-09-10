"""Explicit, offline import of sealed legacy results into an untouched queue.

The exporter uses legacy verification/WAL readers and refuses active generations.
Import never edits old contracts, old WALs or their source-version metadata.
Portable bundle validation is separated from the POSIX-only legacy exporter.
"""
from __future__ import annotations
import json
import hashlib
import math
import os
from pathlib import Path

from .shared_queue import ModelTask, QueueError, atomic_json, canonical, digest, file_digest

OLD_ALGORITHM = "nk-grid-v9-balanced-batch-1"
NEW_ALGORITHM = "nk-grid-v10-svd-gesvd-fallback-1"


def assert_compatible_specs(old, new, *, certificate=None):
    """Permit scheduling identity changes and only the declared SVD transition."""
    a = dict(old); b = dict(new)
    for key in ("git_commit", "require_clean_worktree", "execution_groups"):
        a.pop(key, None); b.pop(key, None)
    if a.get("algorithm_version") != b.get("algorithm_version"):
        if (a.get("algorithm_version"), b.get("algorithm_version")) != (OLD_ALGORITHM, NEW_ALGORITHM):
            raise QueueError("Unapproved numerical algorithm transition")
        b["algorithm_version"] = OLD_ALGORITHM
    if a.get("model_params_sha256") != b.get("model_params_sha256"):
        pairs = (certificate or {}).get("parameter_files", {}).values()
        expected = {"old_sha256": a.get("model_params_sha256"), "new_sha256": b.get("model_params_sha256")}
        if expected not in pairs:
            raise QueueError("Uncertified model parameter file change")
        b["model_params_sha256"] = a["model_params_sha256"]
    if a != b:
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise QueueError("Scientific inputs/settings changed: " + ", ".join(changed))


def source_compatibility(old_repo, new_repo):
    """Fail closed on other numerical edits; allow the exact reviewed Ridge diff.

    New scheduler/cache files are separately recorded by the new Git identity.
    The old numerical module set must remain identical, except for this literal
    SVD replacement. This is stronger than accepting matching hyperparameters.
    """
    old_root = Path(old_repo) / "NK_Grid/src/aleatoric_nk_grid"
    new_root = Path(new_repo) / "NK_Grid/src/aleatoric_nk_grid"
    numerical = sorted(path.name for path in old_root.glob("*.py"))
    if "fold_local.py" not in numerical:
        raise QueueError("Missing original numerical source tree")
    files = {}
    for name in numerical:
        old = (old_root/name).read_text(encoding="utf-8")
        new = (new_root/name).read_text(encoding="utf-8")
        restored = new
        if name == "fold_local.py":
            restored = restored.replace("from .svd_fallback import ridge_svd\n", "")
            restored = restored.replace("        self.svd_fallback_count_ = 0\n", "")
            restored = restored.replace(
                "            (u, singular, vt), fallback_used = ridge_svd(train_X - center)\n            self.svd_fallback_count_ += int(fallback_used)",
                "            u, singular, vt = np.linalg.svd(train_X - center, full_matrices=False)")
        if restored != old:
            raise QueueError(f"Unreviewed numerical source change: {name}")
        files[name] = {"old_sha256": file_digest(old_root/name), "new_sha256": file_digest(new_root/name)}
    # Certificate also binds the complete fallback implementation for review.
    files["svd_fallback.py"] = {"new_sha256": file_digest(new_root/"svd_fallback.py")}
    import yaml
    parameter_files = {}
    for name in ("NK_Grid/model_params.yaml", "FFCWS/model_params.yaml", "SMR/model_params.yaml"):
        old_path, new_path = Path(old_repo)/name, Path(new_repo)/name
        old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
        new = yaml.safe_load(new_path.read_text(encoding="utf-8"))
        if (old.get("algorithm_version"), new.get("algorithm_version")) != (OLD_ALGORITHM, NEW_ALGORITHM):
            raise QueueError("Unexpected parameter-file algorithm transition")
        new["algorithm_version"] = OLD_ALGORITHM
        if old != new:
            raise QueueError(f"Unreviewed hyperparameter change: {name}")
        parameter_files[name] = {"old_sha256": file_digest(old_path), "new_sha256": file_digest(new_path)}
    return {"policy": "exact-scheduler-plus-conditional-gesvd-v1", "files": files,
            "parameter_files": parameter_files}


def validate_scientific_result(row, *, task_kind):
    if task_kind != "regression":
        raise QueueError("First migration implementation supports regression only")
    status = row.get("status")
    if status == "failed":
        return False
    if status == "skipped":
        if not row.get("error"):
            raise QueueError("Skipped result has no declared reason")
        return True
    if status != "ok":
        raise QueueError("Unknown result status")
    metrics = ("mse", "rmse", "mae")
    for metric in metrics:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueError(f"Invalid metric: {metric}") from exc
        if not math.isfinite(value) or value < 0:
            raise QueueError(f"Nonfinite/negative metric: {metric}")
    return True


def export_sealed(snapshot_path, *, target_arguments, output, tmp_dir):
    """Run only AFTER shutdown/sealing. No live-prefix imports are accepted."""
    from . import flat_task_table as legacy
    from .generation_control import GenerationValidationCache, classify_exact_afterany_target_read_only, frozen_sealed_history
    output = Path(output)
    if output.exists():
        raise QueueError("Use a new export directory")
    verified = legacy.verify_rounds(Path(snapshot_path), tmp_dir=Path(tmp_dir), **target_arguments)
    if verified["aborted_tasks"]:
        raise QueueError("Legacy generation contains protocol aborts")
    snapshot = legacy._load_snapshot(Path(snapshot_path))
    args = dict(target_arguments)
    args["prep_token"] = args.pop("expected_prep_token")
    for key in ("expected_previous_generation", "expected_pointer_version", "prep_job_id"):
        args.setdefault(key, None)
    analysis, execution, target = legacy._exact_target(snapshot, **args)
    root = Path(snapshot["output_dir"])
    cache = GenerationValidationCache()
    dispatch = classify_exact_afterany_target_read_only(root, target, cache=cache)
    if dispatch.kind != "sealed-generation":
        raise QueueError("Export requires a sealed generation")
    frontier = frozen_sealed_history(root, target, dispatch)
    if hashlib.sha256(legacy.canonical_json_bytes(list(frontier))).hexdigest() != verified["sealed_history_digest_sha256"]:
        raise QueueError("Verified history changed")
    output.mkdir(parents=True)
    rows_path = output / "results.jsonl"
    count = 0
    with rows_path.open("xb") as f:
        for marker, wal_path, scan in legacy._iter_sealed_wal_scans(root, analysis=analysis, frozen_frontier=frontier, validation_cache=cache):
            for record in scan.records:
                if record.event_type != legacy.TASK_RESULT:
                    continue
                _, rows = legacy.decode_public_rows(record.payload)
                for row in rows:
                    envelope = {"result": row, "origin": {"analysis_id": analysis.analysis_id,
                        "wal": str(wal_path), "sequence": record.sequence, "row_id": record.row_id,
                        "payload_sha256": digest(row)}}
                    f.write(canonical(envelope) + b"\n"); count += 1
        f.flush(); os.fsync(f.fileno())
    receipt = Path(verified["verification_receipt"])
    manifest = {"format": "sealed-legacy-export-v1", "source_analysis_id": analysis.analysis_id,
        "cell_spec": analysis.payload["cell_execution_spec"], "source_snapshot_sha256": file_digest(snapshot_path),
        "public_columns": list(analysis.payload.get("public_result_schema", {}).get("columns", ())),
        "sealed_history_digest_sha256": verified["sealed_history_digest_sha256"],
        "verification_receipt_sha256": file_digest(receipt), "rows": count,
        "results_sha256": file_digest(rows_path), "sealed": True}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def import_bundle(queue, bundle, *, new_spec, certificate, expected_manifest_sha256,
                  task_kind="regression"):
    """Offline operation, two-pass preflight before any completed-state writes.

    The caller must obtain expected_manifest_sha256 from the sealed exporter,
    not from a worker. This function is never exposed through worker RPC.
    """
    bundle = Path(bundle)
    manifest_path = bundle / "manifest.json"
    if file_digest(manifest_path) != expected_manifest_sha256:
        raise QueueError("Unexpected export manifest")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("format") != "sealed-legacy-export-v1" or manifest.get("sealed") is not True:
        raise QueueError("Source is not a sealed legacy export")
    if certificate.get("policy") != "exact-scheduler-plus-conditional-gesvd-v1" or not certificate.get("files"):
        raise QueueError("Source compatibility certificate required")
    assert_compatible_specs(manifest["cell_spec"], new_spec, certificate=certificate)
    identity = queue.manifest["identity"]
    if identity.get("cell_spec") != new_spec or identity.get("compatibility_certificate_sha256") != digest(certificate):
        raise QueueError("Queue identity does not bind the new spec/certificate")
    rows_path = bundle / "results.jsonl"
    if file_digest(rows_path) != manifest["results_sha256"]:
        raise QueueError("Export rows changed")
    # SQLite temporary table keeps uniqueness/conflict checking bounded in RAM.
    with queue.mutex:
        if queue.stats().get("leased", 0):
            raise QueueError("Stop workers before importing")
        queue.db.execute("DROP TABLE IF EXISTS migration_check")
        queue.db.execute("CREATE TEMP TABLE migration_check(id TEXT PRIMARY KEY,result TEXT,origin TEXT)")
        count = 0; failed = 0
        with rows_path.open("rb") as f:
            for line in f:
                envelope = json.loads(line); row = envelope["result"]; origin = envelope["origin"]
                task = ModelTask(int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"]), str(row["model"]))
                queue.validate_result(task.id, row)
                if origin.get("analysis_id") != manifest["source_analysis_id"] or origin.get("payload_sha256") != digest(row):
                    raise QueueError("Legacy result provenance mismatch")
                count += 1
                if not validate_scientific_result(row, task_kind=task_kind):
                    failed += 1; continue
                prior = queue.db.execute("SELECT result FROM migration_check WHERE id=?", (task.id,)).fetchone()
                encoded = canonical(row).decode()
                if prior is not None and prior[0] != encoded:
                    raise QueueError("Conflicting old results")
                complete_origin = {**origin, "export_manifest_sha256": expected_manifest_sha256,
                                   "compatibility_certificate_sha256": digest(certificate)}
                current = queue._get(task.id)
                if current["state"] == "done":
                    if current["result"] != encoded or current["origin"] != canonical(complete_origin).decode():
                        raise QueueError("Existing imported result conflicts")
                elif current["state"] != "pending" or current["attempt"]:
                    raise QueueError("Import would overlap executed tasks")
                queue.db.execute("INSERT OR IGNORE INTO migration_check VALUES(?,?,?)",
                                 (task.id, encoded, canonical(complete_origin).decode()))
                if count % 4096 == 0:
                    queue.db.commit()
        if count != manifest["rows"]:
            raise QueueError("Export row count mismatch")
        queue.db.commit()
        imported = 0
        cursor = queue.db.execute("SELECT id,result,origin FROM migration_check ORDER BY id")
        while True:
            batch = cursor.fetchmany(512)
            if not batch:
                break
            for row in batch:
                imported += int(queue.import_result(row[0], json.loads(row[1]), json.loads(row[2])))
        queue.db.execute("DROP TABLE migration_check"); queue.db.commit()
        receipt = {"imported": imported, "failed_records_left_pending": failed,
                   "source_rows": count, "export_manifest_sha256": expected_manifest_sha256,
                   "compatibility_certificate_sha256": digest(certificate), "queue_id": queue.queue_id}
        atomic_json(queue.root / "migration-receipt.json", receipt)
        return receipt

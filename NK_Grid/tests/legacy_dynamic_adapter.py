"""Retired CSV dynamic queue reference used only by compatibility tests.

This module deliberately lives under ``tests/``.  No production import may
reach the per-task CSV/attempt implementation after the WAL migration.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import replace
import sys
from pathlib import Path
from typing import Iterable, Mapping

import aleatoric_nk_grid.flat_task_table as ft
from aleatoric_nk_grid.nk_grid import run_nk_grid

read_task_table = ft.read_task_table


def manifest_path(output: Path) -> Path:
    """Test-only CSV adapter manifest locator (the production helper is retired)."""
    return Path(output).with_suffix(".manifest.json")


def _read_csv_keys(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Test-only CSV reader retained with the retired compatibility adapter."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _load_snapshot(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != ft.TABLE_FORMAT_VERSION
        or payload.get("result_store_format") != "retired-test-csv-adapter-v1"
    ):
        return ft._load_snapshot(path)
    return payload


def _append_attempt(
    path: Path, *, round_index: int, worker_index: int, sequence: int, row_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "round": round_index,
            "worker_index": worker_index,
            "sequence": sequence,
            "row_id": row_id,
        },
        sort_keys=True,
    ) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def _sweep_slice_temporaries(output: Path, *, round_index: int, worker_index: int) -> None:
    prefix = f".{output.stem}.round-{round_index}.worker-{worker_index}."
    for temporary in output.parent.glob(prefix + "*.csv"):
        temporary.unlink(missing_ok=True)
        manifest_path(temporary).unlink(missing_ok=True)


def _config_from_json(payload: Mapping[str, object]):
    values = dict(payload)
    for field in ("schema", "out", "model_params"):
        values[field] = Path(str(values[field]))
    values["models"] = tuple(str(value) for value in values["models"])
    for field in ("n_grid", "k_grid"):
        if values.get(field) is not None:
            values[field] = tuple(int(value) for value in values[field])
    if values.get("repeat_plan") is not None:
        values["repeat_plan"] = tuple(
            (int(pair[0]), int(pair[1])) for pair in values["repeat_plan"]
        )
    return ft.NKGridConfig(**values)


def _index_queue_completed_shards(connection: sqlite3.Connection, output_dir: Path) -> int:
    statement = f"INSERT OR IGNORE INTO completed ({ft._QUEUE_KEY_SQL}) VALUES (?, ?, ?, ?, ?)"
    for path in sorted(Path(output_dir).glob("round-*/worker-*.csv")):
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("status") in ft.TERMINAL_STATUSES:
                    connection.execute(statement, ft._csv_key(row))
        connection.commit()
    return int(connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])


def _index_queue_attempts(connection: sqlite3.Connection, output_dir: Path) -> None:
    statement = (
        "INSERT INTO attempts (execution_plan_id, round_index, submission_generation, "
        "worker_index, sequence, row_id) VALUES (?, ?, ?, ?, ?, ?)"
    )
    for path in sorted(Path(output_dir).glob("round-*/attempts/worker-*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    connection.execute(statement, (
                        "__legacy__", int(item["round"]), "__legacy__",
                        int(item["worker_index"]), int(item["sequence"]),
                        str(item["row_id"]),
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        connection.commit()


def prepare_round(
    snapshot_path: Path, *, round_index: int, prep_token: str,
    tmp_dir: Path | None = None, **_: object,
) -> dict[str, object]:
    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"]))
    workers = int(payload["workers"])
    table_path = Path(str(payload["task_table"]))
    round_dir = output_dir / f"round-{round_index}"
    round_dir.mkdir(parents=True, exist_ok=True)
    with ft._queue_index(payload, explicit_tmp_dir=tmp_dir, phase="preparation") as connection:
        ft._index_queue_expected_design(connection, table_path)
        completed = _index_queue_completed_shards(connection, output_dir)
        _index_queue_attempts(connection, output_dir)
        ft._index_completed_rows(connection)
        ft._classify_queue_attempts(connection)
        ft._queue_incomplete_rows(connection)
        todo_rows = ft._stage_queue_todo(connection, table_path, workers=workers)
        assignment, assigned_rows, _ = ft._write_assignment_from_queue_index(
            round_dir / "assignment.parquet", connection, workers=workers
        )
        crashed = ft._queue_row_ids(connection, "crashed_rows")
        too_long = ft._queue_row_ids(connection, "too_long_rows")
    stats: dict[str, object] = {
        "round": round_index,
        "workers": workers,
        "todo_rows": todo_rows,
        "assigned_rows": assigned_rows,
        "completed_model_keys": completed,
        "crashed_row_ids": crashed,
        "too_long_row_ids": too_long,
        "assignment": str(assignment),
        "prep_token": prep_token,
    }
    ft.write_json_atomic(round_dir / "crashed.json", {"round": round_index, "row_ids": crashed})
    ft.write_json_atomic(round_dir / "too-long.json", {"round": round_index, "row_ids": too_long})
    ft.write_json_atomic(round_dir / "prep.json", stats)
    ft.write_json_atomic(
        round_dir / "assignment.ready.json",
        {
            "format_version": ft.TABLE_FORMAT_VERSION,
            "round": round_index,
            "workers": workers,
            "assignment": str(assignment.resolve()),
            "prep_token": prep_token,
        },
    )
    return stats


def run_slice(
    snapshot_path: Path, *, round_index: int, worker_index: int,
    expected_prep_token: str, **_: object,
) -> Path:
    payload = _load_snapshot(snapshot_path)
    workers = int(payload["workers"])
    if not 0 <= worker_index < workers:
        raise IndexError("worker_index is outside the frozen worker count")
    round_dir = Path(str(payload["output_dir"])) / f"round-{round_index}"
    ready_path = round_dir / "assignment.ready.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"assignment is not ready for round {round_index}; prep may have failed: {ready_path}"
        ) from exc
    if ready.get("prep_token") != ft._validated_prep_token(expected_prep_token):
        raise RuntimeError(
            f"assignment readiness record is stale for round {round_index}; prep may have failed"
        )
    assignment = Path(str(ready["assignment"]))
    rows = ft.read_row_group(assignment, worker_index)
    output = round_dir / f"worker-{worker_index}.csv"
    _sweep_slice_temporaries(output, round_index=round_index, worker_index=worker_index)
    header: list[str] | None = None
    materialized: list[dict[str, str]] = []
    if output.exists():
        header, materialized = _read_csv_keys(output)
    by_key = {ft._csv_key(row): row for row in materialized}
    completed = {key for key, row in by_key.items() if row.get("status") in ft.TERMINAL_STATUSES}
    config = _config_from_json(payload["config"])
    for sequence, row in enumerate(ft.pending_rows(rows, completed)):
        _append_attempt(
            round_dir / "attempts" / f"worker-{worker_index}.jsonl",
            round_index=round_index, worker_index=worker_index,
            sequence=sequence, row_id=row.row_id,
        )
        row_out = round_dir / f".worker-{worker_index}.{row.row_id}.csv"
        row_config = replace(
            config, out=row_out, models=row.models,
            n_grid=(row.n_samples,), k_grid=(row.k_features,),
            n_seeds=1, n_draws=1,
            repeat_plan=((row.seed, row.draw),), n_jobs=1,
        )
        run_nk_grid(
            row_config, execution_pairs=((row.seed, row.draw),),
            exact_output_path=True, defer_failure_policy=True,
        )
        current_header, current_rows = _read_csv_keys(row_out)
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("worker rows produced inconsistent CSV headers")
        for current in current_rows:
            by_key[ft._csv_key(current)] = current
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header or ()))
            writer.writeheader()
            writer.writerows(by_key[key] for key in sorted(by_key))
        ft.write_json_atomic(
            manifest_path(output),
            {
                "format_version": ft.TABLE_FORMAT_VERSION,
                "execution": {
                    "mode": "slice",
                    "round": round_index,
                    "worker_index": worker_index,
                },
                "completion": {
                    "expected_rows": len(ft.expected_model_keys(rows)),
                    "materialized_rows": len(by_key),
                },
            },
        )
        row_out.unlink(missing_ok=True)
        manifest_path(row_out).unlink(missing_ok=True)
    if header is None and rows:
        raise RuntimeError("slice produced no rows")
    return output


def verify_rounds(snapshot_path: Path, *, tmp_dir: Path | None = None, **_: object) -> dict[str, object]:
    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"]))
    table_path = Path(str(payload["task_table"]))
    with ft._queue_index(payload, explicit_tmp_dir=tmp_dir, phase="verification") as connection:
        expected = ft._index_queue_expected_design(connection, table_path)
        complete = _index_queue_completed_shards(connection, output_dir)
        _index_queue_attempts(connection, output_dir)
        ft._index_completed_rows(connection)
        ft._classify_queue_attempts(connection)
        result = {
            "expected_model_keys": expected,
            "completed_model_keys": complete,
            "missing_model_keys": ft._queue_missing_model_keys(connection),
            "crashed_row_ids": ft._queue_row_ids(connection, "crashed_rows"),
            "too_long_row_ids": ft._queue_row_ids(connection, "too_long_rows"),
        }
    ft.write_json_atomic(output_dir / "verification.json", result)
    return result


def classify_attempts(
    records: Iterable[Mapping[str, int | str]], *, completed_row_ids: Iterable[str] = (),
) -> tuple[set[str], set[str]]:
    attempts = sorted(
        (dict(record) for record in records),
        key=lambda record: (int(record["round"]), int(record["worker_index"]), int(record["sequence"])),
    )
    crashed: set[str] = set()
    finals: dict[tuple[int, int], tuple[int, str]] = {}
    for record in attempts:
        key = (int(record["round"]), int(record["worker_index"]))
        prior = finals.get(key)
        if prior is not None:
            crashed.add(prior[1])
        if prior is None or int(record["sequence"]) > prior[0]:
            finals[key] = (int(record["sequence"]), str(record["row_id"]))
    final_rounds: dict[str, set[int]] = {}
    for (round_index, _), (_, row_id) in finals.items():
        final_rounds.setdefault(row_id, set()).add(round_index)
    too_long = {
        row_id for row_id, rounds in final_rounds.items()
        if any({round_index, round_index + 1, round_index + 2}.issubset(rounds)
               for round_index in rounds)
    }
    completed = set(completed_row_ids)
    return crashed - completed, too_long - completed


def finalize_slice_shards(table_path: Path, worker_outputs: Iterable[Path], output: Path) -> Path:
    expected = ft.expected_model_keys(ft.read_task_table(table_path))
    seen: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    header: list[str] | None = None
    for path in worker_outputs:
        if not Path(path).exists():
            continue
        current_header, rows = _read_csv_keys(Path(path))
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("worker CSV headers differ")
        for row in rows:
            key = ft._csv_key(row)
            if key not in expected or key in seen:
                raise ValueError("merged output has an invalid or duplicate key")
            seen[key] = row
    if expected - set(seen):
        raise ValueError("merged output is missing expected keys")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or (), lineterminator="\n")
        writer.writeheader()
        writer.writerows(seen[key] for key in sorted(seen))
    return output


def finalize_snapshot(snapshot_path: Path, *, tmp_dir: Path | None = None, **_: object) -> dict[str, object]:
    """Reference finalizer for retired CSV fixtures; never imported by production."""
    started = time.perf_counter()
    snapshot = _load_snapshot(snapshot_path)
    table_path = Path(str(snapshot["task_table"]))
    output_dir = Path(str(snapshot["output_dir"]))
    config = snapshot["config"]
    output = Path(str(config["out"])).expanduser().resolve()
    shards = tuple(sorted(output_dir.glob("round-*/worker-*.csv")))
    tmp_base = ft._resolve_finalization_tmp_base(snapshot, tmp_dir)
    available, estimated = ft._preflight_finalization_space(tmp_base, table_path, shards, sealed_wal_bytes=0)
    run_dir = Path(tempfile.mkdtemp(prefix="nk-grid-finalize-", dir=tmp_base))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(run_dir / "index.sqlite")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")
        ft._create_finalization_tables(connection)
        expected = ft._insert_expected_design(connection, table_path)
        header, rows_read, duplicate_rows = ft._ingest_result_shards(connection, shards)
        final_rows, failed_overridden, duplicate_keys = ft._validate_finalization_database(connection, expected)
        ft._write_final_csv(connection, header, output)
        receipt = {
            "format_version": ft.FINALIZATION_FORMAT_VERSION,
            "status": "complete",
            "created_at_utc": ft.utc_now(),
            "backend": "sqlite_streaming",
            "input_shards": len(shards),
            "wal_result_rows": 0,
            "frozen_wal_input_bytes": 0,
            "analysis_id": None,
            "execution_plan_ids": [],
            "verification_receipt": None,
            "rows_read": rows_read,
            "final_rows": final_rows,
            "expected_model_keys": expected,
            "historical_failed_rows_overridden": failed_overridden,
            "duplicate_terminal_keys": duplicate_keys,
            "duplicate_terminal_rows": duplicate_rows,
            "temporary_directory": str(run_dir),
            "temporary_available_bytes": available,
            "estimated_temporary_bytes": estimated,
            "temporary_bytes_used": ft._temporary_directory_bytes(run_dir),
            "wall_time_seconds": time.perf_counter() - started,
            "task_table": str(table_path.resolve()),
            "final_output": str(output),
        }
        ft.write_json_atomic(ft.finalization_manifest_path(output), receipt)
        return receipt
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(run_dir, ignore_errors=True)


def _completed_keys(output_dir: Path) -> set[tuple[str, int, int, int, int]]:
    completed: set[tuple[str, int, int, int, int]] = set()
    for path in sorted(Path(output_dir).glob("round-*/worker-*.csv")):
        _, rows = _read_csv_keys(path)
        completed.update(ft._csv_key(row) for row in rows if row.get("status") in {"ok", "skipped"})
    return completed


def _attempt_records(output_dir: Path) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for path in sorted(Path(output_dir).glob("round-*/attempts/worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                records.append({
                    "round": int(item["round"]),
                    "worker_index": int(item["worker_index"]),
                    "sequence": int(item["sequence"]),
                    "row_id": str(item["row_id"]),
                })
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return records


def main(argv: list[str]) -> None:
    """Tiny CLI shim for tests that exercise the retired CSV reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "finalize"))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_rounds(args.snapshot, tmp_dir=args.tmp_dir)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["missing_model_keys"] or result["crashed_row_ids"] or result["too_long_row_ids"]:
            print(
                "verification incomplete: "
                + "; ".join((
                    f"missing_model_keys={result['missing_model_keys']}",
                    f"crashed_row_ids={len(result['crashed_row_ids'])}",
                    f"too_long_row_ids={len(result['too_long_row_ids'])}",
                )),
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(3)
        return
    print(json.dumps(finalize_snapshot(args.snapshot, tmp_dir=args.tmp_dir), sort_keys=True))

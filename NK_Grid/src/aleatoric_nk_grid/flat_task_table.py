"""Immutable task tables and restartable dynamic-work-queue workers.

The execution rows deliberately have no cost estimate and no precomputed
chunk.  A worker receives one modulo-stratified Parquet row group for one
round, materialises every completed cell group immediately, and a later round
derives its work solely from those durable artefacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .experiment import manifest_path, write_json_atomic
from .nk_grid import NKGridConfig, _process_peak_rss_bytes, resolve_repeat_pairs, run_nk_grid


TABLE_FORMAT_VERSION = 2
TABLE_COLUMNS = ("row_id", "seed", "draw", "N", "K", "group", "models")


@dataclass(frozen=True)
class TaskRow:
    row_id: str
    seed: int
    draw: int
    n_samples: int
    k_features: int
    group: str
    models: tuple[str, ...]

    @property
    def key(self) -> tuple[int, int, int, int, str]:
        return (self.seed, self.draw, self.n_samples, self.k_features, self.group)


def pairs_for(
    n_samples: int, k_features: int, repeat_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return repeat pairs for one grid point without changing their identity."""
    del n_samples, k_features
    return tuple((int(seed), int(draw)) for seed, draw in repeat_pairs)


def execution_groups(models: Sequence[str], *, k_features: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return indivisible preprocessing groups; Super Learner stays imputed."""
    del k_features
    selected = tuple(str(model) for model in models)
    if len(selected) != len(set(selected)):
        raise ValueError("models must not contain duplicates")
    passthrough = tuple(model for model in selected if model in {"lightgbm", "xgboost"})
    imputed = tuple(model for model in selected if model not in passthrough)
    groups: list[tuple[str, tuple[str, ...]]] = []
    if imputed:
        groups.append(("imputed_core", imputed))
    if passthrough:
        groups.append(("passthrough", passthrough))
    if not groups:
        raise ValueError("models must not be empty")
    return tuple(groups)


GROUP_PREPROCESS_MODES: Mapping[str, str] = {
    "imputed_core": "imputed", "passthrough": "passthrough",
}


def preprocess_mode_for_group(group: str) -> str:
    try:
        return GROUP_PREPROCESS_MODES[str(group)]
    except KeyError:
        raise ValueError(f"unknown execution group {group!r}") from None


def _row_id(seed: int, draw: int, n_samples: int, k_features: int, group: str, models: tuple[str, ...]) -> str:
    raw = json.dumps([seed, draw, n_samples, k_features, group, list(models)], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_rows(
    config: NKGridConfig,
    *,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    pairs_provider: Callable[[int, int], Sequence[tuple[int, int]]] | None = None,
) -> tuple[TaskRow, ...]:
    """Build the immutable deterministic cell-group design."""
    repeat_pairs = resolve_repeat_pairs(config)
    provider = pairs_provider or (lambda n, k: pairs_for(n, k, repeat_pairs))
    rows: list[TaskRow] = []
    for k_features in sorted({int(value) for value in k_grid}):
        for n_samples in sorted({int(value) for value in n_grid}):
            point_pairs = tuple((int(seed), int(draw)) for seed, draw in provider(n_samples, k_features))
            if not point_pairs:
                raise ValueError("pairs_for must return at least one pair")
            for seed, draw in point_pairs:
                for group, group_models in execution_groups(config.models, k_features=k_features):
                    rows.append(TaskRow(
                        row_id=_row_id(seed, draw, n_samples, k_features, group, group_models),
                        seed=seed, draw=draw, n_samples=n_samples, k_features=k_features,
                        group=group, models=group_models,
                    ))
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("task table would contain duplicate row IDs")
    return tuple(rows)


def _arrow_table(rows: Sequence[TaskRow]) -> pa.Table:
    return pa.table({
        "row_id": [row.row_id for row in rows],
        "seed": [row.seed for row in rows], "draw": [row.draw for row in rows],
        "N": [row.n_samples for row in rows], "K": [row.k_features for row in rows],
        "group": [row.group for row in rows], "models": [list(row.models) for row in rows],
    })


def _empty_arrow_table() -> pa.Table:
    return pa.table({
        "row_id": pa.array([], type=pa.string()), "seed": pa.array([], type=pa.int64()),
        "draw": pa.array([], type=pa.int64()), "N": pa.array([], type=pa.int64()),
        "K": pa.array([], type=pa.int64()), "group": pa.array([], type=pa.string()),
        "models": pa.array([], type=pa.list_(pa.string())),
    })


def _write_table_groups(path: Path, groups: Sequence[Sequence[TaskRow]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    schema = _arrow_table(groups[0]).schema if groups and groups[0] else _empty_arrow_table().schema
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for group in groups:
            writer.write_table(_arrow_table(group) if group else _empty_arrow_table())
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path


def write_task_table(path: Path, rows: Sequence[TaskRow], *, rows_per_group: int = 100_000) -> Path:
    """Write an immutable v2 main table in fixed-size streaming row groups."""
    if not rows:
        raise ValueError("task table must not be empty")
    if rows_per_group < 1:
        raise ValueError("rows_per_group must be positive")
    ordered = tuple(sorted(rows, key=lambda row: (row.k_features, row.n_samples, row.seed, row.draw, row.group, row.row_id)))
    return _write_table_groups(path, [ordered[start:start + rows_per_group] for start in range(0, len(ordered), rows_per_group)])


def _task_rows(table: pa.Table) -> tuple[TaskRow, ...]:
    payload = table.to_pydict()
    required = set(TABLE_COLUMNS)
    actual = set(payload)
    if required != actual:
        if {"est_cost", "chunk_id"}.issubset(actual):
            raise ValueError("unsupported v1 flat-task table; rebuild a v2 dynamic-work-queue table")
        raise ValueError("unsupported flat-task table schema")
    return tuple(TaskRow(
        row_id=str(payload["row_id"][index]), seed=int(payload["seed"][index]),
        draw=int(payload["draw"][index]), n_samples=int(payload["N"][index]),
        k_features=int(payload["K"][index]), group=str(payload["group"][index]),
        models=tuple(str(value) for value in payload["models"][index]),
    ) for index in range(table.num_rows))


def read_row_group(path: Path, row_group: int) -> tuple[TaskRow, ...]:
    source = pq.ParquetFile(Path(path), memory_map=True)
    if not 0 <= row_group < source.num_row_groups:
        raise IndexError(f"row_group must be between 0 and {source.num_row_groups - 1}")
    return _task_rows(source.read_row_group(row_group))


def read_task_table(path: Path) -> tuple[TaskRow, ...]:
    source = pq.ParquetFile(Path(path), memory_map=True)
    return tuple(row for group in range(source.num_row_groups) for row in read_row_group(path, group))


def _synthetic_arrow_table(count: int) -> pa.Table:
    zeros = np.zeros(count, dtype=np.int32)
    return pa.table({
        "row_id": pa.array(["synthetic"]).take(pa.array(zeros)), "seed": zeros, "draw": zeros,
        "N": np.full(count, 100, dtype=np.int32), "K": np.full(count, 10, dtype=np.int32),
        "group": pa.array(["imputed_core"]).take(pa.array(zeros)),
        "models": pa.array([["ols", "ridge"]]).take(pa.array(zeros)),
    })


def write_synthetic_task_table(path: Path, *, row_count: int, rows_per_group: int = 100_000) -> Path:
    if row_count < 1 or rows_per_group < 1:
        raise ValueError("row_count and rows_per_group must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, _synthetic_arrow_table(1).schema, compression="zstd")
    try:
        for start in range(0, row_count, rows_per_group):
            writer.write_table(_synthetic_arrow_table(min(rows_per_group, row_count - start)))
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path


def _read_group_rss_worker(path: str, row_group: int, connection) -> None:
    try:
        before = _process_peak_rss_bytes()
        rows = read_row_group(Path(path), row_group)
        after = _process_peak_rss_bytes()
        connection.send({"rows": len(rows), "rss_delta_bytes": max(0, after - before)})
    finally:
        connection.close()


def measure_group_read_rss(path: Path, *, row_group: int = 0) -> dict[str, int]:
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_read_group_rss_worker, args=(str(Path(path)), int(row_group), child),
    )
    process.start(); child.close()
    payload = parent.recv(); process.join(); parent.close()
    if process.exitcode != 0:
        raise RuntimeError(f"RSS worker failed with exit code {process.exitcode}")
    return {"rows": int(payload["rows"]), "rss_delta_bytes": int(payload["rss_delta_bytes"])}


def expected_model_keys(rows: Iterable[TaskRow]) -> set[tuple[str, int, int, int, int]]:
    return {(model, row.seed, row.draw, row.n_samples, row.k_features) for row in rows for model in row.models}


def pending_rows(rows: Iterable[TaskRow], completed: Iterable[tuple[str, int, int, int, int]]) -> tuple[TaskRow, ...]:
    done = set(completed)
    return tuple(row for row in rows if any((model, row.seed, row.draw, row.n_samples, row.k_features) not in done for model in row.models))


def _read_csv_keys(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _csv_key(row: Mapping[str, str]) -> tuple[str, int, int, int, int]:
    return (str(row["model"]), int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"]))


def _write_materialized_rows(
    output: Path, header: Sequence[str], by_key: Mapping[tuple[str, int, int, int, int], Mapping[str, str]],
    *, execution: Mapping[str, object], expected_rows: int,
) -> None:
    """Atomically publish a whole worker shard after every successful cell group."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(by_key[key] for key in sorted(by_key))
    os.replace(temporary, output)
    write_json_atomic(manifest_path(output), {
        "format_version": TABLE_FORMAT_VERSION, "execution": dict(execution),
        "completion": {"expected_rows": expected_rows, "materialized_rows": len(by_key)},
    })


def _append_attempt(path: Path, *, round_index: int, worker_index: int, sequence: int, row_id: str) -> None:
    """Append one bounded JSON record before starting a row.

    The record is below PIPE_BUF and therefore an O_APPEND write cannot be
    interleaved with another worker's record (workers have distinct files in
    normal operation).  A trailing malformed record after SIGKILL is ignored
    by the reader below.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"round": round_index, "worker_index": worker_index, "sequence": sequence, "row_id": row_id}, sort_keys=True) + "\n"
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


def run_slice(snapshot_path: Path, *, round_index: int, worker_index: int) -> Path:
    """Run one worker's fixed modulo slice, publishing after every cell group."""
    payload = _load_snapshot(snapshot_path)
    workers = int(payload["workers"])
    if not 0 <= worker_index < workers:
        raise IndexError("worker_index is outside the frozen worker count")
    assignment = _round_directory(payload, round_index) / "assignment.parquet"
    rows = read_row_group(assignment, worker_index)
    output = _round_directory(payload, round_index) / f"worker-{worker_index}.csv"
    _sweep_slice_temporaries(output, round_index=round_index, worker_index=worker_index)
    header: list[str] | None = None
    materialized: list[dict[str, str]] = []
    if output.exists():
        header, materialized = _read_csv_keys(output)
    by_key = {_csv_key(row): row for row in materialized}
    if len(by_key) != len(materialized):
        raise ValueError("worker output contains duplicate keys")
    completed = {key for key, row in by_key.items() if row.get("status") in {"ok", "skipped"}}
    expected = expected_model_keys(rows)
    config = _config_from_json(payload["config"])
    for sequence, row in enumerate(pending_rows(rows, completed)):
        _append_attempt(_round_directory(payload, round_index) / "attempts" / f"worker-{worker_index}.jsonl", round_index=round_index, worker_index=worker_index, sequence=sequence, row_id=row.row_id)
        row_out = output.parent / f".{output.stem}.round-{round_index}.worker-{worker_index}.{row.row_id}.csv"
        # A worker is one CPU.  Keeping this at one also preserves deterministic
        # estimator behaviour; Super Learner remains in imputed_core and its
        # internal model_n_jobs is consequently one as well.
        row_config = replace(
            config, out=row_out, models=row.models, n_grid=(row.n_samples,), k_grid=(row.k_features,),
            n_seeds=1, n_draws=1, repeat_plan=((row.seed, row.draw),), n_jobs=1,
        )
        run_nk_grid(row_config, execution_pairs=((row.seed, row.draw),), exact_output_path=True, defer_failure_policy=True)
        current_header, current_rows = _read_csv_keys(row_out)
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("worker rows produced inconsistent CSV headers")
        for current in current_rows:
            by_key[_csv_key(current)] = current
        _write_materialized_rows(output, header, by_key, execution={"mode": "slice", "round": round_index, "worker_index": worker_index}, expected_rows=len(expected))
        row_out.unlink(missing_ok=True)
        manifest_path(row_out).unlink(missing_ok=True)
    if header is None and rows:
        raise RuntimeError("slice produced no rows")
    if header is None:
        # Empty assignments deliberately have no output file; their manifest is
        # still useful to distinguish an idle worker from an unstarted one.
        write_json_atomic(manifest_path(output), {"format_version": TABLE_FORMAT_VERSION, "execution": {"mode": "slice", "round": round_index, "worker_index": worker_index}, "completion": {"expected_rows": 0, "materialized_rows": 0}})
    return output


def _round_directory(snapshot: Mapping[str, object], round_index: int) -> Path:
    if round_index < 1:
        raise ValueError("round_index must be >= 1")
    return Path(str(snapshot["output_dir"])) / f"round-{round_index}"


def _load_snapshot(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != TABLE_FORMAT_VERSION:
        if payload.get("format_version") == 1:
            raise ValueError("unsupported v1 flat-task snapshot; rebuild a v2 dynamic-work-queue plan")
        raise ValueError("unsupported flat-task snapshot")
    return payload


def _completed_keys(output_dir: Path) -> set[tuple[str, int, int, int, int]]:
    completed: set[tuple[str, int, int, int, int]] = set()
    for path in sorted(output_dir.glob("round-*/worker-*.csv")):
        _, rows = _read_csv_keys(path)
        completed.update(_csv_key(row) for row in rows if row.get("status") in {"ok", "skipped"})
    return completed


def _attempt_records(output_dir: Path) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for path in sorted(output_dir.glob("round-*/attempts/worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                records.append({"round": int(item["round"]), "worker_index": int(item["worker_index"]), "sequence": int(item["sequence"]), "row_id": str(item["row_id"])})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A Slurm SIGKILL can leave the diagnostic tail incomplete.
                continue
    return records


def classify_attempts(
    records: Iterable[Mapping[str, int | str]], *, completed_row_ids: Iterable[str] = (),
) -> tuple[set[str], set[str]]:
    """Return (crashed, too_long) using the two intentionally distinct rules."""
    attempts = sorted(
        (dict(record) for record in records),
        key=lambda record: (int(record["round"]), int(record["worker_index"]), int(record["sequence"])),
    )
    crashed: set[str] = set()
    finals: dict[tuple[int, int], tuple[int, str]] = {}
    for record in attempts:
        round_index = int(record["round"]); worker = int(record["worker_index"]); sequence = int(record["sequence"]); row_id = str(record["row_id"])
        prior = finals.get((round_index, worker))
        if prior is not None:
            crashed.add(prior[1])
        if prior is None or sequence > prior[0]:
            finals[(round_index, worker)] = (sequence, row_id)
    final_rounds: dict[str, set[int]] = {}
    for (round_index, _), (_, row_id) in finals.items():
        final_rounds.setdefault(row_id, set()).add(round_index)
    too_long = {
        row_id for row_id, rounds in final_rounds.items()
        if any({round_index, round_index + 1, round_index + 2}.issubset(rounds) for round_index in rounds)
    }
    completed = set(completed_row_ids)
    return crashed - completed, too_long - completed


def assign_rows_modulo(rows: Sequence[TaskRow], workers: int) -> tuple[tuple[TaskRow, ...], ...]:
    """Assign ordered todo rows by index modulo worker count, never by ranges."""
    if workers < 1:
        raise ValueError("workers must be positive")
    return tuple(tuple(rows[index] for index in range(worker, len(rows), workers)) for worker in range(workers))


def prepare_round(snapshot_path: Path, *, round_index: int) -> dict[str, object]:
    """Build exactly one row group per worker from durable results and attempts."""
    payload = _load_snapshot(snapshot_path)
    output_dir = Path(str(payload["output_dir"])); workers = int(payload["workers"])
    all_rows = read_task_table(Path(str(payload["task_table"])))
    completed_keys = _completed_keys(output_dir)
    completed_row_ids = {
        row.row_id for row in all_rows
        if expected_model_keys((row,)).issubset(completed_keys)
    }
    records = _attempt_records(output_dir)
    crashed, too_long = classify_attempts(records, completed_row_ids=completed_row_ids)
    # Crashed rows stay eligible.  Only the three-consecutive-round diagnosis
    # is excluded so it can be submitted separately with a longer limit.
    todo = [row for row in pending_rows(all_rows, completed_keys) if row.row_id not in too_long]
    groups = assign_rows_modulo(todo, workers)
    round_dir = _round_directory(payload, round_index); round_dir.mkdir(parents=True, exist_ok=True)
    assignment = _write_table_groups(round_dir / "assignment.parquet", groups)
    write_json_atomic(round_dir / "crashed.json", {"round": round_index, "row_ids": sorted(crashed)})
    write_json_atomic(round_dir / "too-long.json", {"round": round_index, "row_ids": sorted(too_long)})
    stats = {"round": round_index, "workers": workers, "todo_rows": len(todo), "assigned_rows": [len(group) for group in groups], "completed_model_keys": len(completed_keys), "crashed_row_ids": sorted(crashed), "too_long_row_ids": sorted(too_long), "assignment": str(assignment)}
    write_json_atomic(round_dir / "prep.json", stats)
    return stats


def verify_rounds(snapshot_path: Path) -> dict[str, object]:
    payload = _load_snapshot(snapshot_path)
    rows = read_task_table(Path(str(payload["task_table"])))
    completed = _completed_keys(Path(str(payload["output_dir"])))
    completed_row_ids = {row.row_id for row in rows if expected_model_keys((row,)).issubset(completed)}
    crashed, too_long = classify_attempts(_attempt_records(Path(str(payload["output_dir"]))), completed_row_ids=completed_row_ids)
    expected = expected_model_keys(rows)
    result = {"expected_model_keys": len(expected), "completed_model_keys": len(completed), "missing_model_keys": len(expected - completed), "crashed_row_ids": sorted(crashed), "too_long_row_ids": sorted(too_long)}
    write_json_atomic(Path(str(payload["output_dir"])) / "verification.json", result)
    return result


def finalize_slice_shards(table_path: Path, worker_outputs: Iterable[Path], output: Path) -> Path:
    """Merge dynamic worker shards after exact complete-design validation."""
    expected = expected_model_keys(read_task_table(table_path))
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
            key = _csv_key(row)
            if key not in expected:
                raise ValueError(f"merged output contains an out-of-design key: {key}")
            if key in seen:
                raise ValueError(f"merged output contains a duplicate key: {key}")
            seen[key] = row
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"merged output is missing {len(missing)} expected keys")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or [])
        writer.writeheader(); writer.writerows(seen[key] for key in sorted(seen))
    return output


@dataclass(frozen=True)
class ResourceRequest:
    cpus_per_task: int
    partition: str
    memory: str
    time_limit: str
    account: str
    constraint: str


def sbatch_resource_args(request: ResourceRequest) -> tuple[str, ...]:
    if request.cpus_per_task != 1:
        raise ValueError("dynamic workers must request exactly one CPU")
    if not all((request.partition, request.memory, request.time_limit, request.account)):
        raise ValueError("partition, memory, time_limit, and account are required")
    args = (f"--partition={request.partition}", "--cpus-per-task=1", f"--mem={request.memory}", f"--time={request.time_limit}", f"--account={request.account}")
    return args if request.constraint == "none" else (*args, f"--constraint={request.constraint}")


def _config_to_json(config: NKGridConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            payload[field.name] = str(value)
        elif isinstance(value, tuple):
            payload[field.name] = [list(item) if isinstance(item, tuple) else item for item in value]
        else:
            payload[field.name] = value
    return payload


def _config_from_json(payload: Mapping[str, object]) -> NKGridConfig:
    values = dict(payload)
    for field in ("schema", "out", "model_params"):
        values[field] = Path(str(values[field]))
    values["models"] = tuple(str(value) for value in values["models"])
    for field in ("n_grid", "k_grid"):
        if values.get(field) is not None:
            values[field] = tuple(int(value) for value in values[field])
    if values.get("repeat_plan") is not None:
        values["repeat_plan"] = tuple((int(pair[0]), int(pair[1])) for pair in values["repeat_plan"])
    return NKGridConfig(**values)


def write_work_snapshot(path: Path, *, table_path: Path, panel: str, config: NKGridConfig, output_dir: Path, workers: int) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    table = Path(table_path).resolve()
    if not read_task_table(table):
        raise ValueError("task table must contain at least one row")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"format_version": TABLE_FORMAT_VERSION, "panel": str(panel), "task_table": str(table), "config": _config_to_json(config), "output_dir": str(Path(output_dir).resolve()), "workers": int(workers)})
    os.chmod(path, 0o444)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dynamic NK-grid work slices")
    parser.add_argument("command", choices=("prep", "run", "verify"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args(argv)
    if args.command == "prep":
        if args.round is None: parser.error("prep requires --round")
        print(json.dumps(prepare_round(args.snapshot, round_index=args.round), sort_keys=True))
    elif args.command == "run":
        if args.round is None or args.worker_index is None: parser.error("run requires --round and --worker-index")
        print(run_slice(args.snapshot, round_index=args.round, worker_index=args.worker_index))
    else:
        print(json.dumps(verify_rounds(args.snapshot), sort_keys=True))


if __name__ == "__main__":
    main()

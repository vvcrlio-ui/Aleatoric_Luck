"""Immutable, cell-group task tables for Slurm-array NK-grid execution.

The table is deliberately independent of calibration.  Stage A only needs a
deterministic *relative* cost so that the layout, resume rules and worker
protocol can be exercised.  Stage B supplies the calibrated cost function and
resource values; this module refuses to invent either of them.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .experiment import manifest_path, write_json_atomic
from .nk_grid import NKGridConfig, resolve_repeat_pairs, run_nk_grid


TABLE_FORMAT_VERSION = 1
TABLE_COLUMNS = (
    "row_id", "seed", "draw", "N", "K", "group", "models", "est_cost",
    "chunk_id",
)
RESOURCE_CLASSES = ("serial", "super_learner")


@dataclass(frozen=True)
class TaskRow:
    row_id: str
    seed: int
    draw: int
    n_samples: int
    k_features: int
    group: str
    models: tuple[str, ...]
    est_cost: float = 0.0
    chunk_id: int = -1

    @property
    def key(self) -> tuple[int, int, int, int, str]:
        return (self.seed, self.draw, self.n_samples, self.k_features, self.group)


CostFunction = Callable[[TaskRow], float]


def pairs_for(
    n_samples: int,
    k_features: int,
    repeat_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return the repeat pairs for one grid point.

    Keeping N/K in the API is intentional: a later design can append repeats
    for selected points without renumbering any existing task.
    """
    del n_samples, k_features
    return tuple((int(seed), int(draw)) for seed, draw in repeat_pairs)


def execution_groups(
    models: Sequence[str],
    *,
    k_features: int,
    split_super_learner_min_k: int | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return indivisible preprocessing groups for one K.

    ``None`` is the Stage-A, calibration-free policy: Super Learner remains in
    the imputed group.  A calibrated threshold may split it in Stage B.
    """
    selected = tuple(str(model) for model in models)
    if len(selected) != len(set(selected)):
        raise ValueError("models must not contain duplicates")
    passthrough = tuple(model for model in selected if model in {"lightgbm", "xgboost"})
    imputed = tuple(model for model in selected if model not in passthrough)
    groups: list[tuple[str, tuple[str, ...]]] = []
    split_super = (
        split_super_learner_min_k is not None
        and k_features >= split_super_learner_min_k
        and "super_learner" in imputed
    )
    core = tuple(model for model in imputed if not (split_super and model == "super_learner"))
    if core:
        groups.append(("imputed_core", core))
    if split_super:
        groups.append(("super_learner", ("super_learner",)))
    if passthrough:
        groups.append(("passthrough", passthrough))
    if not groups:
        raise ValueError("models must not be empty")
    return tuple(groups)


def _row_id(seed: int, draw: int, n_samples: int, k_features: int, group: str, models: tuple[str, ...]) -> str:
    raw = json.dumps([seed, draw, n_samples, k_features, group, list(models)], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_rows(
    config: NKGridConfig,
    *,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    split_super_learner_min_k: int | None = None,
    pairs_provider: Callable[[int, int], Sequence[tuple[int, int]]] | None = None,
) -> tuple[TaskRow, ...]:
    """Build the complete, deterministic cell-group design in memory."""
    repeat_pairs = resolve_repeat_pairs(config)
    provider = pairs_provider or (lambda n, k: pairs_for(n, k, repeat_pairs))
    rows: list[TaskRow] = []
    for k_features in sorted({int(value) for value in k_grid}):
        for n_samples in sorted({int(value) for value in n_grid}):
            point_pairs = tuple((int(seed), int(draw)) for seed, draw in provider(n_samples, k_features))
            if not point_pairs:
                raise ValueError("pairs_for must return at least one pair")
            for seed, draw in point_pairs:
                for group, group_models in execution_groups(
                    config.models,
                    k_features=k_features,
                    split_super_learner_min_k=split_super_learner_min_k,
                ):
                    rows.append(TaskRow(
                        row_id=_row_id(seed, draw, n_samples, k_features, group, group_models),
                        seed=seed, draw=draw, n_samples=n_samples, k_features=k_features,
                        group=group, models=group_models,
                    ))
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("task table would contain duplicate row IDs")
    return tuple(rows)


def unit_cost(row: TaskRow) -> float:
    """Stage-A placeholder cost; B replaces this with a calibrated function."""
    del row
    return 1.0


def pack_lpt(rows: Iterable[TaskRow], *, budget: float, cost_function: CostFunction = unit_cost) -> tuple[TaskRow, ...]:
    """Deterministic LPT packing; an over-budget row is its own chunk."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    measured = [replace(row, est_cost=float(cost_function(row)), chunk_id=-1) for row in rows]
    if any(row.est_cost < 0 for row in measured):
        raise ValueError("cost function must return non-negative values")
    ordered = sorted(measured, key=lambda row: (-row.est_cost, row.row_id))
    packed: list[TaskRow] = []
    chunk_id = 0
    current_cost = 0.0
    for row in ordered:
        if current_cost and current_cost + row.est_cost > budget:
            chunk_id += 1
            current_cost = 0.0
        packed.append(replace(row, chunk_id=chunk_id))
        current_cost += row.est_cost
        # An over-budget task has no neighbour.
        if row.est_cost > budget:
            chunk_id += 1
            current_cost = 0.0
    return tuple(sorted(packed, key=lambda row: (row.chunk_id, row.row_id)))


def _arrow_table(rows: Sequence[TaskRow]) -> pa.Table:
    return pa.table({
        "row_id": [row.row_id for row in rows],
        "seed": [row.seed for row in rows], "draw": [row.draw for row in rows],
        "N": [row.n_samples for row in rows], "K": [row.k_features for row in rows],
        "group": [row.group for row in rows], "models": [list(row.models) for row in rows],
        "est_cost": [row.est_cost for row in rows], "chunk_id": [row.chunk_id for row in rows],
    })


def write_task_table(path: Path, rows: Sequence[TaskRow]) -> Path:
    """Write a read-only Parquet table with exactly one row group per chunk."""
    path = Path(path)
    if not rows:
        raise ValueError("task table must not be empty")
    if any(row.chunk_id < 0 for row in rows):
        raise ValueError("task rows must be packed before they are written")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.chunk_id, row.row_id))
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, _arrow_table(ordered[:1]).schema, compression="zstd")
    try:
        for chunk_id in sorted({row.chunk_id for row in ordered}):
            writer.write_table(_arrow_table([row for row in ordered if row.chunk_id == chunk_id]))
    finally:
        writer.close()
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    return path


def read_chunk(path: Path, chunk_id: int) -> tuple[TaskRow, ...]:
    """Read only the single Parquet row group assigned to a worker."""
    source = pq.ParquetFile(Path(path), memory_map=True)
    if not 0 <= chunk_id < source.num_row_groups:
        raise IndexError(f"chunk_id must be between 0 and {source.num_row_groups - 1}")
    table = source.read_row_group(chunk_id)
    payload = table.to_pydict()
    rows = tuple(TaskRow(
        row_id=str(payload["row_id"][index]), seed=int(payload["seed"][index]),
        draw=int(payload["draw"][index]), n_samples=int(payload["N"][index]),
        k_features=int(payload["K"][index]), group=str(payload["group"][index]),
        models=tuple(str(value) for value in payload["models"][index]),
        est_cost=float(payload["est_cost"][index]), chunk_id=int(payload["chunk_id"][index]),
    ) for index in range(table.num_rows))
    if any(row.chunk_id != chunk_id for row in rows):
        raise ValueError("task table row group does not match its chunk_id")
    return rows


def expected_model_keys(rows: Iterable[TaskRow]) -> set[tuple[str, int, int, int, int]]:
    return {(model, row.seed, row.draw, row.n_samples, row.k_features) for row in rows for model in row.models}


def pending_rows(rows: Iterable[TaskRow], completed: Iterable[tuple[str, int, int, int, int]]) -> tuple[TaskRow, ...]:
    done = set(completed)
    return tuple(row for row in rows if any((model, row.seed, row.draw, row.n_samples, row.k_features) not in done for model in row.models))


def _read_csv_keys(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def run_chunk(table_path: Path, chunk_id: int, config: NKGridConfig, *, output: Path) -> Path:
    """Run one chunk sequentially and resume by the immutable model-key set.

    The existing cell runner is reused unchanged.  Stage A intentionally keeps
    its fit/preprocessing path intact; each task row is an indivisible group.
    """
    rows = read_chunk(table_path, chunk_id)
    output = Path(output)
    header: list[str] | None = None
    materialized: list[dict[str, str]] = []
    if output.exists():
        header, materialized = _read_csv_keys(output)
    by_key: dict[tuple[str, int, int, int, int], dict[str, str]] = {
        (row["model"], int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"])): row
        for row in materialized
    }
    if len(by_key) != len(materialized):
        raise ValueError("chunk output contains duplicate keys")
    completed = {
        (row["model"], int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"]))
        for row in by_key.values() if row.get("status") in {"ok", "skipped"}
    }
    for row in pending_rows(rows, completed):
        row_out = output.parent / f".{output.stem}.chunk-{chunk_id}.{row.row_id}.csv"
        row_config = replace(
            config, out=row_out, models=row.models, n_grid=(row.n_samples,), k_grid=(row.k_features,),
            n_seeds=1, n_draws=1, repeat_plan=((row.seed, row.draw),), n_jobs=1,
        )
        run_nk_grid(row_config, execution_pairs=((row.seed, row.draw),), exact_output_path=True, defer_failure_policy=True)
        current_header, current_rows = _read_csv_keys(row_out)
        if header is None:
            header = current_header
        elif current_header != header:
            raise ValueError("chunk rows produced inconsistent CSV headers")
        for current in current_rows:
            key = (current["model"], int(current["seed"]), int(current["draw"]), int(current["N"]), int(current["K"]))
            by_key[key] = current
        row_out.unlink(missing_ok=True)
        manifest_path(row_out).unlink(missing_ok=True)
    expected = expected_model_keys(rows)
    for key, row in by_key.items():
        if key not in expected:
            raise ValueError(f"chunk output contains an out-of-design key: {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        raise RuntimeError("chunk produced no rows")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(by_key[key] for key in sorted(by_key))
    os.replace(temporary, output)
    write_json_atomic(manifest_path(output), {
        "format_version": TABLE_FORMAT_VERSION, "execution": {"mode": "chunk", "chunk_id": chunk_id},
        "completion": {"expected_rows": len(expected), "materialized_rows": len(by_key)},
    })
    return output


def finalize_chunk_shards(table_path: Path, chunk_outputs: Mapping[int, Path], output: Path) -> Path:
    """Merge chunks after validating the complete table-derived key design."""
    source = pq.ParquetFile(Path(table_path), memory_map=True)
    all_rows = tuple(row for chunk_id in range(source.num_row_groups) for row in read_chunk(table_path, chunk_id))
    expected = expected_model_keys(all_rows)
    seen: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    header: list[str] | None = None
    for chunk_id in range(source.num_row_groups):
        path = Path(chunk_outputs[chunk_id])
        current_header, rows = _read_csv_keys(path)
        if header is None:
            header = current_header
        elif header != current_header:
            raise ValueError("chunk CSV headers differ")
        for row in rows:
            key = (row["model"], int(row["seed"]), int(row["draw"]), int(row["N"]), int(row["K"]))
            if key not in expected:
                raise ValueError(f"merged output contains an out-of-design key: {key}")
            if key in seen:
                raise ValueError(f"merged output contains a duplicate key: {key}")
            seen[key] = row
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"merged output is missing {len(missing)} expected keys")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or [])
        writer.writeheader()
        writer.writerows(seen[key] for key in sorted(seen))
    return output


@dataclass(frozen=True)
class ResourceRequest:
    resource_class: str
    cpus_per_task: int
    partition: str | None = None
    memory: str | None = None
    time_limit: str | None = None


def resource_class_for_rows(rows: Sequence[TaskRow]) -> str:
    groups = {row.group for row in rows}
    return "super_learner" if groups == {"super_learner"} else "serial"


def resource_request(rows: Sequence[TaskRow], values: Mapping[str, ResourceRequest]) -> ResourceRequest:
    """Resolve resources from an explicit per-class mapping, never defaults."""
    resource_class = resource_class_for_rows(rows)
    request = values.get(resource_class)
    if request is None:
        raise ValueError(f"No Stage-B resource request configured for {resource_class}")
    if request.resource_class != resource_class or request.cpus_per_task < 1:
        raise ValueError("invalid resource-class request")
    if not all((request.partition, request.memory, request.time_limit)):
        raise ValueError("Stage-B partition, memory, and time values are required")
    return request


def sbatch_resource_args(rows: Sequence[TaskRow], values: Mapping[str, ResourceRequest]) -> tuple[str, ...]:
    """Generate every resource-sensitive sbatch argument from the class map."""
    request = resource_request(rows, values)
    return (
        f"--partition={request.partition}",
        f"--cpus-per-task={request.cpus_per_task}",
        f"--mem={request.memory}",
        f"--time={request.time_limit}",
    )


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


def write_chunk_snapshot(
    path: Path, *, table_path: Path, panel: str, config: NKGridConfig, output_dir: Path,
) -> Path:
    """Freeze table location, resolved config and chunk-output namespace."""
    table = Path(table_path).resolve()
    source = pq.ParquetFile(table, memory_map=True)
    if source.num_row_groups < 1:
        raise ValueError("task table must contain at least one chunk")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {
        "format_version": TABLE_FORMAT_VERSION,
        "panel": str(panel), "task_table": str(table), "chunk_count": source.num_row_groups,
        "config": _config_to_json(config), "output_dir": str(Path(output_dir).resolve()),
    })
    os.chmod(path, 0o444)
    return path


def run_snapshot_chunk(snapshot_path: Path, chunk_id: int) -> Path:
    """Run the immutable chunk selected by a Slurm array index."""
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if payload.get("format_version") != TABLE_FORMAT_VERSION:
        raise ValueError("unsupported flat-task snapshot")
    if not 0 <= chunk_id < int(payload["chunk_count"]):
        raise IndexError("chunk_id is outside the frozen snapshot")
    return run_chunk(
        Path(str(payload["task_table"])), chunk_id, _config_from_json(payload["config"]),
        output=Path(str(payload["output_dir"])) / f"chunk-{chunk_id}.csv",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run immutable flat NK-grid chunks")
    parser.add_argument("command", choices=("count", "run"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int)
    args = parser.parse_args(argv)
    payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if args.command == "count":
        print(int(payload["chunk_count"]))
        return
    if args.chunk_id is None:
        parser.error("run requires --chunk-id")
    print(run_snapshot_chunk(args.snapshot, args.chunk_id))


if __name__ == "__main__":
    main()

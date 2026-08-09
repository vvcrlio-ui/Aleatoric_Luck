"""Resource-only planning for the dynamic NK-grid work queue.

This module intentionally contains no duration estimate, fitted cost model,
packing heuristic, or multi-resource-class split.  Time is an operational
limit, not a prediction: durable cell shards make repeated rounds safe.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .flat_task_table import (
    ResourceRequest,
    _config_from_json,
    build_rows,
    sbatch_resource_args,
    write_task_table,
    write_work_snapshot,
)
from .nk_grid import NKGridConfig


ENGINE_VALUE_BYTES = 8
MEMORY_BASE_BYTES = int(1.25 * 1024 ** 3)
MEMORY_FRAME_COPIES = 12


def expanded_columns_for_k(schema_path: Path | str, k_features: int) -> int:
    """Return the expanded-column count of the first K schema feature units."""
    if k_features < 1:
        raise ValueError("k_features must be positive")
    try:
        document = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        sources = document["sources"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feature-universe schema: {schema_path}") from exc
    if not isinstance(sources, list) or k_features > len(sources):
        raise ValueError("k_features exceeds schema feature-unit count")
    widths: list[int] = []
    for source in sources[:k_features]:
        features = source.get("features") if isinstance(source, Mapping) else None
        if not isinstance(features, list) or not features:
            raise ValueError("schema source has no expanded features")
        widths.append(len(features))
    return sum(widths)


def peak_memory_bytes(n_samples: int, expanded_columns: int, *, frame_copies: int = MEMORY_FRAME_COPIES) -> int:
    """Conservative one-worker frame formula, expressed in byte units."""
    if n_samples < 1 or expanded_columns < 1 or frame_copies < 1:
        raise ValueError("n_samples, expanded_columns, and frame_copies must be positive")
    return MEMORY_BASE_BYTES + int(frame_copies) * int(n_samples) * int(expanded_columns) * ENGINE_VALUE_BYTES


def implied_frame_copies(measured_bytes: int, *, n_samples: int, expanded_columns: int) -> float:
    """Report how many full N×P frames a real memory probe observed."""
    denominator = int(n_samples) * int(expanded_columns) * ENGINE_VALUE_BYTES
    if measured_bytes < MEMORY_BASE_BYTES or denominator <= 0:
        raise ValueError("measurement cannot imply a positive frame-copy count")
    return (int(measured_bytes) - MEMORY_BASE_BYTES) / denominator


def check_memory_measurement(measured_bytes: int, *, n_samples: int, expanded_columns: int) -> dict[str, float | int]:
    """Fail closed when a probe exceeds the fixed formula rather than retuning it."""
    implied = implied_frame_copies(measured_bytes, n_samples=n_samples, expanded_columns=expanded_columns)
    formula = peak_memory_bytes(n_samples, expanded_columns)
    if implied > MEMORY_FRAME_COPIES:
        raise RuntimeError(
            f"memory probe implies {implied:.3f} frame copies, above fixed "
            f"MEMORY_FRAME_COPIES={MEMORY_FRAME_COPIES}; report and stop"
        )
    return {"measured_bytes": int(measured_bytes), "formula_bytes": formula, "implied_frame_copies": implied}


def format_slurm_memory(num_bytes: int) -> str:
    if num_bytes <= 0:
        raise ValueError("memory must be positive")
    mebibytes = int(math.ceil(num_bytes / 1024 ** 2))
    return f"{mebibytes // 1024}G" if mebibytes % 1024 == 0 else f"{mebibytes}M"


@dataclass(frozen=True)
class ClusterPolicy:
    """Explicit cluster decisions; no account or architecture defaults exist."""

    workers: int
    rounds: int
    partition: str
    time_limit: str
    account: str
    constraint: str
    memory_override: str | None = None
    preparation_memory: str | None = None
    preparation_time_limit: str | None = None
    preparation_tmp_dir: str | None = None
    verification_memory: str | None = None
    verification_time_limit: str | None = None
    verification_tmp_dir: str | None = None
    finalization_memory: str | None = None
    finalization_time_limit: str | None = None
    finalization_tmp_dir: str | None = None
    rows_per_group: int = 100_000

    def validate(self) -> None:
        if self.workers < 1 or self.rounds < 1 or self.rows_per_group < 1:
            raise ValueError("workers, rounds, and rows_per_group must be positive")
        if not self.account:
            raise ValueError("ClusterPolicy.account is required; refusing to emit a submission plan")
        if not self.partition or not self.time_limit or not self.constraint:
            raise ValueError("ClusterPolicy.partition, time_limit, and constraint are required")
        for field in (
            "preparation_memory", "preparation_time_limit", "preparation_tmp_dir",
            "verification_memory", "verification_time_limit", "verification_tmp_dir",
            "finalization_memory", "finalization_time_limit", "finalization_tmp_dir",
        ):
            if getattr(self, field) is not None and not getattr(self, field):
                raise ValueError(f"ClusterPolicy.{field} must be non-empty when set")


def build_dynamic_plan(
    config: NKGridConfig,
    *,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    cluster: ClusterPolicy,
    table_path: Path | str,
    snapshot_path: Path | str,
    output_dir: Path | str,
    panel: str,
) -> dict[str, object]:
    """Freeze one cost-free table, one worker request, and the round count."""
    cluster.validate()
    rows = build_rows(config, n_grid=n_grid, k_grid=k_grid)
    table = write_task_table(Path(table_path), rows, rows_per_group=cluster.rows_per_group)
    max_n = max(row.n_samples for row in rows)
    max_k = max(row.k_features for row in rows)
    expanded = expanded_columns_for_k(config.schema, max_k)
    formula_bytes = peak_memory_bytes(max_n, expanded)
    request = ResourceRequest(
        cpus_per_task=1, partition=cluster.partition,
        memory=cluster.memory_override or format_slurm_memory(formula_bytes),
        time_limit=cluster.time_limit, account=cluster.account, constraint=cluster.constraint,
    )
    preparation_request = ResourceRequest(
        cpus_per_task=1,
        partition=cluster.partition,
        memory=cluster.preparation_memory or request.memory,
        time_limit=cluster.preparation_time_limit or cluster.time_limit,
        account=cluster.account,
        constraint=cluster.constraint,
    )
    verification_request = ResourceRequest(
        cpus_per_task=1,
        partition=cluster.partition,
        memory=cluster.verification_memory or request.memory,
        time_limit=cluster.verification_time_limit or cluster.time_limit,
        account=cluster.account,
        constraint=cluster.constraint,
    )
    finalization_request = ResourceRequest(
        cpus_per_task=1,
        partition=cluster.partition,
        memory=cluster.finalization_memory or request.memory,
        time_limit=cluster.finalization_time_limit or cluster.time_limit,
        account=cluster.account,
        constraint=cluster.constraint,
    )
    def resolve_tmp_dir(value: str | None) -> str | None:
        return None if value is None else str(Path(value).expanduser().resolve())

    preparation_tmp_dir = resolve_tmp_dir(cluster.preparation_tmp_dir)
    verification_tmp_dir = resolve_tmp_dir(cluster.verification_tmp_dir)
    finalization_tmp_dir = resolve_tmp_dir(cluster.finalization_tmp_dir)
    snapshot = write_work_snapshot(
        Path(snapshot_path), table_path=table, panel=panel, config=config,
        output_dir=Path(output_dir), workers=cluster.workers,
        preparation_tmp_dir=preparation_tmp_dir,
        verification_tmp_dir=verification_tmp_dir,
        finalization_tmp_dir=finalization_tmp_dir,
    )
    return {
        "format_version": 2,
        "task_table": str(table), "snapshot": str(snapshot), "workers": cluster.workers,
        "rounds": cluster.rounds,
        "row_count": len(rows),
        "memory": {
            "max_n": max_n, "max_k": max_k, "expanded_columns": expanded,
            "formula_bytes": formula_bytes, "frame_copies": MEMORY_FRAME_COPIES,
            "request": request.memory,
        },
        "submission": {
            "array": f"0-{cluster.workers - 1}%{cluster.workers}",
            "sbatch_args": list(sbatch_resource_args(request)),
            "account": request.account,
            "constraint": request.constraint,
        },
        "preparation": {
            "sbatch_args": list(sbatch_resource_args(preparation_request)),
            "tmp_dir": preparation_tmp_dir,
        },
        "verification": {
            "sbatch_args": list(sbatch_resource_args(verification_request)),
            "tmp_dir": verification_tmp_dir,
        },
        "finalization": {
            "sbatch_args": list(sbatch_resource_args(finalization_request)),
            "tmp_dir": finalization_tmp_dir,
        },
    }


def _cluster_from_payload(payload: Mapping[str, object]) -> ClusterPolicy:
    try:
        return ClusterPolicy(
            workers=int(payload["workers"]), rounds=int(payload["rounds"]),
            partition=str(payload["partition"]), time_limit=str(payload["time_limit"]),
            account=str(payload["account"]),
            constraint=str(payload["constraint"]),
            memory_override=None if payload.get("memory_override") is None else str(payload.get("memory_override")),
            preparation_memory=None if payload.get("preparation_memory") is None else str(payload.get("preparation_memory")),
            preparation_time_limit=None if payload.get("preparation_time_limit") is None else str(payload.get("preparation_time_limit")),
            preparation_tmp_dir=None if payload.get("preparation_tmp_dir") is None else str(payload.get("preparation_tmp_dir")),
            verification_memory=None if payload.get("verification_memory") is None else str(payload.get("verification_memory")),
            verification_time_limit=None if payload.get("verification_time_limit") is None else str(payload.get("verification_time_limit")),
            verification_tmp_dir=None if payload.get("verification_tmp_dir") is None else str(payload.get("verification_tmp_dir")),
            finalization_memory=None if payload.get("finalization_memory") is None else str(payload.get("finalization_memory")),
            finalization_time_limit=None if payload.get("finalization_time_limit") is None else str(payload.get("finalization_time_limit")),
            finalization_tmp_dir=None if payload.get("finalization_tmp_dir") is None else str(payload.get("finalization_tmp_dir")),
            rows_per_group=int(payload.get("rows_per_group", 100_000)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid cluster policy in planning request: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a dynamic NK-grid work-queue plan")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    plan = build_dynamic_plan(
        _config_from_json(payload["config"]), n_grid=[int(value) for value in payload["n_grid"]],
        k_grid=[int(value) for value in payload["k_grid"]], cluster=_cluster_from_payload(payload["cluster"]),
        table_path=payload["task_table"], snapshot_path=payload["snapshot"],
        output_dir=payload["output_dir"], panel=str(payload["panel"]),
    )
    args.plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan["memory"], sort_keys=True))


if __name__ == "__main__":
    main()

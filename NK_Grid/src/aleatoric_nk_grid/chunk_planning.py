"""Resource-only planning for the dynamic NK-grid work queue.

This module intentionally contains no duration estimate, fitted cost model,
packing heuristic, or multi-resource-class split.  Time is an operational
limit, not a prediction: durable cell shards make repeated rounds safe.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .flat_task_table import (
    ResourceRequest,
    _config_from_json,
    execution_groups,
    iter_task_rows_canonical,
    sbatch_resource_args,
    write_task_table_streaming,
    write_work_snapshot,
)
from .execution_contract import (
    AnalysisContract,
    CellExecutionSpec,
    DynamicExecutionContract,
    git_repository_root,
    immutable_json_bytes,
)
from .experiment import git_state, model_run_settings
from .ingest import load_schema
from .model_registry import load_algorithm_version, load_model_params, resolved_model_params
from .nk_grid import LARGE_RUN_THRESHOLD, NKGridConfig, _validate_config, public_result_columns, resolve_repeat_pairs


ENGINE_VALUE_BYTES = 8
MEMORY_BASE_BYTES = int(1.25 * 1024 ** 3)
MEMORY_FRAME_COPIES = 12


def _frozen_input_provenance(schema_path: Path) -> dict[str, dict[str, str]]:
    """Freeze every data/feature file that changes validated numeric input."""

    schema = load_schema(schema_path)
    definition_value = Path(str(schema.feature_universe["definition_file"]))
    definition = definition_value if definition_value.is_absolute() else (schema.path.parent / definition_value).resolve()
    candidates: dict[str, Path | None] = {
        "training_table": schema.table,
        "external_test_table": schema.test_table,
        "feature_manifest": schema.feature_manifest,
        "feature_universe_definition": definition,
        "provenance": schema.table.parent / "provenance.json",
    }
    provenance: dict[str, dict[str, str]] = {}
    for name, candidate in candidates.items():
        if candidate is None:
            continue
        path = Path(candidate).resolve()
        if name == "provenance" and not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"numeric input provenance file is missing: {name}={path}")
        from .execution_contract import sha256_file
        provenance[name] = {"path": str(path), "sha256": sha256_file(path)}
    if not provenance:
        raise ValueError("dynamic planning found no numeric input provenance")
    return provenance


def expanded_columns_for_k(schema_path: Path | str, k_features: int) -> int:
    """Return the expanded-column count of the first K schema feature units."""
    if k_features < 1:
        raise ValueError("k_features must be positive")
    try:
        document_path = Path(schema_path)
        document = json.loads(document_path.read_text(encoding="utf-8"))
        sources = document.get("sources")
        if sources is None:
            feature_universe = document.get("feature_universe", {})
            definition = feature_universe.get("definition_file") if isinstance(feature_universe, Mapping) else None
            if not isinstance(definition, str) or not definition:
                raise KeyError("sources")
            definition_path = Path(definition)
            if not definition_path.is_absolute():
                definition_path = document_path.parent / definition_path
            sources = json.loads(definition_path.read_text(encoding="utf-8"))["sources"]
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
    resolved_n_grid = tuple(sorted({int(value) for value in n_grid}))
    resolved_k_grid = tuple(sorted({int(value) for value in k_grid}))
    if not resolved_n_grid or not resolved_k_grid:
        raise ValueError("dynamic planning requires non-empty resolved N and K grids")
    worker_config = replace(
        config,
        n_jobs=1,
        n_grid=resolved_n_grid,
        k_grid=resolved_k_grid,
        repeat_plan=tuple(resolve_repeat_pairs(config)),
        n_seeds=1,
        n_draws=1,
    )
    _validate_config(worker_config)
    engine_root = Path(__file__).resolve().parents[2]
    source_state = git_state(engine_root)
    if not isinstance(source_state.get("commit"), str) or len(str(source_state["commit"])) != 40:
        raise ValueError("dynamic planning requires a resolvable immutable Git commit")
    if worker_config.preset == "production" and source_state.get("dirty") is not False:
        raise ValueError("Production dynamic planning requires a clean Git worktree")
    # ``CellExecutionSpec`` deliberately freezes dynamic workers at one model
    # job; the local config remains untouched and keeps its original n_jobs.
    # Contracts have one locator root: the real Git top-level.  Using a
    # common parent of inputs makes the same repository acquire a different
    # identity at another mount point and permits locator-root drift.
    repo_root = git_repository_root(engine_root)
    input_schema = load_schema(worker_config.schema)
    selected_params = load_model_params(
        worker_config.model_params, task=input_schema.task, models=worker_config.models,
    )
    frozen_groups = [
        {"k_features": int(k_features), "groups": [
            {"group": group, "models": list(models)}
            for group, models in execution_groups(worker_config.models, k_features=int(k_features))
        ]}
        for k_features in resolved_k_grid
    ]
    cell_spec = CellExecutionSpec.from_config(
        worker_config,
        repo_root=repo_root,
        panel_id=panel,
        resolved_n_grid=resolved_n_grid,
        resolved_k_grid=resolved_k_grid,
        resolved_repeat_plan=worker_config.repeat_plan,
        model_n_jobs=1,
        git_commit=str(source_state["commit"]),
        algorithm_version=load_algorithm_version(worker_config.model_params),
        resolved_model_params=resolved_model_params(selected_params),
        environment_overrides=model_run_settings(worker_config.models),
        execution_groups=frozen_groups,
        input_provenance=_frozen_input_provenance(worker_config.schema),
        require_clean_worktree=worker_config.preset == "production",
    )
    summary = write_task_table_streaming(
        iter_task_rows_canonical(
            worker_config,
            n_grid=resolved_n_grid,
            k_grid=resolved_k_grid,
            repeat_pairs=worker_config.repeat_plan,
        ),
        Path(table_path),
        rows_per_group=cluster.rows_per_group,
    )
    table = summary.path
    if summary.expected_model_rows > LARGE_RUN_THRESHOLD and not worker_config.allow_large_run:
        raise ValueError(
            f"Large dynamic run requires --allow-large-run: {summary.expected_model_rows:,} model cells exceed {LARGE_RUN_THRESHOLD:,}"
        )
    max_n = summary.max_n
    max_k = summary.max_k
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
    schema_payload = json.loads(Path(worker_config.schema).read_text(encoding="utf-8"))
    # Generic legacy test schemas omit ``task``; validated NK-grid input
    # treats that omission as continuous/regression, so the immutable schema
    # codec must make the same deterministic choice.
    task = str(schema_payload.get("task") or "regression")
    public_schema = public_result_columns(task)
    analysis_contract = AnalysisContract.create(
        cell_execution_spec=cell_spec,
        task_design_digest=summary.task_design_digest,
        expected_task_rows=summary.expected_task_rows,
        expected_model_rows=summary.expected_model_rows,
        public_result_schema=public_schema,
        protocol_limits={"result_payload_max_bytes": 8 * 1024 * 1024},
    )
    output_root = Path(output_dir)
    immutable_json_bytes(output_root / "analysis-contract.json", analysis_contract.to_payload())
    execution_contract = DynamicExecutionContract.create(
        analysis_contract=analysis_contract,
        task_table_path=str(table.resolve()),
        task_table_file_sha256=summary.task_table_file_sha256,
        task_table_rows=summary.expected_task_rows,
        worker_count=cluster.workers,
        initial_round_count=cluster.rounds,
        resources={
            "worker": list(sbatch_resource_args(request)),
            "prep": list(sbatch_resource_args(preparation_request)),
            "verify": list(sbatch_resource_args(verification_request)),
            "finalize": list(sbatch_resource_args(finalization_request)),
            "tmp_dirs": {"prep": preparation_tmp_dir, "verify": verification_tmp_dir, "finalize": finalization_tmp_dir},
        },
        output_root=output_root,
        wal_limits={"identity_max_bytes": 64 * 1024, "metadata_max_bytes": 4 * 1024, "payload_max_bytes": 8 * 1024 * 1024, "abort_max_bytes": 4 * 1024},
    )
    immutable_json_bytes(
        output_root / "execution-contracts" / f"{execution_contract.execution_plan_id}.json",
        execution_contract.to_payload(),
    )
    snapshot = write_work_snapshot(
        Path(snapshot_path), table_path=table, panel=panel, config=worker_config,
        output_dir=Path(output_dir), workers=cluster.workers,
        preparation_tmp_dir=preparation_tmp_dir,
        verification_tmp_dir=verification_tmp_dir,
        finalization_tmp_dir=finalization_tmp_dir,
        analysis_contract=analysis_contract,
        execution_contract=execution_contract,
        task_summary=summary,
        cell_spec_repo_root=repo_root,
    )
    return {
        "format_version": 2,
        "task_table": str(table), "snapshot": str(snapshot), "workers": cluster.workers,
        "rounds": cluster.rounds,
        "row_count": summary.expected_task_rows,
        "model_row_count": summary.expected_model_rows,
        "analysis_id": analysis_contract.analysis_id,
        "execution_plan_id": execution_contract.execution_plan_id,
        "task_design_digest": summary.task_design_digest,
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

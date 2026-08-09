"""Canonical, fail-closed contracts for dynamic NK-grid execution.

The dynamic queue has two deliberately different identities.  An analysis
identity says that result rows may be combined; an execution-plan identity
says who owns one immutable assignment and its WAL.  Keeping the codecs here
prevents the worker, closer, and finalizer from each inventing a slightly
different interpretation of JSON or paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CELL_SPEC_FORMAT_VERSION = 1
ANALYSIS_CONTRACT_FORMAT_VERSION = 1
EXECUTION_CONTRACT_FORMAT_VERSION = 1
PUBLIC_RESULT_SERIALIZER_VERSION = 1


class ContractError(ValueError):
    """A contract, identity, or immutable-artifact validation error."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode protocol data with one portable, non-``repr`` codec."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without making its size part of Python memory use."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot hash immutable artefact {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_task_row_payload(row: Any) -> dict[str, object]:
    """Return the logical TaskRow codec shared by planning and assignments."""

    return {
        "K": int(row.k_features),
        "N": int(row.n_samples),
        "draw": int(row.draw),
        "group": str(row.group),
        "models": [str(model) for model in row.models],
        "row_id": str(row.row_id),
        "seed": int(row.seed),
    }


def task_row_digest(rows: Sequence[Any] | Any) -> str:
    """Hash a canonical TaskRow JSON-lines stream without materialising it."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(canonical_task_row_payload(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _inside_root(path: Path, root: Path) -> Path:
    """Resolve a contract locator without accepting symlink or ``..`` escape."""

    root_resolved = Path(root).resolve()
    supplied = Path(path)
    if supplied.is_absolute():
        raise ContractError(f"contract locator must be repository-relative: {path}")
    if ".." in supplied.parts:
        raise ContractError(f"contract locator may not contain '..': {path}")
    resolved = (root_resolved / supplied).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ContractError(f"contract locator escapes repository root: {path}") from exc
    if not resolved.is_file():
        raise ContractError(f"contract locator is not a regular file: {path}")
    return resolved


def canonical_repo_locator(path: Path | str, *, repo_root: Path) -> tuple[str, str]:
    """Return a verified POSIX locator and content hash for immutable input."""

    root = Path(repo_root).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"path is outside repository root: {path}") from exc
    else:
        resolved = _inside_root(candidate, root)
        relative = resolved.relative_to(root)
    if not resolved.is_file():
        raise ContractError(f"contract locator is not a regular file: {path}")
    return relative.as_posix(), sha256_file(resolved)


def resolve_repo_locator(locator: str, expected_sha256: str, *, repo_root: Path) -> Path:
    resolved = _inside_root(Path(locator), Path(repo_root))
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ContractError(
            f"immutable artefact checksum mismatch for {locator}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return resolved


@dataclass(frozen=True)
class CellExecutionSpec:
    """The complete numeric contract consumed by ``NKGridExecutionSession``.

    The dataclass intentionally stores only JSON-shaped fields.  It has no
    assignment, worker, round, result-store, or Slurm field, so a session can
    be used by both the local runner and a dynamic worker.
    """

    payload: Mapping[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CellExecutionSpec":
        value = dict(payload)
        if value.get("cell_spec_format_version") != CELL_SPEC_FORMAT_VERSION:
            raise ContractError("unsupported cell execution spec format")
        if not isinstance(value.get("models"), list) or not value["models"]:
            raise ContractError("cell execution spec requires ordered models")
        model_n_jobs = value.get("model_n_jobs")
        if not isinstance(model_n_jobs, int) or isinstance(model_n_jobs, bool) or model_n_jobs < 1:
            raise ContractError("cell execution spec model_n_jobs must be a positive integer")
        for name in ("resolved_n_grid", "resolved_k_grid", "resolved_repeat_plan"):
            if not isinstance(value.get(name), list) or not value[name]:
                raise ContractError(f"cell execution spec requires frozen {name}")
        return cls(value)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        repo_root: Path,
        panel_id: str | None = None,
        resolved_n_grid: Sequence[int] | None = None,
        resolved_k_grid: Sequence[int] | None = None,
        resolved_repeat_plan: Sequence[tuple[int, int]] | None = None,
        model_n_jobs: int | None = None,
        git_commit: str | None = None,
        algorithm_version: str | None = None,
        resolved_model_params: Mapping[str, object] | None = None,
        environment_overrides: Mapping[str, object] | None = None,
        execution_groups: Sequence[tuple[str, Sequence[str]]] | None = None,
    ) -> "CellExecutionSpec":
        """Freeze a validated config into a session-only identity.

        Callers must supply resolved grids when config values are implicit;
        accepting an unresolved grid would make the same contract execute
        differently after an input change.
        """

        job_count = int(config.n_jobs if model_n_jobs is None else model_n_jobs)
        if job_count < 1:
            raise ContractError("model_n_jobs must be positive")
        n_grid = tuple(int(item) for item in (resolved_n_grid or config.n_grid or ()))
        k_grid = tuple(int(item) for item in (resolved_k_grid or config.k_grid or ()))
        repeats = tuple((int(seed), int(draw)) for seed, draw in (resolved_repeat_plan or config.repeat_plan or ()))
        if not n_grid or not k_grid or not repeats:
            raise ContractError("CellExecutionSpec requires resolved grids and repeat plan")
        schema_locator, schema_sha256 = canonical_repo_locator(config.schema, repo_root=repo_root)
        params_locator, params_sha256 = canonical_repo_locator(config.model_params, repo_root=repo_root)
        groups = [
            {"group": str(group), "models": [str(model) for model in models]}
            for group, models in (execution_groups or ())
        ]
        payload: dict[str, object] = {
            "cell_spec_format_version": CELL_SPEC_FORMAT_VERSION,
            "panel_id": None if panel_id is None else str(panel_id),
            "experiment_id": str(config.experiment_id),
            "data_version": str(config.data_version),
            "model_spec_version": str(config.model_spec_version),
            "outcome": str(config.outcome),
            "preset": None if config.preset is None else str(config.preset),
            "schema_locator": schema_locator,
            "schema_file_sha256": schema_sha256,
            "model_params_locator": params_locator,
            "model_params_sha256": params_sha256,
            "algorithm_version": algorithm_version,
            "resolved_model_params": dict(resolved_model_params or {}),
            "environment_overrides": dict(environment_overrides or {}),
            "split_seed": int(config.seed),
            "test_size": float(config.test_size),
            "models": [str(model) for model in config.models],
            "model_n_jobs": job_count,
            "resolved_n_grid": list(n_grid),
            "resolved_k_grid": list(k_grid),
            "resolved_repeat_plan": [[seed, draw] for seed, draw in repeats],
            "execution_groups": groups,
            "native_process_max_attempts": int(config.native_process_max_attempts),
            "native_process_timeout_seconds": float(config.native_process_timeout_seconds),
            "min_n": int(config.min_n),
            "git_commit": git_commit,
        }
        return cls.from_payload(payload)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(dict(self.payload)))

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)

    def resolve_inputs(self, *, repo_root: Path) -> tuple[Path, Path]:
        payload = self.payload
        return (
            resolve_repo_locator(str(payload["schema_locator"]), str(payload["schema_file_sha256"]), repo_root=repo_root),
            resolve_repo_locator(str(payload["model_params_locator"]), str(payload["model_params_sha256"]), repo_root=repo_root),
        )


@dataclass(frozen=True)
class AnalysisContract:
    payload: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        cell_execution_spec: CellExecutionSpec,
        task_design_digest: str,
        expected_task_rows: int,
        expected_model_rows: int,
        public_result_schema: Sequence[str],
        protocol_limits: Mapping[str, int] | None = None,
    ) -> "AnalysisContract":
        if expected_task_rows < 1 or expected_model_rows < 1:
            raise ContractError("analysis contract expected row counts must be positive")
        schema = [str(column) for column in public_result_schema]
        if not schema or len(schema) != len(set(schema)):
            raise ContractError("public result schema must be ordered and unique")
        schema_payload = {
            "columns": schema,
            "serializer_version": PUBLIC_RESULT_SERIALIZER_VERSION,
            "protocol_limits": dict(protocol_limits or {}),
        }
        payload = {
            "analysis_contract_format_version": ANALYSIS_CONTRACT_FORMAT_VERSION,
            "cell_execution_spec": cell_execution_spec.to_payload(),
            "cell_spec_sha256": cell_execution_spec.sha256,
            "task_design_digest": str(task_design_digest),
            "expected_task_rows": int(expected_task_rows),
            "expected_model_rows": int(expected_model_rows),
            "public_result_schema": schema_payload,
            "public_result_schema_fingerprint": sha256_bytes(canonical_json_bytes(schema_payload)),
        }
        return cls(payload)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(dict(self.payload)))

    @property
    def analysis_id(self) -> str:
        return self.sha256

    def to_payload(self) -> dict[str, object]:
        payload = dict(self.payload)
        payload["analysis_id"] = self.analysis_id
        payload["analysis_contract_sha256"] = self.sha256
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AnalysisContract":
        candidate = dict(payload)
        expected_id = candidate.pop("analysis_id", None)
        expected_sha = candidate.pop("analysis_contract_sha256", None)
        if candidate.get("analysis_contract_format_version") != ANALYSIS_CONTRACT_FORMAT_VERSION:
            raise ContractError("unsupported analysis contract format")
        spec = CellExecutionSpec.from_payload(candidate.get("cell_execution_spec", {}))
        if candidate.get("cell_spec_sha256") != spec.sha256:
            raise ContractError("analysis contract cell execution spec checksum mismatch")
        contract = cls(candidate)
        if expected_id is not None and expected_id != contract.analysis_id:
            raise ContractError("analysis contract ID checksum mismatch")
        if expected_sha is not None and expected_sha != contract.sha256:
            raise ContractError("analysis contract file checksum mismatch")
        return contract


@dataclass(frozen=True)
class DynamicExecutionContract:
    payload: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        analysis_contract: AnalysisContract,
        task_table_path: str,
        task_table_file_sha256: str,
        task_table_rows: int,
        worker_count: int,
        initial_round_count: int,
        resources: Mapping[str, object],
        output_root: Path | str,
        wal_limits: Mapping[str, int],
    ) -> "DynamicExecutionContract":
        if worker_count < 1 or initial_round_count < 1 or task_table_rows < 1:
            raise ContractError("execution contract counts must be positive")
        payload = {
            "execution_contract_format_version": EXECUTION_CONTRACT_FORMAT_VERSION,
            "analysis_id": analysis_contract.analysis_id,
            "analysis_contract_sha256": analysis_contract.sha256,
            "task_table_path": str(task_table_path),
            "task_table_file_sha256": str(task_table_file_sha256),
            "task_table_rows": int(task_table_rows),
            "table_format": "task-table-v2",
            "worker_count": int(worker_count),
            "initial_round_count": int(initial_round_count),
            "result_store_format": "worker-event-wal-v1",
            "wal_protocol_limits": dict(wal_limits),
            "resources": dict(resources),
            "output_root": str(Path(output_root).resolve()),
        }
        return cls(payload)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(dict(self.payload)))

    @property
    def execution_plan_id(self) -> str:
        return self.sha256

    def to_payload(self) -> dict[str, object]:
        payload = dict(self.payload)
        payload["execution_plan_id"] = self.execution_plan_id
        payload["execution_contract_sha256"] = self.sha256
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DynamicExecutionContract":
        candidate = dict(payload)
        expected_id = candidate.pop("execution_plan_id", None)
        expected_sha = candidate.pop("execution_contract_sha256", None)
        if candidate.get("execution_contract_format_version") != EXECUTION_CONTRACT_FORMAT_VERSION:
            raise ContractError("unsupported execution contract format")
        contract = cls(candidate)
        if expected_id is not None and expected_id != contract.execution_plan_id:
            raise ContractError("execution plan ID checksum mismatch")
        if expected_sha is not None and expected_sha != contract.sha256:
            raise ContractError("execution contract checksum mismatch")
        return contract


def immutable_json_bytes(path: Path, payload: Mapping[str, object]) -> None:
    """Write an immutable contract with create-or-byte-identical semantics."""

    target = Path(path)
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        existing = target.read_bytes()
        if existing != encoded:
            raise ContractError(f"immutable artefact differs: {target}")
        return
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

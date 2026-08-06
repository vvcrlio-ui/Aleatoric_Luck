"""Empirical cost/memory calibration harness for the N x K engine.

Measures wall-clock startup cost (t0), per-model fit-time power laws, peak
RSS, and preprocessing cost -- entirely on locally generated synthetic data.
Produces ``NK_Grid/calibration/cost_model_<UTC-date>.json``.

This module is purely additive: it does not change any engine execution
logic, does not touch private observation data (example only, never a dataset-
name guard criterion), and never inspects or reports any metric value (``r2_test``,
``rmse``, etc). See ``plans/cost-calibration.md`` for the full specification.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from aleatoric_nk_grid_measure_worker import (
    cell_worker_target,
    parallel_efficiency_cell_worker_target,
    task_cell_worker_target,
)

from .experiment import (
    SERIAL_OUTER_MODELS,
    core_environment,
    git_state,
    utc_now,
    write_json_atomic,
)
from .ingest import load_input
from .model_registry import DEFAULT_MODEL_PARAMS_PATH, load_model_params, make_model
from .native_process import IsolatedProcessRunner
from .nk_grid import (
    DrawOrders,
    SplitData,
    _fit_predict_model_cell,
    _process_peak_rss_bytes,
    _run_native_model_cell_locked,
    draw_orders,
    split_frame,
)
from .preprocessing import preprocess_cell, source_groups
from .validate_input import canonical_feature_universe, validate_input


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version 5 adds measured parallel efficiency, a runtime t0 import
# measurement/end-to-end consistency check, actual synthetic missingness, and
# calibration-run reporting. Earlier files must never be interpreted as v5.
FORMAT_VERSION = 5
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01
MEMORY_METHOD_CGROUP_PEAK = "cgroup_v2_memory_peak"
MEMORY_METHOD_CGROUP_CURRENT = "cgroup_v2_memory_current_sampled"
MEMORY_METHOD_PROCESS_TREE = "process_tree_rss_sampled_conservative_upper_bound"

MODELS: tuple[str, ...] = (
    "ols",
    "ridge",
    "lasso",
    "random_forest",
    "extra_trees",
    "shallow_neural_network",
    "super_learner",
    "xgboost",
    "lightgbm",
)

# Models that production hands to the isolated native subprocess
# (nk_grid._run_native_model_cell_locked / native_process.IsolatedProcessRunner).
# This is a direct reference to the engine's own constant -- NOT a locally
# maintained copy -- so it can never silently drift from production again
# (round 1 review F1: a hand-copied {"lightgbm", "xgboost"} was wrong on both
# counts; the real set is {"lightgbm", "super_learner"}).
# See test_subprocess_model_set_tracks_the_engine for the anti-drift check.
SUBPROCESS_MODELS: frozenset[str] = SERIAL_OUTER_MODELS

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

DEFAULT_STAGE_A_N: tuple[int, ...] = (10, 100, 1000)
DEFAULT_STAGE_A_K_BASE: tuple[int, ...] = (10, 100, 1000)
DEFAULT_STAGE_A_K_INTERMEDIATE_POINTS = 2
STAGE_A_REPS = 3

DEFAULT_MAX_SECONDS = 3600.0
T0_REPS = 5
PARALLEL_EFFICIENCY_REPS = 3
R2_EXPLANATION_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Private-data guard
# ---------------------------------------------------------------------------


class PrivateDataAccessError(PermissionError):
    """Raised when the measurement path would touch private data."""


def _default_calibration_read_roots() -> tuple[Path, ...]:
    """Locations the calibration harness owns or may safely use by default."""

    return (
        Path(tempfile.gettempdir()).resolve(),
        (repo_root() / "NK_Grid" / "calibration" / "_scratch").resolve(),
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _looks_like_file_path(value: str) -> bool:
    if not value or "\x00" in value:
        return False
    candidate = Path(value)
    return (
        candidate.is_absolute()
        or value.startswith(("./", "../", "~"))
        or "/" in value
        or "\\" in value
        or bool(candidate.suffix)
    )


def _schema_referenced_paths(document: Any, schema_directory: Path) -> set[Path]:
    """Collect path-like strings recursively without depending on field names."""

    paths: set[Path] = set()

    def _visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                _visit(nested)
        elif isinstance(value, list):
            for nested in value:
                _visit(nested)
        elif isinstance(value, str) and _looks_like_file_path(value):
            candidate = Path(value).expanduser()
            paths.add(
                candidate.resolve()
                if candidate.is_absolute()
                else (schema_directory / candidate).resolve()
            )

    _visit(document)
    return paths


def guard_not_private_data(
    schema_path: Path | str,
    *,
    allowed_roots: Sequence[Path | str] = (),
) -> Path:
    """Fail closed unless a schema and every referenced file are allowlisted."""

    resolved_schema = Path(schema_path).resolve()
    whitelist = tuple(
        dict.fromkeys(
            [
                resolved_schema,
                *(_default_calibration_read_roots()),
                *(Path(root).resolve() for root in allowed_roots),
            ]
        )
    )

    def _require_allowed(path: Path) -> None:
        if not any(_is_within(path, root) for root in whitelist):
            rendered = ", ".join(str(root) for root in whitelist) or "<empty>"
            raise PrivateDataAccessError(
                f"refusing calibration read outside whitelist: {path}; "
                f"allowed roots: [{rendered}]"
            )

    _require_allowed(resolved_schema)
    try:
        document = json.loads(resolved_schema.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateDataAccessError(
            f"cannot inspect calibration schema before data access: {resolved_schema}: {exc}"
        ) from exc
    for referenced_path in sorted(
        _schema_referenced_paths(document, resolved_schema.parent), key=str
    ):
        _require_allowed(referenced_path)
    return resolved_schema


def repo_root() -> Path:
    # NK_Grid/src/aleatoric_nk_grid/calibrate_cost.py -> repo root
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Environment guard
# ---------------------------------------------------------------------------


def check_thread_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Report whether the production single-thread env vars are all set to 1."""

    import os

    source = env if env is not None else os.environ
    values = {name: source.get(name) for name in THREAD_ENV_VARS}
    ok = all(values[name] == "1" for name in THREAD_ENV_VARS)
    return {"ok": ok, "values": values, "required": "1"}


def enforce_thread_env(*, strict: bool, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Refuse to run (strict) or warn loudly (non-strict) on the wrong env.

    Always returns the guard report so it can be recorded in the output file
    even when execution is allowed to continue.
    """

    report = check_thread_env(env)
    if not report["ok"]:
        message = (
            "Thread environment variables are not all set to '1' "
            f"({THREAD_ENV_VARS}); measured timings would not be transferable "
            f"to the production Slurm environment. Observed: {report['values']}"
        )
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)
    return report


# ---------------------------------------------------------------------------
# Timing helpers / censoring
# ---------------------------------------------------------------------------


class MeasurementCensored(RuntimeError):
    """Raised when a single measurement exceeds --max-seconds."""


@contextmanager
def time_budget(max_seconds: float | None) -> Iterator[None]:
    """Abort the wrapped block with MeasurementCensored past max_seconds.

    Uses SIGALRM/setitimer (POSIX only); the harness is not intended to run
    on Windows.
    """

    if not max_seconds or max_seconds <= 0:
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise MeasurementCensored(f"measurement exceeded max_seconds={max_seconds}")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(max_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def summarize_reps(values: Sequence[float]) -> dict[str, Any]:
    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_reps": len(values),
    }


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDataParams:
    n_train: int
    shape: "PanelShape"
    seed: int
    missing_rate_continuous: float = 0.10
    missing_rate_group: float = 0.08
    outcome: str = "y"


@dataclass(frozen=True)
class ShapeSource:
    """One source unit and the dtypes of its expanded feature columns."""

    unit_type: str
    feature_dtypes: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.feature_dtypes)


@dataclass(frozen=True)
class PanelShape:
    """A reproducible synthetic-panel shape derived from one structure schema."""

    schema_path: str
    sources: tuple[ShapeSource, ...]
    dtype_source: str
    dtype_metadata_declared: int
    dtype_metadata_total: int
    dtype_profile_path: str | None = None
    dtype_profile_allocation_rule: str = "declared_per_feature"

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def onehot_group_sizes(self) -> tuple[int, ...]:
        return tuple(source.width for source in self.sources if source.unit_type == "onehot_group")

    @property
    def onehot_source_fraction(self) -> float:
        return (
            len(self.onehot_group_sizes) / self.n_sources
            if self.n_sources
            else 0.0
        )

    @property
    def expanded_dtype_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source in self.sources:
            for dtype in source.feature_dtypes:
                counts[dtype] = counts.get(dtype, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def dtype_metadata_coverage(self) -> str:
        return f"{self.dtype_metadata_declared}/{self.dtype_metadata_total}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_path": self.schema_path,
            "n_sources": self.n_sources,
            "onehot_source_fraction": self.onehot_source_fraction,
            "onehot_group_sizes": list(self.onehot_group_sizes),
            "expanded_dtype_counts": self.expanded_dtype_counts,
            "dtype_source": self.dtype_source,
            "dtype_metadata_coverage": self.dtype_metadata_coverage,
            "dtype_metadata_declared": self.dtype_metadata_declared,
            "dtype_metadata_total": self.dtype_metadata_total,
            "dtype_profile_path": self.dtype_profile_path,
            "dtype_profile_allocation_rule": self.dtype_profile_allocation_rule,
        }


def shape_from_schema(
    schema_path: Path | str,
    *,
    feature_dtype_profile: Mapping[str, Any] | None = None,
    assume_feature_dtype: str | None = None,
    dtype_profile_path: Path | str | None = None,
) -> PanelShape:
    """Read an explicit feature-universe schema without observation values.

    Dtype composition is a property of the expanded *data*, not of a structure
    schema, so it can never be inferred from structure alone. It must come from
    either per-feature ``dtype`` metadata declared by the schema, or a read-only
    dtype probe supplied through ``feature_dtype_profile``. A flat profile maps
    dtype names to counts and is allocated in dtype-name alphabetical order then
    schema source/feature order. A structure-aware profile instead maps each
    unit type (``continuous`` or ``onehot_group``) to such a dtype map, and is
    allocated in dtype-name alphabetical order within that unit type while
    preserving schema order inside the unit type. The latter can express a
    realistic allocation such as float continuous variables plus integer
    one-hot groups without relying on source ordering. A caller may instead
    make an explicit uniform assumption with ``assume_feature_dtype``; unlike
    the old implicit float64 fallback, that choice is recorded in the shape and
    JSON.
    """

    if feature_dtype_profile is not None and assume_feature_dtype is not None:
        raise ValueError("feature_dtype_profile and assume_feature_dtype are mutually exclusive")

    resolved = Path(schema_path).resolve()
    guard_not_private_data(resolved, allowed_roots=(resolved.parent,))
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feature-universe schema: {resolved}: {exc}") from exc
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"feature-universe schema has no sources: {resolved}")

    source_rows: list[tuple[str, list[Mapping[str, Any]]]] = []
    declared_count = 0
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"source {index} is not an object in {resolved}")
        unit_type = raw_source.get("unit_type")
        features = raw_source.get("features")
        if unit_type not in {"continuous", "onehot_group"} or not isinstance(features, list) or not features:
            raise ValueError(f"source {index} has invalid unit_type or features in {resolved}")
        for feature in features:
            if not isinstance(feature, Mapping):
                raise ValueError(f"source {index} has a non-object feature in {resolved}")
            declared_count += "dtype" in feature
        source_rows.append((unit_type, features))

    total_count = sum(len(features) for _, features in source_rows)
    def _expand_profile(counts: Mapping[str, Any], *, context: str) -> list[str]:
        profile: list[str] = []
        for raw_dtype, raw_count in sorted(counts.items()):
            try:
                dtype = np.dtype(raw_dtype).name
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid feature dtype profile entry in {context}: "
                    f"{raw_dtype!r}: {raw_count!r}"
                ) from exc
            if count < 0:
                raise ValueError("feature dtype profile counts must be non-negative")
            profile.extend([dtype] * count)
        return profile

    if feature_dtype_profile is not None:
        profile_values = list(feature_dtype_profile.values())
        grouped_profile = bool(profile_values) and all(
            isinstance(value, Mapping) for value in profile_values
        )
        flat_profile = bool(profile_values) and all(
            not isinstance(value, Mapping) for value in profile_values
        )
        if not (grouped_profile or flat_profile):
            raise ValueError(
                "feature dtype profile must be either a flat dtype-count mapping or "
                "a unit_type-to-dtype-count mapping; mixed forms are not allowed"
            )
        if grouped_profile:
            allowed_types = {unit_type for unit_type, _ in source_rows}
            supplied_types = set(feature_dtype_profile)
            if supplied_types != allowed_types:
                raise ValueError(
                    "unit-type dtype profile keys must exactly match schema unit types: "
                    f"expected {sorted(allowed_types)}, got {sorted(supplied_types)}"
                )
            profile_by_unit_type = {
                str(unit_type): _expand_profile(counts, context=f"unit_type {unit_type!r}")
                for unit_type, counts in feature_dtype_profile.items()
            }
            expected_counts = {
                unit_type: sum(len(features) for source_type, features in source_rows if source_type == unit_type)
                for unit_type in allowed_types
            }
            for unit_type, profile in profile_by_unit_type.items():
                if len(profile) != expected_counts[unit_type]:
                    raise ValueError(
                        "unit-type feature dtype profile count does not match expanded "
                        f"feature count for {unit_type!r}: {len(profile)} != "
                        f"{expected_counts[unit_type]}"
                    )
            profile_iterators = {
                unit_type: iter(profile) for unit_type, profile in profile_by_unit_type.items()
            }
            dtype_profile_allocation_rule = (
                "unit_type_grouped_dtype_name_sorted_then_schema_source_order_within_unit_type"
            )
        else:
            profile = _expand_profile(feature_dtype_profile, context="flat profile")
            if len(profile) != total_count:
                raise ValueError(
                    "feature dtype profile count does not match expanded feature count: "
                    f"{len(profile)} != {total_count}"
                )
            profile_iterators = {"__flat__": iter(profile)}
            dtype_profile_allocation_rule = (
                "flat_dtype_name_sorted_then_schema_source_and_feature_order"
            )
        dtype_source = "external_profile"
    else:
        missing_count = total_count - declared_count
        if missing_count and assume_feature_dtype is None:
            raise ValueError(
                f"feature-universe schema declares dtype for {declared_count}/{total_count} "
                "expanded features; pass feature_dtype_profile or explicitly set "
                "assume_feature_dtype"
            )
        assumed_dtype = None
        if assume_feature_dtype is not None:
            try:
                assumed_dtype = np.dtype(assume_feature_dtype).name
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid assumed feature dtype: {assume_feature_dtype!r}") from exc
        if missing_count == 0:
            dtype_source = "declared"
        elif declared_count == 0:
            dtype_source = f"defaulted_{assumed_dtype}"
        else:
            dtype_source = f"declared_with_assumed_{assumed_dtype}"
        profile_iterators = None
        dtype_profile_allocation_rule = (
            "declared_per_feature" if missing_count == 0 else "declared_or_explicit_uniform_assumption_per_feature"
        )

    sources: list[ShapeSource] = []
    for source_index, (unit_type, features) in enumerate(source_rows):
        dtypes: list[str] = []
        for feature in features:
            try:
                if profile_iterators is not None:
                    iterator = profile_iterators.get(unit_type, profile_iterators.get("__flat__"))
                    if iterator is None:
                        raise AssertionError("validated dtype profile lacks a required unit type")
                    dtype = next(iterator)
                else:
                    dtype = np.dtype(feature.get("dtype", assumed_dtype)).name
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"source {source_index} has invalid dtype metadata in {resolved}"
                ) from exc
            dtypes.append(dtype)
        sources.append(ShapeSource(unit_type=unit_type, feature_dtypes=tuple(dtypes)))
    return PanelShape(
        schema_path=str(resolved),
        sources=tuple(sources),
        dtype_source=dtype_source,
        dtype_metadata_declared=declared_count,
        dtype_metadata_total=total_count,
        dtype_profile_path=(
            str(Path(dtype_profile_path).resolve()) if dtype_profile_path is not None else None
        ),
        dtype_profile_allocation_rule=dtype_profile_allocation_rule,
    )


def k_grid_from_shape(
    shape: PanelShape,
    *,
    base_anchors: Sequence[int] = DEFAULT_STAGE_A_K_BASE,
    intermediate_points: int = DEFAULT_STAGE_A_K_INTERMEDIATE_POINTS,
) -> tuple[int, ...]:
    """Return a shape-bounded K grid with geometric interior production probes."""

    if shape.n_sources < 1:
        raise ValueError("panel shape must contain at least one source")
    if intermediate_points < 0:
        raise ValueError("intermediate_points must be non-negative")
    anchors = {int(k) for k in base_anchors if 0 < int(k) < shape.n_sources}
    anchors.add(shape.n_sources)
    lower = max(anchors - {shape.n_sources}, default=1)
    if lower < shape.n_sources:
        log_lower, log_upper = math.log(lower), math.log(shape.n_sources)
        for position in range(1, intermediate_points + 1):
            candidate = int(round(math.exp(log_lower + (log_upper - log_lower) * position / (intermediate_points + 1))))
            if lower < candidate < shape.n_sources:
                anchors.add(candidate)
    return tuple(sorted(anchors))


def default_stage_b_points(
    *, n_train: int, stage_a_n: Sequence[int], k_grid: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Derive validation points from caller-provided N and schema-derived K."""

    if n_train < 1 or not stage_a_n or not k_grid:
        raise ValueError("n_train, stage_a_n, and k_grid must be non-empty positive values")
    points = [(n_train, int(k_grid[-1]))]
    if len(stage_a_n) > 1:
        points.append((min(n_train, max(stage_a_n)), int(k_grid[-1])))
    if len(k_grid) > 1:
        points.append((n_train, int(k_grid[-2])))
    return tuple(dict.fromkeys(points))


def generate_synthetic_bundle(
    root: Path, params: SyntheticDataParams
) -> tuple[Path, dict[str, Any]]:
    """Write a self-contained, deterministic schema bundle to ``root``.

    Returns (schema_path, generation_stats). Uses Parquet for the training
    table (a format already supported by ``ingest.read_table``/``table_columns``)
    to keep wide-dimension synthetic panels tractable on a laptop; a
    deviation from a plain CSV, noted in the report.
    """

    rng = np.random.default_rng(params.seed)
    columns: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    for source_order, source in enumerate(params.shape.sources):
        base = f"S_src{source_order:05d}"
        if source.unit_type == "continuous":
            dtype = np.dtype(source.feature_dtypes[0])
            values = rng.normal(size=params.n_train).astype(dtype)
            if np.issubdtype(dtype, np.floating):
                values[rng.random(params.n_train) < params.missing_rate_continuous] = np.nan
            columns[base] = values
            manifest_rows.append(
                {
                    "source_column": base,
                    "feature_name": base,
                    "keep": True,
                    "source_order": source_order,
                    "feature_order": 0,
                    "unit_type": "continuous",
                    "drop_first": False,
                    "is_reference": False,
                    "reference_level": None,
                    "level_value": None,
                    "ordinal_levels": None,
                    "source_prior": None,
                }
            )
            continue

        size = source.width
        feats = [f"{base}__lvl{k}" for k in range(size)]
        choice = rng.choice(size, size=params.n_train, p=rng.dirichlet(np.ones(size)))
        block = np.zeros((params.n_train, size), dtype=float)
        block[np.arange(params.n_train), choice] = 1.0
        # Integer dummy columns cannot encode a missing group. Preserve their
        # dtype by leaving that source fully observed; float groups retain the
        # existing synthetic missingness behaviour.
        if all(np.issubdtype(np.dtype(dtype), np.floating) for dtype in source.feature_dtypes):
            block[rng.random(params.n_train) < params.missing_rate_group, :] = np.nan
        for feature_order, (feature, dtype_name) in enumerate(zip(feats, source.feature_dtypes)):
            columns[feature] = block[:, feature_order].astype(np.dtype(dtype_name))
            manifest_rows.append(
                {
                    "source_column": base,
                    "feature_name": feature,
                    "keep": True,
                    "source_order": source_order,
                    "feature_order": feature_order,
                    "unit_type": "onehot_group",
                    "drop_first": False,
                    "is_reference": feature_order == 0,
                    "reference_level": 0.0,
                    "level_value": float(feature_order),
                    "ordinal_levels": None,
                    "source_prior": None,
                }
            )

    outcome_values = rng.normal(size=params.n_train)
    frame = pd.DataFrame(columns)
    frame.insert(0, params.outcome, outcome_values)
    manifest = pd.DataFrame(manifest_rows)
    predictors = [column for column in frame.columns if column != params.outcome]
    predictor_values = frame.loc[:, predictors]
    missing_mask = predictor_values.isna()
    columns_with_missing = int(missing_mask.any(axis=0).sum())
    missing_cells = int(missing_mask.to_numpy().sum())
    groups = source_groups(predictors, manifest)
    definition = canonical_feature_universe(predictors, groups, manifest)

    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.parquet"
    manifest_path = root / "feature_manifest.csv"
    definition_path = root / "feature_universe.json"
    frame.to_parquet(train_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    definition_path.write_text(
        json.dumps(definition, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    schema = {
        "schema_version": 1,
        "feature_manifest_version": 1,
        "dataset": "synthetic-calibration",
        "table": "train.parquet",
        "test_table": None,
        "split_mode": "internal_random",
        "task": "regression",
        "outcome_columns": [params.outcome],
        "id_column": None,
        "predictor_columns": predictors,
        "predictor_prefix": None,
        "feature_manifest": "feature_manifest.csv",
        "exchangeable": True,
        "feature_universe": {
            "mode": "fixed_a_priori",
            "definition_file": "feature_universe.json",
        },
        "group_column": None,
        "imputation": {
            "continuous": "median",
            "ordinal": "median_snap",
            "onehot_group": "atomic_mode",
            "model_overrides": {"lightgbm": "passthrough", "xgboost": "passthrough"},
        },
        "max_train_outcome_missing_ratio": 0.5,
        "max_test_outcome_missing_ratio": 0.5,
        "continuous_priors": None,
    }
    schema_path = root / "schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stats = {
        "n_train": params.n_train,
        "n_sources": params.shape.n_sources,
        "n_continuous_sources": sum(source.unit_type == "continuous" for source in params.shape.sources),
        "n_onehot_sources": len(params.shape.onehot_group_sizes),
        "n_expanded_predictors": len(predictors),
        "seed": params.seed,
        "shape": params.shape.as_dict(),
        "missing_rate_continuous": params.missing_rate_continuous,
        "missing_rate_group": params.missing_rate_group,
        # These are observed in the generated panel, not a proxy inferred from
        # configured injection rates. They reveal, for example, a valid but
        # uninformative all-integer profile that necessarily contains no NaNs.
        "observed_missingness": {
            "expanded_columns_with_nan": columns_with_missing,
            "expanded_columns_with_nan_fraction": columns_with_missing / len(predictors),
            "missing_cells": missing_cells,
            "total_predictor_cells": int(predictor_values.size),
            "missing_cells_fraction": missing_cells / int(predictor_values.size),
        },
        "outcome": params.outcome,
        "table_format": "parquet",
    }
    return schema_path, stats


def assert_frame_determinism(params: SyntheticDataParams, tmp_a: Path, tmp_b: Path) -> None:
    """Generate the same bundle twice and assert exact value/dtype equality."""

    _, stats_a = generate_synthetic_bundle(tmp_a, params)
    _, stats_b = generate_synthetic_bundle(tmp_b, params)
    frame_a = pd.read_parquet(tmp_a / "train.parquet")
    frame_b = pd.read_parquet(tmp_b / "train.parquet")
    pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)
    assert stats_a == stats_b


# ---------------------------------------------------------------------------
# t0 measurement (fresh-process, component split)
# ---------------------------------------------------------------------------

_T0_IMPORT_SUBPROCESS_SCRIPT = """
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.nk_grid import draw_orders, split_frame
from aleatoric_nk_grid.preprocessing import source_groups
"""

_T0_COMPONENTS_SUBPROCESS_SCRIPT = """
import json
import sys
import time

from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.nk_grid import draw_orders, split_frame
from aleatoric_nk_grid.preprocessing import source_groups

schema_path = sys.argv[1]
outcome = sys.argv[2]

t_load_start = time.perf_counter()
loaded = load_input(schema_path, outcome)
t_load_end = time.perf_counter()

t_split_start = time.perf_counter()
split = split_frame(
    loaded.train, loaded.predictors, outcome, test_size=0.2, seed=0, task="regression"
)
t_split_end = time.perf_counter()

groups = source_groups(loaded.predictors, loaded.manifest)
feature_units = [group.name for group in groups]

t_orders_start = time.perf_counter()
draw_orders(split.X_train.index, feature_units, seed=0, draw=0)
t_orders_end = time.perf_counter()

print(json.dumps({
    "t_load": t_load_end - t_load_start,
    "t_split": t_split_end - t_split_start,
    "t_orders": t_orders_end - t_orders_start,
}))
"""


def t0_component_consistency(
    component_total_seconds: float,
    end_to_end_seconds: float,
    *,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    """Describe whether independent startup timing agrees with component sums."""

    if component_total_seconds < 0 or end_to_end_seconds <= 0:
        raise ValueError("t0 consistency requires non-negative component and positive end-to-end times")
    if tolerance < 0:
        raise ValueError("t0 consistency tolerance must be non-negative")
    difference = abs(component_total_seconds - end_to_end_seconds)
    relative_difference = difference / end_to_end_seconds
    return {
        "component_total_median": component_total_seconds,
        "end_to_end_median": end_to_end_seconds,
        "absolute_difference_seconds": difference,
        "relative_difference": relative_difference,
        "tolerance": tolerance,
        "exceeds_tolerance": relative_difference > tolerance,
    }


def measure_t0(
    schema_path: Path,
    outcome: str,
    *,
    n_reps: int = T0_REPS,
    python_executable: str | None = None,
    env: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure all t0 components and an independent end-to-end startup path.

    Each observation uses a fresh interpreter. ``import`` is an outer
    wall-clock measurement of interpreter launch plus the engine imports;
    ``load``, ``split``, and ``orders`` are measured within separate full-path
    processes. ``end_to_end`` independently times that full path from before
    interpreter launch through order construction, exposing unaccounted setup.
    """

    import os

    executable = python_executable or sys.executable
    run_env = dict(env if env is not None else os.environ)
    src_root = str(Path(__file__).resolve().parents[1])
    run_env["PYTHONPATH"] = src_root + (
        (":" + run_env["PYTHONPATH"]) if run_env.get("PYTHONPATH") else ""
    )

    t_load: list[float] = []
    t_split: list[float] = []
    t_orders: list[float] = []
    t_import: list[float] = []
    t_end_to_end: list[float] = []
    for _ in range(n_reps):
        import_started = clock()
        subprocess.run(
            [executable, "-c", _T0_IMPORT_SUBPROCESS_SCRIPT],
            capture_output=True,
            text=True,
            env=run_env,
            check=True,
        )
        t_import.append(clock() - import_started)

        end_to_end_started = clock()
        result = subprocess.run(
            [executable, "-c", _T0_COMPONENTS_SUBPROCESS_SCRIPT, str(schema_path), outcome],
            capture_output=True,
            text=True,
            env=run_env,
            check=True,
        )
        t_end_to_end.append(clock() - end_to_end_started)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        t_load.append(float(payload["t_load"]))
        t_split.append(float(payload["t_split"]))
        t_orders.append(float(payload["t_orders"]))

    import_summary = summarize_reps(t_import)
    import_summary["source"] = "measured_in_current_runtime_fresh_interpreter"
    load_summary = summarize_reps(t_load)
    split_summary = summarize_reps(t_split)
    orders_summary = summarize_reps(t_orders)
    component_total_median = (
        import_summary["median"]
        + load_summary["median"]
        + split_summary["median"]
        + orders_summary["median"]
    )
    return {
        "import": import_summary,
        "load": load_summary,
        "split": split_summary,
        "orders": orders_summary,
        "total": {"median": component_total_median, "n_reps": n_reps},
        "end_to_end": summarize_reps(t_end_to_end),
        "component_vs_end_to_end": t0_component_consistency(
            component_total_median, float(np.median(t_end_to_end))
        ),
    }


# ---------------------------------------------------------------------------
# Power-law regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerLawFit:
    log_c: float
    a: float
    b: float
    r2: float
    residual_range: tuple[float, float]
    n_points: int


def fit_power_law(
    n_values: Sequence[float], k_values: Sequence[float], t_values: Sequence[float]
) -> PowerLawFit:
    """Fit log(t) = log(c) + a*log(K) + b*log(N) via ordinary least squares."""

    n_arr = np.asarray(n_values, dtype=float)
    k_arr = np.asarray(k_values, dtype=float)
    t_arr = np.asarray(t_values, dtype=float)
    if len(n_arr) != len(k_arr) or len(n_arr) != len(t_arr):
        raise ValueError("n_values, k_values, t_values must have equal length")
    if len(n_arr) < 3:
        raise ValueError("fit_power_law requires at least 3 points")
    if np.any(t_arr <= 0) or np.any(n_arr <= 0) or np.any(k_arr <= 0):
        raise ValueError("fit_power_law requires strictly positive N, K, t values")

    y = np.log(t_arr)
    design = np.column_stack([np.ones_like(y), np.log(k_arr), np.log(n_arr)])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    log_c, a, b = (float(v) for v in coefficients)

    predicted = design @ coefficients
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return PowerLawFit(
        log_c=log_c,
        a=a,
        b=b,
        r2=r2,
        residual_range=(float(np.min(residuals)), float(np.max(residuals))),
        n_points=len(y),
    )


def predict_power_law(fit: PowerLawFit, n: float, k: float) -> float:
    return float(math.exp(fit.log_c) * (k**fit.a) * (n**fit.b))


# ---------------------------------------------------------------------------
# Measurement harness (Stage A / Stage B / peak RSS)
# ---------------------------------------------------------------------------


@dataclass
class RawMeasurement:
    model: str
    n: int
    k: int
    rep: int
    fit_seconds: float
    preprocess_seconds: float
    preprocess_mode: str
    peak_rss_bytes: int
    stage: str
    # ``peak_rss_bytes`` remains the fitted, whole-tree cell peak for
    # compatibility with the cost model name.  The following fields preserve
    # the distinct measurement scopes instead of silently conflating them.
    process_peak_rss_bytes: int = 0
    cell_cgroup_peak_bytes: int = 0
    cell_memory_method: str = MEMORY_METHOD_PROCESS_TREE
    cell_memory_sampling_interval_seconds: float | None = MEMORY_SAMPLE_INTERVAL_SECONDS
    cell_memory_sampling_interval_max_seconds: float | None = MEMORY_SAMPLE_INTERVAL_SECONDS
    memory_scope_suspect: bool = False
    preprocess_vectorized: bool = False


@dataclass(frozen=True)
class MemoryPeak:
    """A whole-workload peak with an explicit collection method.

    ``process_tree_rss_sampled_conservative_upper_bound`` is intentionally
    verbose: summing RSS across a tree double-counts shared pages and must not
    be presented as an exact cgroup number.
    """

    bytes: int
    method: str
    sampling_interval_seconds: float | None
    sampling_interval_max_seconds: float | None = None
    samples: int = 0


@dataclass(frozen=True)
class CellMeasurementRequest:
    """One complete cell workload for the n_jobs=8 memory probe."""

    model_name: str
    n: int
    k: int
    seed: int = 0
    draw: int = 0
    max_seconds: float = DEFAULT_MAX_SECONDS


@dataclass
class CalibrationSession:
    """Loaded/validated synthetic panel plus per-seed splits, reused across
    every (N, K, model) measurement so generation/validation cost is paid
    once, matching how the production engine amortizes it across a run."""

    schema_path: Path
    outcome: str
    task: str
    predictors: tuple[str, ...]
    frame: pd.DataFrame
    groups: tuple[Any, ...]
    feature_units: tuple[str, ...]
    feature_groups: dict[str, tuple[str, ...]]
    groups_by_name: dict[str, Any]
    imputation: Mapping[str, Any]
    split: SplitData
    model_params: dict[str, dict[str, Any]]
    # Lazily started: only spawned the first time a SUBPROCESS_MODELS model is
    # measured, mirroring production's one-reusable-worker design (see
    # nk_grid._run_native_model_cell_locked / native_process.IsolatedProcessRunner).
    native_runner: IsolatedProcessRunner | None = None

    def orders_for(self, seed: int, draw: int) -> DrawOrders:
        return draw_orders(
            self.split.X_train.index, list(self.feature_units), seed=seed, draw=draw
        )


def _native_runner(session: CalibrationSession) -> IsolatedProcessRunner:
    """Return the session's reusable isolated-subprocess worker, starting it lazily."""

    if session.native_runner is None:
        session.native_runner = IsolatedProcessRunner()
    return session.native_runner


def close_session(session: CalibrationSession) -> None:
    """Release the isolated subprocess worker, if one was ever started."""

    if session.native_runner is not None:
        session.native_runner.close()
        session.native_runner = None


def build_session(
    schema_path: Path,
    outcome: str,
    *,
    seed: int = 0,
    test_size: float = 0.2,
    allowed_roots: Sequence[Path | str] = (),
) -> CalibrationSession:
    guard_not_private_data(schema_path, allowed_roots=allowed_roots)
    raw_loaded = load_input(schema_path, outcome)
    loaded, groups = validate_input(
        raw_loaded,
        outcome,
        models=list(MODELS),
        min_n=10,
        test_size=test_size,
        seed=seed,
    )
    task = loaded.schema.task
    predictors = list(loaded.predictors)
    feature_units = tuple(group.name for group in groups)
    feature_groups = {group.name: tuple(group.features) for group in groups}
    groups_by_name = {group.name: group for group in groups}
    split = split_frame(
        loaded.train, predictors, outcome, test_size=test_size, seed=seed, task=task
    )
    model_params = {
        model_name: load_model_params(
            DEFAULT_MODEL_PARAMS_PATH, task=task, models=[model_name]
        )[model_name]
        for model_name in MODELS
    }
    return CalibrationSession(
        schema_path=schema_path,
        outcome=outcome,
        task=task,
        predictors=tuple(predictors),
        frame=loaded.train,
        groups=groups,
        feature_units=feature_units,
        feature_groups=feature_groups,
        groups_by_name=groups_by_name,
        imputation=loaded.schema.imputation,
        split=split,
        model_params=model_params,
    )


def _measure_one_cell_in_process(
    session: CalibrationSession,
    *,
    model_name: str,
    n: int,
    k: int,
    seed: int,
    draw: int,
    max_seconds: float,
) -> RawMeasurement:
    """Run the cell in the current process.

    This is deliberately private.  Public calibration measurements call it
    only from a new ``spawn`` child so ``RUSAGE_SELF`` cannot inherit a prior
    grid point's high-water mark.
    """

    orders = session.orders_for(seed, draw)
    selected_rows = orders.row_index[:n]
    selected_units = [str(unit) for unit in orders.feature_names[:k]]
    selected_cols = [
        feature
        for unit in selected_units
        for feature in session.feature_groups[unit]
    ]
    selected_groups = [session.groups_by_name[unit] for unit in selected_units]

    X_sub_raw = session.split.X_train.loc[selected_rows, selected_cols]
    y_sub = session.split.y_train.loc[selected_rows]
    X_test_raw = session.split.X_test.loc[:, selected_cols]

    mode = (
        "passthrough"
        if session.imputation["model_overrides"].get(model_name) == "passthrough"
        else "imputed"
    )

    with time_budget(max_seconds):
        preprocess_started = time.perf_counter()
        prepared = preprocess_cell(
            X_sub_raw, X_test_raw, selected_groups, session.imputation, model_name=model_name
        )
        preprocess_seconds = time.perf_counter() - preprocess_started
        preprocess_vectorized = bool(
            prepared.X_train.attrs.get("_preprocess_vectorized", False)
        )

        params = session.model_params[model_name]
        fit_arguments = {
            "model_name": model_name,
            "model_seed": seed,
            "task": session.task,
            "params": params,
            "X_train": prepared.X_train,
            "y_train": y_sub,
            "X_test": prepared.X_test,
            "model_n_jobs": 1,
        }
        if model_name in SUBPROCESS_MODELS:
            # Route through the real isolated-subprocess path so peak RSS
            # reflects the child process that actually does the fitting (as
            # production does via _run_native_model_cell_locked), not just
            # this harness's own RUSAGE_SELF. _fit_predict_model_cell runs
            # inside that child and reports its own RUSAGE_SELF back over the
            # pipe, which is exactly what production's diagnostics column
            # consumes (see nk_grid.py: fit_result["peak_rss_bytes"]).
            try:
                result = _run_native_model_cell_locked(
                    _native_runner(session),
                    fit_arguments=fit_arguments,
                    on_native_crash=lambda attempt, exc: None,
                    on_native_timeout=lambda attempt, exc: None,
                )
            except BaseException:
                # A censored (SIGALRM-interrupted) or crashed call can leave
                # the reused worker mid-request; discard it so the next
                # measurement starts a fresh, known-good subprocess instead
                # of reusing a pipe with an outstanding/mismatched response.
                close_session(session)
                raise
        else:
            result = _fit_predict_model_cell(**fit_arguments)

    return RawMeasurement(
        model=model_name,
        n=n,
        k=k,
        rep=draw,
        fit_seconds=float(result["fit_seconds"]),
        preprocess_seconds=float(preprocess_seconds),
        preprocess_mode=mode,
        # The production diagnostic is intentionally not used for calibration:
        # it is a lifetime high-water mark in a reusable process.  The parent
        # sampler fills the real cell peak after this private call returns.
        peak_rss_bytes=0,
        stage="",
        preprocess_vectorized=preprocess_vectorized,
    )


def _cgroup_v2_path_for_pid(pid: int) -> Path | None:
    """Return the v2 cgroup path for *pid*, or None off Linux/cgroup v2."""

    cgroup_file = Path(f"/proc/{pid}/cgroup")
    root = Path("/sys/fs/cgroup")
    if not cgroup_file.exists() or not root.exists():
        return None
    try:
        for line in cgroup_file.read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                path = root / relative.lstrip("/")
                return path if path.exists() else None
    except (OSError, ValueError):
        return None
    return None


def _create_measurement_cgroup() -> Path | None:
    """Create an empty cgroup before the spawned worker exists.

    The child joins itself as its first action.  Parent-side post-start
    migration loses allocations made before the move from cgroup accounting.
    """

    parent = _cgroup_v2_path_for_pid(os.getpid())
    if parent is None:
        return None
    child = parent / f"cost-calibration-{os.getpid()}-{time.monotonic_ns()}"
    try:
        child.mkdir()
        return child
    except OSError:
        try:
            child.rmdir()
        except OSError:
            pass
        return None


def _remove_measurement_cgroup(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.rmdir()
    except OSError:
        # A process can briefly remain while its multiprocessing handle has
        # exited.  Leaving an empty, uniquely-named cgroup is safer than
        # moving another task's processes while cleaning up.
        pass


def _read_cgroup_bytes(path: Path, filename: str) -> int | None:
    try:
        value = (path / filename).read_text(encoding="ascii").strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def _process_tree_rss_bytes(root_pid: int) -> int:
    """Return RSS summed over a process tree (a conservative upper bound)."""

    proc_root = Path("/proc")
    if proc_root.exists():
        return _linux_process_tree_rss_bytes(root_pid, proc_root)

    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    parents: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        parents.setdefault(ppid, []).append(pid)
        rss[pid] = rss_kib * 1024
    descendants = [root_pid]
    seen: set[int] = set()
    total = 0
    while descendants:
        pid = descendants.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        descendants.extend(parents.get(pid, ()))
    return total


def _linux_process_tree_rss_bytes(root_pid: int, proc_root: Path = Path("/proc")) -> int:
    """Read Linux procfs directly; never fork ``ps`` on a sampling path."""

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        entries = tuple(proc_root.iterdir())
    except OSError:
        return 0
    parents: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat = (entry / "stat").read_text(encoding="utf-8")
            # comm may contain spaces or parentheses; fields after the final
            # ')' start with state (field 3), then ppid (field 4).
            fields = stat.rsplit(")", 1)[1].split()
            ppid = int(fields[1])
            statm_fields = (entry / "statm").read_text(encoding="ascii").split()
            rss[pid] = int(statm_fields[1]) * page_size
        except (IndexError, OSError, ValueError):
            continue
        parents.setdefault(ppid, []).append(pid)
    descendants = [root_pid]
    seen: set[int] = set()
    total = 0
    while descendants:
        pid = descendants.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        descendants.extend(parents.get(pid, ()))
    return total


def _sampling_summary(timestamps: Sequence[float]) -> tuple[float | None, float | None]:
    if len(timestamps) < 2:
        return None, None
    intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    return float(sum(intervals) / len(intervals)), float(max(intervals))


def _memory_scope_suspect(cell_cgroup_peak_bytes: int, process_peak_rss_bytes: int) -> bool:
    """Flag, rather than reject, a cgroup result smaller than one process."""

    return cell_cgroup_peak_bytes < process_peak_rss_bytes


def _monitor_peak(process: mp.Process, cgroup: Path | None, *, interval: float) -> MemoryPeak:
    """Monitor one spawned process and all of its descendants until exit."""

    if cgroup is not None and (cgroup / "memory.peak").exists():
        process.join()
        peak = _read_cgroup_bytes(cgroup, "memory.peak")
        if peak is not None:
            return MemoryPeak(peak, MEMORY_METHOD_CGROUP_PEAK, None, None, 0)

    if cgroup is not None and (cgroup / "memory.current").exists():
        method = MEMORY_METHOD_CGROUP_CURRENT
        sample = lambda: _read_cgroup_bytes(cgroup, "memory.current") or 0
    else:
        method = MEMORY_METHOD_PROCESS_TREE
        sample = lambda: _process_tree_rss_bytes(process.pid or 0)

    peak = 0
    timestamps: list[float] = []
    while process.is_alive():
        timestamps.append(time.monotonic())
        peak = max(peak, sample())
        time.sleep(interval)
    # Take one final sample to include a short-lived child which completed
    # between polls where the platform still exposes it/cgroup accounting.
    timestamps.append(time.monotonic())
    peak = max(peak, sample())
    process.join()
    mean_interval, max_interval = _sampling_summary(timestamps)
    return MemoryPeak(peak, method, mean_interval, max_interval, len(timestamps))


def _process_peak_worker(connection: Any, target: Callable[..., None], args: tuple[Any, ...]) -> None:
    """Execute a picklable workload and report this fresh process's RSS peak."""

    try:
        target(*args)
        connection.send({"ok": True, "process_peak_rss_bytes": _process_peak_rss_bytes()})
    except BaseException as exc:
        connection.send({"ok": False, "type": type(exc).__name__, "message": str(exc)})
    finally:
        connection.close()


def measure_process_peak_rss(
    target: Callable[..., None], *args: Any
) -> int:
    """Measure one workload in a fresh ``spawn`` worker, then discard it.

    This diagnostic deliberately says nothing about worker children; callers
    needing allocation-level memory must use one of the cgroup/tree routines.
    """

    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_process_peak_worker, args=(child_connection, target, args))
    process.start()
    child_connection.close()
    process.join()
    try:
        if not parent_connection.poll():
            raise RuntimeError(f"spawned RSS worker exited without a result (exitcode={process.exitcode})")
        response = parent_connection.recv()
    finally:
        parent_connection.close()
    if not response["ok"]:
        raise RuntimeError(f"spawned RSS worker {response['type']}: {response['message']}")
    return int(response["process_peak_rss_bytes"])


def _run_cell_worker_after_join(
    result_path: str,
    cgroup_joined: bool,
    schema_path: str,
    outcome: str,
    seed: int,
    model_name: str,
    n: int,
    k: int,
    draw: int,
    max_seconds: float,
) -> None:
    """Heavy cell implementation, imported only after the worker joins."""

    session: CalibrationSession | None = None
    try:
        session = build_session(Path(schema_path), outcome, seed=seed)
        measurement = _measure_one_cell_in_process(
            session, model_name=model_name, n=n, k=k, seed=seed, draw=draw,
            max_seconds=max_seconds,
        )
        response = {
            "ok": True,
            "measurement": _raw_to_dict(measurement),
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "cgroup_joined": cgroup_joined,
        }
    except BaseException as exc:
        response = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    finally:
        if session is not None:
            close_session(session)
    Path(result_path).write_text(json.dumps(response), encoding="utf-8")


def _raw_measurement_from_dict(row: Mapping[str, Any]) -> RawMeasurement:
    return RawMeasurement(
        model=str(row["model"]), n=int(row["n"]), k=int(row["k"]), rep=int(row["rep"]),
        fit_seconds=float(row["fit_seconds"]), preprocess_seconds=float(row["preprocess_seconds"]),
        preprocess_mode=str(row["preprocess_mode"]), peak_rss_bytes=int(row["peak_rss_bytes"]),
        stage=str(row["stage"]),
        process_peak_rss_bytes=int(row.get("process_peak_rss_bytes", 0)),
        cell_cgroup_peak_bytes=int(row.get("cell_cgroup_peak_bytes", 0)),
        cell_memory_method=str(row.get("cell_memory_method", MEMORY_METHOD_PROCESS_TREE)),
        cell_memory_sampling_interval_seconds=row.get("cell_memory_sampling_interval_seconds"),
        cell_memory_sampling_interval_max_seconds=row.get(
            "cell_memory_sampling_interval_max_seconds"
        ),
        memory_scope_suspect=bool(row.get("memory_scope_suspect", False)),
        preprocess_vectorized=bool(row.get("preprocess_vectorized", False)),
    )


def measure_one_cell(
    session: CalibrationSession,
    *,
    model_name: str,
    n: int,
    k: int,
    seed: int,
    draw: int,
    max_seconds: float,
) -> RawMeasurement:
    """Measure a cell in a fresh ``spawn`` process, never a forked child."""

    context = mp.get_context("spawn")
    cgroup = _create_measurement_cgroup()
    with tempfile.TemporaryDirectory(prefix="nk-grid-cell-memory-") as temporary_dir:
        result_path = str(Path(temporary_dir) / "result.json")
        process = context.Process(
            target=cell_worker_target,
            args=(None if cgroup is None else str(cgroup), result_path,
                  str(session.schema_path), session.outcome, seed,
                  model_name, n, k, draw, max_seconds, None, 0),
        )
        process.start()
        try:
            cell_peak = _monitor_peak(process, cgroup, interval=MEMORY_SAMPLE_INTERVAL_SECONDS)
            if not Path(result_path).exists():
                raise RuntimeError(
                    f"spawned cell worker exited without a result (exitcode={process.exitcode})"
                )
            response = json.loads(Path(result_path).read_text(encoding="utf-8"))
        finally:
            _remove_measurement_cgroup(cgroup)
    if not response["ok"]:
        if response["type"] == "MeasurementCensored":
            raise MeasurementCensored(response["message"])
        raise RuntimeError(f"spawned cell worker {response['type']}: {response['message']}")
    measurement = _raw_measurement_from_dict(response["measurement"])
    measurement.process_peak_rss_bytes = int(response["process_peak_rss_bytes"])
    measurement.cell_cgroup_peak_bytes = cell_peak.bytes
    measurement.peak_rss_bytes = cell_peak.bytes
    measurement.cell_memory_method = cell_peak.method
    measurement.cell_memory_sampling_interval_seconds = cell_peak.sampling_interval_seconds
    measurement.cell_memory_sampling_interval_max_seconds = cell_peak.sampling_interval_max_seconds
    measurement.memory_scope_suspect = _memory_scope_suspect(
        measurement.cell_cgroup_peak_bytes, measurement.process_peak_rss_bytes
    ) or (cgroup is not None and not bool(response.get("cgroup_joined")))
    return measurement


def _monitor_processes(
    processes: Sequence[mp.Process], cgroup: Path | None, *, interval: float
) -> MemoryPeak:
    """As ``_monitor_peak``, but for the complete concurrent task scope."""

    if cgroup is not None and (cgroup / "memory.peak").exists():
        for process in processes:
            process.join()
        peak = _read_cgroup_bytes(cgroup, "memory.peak")
        if peak is not None:
            return MemoryPeak(peak, MEMORY_METHOD_CGROUP_PEAK, None, None, 0)
    if cgroup is not None and (cgroup / "memory.current").exists():
        method = MEMORY_METHOD_CGROUP_CURRENT
        sample = lambda: _read_cgroup_bytes(cgroup, "memory.current") or 0
    else:
        method = MEMORY_METHOD_PROCESS_TREE
        sample = lambda: sum(_process_tree_rss_bytes(process.pid or 0) for process in processes)
    peak = 0
    timestamps: list[float] = []
    while any(process.is_alive() for process in processes):
        timestamps.append(time.monotonic())
        peak = max(peak, sample())
        time.sleep(interval)
    timestamps.append(time.monotonic())
    peak = max(peak, sample())
    for process in processes:
        process.join()
    mean_interval, max_interval = _sampling_summary(timestamps)
    return MemoryPeak(peak, method, mean_interval, max_interval, len(timestamps))


def _measure_task_peak_n_jobs_8(
    worker_args: Sequence[tuple[Any, ...]],
    *,
    sample_interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS,
    observation_paths: Sequence[str | None] | None = None,
    probe_only: int = 0,
) -> MemoryPeak:
    """Measure the memory of one complete eight-worker task.

    Every worker argument is a primitive value.  The lightweight fixed target
    joins the cgroup before importing this module and reconstructing the cell.
    """

    if len(worker_args) != 8:
        raise ValueError("task_cgroup_peak_n_jobs_8 requires exactly eight workers")
    if observation_paths is None:
        observation_paths = (None,) * 8
    if len(observation_paths) != 8:
        raise ValueError("task memory observation_paths must contain eight entries")
    context = mp.get_context("spawn")
    cgroup = _create_measurement_cgroup()
    cgroup_path = None if cgroup is None else str(cgroup)
    processes = [
        context.Process(
            target=task_cell_worker_target,
            args=(cgroup_path, *args, observation_path, probe_only),
        )
        for args, observation_path in zip(worker_args, observation_paths)
    ]
    for process in processes:
        process.start()
    try:
        return _monitor_processes(processes, cgroup, interval=sample_interval_seconds)
    finally:
        _remove_measurement_cgroup(cgroup)


def _run_task_cell_after_join(
    schema_path: str,
    outcome: str,
    model_name: str,
    n: int,
    k: int,
    seed: int,
    draw: int,
    max_seconds: float,
) -> None:
    """Rebuild a task cell only after its lightweight target has joined."""

    session: CalibrationSession | None = None
    try:
        session = build_session(Path(schema_path), outcome, seed=seed)
        _measure_one_cell_in_process(
            session,
            model_name=model_name,
            n=n,
            k=k,
            seed=seed,
            draw=draw,
            max_seconds=max_seconds,
        )
    finally:
        if session is not None:
            close_session(session)


def measure_task_cgroup_peak_n_jobs_8(
    session: CalibrationSession,
    requests: Sequence[CellMeasurementRequest],
    *,
    sample_interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS,
) -> MemoryPeak:
    """Measure a real complete eight-cell calibration task.

    This is intentionally an explicit caller-controlled task composition:
    only the scheduler knows which eight cells will actually overlap in the
    production resource class.  Its result is the sole memory number suitable
    for ``--mem``.
    """

    return _measure_task_peak_n_jobs_8(
        [
            (
                str(session.schema_path),
                session.outcome,
                request.model_name,
                request.n,
                request.k,
                request.seed,
                request.draw,
                request.max_seconds,
            )
            for request in requests
        ],
        sample_interval_seconds=sample_interval_seconds,
    )


# ---------------------------------------------------------------------------
# Parallel-efficiency measurement (independent from memory probes)
# ---------------------------------------------------------------------------


def parallel_efficiency(
    t1_seconds: float, t8_seconds: float, *, n_jobs: int = 8
) -> float:
    """Return eta = T1 / (n_jobs * Tn), with no machine-dependent defaults."""

    if n_jobs < 2:
        raise ValueError("parallel efficiency requires at least two jobs")
    if t1_seconds <= 0 or t8_seconds <= 0:
        raise ValueError("parallel efficiency requires positive wall-clock measurements")
    return float(t1_seconds / (n_jobs * t8_seconds))


def _parallel_efficiency_worker_args(
    session: CalibrationSession, requests: Sequence[CellMeasurementRequest]
) -> list[tuple[Any, ...]]:
    if len(requests) != 8:
        raise ValueError("parallel efficiency requires exactly eight fixed cell workloads")
    return [
        (
            str(session.schema_path),
            session.outcome,
            request.model_name,
            request.n,
            request.k,
            request.seed,
            request.draw,
            request.max_seconds,
        )
        for request in requests
    ]


def _run_parallel_efficiency_cells(
    worker_args: Sequence[tuple[Any, ...]], *, n_jobs: int
) -> None:
    """Run fixed cell workloads via spawn without cgroups or memory sampling."""

    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    context = mp.get_context("spawn")
    pending = iter(worker_args)
    while True:
        batch = list()
        for _ in range(n_jobs):
            try:
                args = next(pending)
            except StopIteration:
                break
            process = context.Process(target=parallel_efficiency_cell_worker_target, args=args)
            process.start()
            batch.append(process)
        if not batch:
            return
        for process in batch:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    "parallel-efficiency spawn worker failed "
                    f"(pid={process.pid}, exitcode={process.exitcode})"
                )


def measure_parallel_efficiency(
    session: CalibrationSession,
    requests: Sequence[CellMeasurementRequest],
    *,
    n_reps: int = PARALLEL_EFFICIENCY_REPS,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure serial and n_jobs=8 wall clocks for the same fixed eight cells.

    This deliberately shares neither a cgroup nor a memory sampler with the
    item-1 task-memory probe.  It exists solely to estimate the time-model
    denominator used by Stage B's multi-CPU split decision.
    """

    if n_reps < 1:
        raise ValueError("parallel efficiency n_reps must be positive")
    worker_args = _parallel_efficiency_worker_args(session, requests)
    t1_values: list[float] = []
    t8_values: list[float] = []
    for _ in range(n_reps):
        started = clock()
        _run_parallel_efficiency_cells(worker_args, n_jobs=1)
        t1_values.append(clock() - started)
        started = clock()
        _run_parallel_efficiency_cells(worker_args, n_jobs=8)
        t8_values.append(clock() - started)
    t1_summary = summarize_reps(t1_values)
    t8_summary = summarize_reps(t8_values)
    return {
        "worker_start_method": "spawn",
        "n_jobs": 8,
        "n_reps": n_reps,
        "cell_workloads": [
            {"model": request.model_name, "n": request.n, "k": request.k,
             "seed": request.seed, "draw": request.draw}
            for request in requests
        ],
        "t1_seconds": t1_summary,
        "t8_seconds": t8_summary,
        "eta": parallel_efficiency(t1_summary["median"], t8_summary["median"]),
    }


def run_stage_a(
    session: CalibrationSession,
    *,
    n_grid: Sequence[int] = DEFAULT_STAGE_A_N,
    k_grid: Sequence[int] = DEFAULT_STAGE_A_K_BASE,
    n_reps: int = STAGE_A_REPS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    models: Sequence[str] = MODELS,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[RawMeasurement], list[dict[str, Any]]]:
    """Cheap grid used to fit the per-model power laws. Returns (raw, censored)."""

    raw: list[RawMeasurement] = []
    censored: list[dict[str, Any]] = []
    for model_name in models:
        for n in n_grid:
            for k in k_grid:
                for rep in range(n_reps):
                    if progress:
                        progress(f"stage A: model={model_name} N={n} K={k} rep={rep}")
                    try:
                        measurement = measure_one_cell(
                            session,
                            model_name=model_name,
                            n=n,
                            k=k,
                            seed=0,
                            draw=rep,
                            max_seconds=max_seconds,
                        )
                        measurement.stage = "A"
                        raw.append(measurement)
                    except MeasurementCensored:
                        censored.append(
                            {
                                "n": n,
                                "k": k,
                                "model": model_name,
                                "rep": rep,
                                "max_seconds": max_seconds,
                                "stage": "A",
                            }
                        )
    return raw, censored


def run_stage_b(
    session: CalibrationSession,
    *,
    points: Sequence[tuple[int, int]],
    max_seconds: float = DEFAULT_MAX_SECONDS,
    models: Sequence[str] = MODELS,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[RawMeasurement], list[dict[str, Any]]]:
    """One measurement per (N, K) validation point per model. Returns (raw, censored)."""

    raw: list[RawMeasurement] = []
    censored: list[dict[str, Any]] = []
    for n, k in points:
        for model_name in models:
            if progress:
                progress(f"stage B: model={model_name} N={n} K={k}")
            try:
                measurement = measure_one_cell(
                    session,
                    model_name=model_name,
                    n=n,
                    k=k,
                    seed=0,
                    draw=0,
                    max_seconds=max_seconds,
                )
                measurement.stage = "B"
                raw.append(measurement)
            except MeasurementCensored:
                censored.append(
                    {
                        "n": n,
                        "k": k,
                        "model": model_name,
                        "rep": 0,
                        "max_seconds": max_seconds,
                        "stage": "B",
                    }
                )
    return raw, censored


# ---------------------------------------------------------------------------
# Fitting / assembling the calibration payload
# ---------------------------------------------------------------------------


def fit_all_models(raw: Sequence[RawMeasurement], models: Sequence[str] = MODELS) -> dict[str, PowerLawFit | None]:
    fits: dict[str, PowerLawFit | None] = {}
    for model_name in models:
        points = [m for m in raw if m.model == model_name and m.stage == "A"]
        if len(points) < 3:
            fits[model_name] = None
            continue
        grouped: dict[tuple[int, int], list[float]] = {}
        for point in points:
            grouped.setdefault((point.n, point.k), []).append(point.fit_seconds)
        ns = [pair[0] for pair in grouped]
        ks = [pair[1] for pair in grouped]
        medians = [float(np.median(values)) for values in grouped.values()]
        fits[model_name] = fit_power_law(ns, ks, medians)
    return fits


def fit_preprocess_by_mode(raw: Sequence[RawMeasurement]) -> dict[str, PowerLawFit | None]:
    fits: dict[str, PowerLawFit | None] = {}
    modes = sorted({m.preprocess_mode for m in raw if m.stage == "A"})
    for mode in modes:
        points = [m for m in raw if m.stage == "A" and m.preprocess_mode == mode]
        grouped: dict[tuple[int, int], list[float]] = {}
        for point in points:
            grouped.setdefault((point.n, point.k), []).append(point.preprocess_seconds)
        ns = [pair[0] for pair in grouped]
        ks = [pair[1] for pair in grouped]
        medians = [float(np.median(values)) for values in grouped.values()]
        positive = [v > 0 for v in medians]
        if len(medians) < 3 or not all(positive):
            fits[mode] = None
            continue
        fits[mode] = fit_power_law(ns, ks, medians)
    return fits


def fit_peak_rss(raw: Sequence[RawMeasurement], models: Sequence[str] = MODELS) -> dict[str, PowerLawFit | None]:
    fits: dict[str, PowerLawFit | None] = {}
    for model_name in models:
        points = [m for m in raw if m.model == model_name and m.stage == "A"]
        grouped: dict[tuple[int, int], list[float]] = {}
        for point in points:
            grouped.setdefault((point.n, point.k), []).append(float(point.peak_rss_bytes))
        ns = [pair[0] for pair in grouped]
        ks = [pair[1] for pair in grouped]
        medians = [float(np.median(values)) for values in grouped.values()]
        if len(medians) < 3 or any(v <= 0 for v in medians):
            fits[model_name] = None
            continue
        fits[model_name] = fit_power_law(ns, ks, medians)
    return fits


def build_validation_rows(
    raw_b: Sequence[RawMeasurement],
    censored_b: Sequence[Mapping[str, Any]],
    fits: Mapping[str, PowerLawFit | None],
    points: Sequence[tuple[int, int]],
    models: Sequence[str],
    *,
    not_measured_points: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    censored_keys = {(c["n"], c["k"], c["model"]) for c in censored_b}
    measured_by_key = {(m.n, m.k, m.model): m for m in raw_b}
    for n, k in points:
        for model_name in models:
            if (n, k) in not_measured_points:
                rows.append(
                    {
                        "n": n,
                        "k": k,
                        "model": model_name,
                        "predicted": None,
                        "actual": None,
                        "ratio": None,
                        "status": "not_measured",
                    }
                )
                continue
            key = (n, k, model_name)
            if key in censored_keys:
                rows.append(
                    {
                        "n": n,
                        "k": k,
                        "model": model_name,
                        "predicted": None,
                        "actual": None,
                        "ratio": None,
                        "status": "censored",
                    }
                )
                continue
            measurement = measured_by_key.get(key)
            fit = fits.get(model_name)
            if measurement is None or fit is None:
                rows.append(
                    {
                        "n": n,
                        "k": k,
                        "model": model_name,
                        "predicted": None,
                        "actual": None,
                        "ratio": None,
                        "status": "missing",
                    }
                )
                continue
            predicted = predict_power_law(fit, n, k)
            actual = measurement.fit_seconds
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "model": model_name,
                    "predicted": predicted,
                    "actual": actual,
                    "ratio": predicted / actual if actual else None,
                    "status": "measured",
                }
            )
    return rows


def _fit_to_dict(fit: PowerLawFit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "log_c": fit.log_c,
        "a": fit.a,
        "b": fit.b,
        "r2": fit.r2,
        "residual_range": list(fit.residual_range),
        "n_points": fit.n_points,
    }


def _raw_to_dict(measurement: RawMeasurement) -> dict[str, Any]:
    return {
        "model": measurement.model,
        "n": measurement.n,
        "k": measurement.k,
        "rep": measurement.rep,
        "fit_seconds": measurement.fit_seconds,
        "preprocess_seconds": measurement.preprocess_seconds,
        "preprocess_mode": measurement.preprocess_mode,
        "peak_rss_bytes": measurement.peak_rss_bytes,
        "process_peak_rss_bytes": measurement.process_peak_rss_bytes,
        "cell_cgroup_peak_bytes": measurement.cell_cgroup_peak_bytes,
        "cell_memory_method": measurement.cell_memory_method,
        "cell_memory_sampling_interval_seconds": measurement.cell_memory_sampling_interval_seconds,
        "cell_memory_sampling_interval_max_seconds": measurement.cell_memory_sampling_interval_max_seconds,
        "memory_scope_suspect": measurement.memory_scope_suspect,
        "preprocess_vectorized": measurement.preprocess_vectorized,
        "stage": measurement.stage,
    }


def recompute_fit_cost_from_raw(
    raw_measurements: Sequence[Mapping[str, Any]], models: Sequence[str] = MODELS
) -> dict[str, PowerLawFit | None]:
    """Recompute fit_cost coefficients from raw_measurements alone (round-trip check)."""

    reconstructed = [_raw_measurement_from_dict(row) for row in raw_measurements]
    return fit_all_models(reconstructed, models)


def build_calibration_payload(
    *,
    synthetic_params: SyntheticDataParams,
    synthetic_stats: Mapping[str, Any],
    t0_seconds: Mapping[str, Any],
    fit_cost: Mapping[str, PowerLawFit | None],
    preprocess_cost: Mapping[str, PowerLawFit | None],
    peak_rss: Mapping[str, PowerLawFit | None],
    validation: Sequence[Mapping[str, Any]],
    censored: Sequence[Mapping[str, Any]],
    raw_measurements: Sequence[RawMeasurement],
    thread_env_report: Mapping[str, Any],
    scope_reduction: Mapping[str, Any] | None = None,
    task_cgroup_peak_n_jobs_8: MemoryPeak | None = None,
    stage_a_k_grid: Sequence[int] | None = None,
    parallel_efficiency_measurement: Mapping[str, Any] | None = None,
    wall_clock_seconds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    environment = core_environment()
    environment["platform"] = _platform_string()
    environment["thread_env"] = thread_env_report["values"]
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "git_commit": git_state(repo_root()).get("commit"),
        "environment": environment,
        "synthetic_data": {
            # Every SyntheticDataParams field, written explicitly (not just
            # via the **synthetic_stats spread below) so a reader months from
            # now can see exactly what assumptions the fitted coefficients
            # rest on without cross-referencing the generator's source code
            # (round 1 review F3).
            "n_train": synthetic_params.n_train,
            "n_feature_units": synthetic_params.shape.n_sources,
            "p_onehot": synthetic_stats["n_expanded_predictors"],
            "seed": synthetic_params.seed,
            "panel_shape": synthetic_params.shape.as_dict(),
            "stage_a_k_grid": list(
                stage_a_k_grid if stage_a_k_grid is not None else k_grid_from_shape(synthetic_params.shape)
            ),
            "missing_rate_continuous": synthetic_params.missing_rate_continuous,
            "missing_rate_group": synthetic_params.missing_rate_group,
            "outcome": synthetic_params.outcome,
            **synthetic_stats,
        },
        "t0_seconds": t0_seconds,
        "fit_cost": {name: _fit_to_dict(fit) for name, fit in fit_cost.items()},
        "preprocess_cost": {name: _fit_to_dict(fit) for name, fit in preprocess_cost.items()},
        "peak_rss_bytes": {name: _fit_to_dict(fit) for name, fit in peak_rss.items()},
        "memory_measurement": {
            "process_peak_rss": "fresh_spawn_worker_RUSAGE_SELF",
            "cell_cgroup_peak": "per-raw-measurement fields",
            "cell_memory_scope_suspect_count": sum(
                measurement.memory_scope_suspect for measurement in raw_measurements
            ),
            "task_cgroup_peak_n_jobs_8": (
                {
                    "status": "not_measured",
                    "bytes": None,
                    "method": None,
                    "sampling_interval_seconds": None,
                    "sampling_interval_max_seconds": None,
                    "samples": 0,
                }
                if task_cgroup_peak_n_jobs_8 is None
                else {
                    "status": "measured",
                    "bytes": task_cgroup_peak_n_jobs_8.bytes,
                    "method": task_cgroup_peak_n_jobs_8.method,
                    "sampling_interval_seconds": task_cgroup_peak_n_jobs_8.sampling_interval_seconds,
                    "sampling_interval_max_seconds": task_cgroup_peak_n_jobs_8.sampling_interval_max_seconds,
                    "samples": task_cgroup_peak_n_jobs_8.samples,
                }
            ),
        },
        "parallel_efficiency": (
            {"status": "not_measured"}
            if parallel_efficiency_measurement is None
            else {"status": "measured", **dict(parallel_efficiency_measurement)}
        ),
        # Fitted estimators are owned inside nk_grid._fit_predict_model_cell
        # and that production entry point intentionally returns only numeric
        # results. This harness must not change production model code merely
        # to inspect lasso.n_iter_ / RidgeCV solver selection, so the absence
        # is explicit rather than silently omitted.
        "telemetry": {
            "status": "unavailable_without_production_entrypoint_change",
            "requested": ["lasso.n_iter_", "ridge.selected_solver"],
            "reason": (
                "fitted estimators are not exposed by the existing numeric-only "
                "production fit result; calibrate_cost does not modify nk_grid.py "
                "or model_registry.py"
            ),
        },
        "fit_quality": {
            "r2_threshold": R2_EXPLANATION_THRESHOLD,
            "models_below_r2_threshold": [
                {"model": name, "r2": fit.r2}
                for name, fit in sorted(fit_cost.items())
                if fit is not None and fit.r2 < R2_EXPLANATION_THRESHOLD
            ],
        },
        "wall_clock_seconds": (
            {"status": "not_measured"}
            if wall_clock_seconds is None
            else {"status": "measured", **dict(wall_clock_seconds)}
        ),
        "validation": list(validation),
        "censored": list(censored),
        "raw_measurements": [_raw_to_dict(m) for m in raw_measurements],
    }
    if scope_reduction is not None:
        payload["scope_reduction"] = dict(scope_reduction)
    return payload


def _platform_string() -> str:
    import platform

    return platform.platform()


def write_calibration_file(payload: Mapping[str, Any], out_dir: Path, *, date: str | None = None) -> Path:
    from datetime import datetime, timezone

    utc_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cost_model_{utc_date}.json"
    write_json_atomic(out_path, dict(payload))
    return out_path


def read_calibration_file(path: Path) -> dict[str, Any]:
    """Load only the current calibration schema; older RSS files are invalid."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported calibration format_version={payload.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}. Earlier peak_rss_bytes formats are invalid."
        )
    required = {
        "memory_measurement", "raw_measurements", "peak_rss_bytes", "synthetic_data",
        "parallel_efficiency", "telemetry", "fit_quality", "wall_clock_seconds",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"calibration file is missing required fields: {sorted(missing)}")
    shape_fields = {"panel_shape", "stage_a_k_grid"}
    missing_shape = shape_fields.difference(payload["synthetic_data"])
    if missing_shape:
        raise ValueError(f"calibration file is missing schema-derived shape fields: {sorted(missing_shape)}")
    missing_missingness = {"observed_missingness"}.difference(payload["synthetic_data"])
    if missing_missingness:
        raise ValueError(
            "calibration file is missing observed synthetic missingness fields: "
            f"{sorted(missing_missingness)}"
        )
    return payload


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure NK-Grid startup cost, per-model fit-time power laws, "
        "and peak RSS on synthetic data; never reads private data or model metrics."
    )
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--shape-schema",
        type=Path,
        required=True,
        help="Explicit feature-universe schema used to derive the synthetic panel shape.",
    )
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument(
        "--feature-dtype-profile",
        type=Path,
        default=None,
        help="JSON object mapping dtype names to expanded-feature counts from a read-only dtype probe.",
    )
    dtype_group.add_argument(
        "--assume-feature-dtype",
        type=str,
        default=None,
        help="Explicit uniform dtype assumption for schema features lacking dtype metadata.",
    )
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument(
        "--stage-a-n",
        type=str,
        default=",".join(str(value) for value in DEFAULT_STAGE_A_N),
        help="Comma-separated N values for the stage-A fit grid.",
    )
    parser.add_argument(
        "--stage-a-k-base",
        type=str,
        default=",".join(str(value) for value in DEFAULT_STAGE_A_K_BASE),
        help="Lower K anchors; the schema-derived source maximum is always included.",
    )
    parser.add_argument(
        "--stage-a-k-intermediate-points",
        type=int,
        default=DEFAULT_STAGE_A_K_INTERMEDIATE_POINTS,
        help="Number of geometric K probes between the largest lower anchor and schema maximum.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-nonproduction-threads",
        action="store_true",
        help="Warn instead of refusing to start when thread env vars != 1.",
    )
    parser.add_argument(
        "--stage-b-points",
        type=str,
        default=None,
        help="Comma-separated N:K pairs to actually measure in stage B. "
        "Default is derived from --n-train and the schema-derived K grid.",
    )
    parser.add_argument("--generation-time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--generation-rss-budget-bytes", type=int, default=6 * 1024**3)
    parser.add_argument(
        "--task-memory-cells",
        type=str,
        default=None,
        help=(
            "Exactly eight model:N:K cell workloads that will overlap in one "
            "n_jobs=8 production task, comma-separated. Measures and records "
            "task_cgroup_peak_n_jobs_8; omit only when no task composition is known."
        ),
    )
    parser.add_argument(
        "--parallel-efficiency-cells",
        type=str,
        default=None,
        help=(
            "Exactly eight fixed model:N:K cell workloads used for both the serial "
            "and n_jobs=8 spawn-only eta measurement; independent from --task-memory-cells."
        ),
    )
    parser.add_argument(
        "--parallel-efficiency-reps",
        type=int,
        default=PARALLEL_EFFICIENCY_REPS,
        help="Repeated serial/concurrent wall-clock pairs for eta.",
    )
    return parser.parse_args(argv)


def parse_task_memory_cells(specification: str, *, max_seconds: float) -> tuple[CellMeasurementRequest, ...]:
    """Parse the explicit eight-cell task shape used for the --mem probe."""

    requests: list[CellMeasurementRequest] = []
    for token in specification.split(","):
        try:
            model_name, n_text, k_text = token.split(":")
            request = CellMeasurementRequest(
                model_name=model_name, n=int(n_text), k=int(k_text), max_seconds=max_seconds
            )
        except ValueError as exc:
            raise ValueError(
                "--task-memory-cells must be eight comma-separated model:N:K entries"
            ) from exc
        if request.model_name not in MODELS or request.n < 1 or request.k < 1:
            raise ValueError(f"invalid task-memory cell {token!r}")
        requests.append(request)
    if len(requests) != 8:
        raise ValueError("--task-memory-cells must describe exactly eight n_jobs=8 workers")
    return tuple(requests)


def parse_positive_int_grid(specification: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in specification.split(","))
    except ValueError as exc:
        raise ValueError(f"{option} must be comma-separated positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{option} must be comma-separated positive integers")
    return tuple(dict.fromkeys(values))


def read_feature_dtype_profile(path: Path | str) -> dict[str, int]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feature dtype profile JSON: {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("feature dtype profile must be a non-empty JSON object")
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> None:
    calibration_started = time.perf_counter()
    args = parse_args(argv)
    if args.parallel_efficiency_cells is None:
        raise ValueError("--parallel-efficiency-cells is required for format-v5 calibration")
    thread_report = enforce_thread_env(strict=not args.allow_nonproduction_threads)
    phase_seconds: dict[str, float] = {}

    work_dir = args.work_dir or (repo_root() / "NK_Grid" / "calibration" / "_scratch")
    out_dir = args.out_dir or (repo_root() / "NK_Grid" / "calibration")
    dtype_profile = (
        read_feature_dtype_profile(args.feature_dtype_profile)
        if args.feature_dtype_profile is not None
        else None
    )
    shape = shape_from_schema(
        args.shape_schema,
        feature_dtype_profile=dtype_profile,
        assume_feature_dtype=args.assume_feature_dtype,
        dtype_profile_path=args.feature_dtype_profile,
    )
    stage_a_n = parse_positive_int_grid(args.stage_a_n, option="--stage-a-n")
    if max(stage_a_n) > args.n_train:
        raise ValueError("--stage-a-n values cannot exceed --n-train")
    stage_a_k = k_grid_from_shape(
        shape,
        base_anchors=parse_positive_int_grid(args.stage_a_k_base, option="--stage-a-k-base"),
        intermediate_points=args.stage_a_k_intermediate_points,
    )
    params = SyntheticDataParams(n_train=args.n_train, shape=shape, seed=args.seed)

    generation_start = time.perf_counter()
    peak_rss_before = _process_peak_rss_bytes()
    bundle_root = work_dir / "full"
    schema_path, stats = generate_synthetic_bundle(bundle_root, params)
    generation_seconds = time.perf_counter() - generation_start
    phase_seconds["synthetic_generation"] = generation_seconds
    peak_rss_after = _process_peak_rss_bytes()

    scope_reduction: dict[str, Any] = {
        "max_seconds": args.max_seconds,
        "generation_seconds": generation_seconds,
        "generation_peak_rss_bytes": peak_rss_after,
    }

    if (
        generation_seconds > args.generation_time_budget_seconds
        or peak_rss_after > args.generation_rss_budget_bytes
    ):
        raise RuntimeError(
            "synthetic panel generation exceeded its feasibility budget; "
            "choose an explicitly smaller shape schema instead of silently changing dimensions"
        )

    session = build_session(schema_path, params.outcome, seed=args.seed)
    try:
        phase_started = time.perf_counter()
        t0 = measure_t0(schema_path, params.outcome)
        phase_seconds["t0"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        eta_measurement = measure_parallel_efficiency(
            session,
            parse_task_memory_cells(
                args.parallel_efficiency_cells, max_seconds=args.max_seconds
            ),
            n_reps=args.parallel_efficiency_reps,
        )
        phase_seconds["parallel_efficiency"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        task_peak = (
            None
            if args.task_memory_cells is None
            else measure_task_cgroup_peak_n_jobs_8(
                session,
                parse_task_memory_cells(
                    args.task_memory_cells, max_seconds=args.max_seconds
                ),
            )
        )
        phase_seconds["task_memory"] = time.perf_counter() - phase_started

        def _progress(message: str) -> None:
            print(message, file=sys.stderr)

        phase_started = time.perf_counter()
        raw_a, censored_a = run_stage_a(
            session,
            n_grid=stage_a_n,
            k_grid=stage_a_k,
            max_seconds=args.max_seconds,
            progress=_progress,
        )
        phase_seconds["stage_a"] = time.perf_counter() - phase_started

        if args.stage_b_points:
            points_to_measure = tuple(
                tuple(int(v) for v in pair.split(":"))
                for pair in args.stage_b_points.split(",")
            )
        else:
            points_to_measure = default_stage_b_points(
                n_train=args.n_train, stage_a_n=stage_a_n, k_grid=stage_a_k
            )
        validation_points = default_stage_b_points(
            n_train=args.n_train, stage_a_n=stage_a_n, k_grid=stage_a_k
        )
        not_measured = tuple(p for p in validation_points if p not in points_to_measure)

        phase_started = time.perf_counter()
        raw_b, censored_b = run_stage_b(
            session, points=points_to_measure, max_seconds=args.max_seconds, progress=_progress
        )
        phase_seconds["stage_b"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        fit_cost = fit_all_models(raw_a)
        preprocess_cost = fit_preprocess_by_mode(raw_a)
        peak_rss_fits = fit_peak_rss(raw_a)
        validation = build_validation_rows(
            raw_b, censored_b, fit_cost, validation_points, MODELS, not_measured_points=not_measured
        )
        phase_seconds["fit_and_validation"] = time.perf_counter() - phase_started

        wall_clock_seconds = {
            "phases": phase_seconds,
            "total": time.perf_counter() - calibration_started,
        }

        payload = build_calibration_payload(
            synthetic_params=params,
            synthetic_stats=stats,
            t0_seconds=t0,
            fit_cost=fit_cost,
            preprocess_cost=preprocess_cost,
            peak_rss=peak_rss_fits,
            validation=validation,
            censored=[*censored_a, *censored_b],
            raw_measurements=[*raw_a, *raw_b],
            thread_env_report=thread_report,
            scope_reduction=scope_reduction,
            task_cgroup_peak_n_jobs_8=task_peak,
            stage_a_k_grid=stage_a_k,
            parallel_efficiency_measurement=eta_measurement,
            wall_clock_seconds=wall_clock_seconds,
        )
    finally:
        close_session(session)
    out_path = write_calibration_file(payload, out_dir)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

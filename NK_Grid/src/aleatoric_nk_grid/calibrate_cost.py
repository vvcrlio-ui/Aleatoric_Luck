"""Empirical cost/memory calibration harness for the N x K engine.

Measures wall-clock startup cost (t0), per-model fit-time power laws, peak
RSS, and preprocessing cost -- entirely on locally generated synthetic data.
Produces ``NK_Grid/calibration/cost_model_<UTC-date>.json``.

This module is purely additive: it does not change any engine execution
logic, does not touch ``SMR/data/`` or ``FFCWS/data/`` (private data, not in
the repository), and never inspects or reports any metric value (``r2_test``,
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
from aleatoric_nk_grid_measure_worker import cell_worker_target, task_cell_worker_target

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

# Version 3 rejects both the first process-lifetime RSS format and v2, which
# lacked self-migration verification, scope-suspect flags, and actual sampling
# cadence. Neither can be safely reinterpreted as this format.
FORMAT_VERSION = 3
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

# Directory fragments that must never be read by this module. Checked with
# forward slashes after normalizing the resolved path.
FORBIDDEN_DATA_FRAGMENTS: tuple[str, ...] = ("SMR/data", "FFCWS/data")

# Human-pre-measured t_import value (see plans/cost-calibration.md sec 2 and
# the tonight-only scope reduction). Not remeasured by this module.
T_IMPORT_PREMEASURED: dict[str, Any] = {
    "median": 1.18,
    "min": 1.156,
    "max": 1.234,
    "n_reps": 5,
    "bare_interpreter_seconds": 0.019,
    "source": (
        "pre-measured by human operator, 2026-07-31, local SSD, "
        "not remeasured this run"
    ),
}

STAGE_A_N: tuple[int, ...] = (10, 100, 1000, 4242)
STAGE_A_K: tuple[int, ...] = (10, 100, 1000)
STAGE_A_REPS = 3

STAGE_B_POINTS: tuple[tuple[int, int], ...] = (
    (4242, 8053),
    (1000, 8053),
    (4242, 3125),
)

DEFAULT_MAX_SECONDS = 3600.0
T0_REPS = 5


# ---------------------------------------------------------------------------
# Private-data guard
# ---------------------------------------------------------------------------


class PrivateDataAccessError(PermissionError):
    """Raised when the measurement path would touch private data."""


def guard_not_private_data(path: Path | str) -> Path:
    """Refuse any path under SMR/data or FFCWS/data (private, not in repo)."""

    resolved = Path(path).resolve()
    normalized = resolved.as_posix()
    for fragment in FORBIDDEN_DATA_FRAGMENTS:
        if f"/{fragment}/" in f"{normalized}/" or normalized.endswith(f"/{fragment}"):
            raise PrivateDataAccessError(
                f"refusing to read private data path: {resolved}"
            )
    return resolved


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
    n_sources: int
    seed: int
    continuous_fraction: float = 0.622
    missing_rate_continuous: float = 0.10
    missing_rate_group: float = 0.08
    outcome: str = "y"


def onehot_group_size_pool(root: Path | None = None) -> tuple[tuple[int, ...], str]:
    """Return (sizes, provenance) for one-hot group widths.

    Informed by FFCWS/schema/ffc_median_missing_indicator.feature_universe.json
    (a schema file, not private data: it holds only feature/source structure,
    no observations) when available; otherwise a documented hardcoded fallback.
    """

    schema_path = (root or repo_root()) / "FFCWS" / "schema" / (
        "ffc_median_missing_indicator.feature_universe.json"
    )
    guard_not_private_data(schema_path)
    try:
        document = json.loads(schema_path.read_text(encoding="utf-8"))
        sizes = tuple(
            len(source["features"])
            for source in document.get("sources", [])
            if source.get("unit_type") == "onehot_group"
        )
        if sizes:
            return sizes, (
                f"FFCWS/schema/{schema_path.name} onehot_group feature counts "
                f"(n_groups={len(sizes)}, mean={float(np.mean(sizes)):.3f})"
            )
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    fallback = (2, 3, 3, 4, 4, 5)
    return fallback, "fallback hardcoded distribution (schema file unavailable)"


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
    n_continuous = int(round(params.n_sources * params.continuous_fraction))
    n_onehot = params.n_sources - n_continuous
    sizes_pool, sizes_provenance = onehot_group_size_pool()
    sizes = (
        rng.choice(np.array(sizes_pool, dtype=int), size=n_onehot, replace=True)
        if n_onehot > 0
        else np.array([], dtype=int)
    )

    columns: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    source_order = 0

    for i in range(n_continuous):
        name = f"N_src{i:05d}"
        values = rng.normal(size=params.n_train).astype(float)
        missing_mask = rng.random(params.n_train) < params.missing_rate_continuous
        values[missing_mask] = np.nan
        columns[name] = values
        manifest_rows.append(
            {
                "source_column": name,
                "feature_name": name,
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
        source_order += 1

    for j, raw_size in enumerate(sizes):
        size = int(raw_size)
        base = f"C_src{j:05d}"
        feats = [f"{base}__lvl{k}" for k in range(size)]
        weights = rng.dirichlet(np.ones(size))
        choice = rng.choice(size, size=params.n_train, p=weights)
        missing_mask = rng.random(params.n_train) < params.missing_rate_group
        block = np.zeros((params.n_train, size), dtype=float)
        block[np.arange(params.n_train), choice] = 1.0
        block[missing_mask, :] = np.nan
        for k, feat in enumerate(feats):
            columns[feat] = block[:, k]
            manifest_rows.append(
                {
                    "source_column": base,
                    "feature_name": feat,
                    "keep": True,
                    "source_order": source_order,
                    "feature_order": k,
                    "unit_type": "onehot_group",
                    "drop_first": False,
                    "is_reference": k == 0,
                    "reference_level": 0.0,
                    "level_value": float(k),
                    "ordinal_levels": None,
                    "source_prior": None,
                }
            )
        source_order += 1

    outcome_values = rng.normal(size=params.n_train)
    frame = pd.DataFrame(columns)
    frame.insert(0, params.outcome, outcome_values)
    manifest = pd.DataFrame(manifest_rows)
    predictors = [column for column in frame.columns if column != params.outcome]
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
        "n_sources": params.n_sources,
        "n_continuous_sources": n_continuous,
        "n_onehot_sources": n_onehot,
        "n_expanded_predictors": len(predictors),
        "seed": params.seed,
        "onehot_group_size_source": sizes_provenance,
        "continuous_fraction": params.continuous_fraction,
        "missing_rate_continuous": params.missing_rate_continuous,
        "missing_rate_group": params.missing_rate_group,
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

_T0_SUBPROCESS_SCRIPT = """
import json
import sys
import time

t_after_import_start = time.perf_counter()
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


def measure_t0(
    schema_path: Path,
    outcome: str,
    *,
    n_reps: int = T0_REPS,
    python_executable: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure t_load/t_split/t_orders across n_reps fresh interpreter starts.

    t_import is NOT remeasured; the pre-measured value is substituted (see
    plans/cost-calibration.md's tonight-only scope reduction).
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
    for _ in range(n_reps):
        result = subprocess.run(
            [executable, "-c", _T0_SUBPROCESS_SCRIPT, str(schema_path), outcome],
            capture_output=True,
            text=True,
            env=run_env,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        t_load.append(float(payload["t_load"]))
        t_split.append(float(payload["t_split"]))
        t_orders.append(float(payload["t_orders"]))

    import_summary = dict(T_IMPORT_PREMEASURED)
    load_summary = summarize_reps(t_load)
    split_summary = summarize_reps(t_split)
    orders_summary = summarize_reps(t_orders)
    total_median = (
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
        "total": {"median": total_median, "n_reps": n_reps},
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


def build_session(schema_path: Path, outcome: str, *, seed: int = 0, test_size: float = 0.2) -> CalibrationSession:
    guard_not_private_data(schema_path)
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


def run_stage_a(
    session: CalibrationSession,
    *,
    n_grid: Sequence[int] = STAGE_A_N,
    k_grid: Sequence[int] = STAGE_A_K,
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
    points: Sequence[tuple[int, int]] = STAGE_B_POINTS,
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
            "n_feature_units": synthetic_params.n_sources,
            "p_onehot": synthetic_stats["n_expanded_predictors"],
            "seed": synthetic_params.seed,
            "continuous_fraction": synthetic_params.continuous_fraction,
            "missing_rate_continuous": synthetic_params.missing_rate_continuous,
            "missing_rate_group": synthetic_params.missing_rate_group,
            "outcome": synthetic_params.outcome,
            # The two missing-rate fields above have no empirical basis (real
            # missingness lives only in FFCWS/data/, which this module never
            # reads); they are placeholders chosen for plausibility, not
            # measured. Their effect on *timing* is second-order (imputer
            # cost is dominated by data scale, and lightgbm/xgboost handle
            # NaN natively without imputation), so this flag documents the
            # assumption rather than blocking on finding a real value.
            "missing_rates_are_unverified_placeholders": True,
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
    required = {"memory_measurement", "raw_measurements", "peak_rss_bytes"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"calibration file is missing required fields: {sorted(missing)}")
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
    parser.add_argument("--n-train", type=int, default=4242)
    parser.add_argument("--n-feature-units", type=int, default=8053)
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
        help="Comma-separated N:K pairs to actually measure in stage B "
        "(default: all three plan points). Points not listed are recorded "
        "as not_measured, never fabricated.",
    )
    parser.add_argument(
        "--fallback-feature-units",
        type=int,
        default=None,
        help="If full-dimension generation exceeds the feasibility budget, "
        "regenerate at this many feature units instead.",
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    thread_report = enforce_thread_env(strict=not args.allow_nonproduction_threads)

    work_dir = args.work_dir or (repo_root() / "NK_Grid" / "calibration" / "_scratch")
    out_dir = args.out_dir or (repo_root() / "NK_Grid" / "calibration")

    params = SyntheticDataParams(
        n_train=args.n_train, n_sources=args.n_feature_units, seed=args.seed
    )

    generation_start = time.perf_counter()
    peak_rss_before = _process_peak_rss_bytes()
    bundle_root = work_dir / "full"
    schema_path, stats = generate_synthetic_bundle(bundle_root, params)
    generation_seconds = time.perf_counter() - generation_start
    peak_rss_after = _process_peak_rss_bytes()

    scope_reduction: dict[str, Any] = {
        "max_seconds": args.max_seconds,
        "generation_seconds": generation_seconds,
        "generation_peak_rss_bytes": peak_rss_after,
    }

    if (
        generation_seconds > args.generation_time_budget_seconds
        or peak_rss_after > args.generation_rss_budget_bytes
    ) and args.fallback_feature_units:
        scope_reduction["dimension_fallback_triggered"] = True
        scope_reduction["dimensions_scaled_down"] = True
        scope_reduction["fallback_note"] = (
            "维度已缩放，系数需在全维度上重新标定"
        )
        params = SyntheticDataParams(
            n_train=args.n_train, n_sources=args.fallback_feature_units, seed=args.seed
        )
        bundle_root = work_dir / "fallback"
        schema_path, stats = generate_synthetic_bundle(bundle_root, params)
    else:
        scope_reduction["dimension_fallback_triggered"] = False
        scope_reduction["dimensions_scaled_down"] = False

    session = build_session(schema_path, params.outcome, seed=args.seed)
    try:
        t0 = measure_t0(schema_path, params.outcome)
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

        def _progress(message: str) -> None:
            print(message, file=sys.stderr)

        raw_a, censored_a = run_stage_a(
            session, max_seconds=args.max_seconds, progress=_progress
        )

        if args.stage_b_points:
            points_to_measure = tuple(
                tuple(int(v) for v in pair.split(":"))
                for pair in args.stage_b_points.split(",")
            )
        else:
            points_to_measure = STAGE_B_POINTS
        not_measured = tuple(p for p in STAGE_B_POINTS if p not in points_to_measure)

        raw_b, censored_b = run_stage_b(
            session, points=points_to_measure, max_seconds=args.max_seconds, progress=_progress
        )

        fit_cost = fit_all_models(raw_a)
        preprocess_cost = fit_preprocess_by_mode(raw_a)
        peak_rss_fits = fit_peak_rss(raw_a)
        validation = build_validation_rows(
            raw_b, censored_b, fit_cost, STAGE_B_POINTS, MODELS, not_measured_points=not_measured
        )

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
        )
    finally:
        close_session(session)
    out_path = write_calibration_file(payload, out_dir)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

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
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiment import core_environment, git_state, utc_now, write_json_atomic
from .ingest import load_input
from .model_registry import DEFAULT_MODEL_PARAMS_PATH, load_model_params, make_model
from .nk_grid import (
    DrawOrders,
    SplitData,
    _fit_predict_model_cell,
    _process_peak_rss_bytes,
    draw_orders,
    split_frame,
)
from .preprocessing import preprocess_cell, source_groups
from .validate_input import canonical_feature_universe, validate_input


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMAT_VERSION = 1

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

# Subprocess-based models that hand fitting to a native isolated process in
# production (see nk_grid.IsolatedProcessRunner). Their peak RSS must be
# measured via RUSAGE_CHILDREN in production; the calibration harness runs
# them in-process for simplicity and records that deviation in the report.
SUBPROCESS_MODELS: frozenset[str] = frozenset({"lightgbm", "xgboost"})

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

    def orders_for(self, seed: int, draw: int) -> DrawOrders:
        return draw_orders(
            self.split.X_train.index, list(self.feature_units), seed=seed, draw=draw
        )


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
    """Fit one (model, N, K) cell once, returning fit+preprocess timing and RSS.

    Raises MeasurementCensored if fitting exceeds max_seconds.
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
        result = _fit_predict_model_cell(
            model_name=model_name,
            model_seed=seed,
            task=session.task,
            params=params,
            X_train=prepared.X_train,
            y_train=y_sub,
            X_test=prepared.X_test,
            model_n_jobs=1,
        )

    return RawMeasurement(
        model=model_name,
        n=n,
        k=k,
        rep=draw,
        fit_seconds=float(result["fit_seconds"]),
        preprocess_seconds=float(preprocess_seconds),
        preprocess_mode=mode,
        peak_rss_bytes=int(result["peak_rss_bytes"]),
        stage="",
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
        "stage": measurement.stage,
    }


def recompute_fit_cost_from_raw(
    raw_measurements: Sequence[Mapping[str, Any]], models: Sequence[str] = MODELS
) -> dict[str, PowerLawFit | None]:
    """Recompute fit_cost coefficients from raw_measurements alone (round-trip check)."""

    reconstructed = [
        RawMeasurement(
            model=str(row["model"]),
            n=int(row["n"]),
            k=int(row["k"]),
            rep=int(row["rep"]),
            fit_seconds=float(row["fit_seconds"]),
            preprocess_seconds=float(row["preprocess_seconds"]),
            preprocess_mode=str(row["preprocess_mode"]),
            peak_rss_bytes=int(row["peak_rss_bytes"]),
            stage=str(row["stage"]),
        )
        for row in raw_measurements
    ]
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
            "n_train": synthetic_params.n_train,
            "n_feature_units": synthetic_params.n_sources,
            "p_onehot": synthetic_stats["n_expanded_predictors"],
            "seed": synthetic_params.seed,
            **synthetic_stats,
        },
        "t0_seconds": t0_seconds,
        "fit_cost": {name: _fit_to_dict(fit) for name, fit in fit_cost.items()},
        "preprocess_cost": {name: _fit_to_dict(fit) for name, fit in preprocess_cost.items()},
        "peak_rss_bytes": {name: _fit_to_dict(fit) for name, fit in peak_rss.items()},
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
    return parser.parse_args(argv)


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

    t0 = measure_t0(schema_path, params.outcome)

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
    )
    out_path = write_calibration_file(payload, out_dir)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

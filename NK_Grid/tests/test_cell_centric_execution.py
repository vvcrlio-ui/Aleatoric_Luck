from __future__ import annotations

import gc
import ctypes
import json
import os
import sys
import time
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.experiment import load_checkpoint, write_checkpoint_part
from aleatoric_nk_grid.nk_grid import (
    METRIC_COLUMNS,
    NKGridConfig,
    run_nk_grid,
)
from aleatoric_nk_grid.preprocessing import preprocess_cell

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


class _MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def _run_isolated_inline(_runner, function, /, *args, **kwargs):
    kwargs.pop("on_native_crash", None)
    kwargs.pop("on_native_timeout", None)
    return function(*args, **kwargs)


def _frame(rows: int = 48, features: int = 4) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    frame = pd.DataFrame({"y": 1.75 * values + (values % 3)})
    for column in range(features):
        data = (values + column) % (column + 5)
        data[(np.arange(rows) + column) % 11 == 0] = np.nan
        frame[f"X_{column:03d}"] = data
    return frame


def _config(
    schema: Path,
    out: Path,
    *,
    models: tuple[str, ...] = ("ols", "ridge"),
    **overrides,
) -> NKGridConfig:
    values = {
        "schema": schema,
        "out": out,
        "outcome": "y",
        "models": models,
        "seed": 321,
        "test_size": 0.25,
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 1,
        "n_sizes_k": 1,
        "min_n": 10,
        "max_n": 0,
        "max_k": 0,
        "batch_size": 20,
        "n_jobs": 1,
        "model_params": MODEL_PARAMS,
        "rerun_completed": False,
    }
    values.update(overrides)
    return NKGridConfig(**values)


def _metric_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = ["model", "seed", "draw", "N", "K", *METRIC_COLUMNS]
    return frame.loc[:, columns].sort_values(
        ["seed", "draw", "K", "N", "model"], kind="stable"
    ).reset_index(drop=True)


def test_cell_group_metrics_exactly_match_model_at_a_time_execution(tmp_path):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input",
        frame,
        predictors=predictors,
        imputation={
            "continuous": "median",
            "ordinal": "most_frequent",
            "onehot_group": "atomic_mode",
            "model_overrides": {"lightgbm": "passthrough"},
        },
    )
    models = ("ols", "ridge", "extra_trees")
    grouped_out = tmp_path / "grouped.csv"
    grouped = _config(
        schema,
        grouped_out,
        models=models,
        n_draws=2,
        n_sizes_n=2,
        n_sizes_k=2,
    )
    run_nk_grid(grouped)

    legacy_rows = []
    for model in models:
        model_out = tmp_path / f"legacy-{model}.csv"
        run_nk_grid(
            _config(
                schema,
                model_out,
                models=(model,),
                n_draws=2,
                n_sizes_n=2,
                n_sizes_k=2,
            )
        )
        legacy_rows.append(pd.read_csv(model_out))
    legacy = pd.concat(legacy_rows, ignore_index=True)
    legacy_path = tmp_path / "legacy.csv"
    legacy.to_csv(legacy_path, index=False)

    pd.testing.assert_frame_equal(
        _metric_rows(grouped_out),
        _metric_rows(legacy_path),
        check_exact=True,
    )


def test_preprocess_cell_runs_at_most_once_per_mode_in_cell_group(tmp_path):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input",
        frame,
        predictors=predictors,
        imputation={
            "continuous": "median",
            "ordinal": "most_frequent",
            "onehot_group": "atomic_mode",
            "model_overrides": {"lightgbm": "passthrough"},
        },
    )
    calls: list[str] = []
    checkpoint_sizes: list[int] = []

    def counting_preprocess(*args, model_name, **kwargs):
        calls.append(model_name)
        return preprocess_cell(*args, model_name=model_name, **kwargs)

    def recording_checkpoint(rows, out_path):
        checkpoint_sizes.append(len(rows))
        return write_checkpoint_part(rows, out_path)

    with (
        patch(
            "aleatoric_nk_grid.nk_grid.preprocess_cell",
            new=counting_preprocess,
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.write_checkpoint_part",
            new=recording_checkpoint,
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=lambda *args, **kwargs: _MeanRegressor(),
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.IsolatedProcessRunner.run",
            autospec=True,
            side_effect=_run_isolated_inline,
        ),
    ):
        run_nk_grid(
            _config(
                schema,
                tmp_path / "modes.csv",
                models=("ols", "ridge", "lightgbm"),
                batch_size=1,
            )
        )

    assert calls == ["ols", "lightgbm"]
    assert checkpoint_sizes == [1, 1, 1]


def test_estimator_mutation_cannot_change_or_retain_shared_prepared_cell(
    tmp_path,
):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=predictors
    )
    prepared_ref = None
    prepared_train = None
    prepared_test = None

    def tracking_preprocess(*args, **kwargs):
        nonlocal prepared_ref, prepared_train, prepared_test
        prepared = preprocess_cell(*args, **kwargs)
        prepared_ref = weakref.ref(prepared)
        prepared_train = prepared.X_train.copy(deep=True)
        prepared_test = prepared.X_test.copy(deep=True)
        return prepared

    class MutatingRegressor(_MeanRegressor):
        def fit(self, X, y):
            X.iloc[:, :] = -999.0
            return super().fit(X, y)

        def predict(self, X):
            X.iloc[:, :] = -777.0
            return super().predict(X)

    class ObservingRegressor(_MeanRegressor):
        def fit(self, X, y):
            prepared = prepared_ref()
            assert prepared is not None
            pd.testing.assert_frame_equal(
                prepared.X_train, prepared_train, check_exact=True
            )
            pd.testing.assert_frame_equal(
                prepared.X_test, prepared_test, check_exact=True
            )
            assert not X.eq(-999.0).any().any()
            return super().fit(X, y)

    def make_fake_model(model_name, **_kwargs):
        return (
            MutatingRegressor()
            if model_name == "ols"
            else ObservingRegressor()
        )

    with (
        patch(
            "aleatoric_nk_grid.nk_grid.preprocess_cell",
            new=tracking_preprocess,
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=make_fake_model,
        ),
    ):
        run_nk_grid(
            _config(
                schema,
                tmp_path / "mutation.csv",
                models=("ols", "ridge"),
            )
        )

    gc.collect()
    assert prepared_ref is not None
    assert prepared_ref() is None


def test_cell_groups_are_deterministic_with_concurrent_outer_jobs(tmp_path):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=predictors
    )
    first = _config(
        schema,
        tmp_path / "first.csv",
        models=("ols", "ridge"),
        n_draws=3,
        n_sizes_n=2,
        n_sizes_k=2,
        n_jobs=2,
    )
    second = _config(
        schema,
        tmp_path / "second.csv",
        models=("ols", "ridge"),
        n_draws=3,
        n_sizes_n=2,
        n_sizes_k=2,
        n_jobs=2,
    )

    run_nk_grid(first)
    run_nk_grid(second)

    pd.testing.assert_frame_equal(
        _metric_rows(first.out),
        _metric_rows(second.out),
        check_exact=True,
    )


def test_preprocess_telemetry_counts_only_mode_misses_and_matches_manifest(
    tmp_path,
):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=predictors
    )

    def measured_run(models: tuple[str, ...], name: str):
        observed_seconds: list[float] = []

        def measured_preprocess(*args, **kwargs):
            started = time.perf_counter()
            time.sleep(0.05)
            result = preprocess_cell(*args, **kwargs)
            observed_seconds.append(time.perf_counter() - started)
            return result

        out = tmp_path / f"{name}.csv"
        with (
            patch(
                "aleatoric_nk_grid.nk_grid.preprocess_cell",
                new=measured_preprocess,
            ),
            patch(
                "aleatoric_nk_grid.nk_grid.make_model",
                side_effect=lambda *args, **kwargs: _MeanRegressor(),
            ),
            patch(
                "aleatoric_nk_grid.nk_grid._prune_checkpoint_parts",
                return_value=False,
            ),
        ):
            run_nk_grid(_config(schema, out, models=models))
        checkpoint = load_checkpoint(out)
        manifest = json.loads(
            out.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        by_model = manifest["diagnostics"]["by_model"]
        manifest_total = sum(
            model["preprocess_seconds_total"]
            for model in by_model.values()
        )
        return checkpoint, manifest_total, sum(observed_seconds)

    one, one_total, one_observed = measured_run(("ols",), "one")
    three, three_total, three_observed = measured_run(
        ("ols", "ridge", "lasso"), "three"
    )

    assert int(one["_preprocess_computed"].astype(bool).sum()) == 1
    assert int(three["_preprocess_computed"].astype(bool).sum()) == 1
    assert int(three["_preprocess_seconds"].gt(0).sum()) == 1
    assert int(three["_slice_seconds"].gt(0).sum()) == 1
    assert three["_cell_wall_seconds"].gt(0).all()
    assert three["_peak_rss_bytes"].gt(0).all()
    assert one_total == pytest.approx(
        float(one["_preprocess_seconds"].sum()), rel=1e-12
    )
    assert three_total == pytest.approx(
        float(three["_preprocess_seconds"].sum()), rel=1e-12
    )
    assert one_total == pytest.approx(one_observed, rel=0.01)
    assert three_total == pytest.approx(three_observed, rel=0.01)
    assert three_total == pytest.approx(one_total, rel=0.20)
    three_manifest = json.loads(
        (tmp_path / "three.manifest.json").read_text(encoding="utf-8")
    )
    for model_name, summary in three_manifest["diagnostics"]["by_model"].items():
        model_rows = three.loc[three["model"].eq(model_name)]
        assert summary["cell_wall_seconds_total"] == pytest.approx(
            float(model_rows["_cell_wall_seconds"].sum()), rel=1e-12
        )
        assert summary["peak_rss_bytes_max"] == int(
            model_rows["_peak_rss_bytes"].max()
        )


def test_max_k_cell_groups_do_not_show_strictly_increasing_rss(tmp_path):
    frame = _frame(rows=64, features=64)
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=predictors
    )
    rss_samples: list[int] = []

    def current_rss_bytes() -> int:
        if sys.platform == "darwin":
            class ProcTaskInfo(ctypes.Structure):
                _fields_ = [
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("total_user", ctypes.c_uint64),
                    ("total_system", ctypes.c_uint64),
                    ("threads_user", ctypes.c_uint64),
                    ("threads_system", ctypes.c_uint64),
                    ("policy", ctypes.c_int32),
                    ("faults", ctypes.c_int32),
                    ("pageins", ctypes.c_int32),
                    ("cow_faults", ctypes.c_int32),
                    ("messages_sent", ctypes.c_int32),
                    ("messages_received", ctypes.c_int32),
                    ("syscalls_mach", ctypes.c_int32),
                    ("syscalls_unix", ctypes.c_int32),
                    ("csw", ctypes.c_int32),
                    ("threadnum", ctypes.c_int32),
                    ("numrunning", ctypes.c_int32),
                    ("priority", ctypes.c_int32),
                ]

            info = ProcTaskInfo()
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            size = libproc.proc_pidinfo(
                os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if size != ctypes.sizeof(info):
                raise RuntimeError("proc_pidinfo did not return PROC_PIDTASKINFO")
            return int(info.resident_size)
        resident_pages = int(
            Path("/proc/self/statm").read_text(encoding="utf-8").split()[1]
        )
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    def recording_checkpoint(rows, out_path):
        part = write_checkpoint_part(rows, out_path)
        gc.collect()
        rss_samples.append(current_rss_bytes())
        return part

    with (
        patch(
            "aleatoric_nk_grid.nk_grid.write_checkpoint_part",
            new=recording_checkpoint,
        ),
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=lambda *args, **kwargs: _MeanRegressor(),
        ),
    ):
        run_nk_grid(
            _config(
                schema,
                tmp_path / "rss.csv",
                models=("ols", "ridge"),
                n_draws=6,
                max_k=0,
                batch_size=2,
            )
        )

    assert len(rss_samples) == 6
    assert not all(
        right > left for left, right in zip(rss_samples, rss_samples[1:])
    ), rss_samples


def test_max_jobs_batch_boundaries_and_resume_match_uninterrupted_output(
    tmp_path,
):
    frame = _frame()
    predictors = [column for column in frame if column.startswith("X_")]
    schema = write_schema_bundle(
        tmp_path / "input", frame, predictors=predictors
    )
    complete = _config(
        schema,
        tmp_path / "complete.csv",
        models=("ols", "ridge"),
        n_draws=3,
        batch_size=3,
    )
    resumed = _config(
        schema,
        tmp_path / "resumed.csv",
        models=("ols", "ridge"),
        n_draws=3,
        batch_size=3,
    )

    run_nk_grid(complete)
    run_nk_grid(resumed, max_jobs=3)
    partial = pd.read_csv(resumed.out)
    assert len(partial) == 3
    run_nk_grid(resumed)

    pd.testing.assert_frame_equal(
        _metric_rows(complete.out),
        _metric_rows(resumed.out),
        check_exact=True,
    )

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from joblib.externals.cloudpickle import dumps as cloudpickle_dumps

import aleatoric_nk_grid.experiment as experiment_module
from aleatoric_nk_grid.experiment import (
    checkpoint_loose_parts_dir,
    checkpoint_parts_dir,
    load_checkpoint,
    load_checkpoint_index,
    output_run_lock,
    write_checkpoint_part,
)
from aleatoric_nk_grid.nk_grid import (
    NKGridConfig,
    _freeze_draw_orders,
    draw_orders,
    estimate_run_size,
    run_nk_grid,
)
from aleatoric_nk_grid.run_panels import PRESETS

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


def _frame(rows: int = 40) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "y": values * 1.5 + 2.0,
            "X_a": values,
            "X_b": values % 5,
        }
    )


def test_draw_orders_is_deterministic_and_only_cached_payload_is_frozen():
    first = draw_orders(
        [8, 3, 5, 1],
        ["β", "feature_a", "feature_z"],
        seed=17,
        draw=4,
    )
    second = draw_orders(
        [8, 3, 5, 1],
        ["β", "feature_a", "feature_z"],
        seed=17,
        draw=4,
    )

    assert first.row_index.tobytes() == second.row_index.tobytes()
    assert first.feature_names.tobytes() == second.feature_names.tobytes()
    assert first.row_index.flags.writeable is True
    assert first.feature_names.flags.writeable is True
    frozen = _freeze_draw_orders(second)
    assert frozen.row_index.flags.writeable is False
    assert frozen.feature_names.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        frozen.row_index[0] = -1
    with pytest.raises(ValueError, match="read-only"):
        frozen.feature_names[0] = "changed"


def test_thread_and_serial_groups_reuse_one_order_per_seed_draw(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _frame(),
        predictors=["X_a", "X_b"],
    )
    config = NKGridConfig(
        schema=schema,
        out=tmp_path / "result.csv",
        outcome="y",
        models=("ols", "lightgbm"),
        seed=123,
        test_size=0.3,
        n_seeds=1,
        n_draws=2,
        n_sizes_n=2,
        n_sizes_k=2,
        min_n=10,
        max_n=0,
        max_k=0,
        batch_size=3,
        n_jobs=2,
        model_params=MODEL_PARAMS,
    )

    with (
        patch(
            "aleatoric_nk_grid.nk_grid.draw_orders",
            wraps=draw_orders,
        ) as order_spy,
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
        run_nk_grid(config)

    # Multiple N/K values and both execution groups share each draw order.
    assert order_spy.call_count == config.n_seeds * config.n_draws
    result = pd.read_csv(config.out)
    assert len(result) == 16
    assert result["status"].eq("ok").all()


def test_bart_process_task_remains_pickleable_without_cached_order_ipc(tmp_path):
    schema = write_schema_bundle(
        tmp_path / "input",
        _frame(),
        predictors=["X_a", "X_b"],
    )
    config = NKGridConfig(
        schema=schema,
        out=tmp_path / "bart.csv",
        outcome="y",
        models=("bart",),
        seed=123,
        test_size=0.3,
        n_seeds=1,
        n_draws=1,
        n_sizes_n=1,
        n_sizes_k=1,
        min_n=10,
        max_n=0,
        max_k=0,
        batch_size=1,
        n_jobs=2,
        model_params=MODEL_PARAMS,
    )
    observed_orders = []

    class PicklingParallel:
        def __init__(self, **kwargs):
            assert kwargs["prefer"] == "processes"

        def __call__(self, tasks):
            rows = []
            for function, args, kwargs in tasks:
                cloudpickle_dumps(function)
                observed_orders.append(kwargs["orders"])
                rows.append(function(*args, **kwargs))
            return rows

    with (
        patch("aleatoric_nk_grid.nk_grid.Parallel", PicklingParallel),
        patch(
            "aleatoric_nk_grid.nk_grid.make_model",
            side_effect=lambda *args, **kwargs: _MeanRegressor(),
        ),
    ):
        run_nk_grid(config)

    assert observed_orders == [None]
    assert pd.read_csv(config.out).loc[0, "status"] == "ok"


def test_checkpoint_index_projects_metrics_and_preserves_success_priority(
    tmp_path,
):
    out = tmp_path / "result.csv"
    base = {
        "experiment_id": "experiment",
        "model": "ols",
        "seed": 1,
        "draw": 2,
        "N": 10,
        "K": 3,
    }
    write_checkpoint_part(
        [{**base, "status": "ok", "rmse": 1.25, "large_metric": "x" * 1000}],
        out,
    )
    write_checkpoint_part(
        [{**base, "status": "failed", "rmse": 99.0, "large_metric": "y" * 1000}],
        out,
    )

    with patch(
        "aleatoric_nk_grid.experiment.pd.read_csv",
        wraps=pd.read_csv,
    ) as read_csv_spy:
        projected = load_checkpoint_index(out)

    assert set(projected) == {
        "experiment_id",
        "model",
        "seed",
        "draw",
        "N",
        "K",
        "status",
    }
    assert projected.loc[0, "status"] == "ok"
    assert all(
        call.kwargs.get("usecols") is not None
        for call in read_csv_spy.call_args_list
    )
    assert all(
        call.kwargs["usecols"]("rmse") is False
        for call in read_csv_spy.call_args_list
    )

    full = load_checkpoint(out)
    assert "rmse" in full
    assert full.loc[0, "status"] == "ok"


def test_checkpoint_index_accepts_legacy_rows_without_status(tmp_path):
    out = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "legacy",
                "model": "ridge",
                "seed": 1,
                "draw": 0,
                "N": 20,
                "K": 4,
                "rmse": 0.5,
            }
        ]
    ).to_csv(out, index=False)

    projected = load_checkpoint_index(out)

    assert projected.loc[0, "status"] == "ok"
    assert "rmse" not in projected
    assert projected.loc[0, "model"] == "ridge"


def test_checkpoint_index_rejects_zero_byte_and_missing_key_sources(tmp_path):
    zero = tmp_path / "zero.csv"
    zero.write_bytes(b"")
    with pytest.raises(ValueError, match=r"Malformed checkpoint source .*zero\.csv"):
        load_checkpoint_index(zero)

    missing = tmp_path / "missing.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "bad",
                "model": "ols",
                "seed": 1,
                "draw": 0,
                "N": 10,
                "status": "ok",
            }
        ]
    ).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required columns.*K"):
        load_checkpoint_index(missing)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("seed", 1.5, "seed must contain finite integers"),
        ("draw", np.inf, "draw must contain finite integers"),
        ("status", "finished", "status must contain only"),
    ],
)
def test_checkpoint_index_rejects_invalid_keys_and_status(
    tmp_path,
    column,
    value,
    match,
):
    row = {
        "experiment_id": "bad",
        "model": "ols",
        "seed": 1,
        "draw": 0,
        "N": 10,
        "K": 2,
        "status": "ok",
    }
    row[column] = value
    out = tmp_path / f"invalid-{column}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)

    with pytest.raises(ValueError, match=match):
        load_checkpoint_index(out)


def test_checkpoint_index_normalizes_mixed_legacy_and_status_shards(tmp_path):
    out = tmp_path / "mixed.csv"
    row = {
        "experiment_id": "mixed",
        "model": "ols",
        "seed": 1,
        "draw": 0,
        "N": 10,
        "K": 2,
    }
    write_checkpoint_part([{**row, "status": "ok"}], out)
    legacy = out.with_suffix(".parts") / "part-legacy.csv"
    pd.DataFrame([{**row, "draw": 1}]).to_csv(legacy, index=False)

    projected = load_checkpoint_index(out)

    assert projected["draw"].tolist() == [0, 1]
    assert projected["status"].tolist() == ["ok", "ok"]


def test_output_writer_lease_rejects_duplicate_process_owner(tmp_path):
    out = tmp_path / "result.csv"

    with output_run_lock(out):
        with pytest.raises(RuntimeError, match="already holds the output lease"):
            with output_run_lock(out):
                pass


def test_output_writer_lease_durably_creates_nested_parent(tmp_path):
    out = tmp_path / "new" / "nested" / "result.csv"

    with patch(
        "aleatoric_nk_grid.experiment._fsync_directory",
        wraps=experiment_module._fsync_directory,
    ) as fsync_directory:
        with output_run_lock(out):
            pass

    synced = {call.args[0] for call in fsync_directory.call_args_list}
    assert out.parent in synced
    assert out.parent.parent in synced
    assert tmp_path in synced


def test_first_checkpoint_durably_publishes_each_new_directory_level(tmp_path):
    out = tmp_path / "nested" / "result.csv"
    row = {
        "experiment_id": "durable",
        "model": "ols",
        "seed": 1,
        "draw": 0,
        "N": 10,
        "K": 2,
        "status": "ok",
    }

    with patch(
        "aleatoric_nk_grid.experiment._fsync_directory",
        wraps=experiment_module._fsync_directory,
    ) as fsync_directory:
        write_checkpoint_part([row], out)

    synced = {call.args[0] for call in fsync_directory.call_args_list}
    assert out.parent in synced
    assert checkpoint_parts_dir(out) in synced
    assert checkpoint_loose_parts_dir(out) in synced


def test_production_checkpoint_estimate_keeps_small_batches_but_compacts_parts():
    assert PRESETS["production"]["batch_size"] == 20
    assert PRESETS["timing_full"] == {
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 20,
        "n_sizes_k": 20,
        "min_n": 10,
        "max_n": 0,
        "max_k": 0,
        "batch_size": 20,
    }
    config = NKGridConfig(
        schema=Path("schema.json"),
        out=Path("result.csv"),
        outcome="y",
        models=("ols",),
        seed=123,
        test_size=0.3,
        n_seeds=100,
        n_draws=50,
        n_sizes_n=20,
        n_sizes_k=20,
        min_n=10,
        max_n=0,
        max_k=0,
        batch_size=20,
        n_jobs=8,
        model_params=MODEL_PARAMS,
        preset="production",
    )

    estimate = estimate_run_size(config)

    assert estimate["top_level_model_cells"] == 2_000_000
    assert estimate["estimated_checkpoint_writes"] == 100_000
    assert estimate["checkpoint_compaction_loose_parts"] == 50
    assert estimate["estimated_checkpoint_parts"] == 2_000
    assert estimate["estimated_peak_checkpoint_parts"] == 2_050
    assert estimate["max_uncheckpointed_cells"] == 20
    assert estimate["materialization_backend"] == "sqlite_streaming"

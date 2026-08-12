from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from aleatoric_nk_grid import nk_grid
from aleatoric_nk_grid.chunk_planning import build_dynamic_plan
from aleatoric_nk_grid.experiment import load_checkpoint
from aleatoric_nk_grid.flat_task_table import _config_from_json
from aleatoric_nk_grid.nk_grid import NKGridConfig, run_nk_grid
from aleatoric_nk_grid.prediction_export import (
    PREDICTION_EXPORT_COLUMNS,
    materialize_prediction_export_atomic,
    prediction_export_parts_dir,
    prediction_export_path,
    prediction_export_schema,
    write_prediction_part_atomic,
)
from aleatoric_nk_grid.run_panels import resolve_panel

from conftest import write_schema_bundle


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def _regression_frames(
    *, train_rows: int = 36, test_rows: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = np.arange(train_rows, dtype=float)
    test_x = np.arange(test_rows, dtype=float) + 0.25
    train = pd.DataFrame(
        {
            "row_key": np.arange(1000, 1000 + train_rows),
            "y": 1.5 * train_x + (train_x % 4),
            "X_a": train_x,
            "X_b": train_x % 5,
        }
    )
    test = pd.DataFrame(
        {
            "row_key": np.arange(5000, 5000 + test_rows),
            "y": 1.5 * test_x + (test_x % 4),
            "X_a": test_x,
            "X_b": test_x % 5,
        }
    )
    return train, test


def _classification_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = np.arange(30, dtype=float)
    test_x = np.arange(6, dtype=float) + 0.5
    train = pd.DataFrame(
        {
            "row_key": np.arange(100, 130),
            "y": np.tile([0, 1], 15),
            "X_a": train_x,
        }
    )
    test = pd.DataFrame(
        {
            "row_key": np.arange(800, 806),
            "y": [0, 1, 0, 1, 0, 1],
            "X_a": test_x,
        }
    )
    return train, test


def _external_schema(
    root: Path, *, task: str = "regression"
) -> tuple[Path, pd.DataFrame]:
    train, test = (
        _classification_frames()
        if task == "classification"
        else _regression_frames()
    )
    schema = write_schema_bundle(
        root,
        train,
        test=test,
        task=task,
        split_mode="external_test",
        predictors=[column for column in train if column.startswith("X_")],
        id_column="row_key",
    )
    return schema, test


def _config(
    schema: Path,
    out: Path,
    *,
    models: tuple[str, ...] = ("ols",),
    n_grid: tuple[int, ...] = (20,),
    k_grid: tuple[int, ...] = (1,),
    export_cells: tuple[tuple[str, int, int], ...] = (),
    **overrides,
) -> NKGridConfig:
    values = {
        "schema": schema,
        "out": out,
        "outcome": "y",
        "models": models,
        "seed": 71,
        "test_size": 0.3,
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": len(n_grid),
        "n_sizes_k": len(k_grid),
        "max_n": max(n_grid),
        "max_k": max(k_grid),
        "batch_size": 2,
        "n_jobs": 1,
        "min_n": 10,
        "model_params": MODEL_PARAMS,
        "rerun_completed": False,
        "n_grid": n_grid,
        "k_grid": k_grid,
        "prediction_export_cells": export_cells,
    }
    values.update(overrides)
    return NKGridConfig(**values)


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_disabled_export_preserves_legacy_main_csv_bytes(
    tmp_path: Path, explicit_empty: bool
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    baseline = _config(schema, tmp_path / "baseline.csv")
    candidate_values = asdict(baseline)
    candidate_values["out"] = tmp_path / "candidate.csv"
    if not explicit_empty:
        candidate_values.pop("prediction_export_cells")
    candidate = NKGridConfig(**candidate_values)

    run_nk_grid(baseline)
    run_nk_grid(candidate)

    assert baseline.out.read_bytes() == candidate.out.read_bytes()
    assert not prediction_export_path(candidate.out).exists()
    assert not prediction_export_parts_dir(candidate.out).exists()


def test_exact_whitelist_exports_ids_truth_predictions_and_preserves_main_csv(
    tmp_path: Path,
) -> None:
    schema, test = _external_schema(tmp_path / "input")
    disabled = _config(
        schema,
        tmp_path / "disabled.csv",
        models=("ols", "ridge"),
        n_grid=(20, 24),
        k_grid=(1, 2),
    )
    enabled = _config(
        schema,
        tmp_path / "enabled.csv",
        models=("ols", "ridge"),
        n_grid=(20, 24),
        k_grid=(1, 2),
        export_cells=(("ols", 20, 1), ("ridge", 24, 2)),
    )

    run_nk_grid(disabled)
    run_nk_grid(enabled)

    assert disabled.out.read_bytes() == enabled.out.read_bytes()
    exported = pd.read_parquet(prediction_export_path(enabled.out))
    assert tuple(exported.columns) == PREDICTION_EXPORT_COLUMNS
    assert len(exported) == 2 * len(test)
    assert set(zip(exported["model"], exported["N"], exported["K"])) == {
        ("ols", 20, 1),
        ("ridge", 24, 2),
    }
    for _, cell in exported.groupby(["model", "seed", "draw", "N", "K"]):
        pd.testing.assert_series_equal(
            cell["row_id"].reset_index(drop=True),
            test["row_key"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            cell["y_true"].reset_index(drop=True),
            test["y"].reset_index(drop=True),
            check_names=False,
        )
        mse = float(np.mean((cell["y_true"] - cell["y_pred"]) ** 2))
        main = pd.read_csv(enabled.out)
        match = main[
            main["model"].eq(cell["model"].iloc[0])
            & main["seed"].eq(cell["seed"].iloc[0])
            & main["draw"].eq(cell["draw"].iloc[0])
            & main["N"].eq(cell["N"].iloc[0])
            & main["K"].eq(cell["K"].iloc[0])
        ].iloc[0]
        assert mse == pytest.approx(float(match["rmse"]) ** 2)


def test_classification_export_uses_metric_probability_not_hard_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, test = _external_schema(tmp_path / "input", task="classification")
    scores = np.array([0.91, 0.12, 0.73, 0.44, 0.65, 0.28])

    def fake_fit(**kwargs):
        assert len(kwargs["X_test"]) == len(scores)
        return {
            "predictions": scores.copy(),
            "fit_seconds": 0.0,
            "best_rounds": np.nan,
            "converged": True,
            "solver": None,
            "iterations": np.nan,
            "alpha": np.nan,
            "peak_rss_bytes": 0,
        }

    monkeypatch.setattr(nk_grid, "_fit_predict_model_cell", fake_fit)
    config = _config(
        schema,
        tmp_path / "classification.csv",
        export_cells=(("ols", 20, 1),),
    )
    run_nk_grid(config)

    exported = pd.read_parquet(prediction_export_path(config.out))
    np.testing.assert_array_equal(exported["y_pred"], scores)
    assert not set(exported["y_pred"]).issubset({0.0, 1.0})
    np.testing.assert_array_equal(exported["row_id"], test["row_key"])


@pytest.mark.parametrize("enabled", [False, True])
def test_internal_random_requires_explicit_id_only_when_export_enabled(
    tmp_path: Path, enabled: bool
) -> None:
    train, _ = _regression_frames()
    train = train.drop(columns="row_key")
    schema = write_schema_bundle(
        tmp_path / "input", train, predictors=["X_a", "X_b"]
    )
    export_cells = (("ols", 20, 1),) if enabled else ()
    config = _config(
        schema,
        tmp_path / "result.csv",
        export_cells=export_cells,
    )
    if enabled:
        with pytest.raises(ValueError, match=r"requires schema\.id_column"):
            run_nk_grid(config)
    else:
        assert run_nk_grid(config) == config.out


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "contains missing IDs"),
        ("duplicate", "contains duplicate IDs"),
    ],
)
def test_internal_prediction_export_validates_id_integrity(
    tmp_path: Path, mutation: str, message: str
) -> None:
    train, _ = _regression_frames()
    if mutation == "missing":
        train.loc[0, "row_key"] = np.nan
    else:
        train.loc[0, "row_key"] = train.loc[1, "row_key"]
    schema = write_schema_bundle(
        tmp_path / "input",
        train,
        predictors=["X_a", "X_b"],
        id_column="row_key",
    )
    config = _config(
        schema,
        tmp_path / "result.csv",
        export_cells=(("ols", 20, 1),),
    )
    with pytest.raises(ValueError, match=message):
        run_nk_grid(config)


def test_dynamic_planning_rejects_prediction_export_with_plan_reference(
    tmp_path: Path,
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    config = _config(
        schema,
        tmp_path / "result.csv",
        export_cells=(("ols", 20, 1),),
    )
    with pytest.raises(
        ValueError, match=r"plans/per-row-prediction-export\.md"
    ):
        build_dynamic_plan(
            config,
            n_grid=[20],
            k_grid=[1],
            cluster=None,  # rejection precedes every dynamic artefact operation
            table_path=tmp_path / "tasks.parquet",
            snapshot_path=tmp_path / "snapshot.json",
            output_dir=tmp_path / "dynamic",
            panel="synthetic-panel",
        )


def test_panel_mapping_resolves_to_exact_model_n_k_keys(tmp_path: Path) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    _, config = resolve_panel(
        {
            "name": "prediction-panel",
            "schema": str(schema),
            "outcome": "y",
            "models": ["ols", "ridge"],
            "out": "result.csv",
            "prediction_export_cells": [
                {"model": "ols", "N": 20, "K": 1},
                {"model": "ridge", "N": 24, "K": 2},
            ],
        },
        tmp_path,
    )
    assert config.prediction_export_cells == (
        ("ols", 20, 1),
        ("ridge", 24, 2),
    )


def test_unmatched_whitelist_entries_fail_before_creating_output(
    tmp_path: Path,
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    config = _config(
        schema,
        tmp_path / "result.csv",
        models=("ols", "ridge"),
        n_grid=(20, 24),
        k_grid=(1, 2),
        export_cells=(("ols", 999, 1), ("ridge", 20, 8)),
    )
    with pytest.raises(ValueError) as error:
        run_nk_grid(config)
    message = str(error.value)
    assert "do not match the resolved N/K grid" in message
    assert "(model='ols', N=999, K=1)" in message
    assert "(model='ridge', N=20, K=8)" in message
    assert "resolved N=[20, 24], K=[1, 2]" in message
    assert not config.out.exists()
    assert not config.out.with_suffix(".manifest.json").exists()
    assert not prediction_export_path(config.out).exists()


def test_changed_whitelist_rematerializes_sidecar_without_stale_cells(
    tmp_path: Path,
) -> None:
    schema, test = _external_schema(tmp_path / "input")
    out = tmp_path / "result.csv"
    first = _config(
        schema,
        out,
        models=("ols", "ridge"),
        export_cells=(("ols", 20, 1), ("ridge", 20, 1)),
    )
    run_nk_grid(first)
    assert len(pd.read_parquet(prediction_export_path(out))) == 2 * len(test)

    narrowed = _config(
        schema,
        out,
        models=("ols", "ridge"),
        export_cells=(("ols", 20, 1),),
    )
    run_nk_grid(narrowed)

    exported = pd.read_parquet(prediction_export_path(out))
    assert len(exported) == len(test)
    assert set(exported["model"]) == {"ols"}


def test_legacy_config_without_export_field_resumes_checkpoint_disabled(
    tmp_path: Path,
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    original = _config(
        schema,
        tmp_path / "result.csv",
        models=("ols", "ridge"),
    )
    run_nk_grid(original, max_jobs=1)
    legacy_payload = asdict(original)
    legacy_payload.pop("prediction_export_cells")
    resumed = _config_from_json(legacy_payload)

    run_nk_grid(resumed)

    result = pd.read_csv(resumed.out)
    assert set(result["model"]) == {"ols", "ridge"}
    assert result["status"].eq("ok").all()
    assert not prediction_export_path(resumed.out).exists()


def test_part_write_failure_is_failed_checkpoint_and_resume_recomputes_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    config = _config(
        schema,
        tmp_path / "result.csv",
        models=("ols", "ridge"),
        export_cells=(("ols", 20, 1),),
        failed_abs_threshold=10,
        failed_ratio_threshold=1.0,
    )
    real_writer = nk_grid.write_prediction_part_atomic
    attempts = 0

    def fail_once(rows, target, *, schema):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected prediction part failure")
        return real_writer(rows, target, schema=schema)

    monkeypatch.setattr(nk_grid, "write_prediction_part_atomic", fail_once)
    run_nk_grid(config)
    first = load_checkpoint(config.out)
    assert first.loc[first["model"].eq("ols"), "status"].iloc[0] == "failed"
    assert first.loc[first["model"].eq("ridge"), "status"].iloc[0] == "ok"
    assert not any(prediction_export_parts_dir(config.out).glob("*.parquet"))

    run_nk_grid(config)

    resumed = pd.read_csv(config.out)
    assert resumed["status"].eq("ok").all()
    assert attempts == 2
    assert len(pd.read_parquet(prediction_export_path(config.out))) == 7


def test_failed_model_cell_produces_no_prediction_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema, _ = _external_schema(tmp_path / "input")
    config = _config(
        schema,
        tmp_path / "result.csv",
        export_cells=(("ols", 20, 1),),
        failed_abs_threshold=10,
        failed_ratio_threshold=1.0,
    )

    def fail_fit(**_kwargs):
        raise RuntimeError("injected model failure")

    monkeypatch.setattr(nk_grid, "_fit_predict_model_cell", fail_fit)
    run_nk_grid(config)

    result = pd.read_csv(config.out)
    assert result.loc[0, "status"] == "failed"
    assert not any(prediction_export_parts_dir(config.out).glob("*.parquet"))
    assert pd.read_parquet(prediction_export_path(config.out)).empty


def test_atomic_part_failure_leaves_neither_target_nor_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "parts" / "cell.parquet"

    def partial_then_fail(table, path, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected partial parquet write")

    monkeypatch.setattr(pq, "write_table", partial_then_fail)
    rows = [
        {
            "dataset": "synthetic",
            "model": "ols",
            "seed": 1,
            "draw": 0,
            "N": 10,
            "K": 1,
            "row_id": "row-1",
            "y_true": 1.0,
            "y_pred": 0.75,
        }
    ]
    schema = prediction_export_schema(pd.Series(["row-1"]))
    with pytest.raises(OSError, match="partial parquet"):
        write_prediction_part_atomic(rows, target, schema=schema)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize(
    "row_ids",
    [
        pd.Series([101, 102], dtype="int64"),
        pd.Series(["row-101", "row-102"], dtype="string"),
    ],
)
def test_empty_and_populated_exports_share_the_exact_parquet_schema(
    tmp_path: Path, row_ids: pd.Series
) -> None:
    schema = prediction_export_schema(row_ids)
    rows = [
        {
            "dataset": "synthetic",
            "model": "ols",
            "seed": 1,
            "draw": 0,
            "N": 10,
            "K": 1,
            "row_id": row_ids.iloc[0],
            "y_true": 1.0,
            "y_pred": 0.75,
        }
    ]
    part = tmp_path / "populated.prediction-parts" / "cell.parquet"
    write_prediction_part_atomic(rows, part, schema=schema)
    populated_out = tmp_path / "populated.csv"
    empty_out = tmp_path / "empty.csv"
    populated = materialize_prediction_export_atomic(
        [part], populated_out, schema=schema
    )
    empty = materialize_prediction_export_atomic([], empty_out, schema=schema)

    assert pq.read_schema(populated) == pq.read_schema(empty) == schema

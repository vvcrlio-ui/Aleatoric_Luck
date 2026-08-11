from __future__ import annotations

import csv
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import write_legacy_dynamic_fixture as write_work_snapshot
from legacy_dynamic_adapter import (
    classify_attempts,
    finalize_snapshot as legacy_finalize_snapshot,
    finalize_slice_shards,
    main as legacy_main,
    prepare_round,
    run_slice,
    verify_rounds,
)
import legacy_dynamic_adapter as legacy

import aleatoric_nk_grid.flat_task_table as ft
from aleatoric_nk_grid.flat_task_table import (
    ResourceRequest,
    TaskRow,
    FinalizationError,
    assign_rows_modulo,
    build_rows,
    expected_model_keys,
    finalization_manifest_path,
    read_row_group,
    read_task_table,
    sbatch_resource_args,
    write_task_table,
)
from aleatoric_nk_grid.nk_grid import NKGridConfig
from aleatoric_nk_grid.generation_control import INCOMPLETE_EXIT_CODE


def _config(tmp_path: Path, *, models: tuple[str, ...] = ("ols",)) -> NKGridConfig:
    return NKGridConfig(
        schema=tmp_path / "schema.json", out=tmp_path / "unused.csv", outcome="y", models=models,
        seed=11, test_size=0.3, n_seeds=1, n_draws=1, n_sizes_n=1, n_sizes_k=1,
        max_n=20, max_k=2, batch_size=4, n_jobs=8, repeat_plan=((11, 0),),
    )


def _fake_run(config: NKGridConfig, **_: object) -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"])
        writer.writeheader()
        for model in config.models:
            writer.writerow({"model": model, "seed": config.repeat_plan[0][0], "draw": config.repeat_plan[0][1], "N": config.n_grid[0], "K": config.k_grid[0], "status": "ok", "metric": f"{config.n_grid[0]}:{config.k_grid[0]}"})


def _snapshot(tmp_path: Path, *, workers: int = 2, n_grid: tuple[int, ...] = (10, 12), k_grid: tuple[int, ...] = (1, 2)) -> tuple[Path, tuple[TaskRow, ...]]:
    config = _config(tmp_path)
    rows = build_rows(config, n_grid=n_grid, k_grid=k_grid)
    table = write_task_table(tmp_path / "tasks.parquet", rows, rows_per_group=2)
    snapshot = write_work_snapshot(tmp_path / "snapshot.json", table_path=table, panel="test", config=config, output_dir=tmp_path / "outputs", workers=workers)
    return snapshot, rows


RESULT_HEADER = ["model", "seed", "draw", "N", "K", "status", "metric"]


def _result_row(row: TaskRow, *, status: str = "ok", metric: str | None = None) -> dict[str, object]:
    return {
        "model": row.models[0], "seed": row.seed, "draw": row.draw,
        "N": row.n_samples, "K": row.k_features, "status": status,
        "metric": metric if metric is not None else f"{row.n_samples}:{row.k_features}",
    }


def _write_shard(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_all_terminal(snapshot: Path, rows: tuple[TaskRow, ...]) -> Path:
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    return _write_shard(
        Path(payload["output_dir"]) / "round-1" / "worker-0.csv",
        [_result_row(row) for row in rows],
    )


def test_v2_table_is_immutable_cost_free_and_streamed(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=3)
    table = Path(json.loads(snapshot.read_text())["task_table"])
    source = pq.ParquetFile(table)
    assert source.schema.names == ["row_id", "seed", "draw", "N", "K", "group", "element"] or source.schema.names == ["row_id", "seed", "draw", "N", "K", "group", "models"]
    assert all(column not in source.schema.names for column in ("est_cost", "chunk_id"))
    assert read_task_table(table) == tuple(sorted(rows, key=lambda row: (row.k_features, row.n_samples, row.seed, row.draw, row.group, row.row_id)))
    assert not table.stat().st_mode & stat.S_IWUSR


def test_v1_table_is_rejected_fail_closed(tmp_path):
    path = tmp_path / "old.parquet"
    pq.write_table(pa.table({"row_id": ["a"], "seed": [1], "draw": [0], "N": [10], "K": [1], "group": ["imputed_core"], "models": [["ols"]], "est_cost": [1.0], "chunk_id": [0]}), path)
    with pytest.raises(ValueError, match="v1"):
        read_task_table(path)


def test_snapshot_freeze_rejects_v1_table_without_publishing(tmp_path):
    table = tmp_path / "v1.parquet"
    pq.write_table(pa.table({
        "row_id": ["a"], "seed": [1], "draw": [0], "N": [10], "K": [1],
        "group": ["imputed_core"], "models": [["ols"]],
        "est_cost": [1.0], "chunk_id": [0],
    }), table)
    snapshot = tmp_path / "snapshot.json"
    with pytest.raises(ValueError, match="v1"):
        write_work_snapshot(
            snapshot, table_path=table, panel="test", config=_config(tmp_path),
            output_dir=tmp_path / "outputs", workers=1,
        )
    assert not snapshot.exists()


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_snapshot_freeze_rejects_non_v2_schema_without_publishing(tmp_path, mutation):
    columns = {
        "row_id": ["a"], "seed": [1], "draw": [0], "N": [10], "K": [1],
        "group": ["imputed_core"], "models": [["ols"]],
    }
    if mutation == "missing":
        del columns["models"]
    else:
        columns["unexpected"] = ["value"]
    table = tmp_path / f"{mutation}.parquet"
    pq.write_table(pa.table(columns), table)
    snapshot = tmp_path / "snapshot.json"
    with pytest.raises(ValueError, match="schema"):
        write_work_snapshot(
            snapshot, table_path=table, panel="test", config=_config(tmp_path),
            output_dir=tmp_path / "outputs", workers=1,
        )
    assert not snapshot.exists()


@pytest.mark.parametrize("workers,row_count", [(1, 5), (2, 5), (7, 5), (4, 0)])
def test_prepare_round_assigns_every_todo_row_by_index_modulo(tmp_path, workers, row_count):
    snapshot, rows = _snapshot(tmp_path, workers=workers, n_grid=(10, 12, 14, 16, 18), k_grid=(1,))
    if row_count:
        rows = rows[:row_count]
        table = write_task_table(tmp_path / "short.parquet", rows)
        payload = json.loads(snapshot.read_text()); payload["task_table"] = str(table); Path(snapshot).chmod(0o644); Path(snapshot).write_text(json.dumps(payload)); Path(snapshot).chmod(0o444)
    else:
        # Mark the only model key of every row complete without manufacturing a
        # special empty main table, which production deliberately forbids.
        output = tmp_path / "outputs" / "round-0"; output.mkdir(parents=True)
        with (output / "worker-0.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status"]); writer.writeheader()
            for row in rows: writer.writerow({"model": "ols", "seed": row.seed, "draw": row.draw, "N": row.n_samples, "K": row.k_features, "status": "ok"})
    stats = prepare_round(snapshot, round_index=1, prep_token="prep-1")
    assignment = Path(stats["assignment"])
    groups = [read_row_group(assignment, index) for index in range(workers)]
    assigned = [row for group in groups for row in group]
    assert len(assigned) == len({row.row_id for row in assigned})
    expected = [] if row_count == 0 else list(rows)
    assert {row.row_id for row in assigned} == {row.row_id for row in expected}
    assert all(tuple(row.row_id for row in group) == tuple(row.row_id for row in expected[index::workers]) for index, group in enumerate(groups))


def test_attempt_classification_keeps_crash_and_too_long_separate():
    ordered = [
        {"round": 1, "worker_index": 0, "sequence": 0, "row_id": "crashed"},
        {"round": 1, "worker_index": 0, "sequence": 1, "row_id": "later"},
        {"round": 1, "worker_index": 1, "sequence": 0, "row_id": "long"},
        {"round": 2, "worker_index": 1, "sequence": 0, "row_id": "long"},
        {"round": 3, "worker_index": 1, "sequence": 0, "row_id": "long"},
    ]
    crashed, too_long = classify_attempts(ordered)
    assert crashed == {"crashed"}
    assert too_long == {"long"}
    shuffled = [ordered[index] for index in (4, 1, 3, 0, 2)]
    assert classify_attempts(shuffled) == (crashed, too_long)


def test_worker_refuses_an_unready_assignment_after_failed_prep(tmp_path):
    snapshot, _ = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    with pytest.raises(RuntimeError, match="assignment is not ready.*prep may have failed"):
        run_slice(snapshot, round_index=1, worker_index=0, expected_prep_token="prep-1")


def test_worker_rejects_stale_ready_marker_from_prior_prep_job(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_all_terminal(snapshot, rows)
    stats = prepare_round(snapshot, round_index=1, prep_token="old-prep-job")
    marker = json.loads((Path(stats["assignment"]).parent / "assignment.ready.json").read_text())
    assert marker["prep_token"] == "old-prep-job"
    # The current token succeeds even for the todo=0 path.
    run_slice(
        snapshot, round_index=1, worker_index=0,
        expected_prep_token="old-prep-job",
    )
    monkeypatch.setattr(
        ft, "read_row_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stale assignment was read before token validation")
        ),
    )
    # Simulate a newly submitted prep dying in its shell before Python can
    # replace the old marker. The dependent worker knows the new Slurm job ID.
    with pytest.raises(RuntimeError, match="prep may have failed"):
        run_slice(
            snapshot, round_index=1, worker_index=0,
            expected_prep_token="new-prep-job",
        )


def test_verify_cli_writes_complete_json_and_exits_zero(tmp_path, capsys):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_all_terminal(snapshot, rows)
    legacy_main(["verify", "--snapshot", str(snapshot)])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["missing_model_keys"] == 0
    assert result["crashed_row_ids"] == []
    assert result["too_long_row_ids"] == []
    assert json.loads((tmp_path / "outputs" / "verification.json").read_text()) == result
    assert captured.err == ""


@pytest.mark.parametrize("failure", ("missing_model_keys", "crashed_row_ids", "too_long_row_ids"))
def test_verify_cli_writes_json_then_uses_stable_incomplete_exit_code(
    tmp_path, capsys, failure,
):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    output_dir = tmp_path / "outputs"
    if failure != "missing_model_keys":
        _write_all_terminal(snapshot, rows)
    if failure == "crashed_row_ids":
        attempts = output_dir / "round-2" / "attempts" / "worker-0.jsonl"
        attempts.parent.mkdir(parents=True)
        attempts.write_text(
            '\n'.join((
                json.dumps({"round": 2, "worker_index": 0, "sequence": 0, "row_id": "crashed"}),
                json.dumps({"round": 2, "worker_index": 0, "sequence": 1, "row_id": "later"}),
            )) + '\n',
            encoding="utf-8",
        )
    elif failure == "too_long_row_ids":
        for round_index in (1, 2, 3):
            attempts = output_dir / f"round-{round_index}" / "attempts" / "worker-0.jsonl"
            attempts.parent.mkdir(parents=True, exist_ok=True)
            attempts.write_text(
                json.dumps({
                    "round": round_index, "worker_index": 0,
                    "sequence": 0, "row_id": "too-long",
                }) + '\n',
                encoding="utf-8",
            )
    with pytest.raises(SystemExit) as stopped:
        legacy_main(["verify", "--snapshot", str(snapshot)])
    assert stopped.value.code == INCOMPLETE_EXIT_CODE
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    persisted = json.loads((output_dir / "verification.json").read_text())
    assert persisted == result
    count = result[failure] if failure == "missing_model_keys" else len(result[failure])
    assert count > 0
    assert f"{failure}={count}" in captured.err
    assert "missing_model_keys=" in captured.err
    assert "crashed_row_ids=" in captured.err
    assert "too_long_row_ids=" in captured.err


def test_verify_cli_keeps_malformed_snapshot_as_an_execution_error(tmp_path):
    snapshot = tmp_path / "malformed.json"
    snapshot.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        legacy_main(["verify", "--snapshot", str(snapshot)])


@pytest.mark.parametrize("workers,row_count", [(workers, rows) for workers in range(1, 11) for rows in (0, 1, 2, 7, 19)])
def test_modulo_assignment_is_complete_balanced_and_not_contiguous(workers, row_count):
    rows = tuple(TaskRow(str(index), 1, 0, 10, index + 1, "imputed_core", ("ols",)) for index in range(row_count))
    groups = assign_rows_modulo(rows, workers)
    assert len(groups) == workers
    assert {row.row_id for group in groups for row in group} == {row.row_id for row in rows}
    assert max((len(group) for group in groups), default=0) - min((len(group) for group in groups), default=0) <= 1
    assert all(tuple(row.row_id for row in group) == tuple(str(index) for index in range(worker, row_count, workers)) for worker, group in enumerate(groups))


def test_each_successful_cell_is_atomically_persisted_before_interruption(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10, 12, 14), k_grid=(1,))
    prepare_round(snapshot, round_index=1, prep_token="prep-1")
    calls = 0

    def interrupted(config, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated SIGKILL boundary")
        _fake_run(config, **kwargs)

    monkeypatch.setattr(legacy, "run_nk_grid", interrupted)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        run_slice(snapshot, round_index=1, worker_index=0, expected_prep_token="prep-1")
    shard = tmp_path / "outputs" / "round-1" / "worker-0.csv"
    persisted = list(csv.DictReader(shard.open(encoding="utf-8")))
    assert len(persisted) == 2
    assert json.loads(legacy.manifest_path(shard).read_text())["completion"]["materialized_rows"] == 2
    assert all(not path.name.endswith(".tmp") for path in shard.parent.iterdir())
    monkeypatch.setattr(legacy, "run_nk_grid", _fake_run)
    run_slice(snapshot, round_index=1, worker_index=0, expected_prep_token="prep-1")
    assert len(list(csv.DictReader(shard.open(encoding="utf-8")))) == len(expected_model_keys(rows))


def test_real_two_round_recovery_converges_to_one_shot_output(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=2)
    monkeypatch.setattr(legacy, "run_nk_grid", _fake_run)
    first = prepare_round(snapshot, round_index=1, prep_token="prep-1")
    assert first["todo_rows"] == len(rows)
    run_slice(snapshot, round_index=1, worker_index=0, expected_prep_token="prep-1")
    second = prepare_round(snapshot, round_index=2, prep_token="prep-2")
    assert second["todo_rows"] == len(rows) - len(read_row_group(Path(first["assignment"]), 0))
    run_slice(snapshot, round_index=2, worker_index=0, expected_prep_token="prep-2")
    run_slice(snapshot, round_index=2, worker_index=1, expected_prep_token="prep-2")
    result = verify_rounds(snapshot)
    assert result["missing_model_keys"] == 0
    merged = finalize_slice_shards(Path(json.loads(snapshot.read_text())["task_table"]), (tmp_path / "outputs" / "round-1" / "worker-0.csv", tmp_path / "outputs" / "round-2" / "worker-0.csv", tmp_path / "outputs" / "round-2" / "worker-1.csv"), tmp_path / "merged.csv")
    one_shot = tmp_path / "one-shot.csv"
    with one_shot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "draw", "N", "K", "status", "metric"]); writer.writeheader()
        for row in sorted(rows, key=lambda row: (row.n_samples, row.k_features)):
            writer.writerow({"model": "ols", "seed": row.seed, "draw": row.draw, "N": row.n_samples, "K": row.k_features, "status": "ok", "metric": f"{row.n_samples}:{row.k_features}"})
    pd.testing.assert_frame_equal(pd.read_csv(merged).sort_values(["N", "K"]).reset_index(drop=True), pd.read_csv(one_shot).sort_values(["N", "K"]).reset_index(drop=True), check_exact=True)


def test_streaming_finalizer_prefers_later_terminal_over_historical_failure(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_shard(tmp_path / "outputs" / "round-1" / "worker-0.csv", [
        _result_row(rows[0], status="failed", metric="old failure"),
    ])
    _write_shard(tmp_path / "outputs" / "round-2" / "worker-0.csv", [
        _result_row(rows[0], status="ok", metric="recovered"),
    ])
    receipt = legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    final_rows = list(csv.DictReader((tmp_path / "unused.csv").open(encoding="utf-8")))
    assert len(final_rows) == 1
    assert final_rows[0]["status"] == "ok"
    assert final_rows[0]["metric"] == "recovered"
    assert receipt["historical_failed_rows_overridden"] == 1
    assert receipt["final_rows"] == receipt["expected_model_keys"] == 1
    assert receipt["input_shards"] == receipt["rows_read"] == 2
    assert receipt["duplicate_terminal_keys"] == 0
    assert receipt["temporary_bytes_used"] > 0
    assert receipt["wall_time_seconds"] > 0
    assert receipt["final_output"] == str((tmp_path / "unused.csv").resolve())
    assert not Path(receipt["temporary_directory"]).exists()
    assert json.loads(finalization_manifest_path(tmp_path / "unused.csv").read_text()) == receipt


def test_streaming_finalizer_rejects_conflicting_terminal_rows(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_shard(tmp_path / "outputs" / "round-1" / "worker-0.csv", [
        _result_row(rows[0], metric="first"),
    ])
    _write_shard(tmp_path / "outputs" / "round-2" / "worker-0.csv", [
        _result_row(rows[0], metric="different"),
    ])
    with pytest.raises(FinalizationError, match="conflicting terminal"):
        legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert not (tmp_path / "unused.csv").exists()


def test_streaming_finalizer_rejects_out_of_design_key(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    outside = {**_result_row(rows[0]), "N": 999}
    _write_shard(
        tmp_path / "outputs" / "round-1" / "worker-0.csv",
        [_result_row(rows[0]), outside],
    )
    with pytest.raises(FinalizationError, match="out-of-design"):
        legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert not (tmp_path / "unused.csv").exists()


def test_streaming_finalizer_rejects_missing_terminal_without_publication(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10, 12), k_grid=(1,))
    _write_shard(
        tmp_path / "outputs" / "round-1" / "worker-0.csv",
        [_result_row(rows[0])],
    )
    with pytest.raises(FinalizationError, match="missing 1 expected terminal"):
        legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert not (tmp_path / "unused.csv").exists()


def test_streaming_finalizer_is_atomic_and_rerunnable_after_injected_failure(
    tmp_path, monkeypatch,
):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_all_terminal(snapshot, rows)
    output = tmp_path / "unused.csv"
    output.write_bytes(b"previous-complete-output\n")
    real_publish = ft._publish_final_csv

    def interrupted(temporary, target):
        assert temporary.exists()
        assert target == output.resolve()
        raise RuntimeError("injected before replace")

    monkeypatch.setattr(ft, "_publish_final_csv", interrupted)
    with pytest.raises(RuntimeError, match="before replace"):
        legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert output.read_bytes() == b"previous-complete-output\n"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    monkeypatch.setattr(ft, "_publish_final_csv", real_publish)
    first = legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    first_bytes = output.read_bytes()
    second = legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert output.read_bytes() == first_bytes
    assert first["final_rows"] == second["final_rows"] == 1


def test_streaming_finalizer_matches_small_reference_byte_for_byte(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=1)
    shard = _write_all_terminal(snapshot, rows)
    legacy = finalize_slice_shards(
        Path(json.loads(snapshot.read_text())["task_table"]), (shard,), tmp_path / "legacy.csv",
    )
    monkeypatch.setattr(
        ft, "read_task_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("production finalizer loaded all tasks")),
    )
    legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert (tmp_path / "unused.csv").read_bytes() == legacy.read_bytes()


def test_streaming_finalizer_counts_idempotent_terminal_duplicates(tmp_path):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    duplicate = _result_row(rows[0])
    _write_shard(tmp_path / "outputs" / "round-1" / "worker-0.csv", [duplicate])
    _write_shard(tmp_path / "outputs" / "round-2" / "worker-0.csv", [duplicate])
    receipt = legacy_finalize_snapshot(snapshot, tmp_dir=tmp_path / "scratch")
    assert receipt["duplicate_terminal_keys"] == 1
    assert receipt["duplicate_terminal_rows"] == 1
    assert receipt["final_rows"] == 1


def test_finalize_cli_ignores_stale_unique_temp_directory(tmp_path, capsys):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_all_terminal(snapshot, rows)
    scratch = tmp_path / "scratch"
    stale = scratch / "nk-grid-finalize-stale"
    stale.mkdir(parents=True)
    (stale / "index.sqlite").write_text("interrupted old run", encoding="utf-8")
    legacy_main(["finalize", "--snapshot", str(snapshot), "--tmp-dir", str(scratch)])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "complete"
    assert receipt["final_rows"] == 1
    assert (tmp_path / "unused.csv").exists()
    assert stale.exists()
    assert Path(receipt["temporary_directory"]).name != stale.name
    assert not Path(receipt["temporary_directory"]).exists()


def test_finalization_tmp_directory_priority(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    nk_grid_env = tmp_path / "nk-grid-env"
    tmp_env = tmp_path / "tmp-env"
    monkeypatch.setenv("NK_GRID_TMPDIR", str(nk_grid_env))
    monkeypatch.setenv("TMPDIR", str(tmp_env))
    snapshot = {"finalization": {"tmp_dir": str(configured)}}
    assert ft._resolve_finalization_tmp_base(snapshot, explicit) == explicit.resolve()
    assert ft._resolve_finalization_tmp_base(snapshot, None) == configured.resolve()
    assert ft._resolve_finalization_tmp_base({}, None) == nk_grid_env.resolve()
    monkeypatch.delenv("NK_GRID_TMPDIR")
    assert ft._resolve_finalization_tmp_base({}, None) == tmp_env.resolve()


def test_finalizer_temp_space_preflight_reports_directory_and_bytes(tmp_path, monkeypatch):
    snapshot, rows = _snapshot(tmp_path, workers=1, n_grid=(10,), k_grid=(1,))
    _write_all_terminal(snapshot, rows)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(ft.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(FinalizationError) as failed:
        legacy_finalize_snapshot(snapshot, tmp_dir=scratch)
    message = str(failed.value)
    assert f"temporary_directory={scratch.resolve()}" in message
    assert "available_bytes=1" in message
    assert "estimated_required_bytes=" in message
    assert not list(scratch.glob("nk-grid-finalize-*"))


def test_resource_request_is_single_core_and_records_account_constraint():
    request = ResourceRequest(1, "long", "8G", "12:00:00", "proj", "cpu-a")
    assert sbatch_resource_args(request) == ("--partition=long", "--cpus-per-task=1", "--mem=8G", "--time=12:00:00", "--account=proj", "--constraint=cpu-a")
    with pytest.raises(ValueError, match="one CPU"):
        sbatch_resource_args(ResourceRequest(2, "long", "8G", "12:00:00", "proj", "none"))

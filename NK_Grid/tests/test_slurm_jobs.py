from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aleatoric_nk_grid import slurm_jobs
from aleatoric_nk_grid.model_registry import SUPPORTED_MODEL_NAMES
from aleatoric_nk_grid.nk_grid import NKGridConfig
from aleatoric_nk_grid.run_panels import config_to_json
from aleatoric_nk_grid.slurm_jobs import (
    RESOURCE_CLASSES,
    SlurmJob,
    _expand_array_indices,
    apply_worker_overrides,
    build_slurm_jobs,
    chunk_master_indices,
    dependency_never_satisfied_diagnostics,
    load_chunk_map,
    load_job_snapshot,
    require_large_run_authorization,
    resource_class_for_model,
    resource_class_indices,
    write_chunk_map,
    write_job_snapshot,
    write_submission_receipt,
)


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"
ENGINE_DIR = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **overrides) -> NKGridConfig:
    values = {
        "schema": tmp_path / "schema.json",
        "out": tmp_path / "result.csv",
        "outcome": "y",
        "models": ("ols",),
        "seed": 123,
        "test_size": 0.3,
        "n_seeds": 1,
        "n_draws": 1,
        "n_sizes_n": 1,
        "n_sizes_k": 1,
        "max_n": 20,
        "max_k": 2,
        "batch_size": 1,
        "n_jobs": 4,
        "min_n": 10,
        "model_params": MODEL_PARAMS,
        "rerun_completed": True,
    }
    values.update(overrides)
    return NKGridConfig(**values)


def _write_snapshot(path: Path, jobs: list[SlurmJob]) -> None:
    payload = {
        "format_version": 1,
        "source_manifest": str(path.parent / "panels.yaml"),
        "jobs": [
            {
                "panel": job.panel,
                "model": job.model,
                "seed": job.seed,
                "draws": list(job.draws),
                "config": config_to_json(job.config),
                "final_out": str(job.final_out),
            }
            for job in jobs
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jobs_for_models(tmp_path: Path, models: tuple[str, ...]) -> list[SlurmJob]:
    return [
        SlurmJob(
            panel=f"panel-{index}",
            model=model,
            seed=123,
            draws=(0,),
            config=_config(
                tmp_path,
                models=(model,),
                out=tmp_path / f"result-{index}-{model}.csv",
            ),
            final_out=tmp_path / f"final-{index}-{model}.csv",
        )
        for index, model in enumerate(models)
    ]


def test_worker_uses_runtime_slurm_cpu_allocation_without_mutating_snapshot(
    tmp_path,
):
    frozen = _config(tmp_path, n_jobs=4, rerun_completed=True)

    runtime = apply_worker_overrides(
        frozen,
        rerun_completed=False,
        environ={"SLURM_CPUS_PER_TASK": "8"},
    )

    assert runtime.n_jobs == 8
    assert runtime.rerun_completed is False
    assert frozen.n_jobs == 4
    assert frozen.rerun_completed is True


def test_worker_retains_snapshot_cpu_count_without_slurm_environment(tmp_path):
    frozen = _config(tmp_path, n_jobs=3)
    runtime = apply_worker_overrides(
        frozen,
        rerun_completed=True,
        environ={},
    )
    assert runtime.n_jobs == 3
    assert runtime.rerun_completed is True


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_worker_rejects_invalid_slurm_cpu_allocation(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        apply_worker_overrides(
            _config(tmp_path),
            rerun_completed=False,
            environ={"SLURM_CPUS_PER_TASK": value},
        )


@pytest.mark.parametrize("rerun_completed", [False, True])
def test_snapshot_freezes_explicit_slurm_rerun_policy(
    tmp_path,
    monkeypatch,
    rerun_completed,
):
    frozen = SlurmJob("panel", "ols", 123, (0,), _config(tmp_path), tmp_path / "final.csv")
    monkeypatch.setattr(slurm_jobs, "build_slurm_jobs", lambda path: [frozen])
    monkeypatch.setattr(
        slurm_jobs,
        "resolved_panels",
        lambda path: [("panel", _config(tmp_path))],
    )
    snapshot = tmp_path / "jobs.json"

    written = write_job_snapshot(
        tmp_path / "panels.yaml",
        snapshot,
        allow_large_run=False,
        rerun_completed=rerun_completed,
    )

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    loaded = load_job_snapshot(snapshot)
    assert written[0].config.rerun_completed is rerun_completed
    assert loaded[0].config.rerun_completed is rerun_completed
    assert payload["execution_policy"] == {
        "rerun_completed": rerun_completed
    }
    assert payload["panel_outputs"] == {
        "panel": str(tmp_path / "result.csv")
    }


def test_snapshot_requires_explicit_seed_draws_and_final_output(tmp_path):
    config_payload = config_to_json(_config(tmp_path))
    config_payload.pop("rerun_completed")
    snapshot = tmp_path / "legacy.json"
    snapshot.write_text(
        json.dumps(
            {
                "format_version": 1,
                "jobs": [
                    {
                        "panel": "panel",
                        "model": "ols",
                        "config": config_payload,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_job_snapshot(snapshot)


def test_resource_classes_are_mutually_exclusive_and_exhaust_all_models(
    tmp_path,
):
    jobs = _jobs_for_models(tmp_path, SUPPORTED_MODEL_NAMES)
    grouped = {
        resource_class: set(resource_class_indices(jobs, resource_class))
        for resource_class in RESOURCE_CLASSES
    }

    assert set().union(*grouped.values()) == set(range(len(jobs)))
    for left_index, left_class in enumerate(RESOURCE_CLASSES):
        for right_class in RESOURCE_CLASSES[left_index + 1 :]:
            assert grouped[left_class].isdisjoint(grouped[right_class])
    for index, job in enumerate(jobs):
        assert index in grouped[resource_class_for_model(job.model)]

    assert {jobs[index].model for index in grouped["serial"]} == {"lightgbm"}
    assert {jobs[index].model for index in grouped["super_learner"]} == {"super_learner"}
    assert grouped["parallel"]
    assert RESOURCE_CLASSES == ("parallel", "serial", "super_learner")


def test_resource_class_indices_rejects_unknown_class(tmp_path):
    with pytest.raises(ValueError, match="Unknown Slurm resource class"):
        resource_class_indices(
            _jobs_for_models(tmp_path, ("ols",)),
            "gpu",
        )


def test_per_model_final_outputs_are_unique_within_one_panel(
    tmp_path, monkeypatch
):
    panel_config = _config(
        tmp_path,
        models=("ols", "ridge"),
        out=tmp_path / "panel.csv",
    )
    monkeypatch.setattr(
        slurm_jobs,
        "resolved_panels",
        lambda path: [("panel", panel_config)],
    )

    jobs = build_slurm_jobs(tmp_path / "panels.yaml")

    final_by_model = {job.model: job.final_out for job in jobs}
    assert final_by_model == {
        "ols": tmp_path / ".seed-final" / "panel" / "ols.csv",
        "ridge": tmp_path / ".seed-final" / "panel" / "ridge.csv",
    }


@pytest.mark.parametrize(
    ("indices", "maximum", "expected"),
    [
        ((), 4, ()),
        ((7,), 4, ((7,),)),
        ((0, 1, 2, 3), 4, ((0, 1, 2, 3),)),
        ((0, 1, 2, 3, 4), 4, ((0, 1, 2, 3), (4,))),
        ((12, 17, 25, 90, 103), 2, ((12, 17), (25, 90), (103,))),
    ],
)
def test_chunk_master_indices_obeys_max_array_size(
    indices, maximum, expected
):
    assert chunk_master_indices(indices, maximum) == expected


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1, 1],
        [-1],
        [True],
        [1.5],
        ["1"],
    ],
)
def test_chunk_map_rejects_empty_duplicate_negative_or_noninteger(
    tmp_path, values
):
    chunk_map = tmp_path / "chunk.json"
    chunk_map.write_text(
        json.dumps(
            {
                "format_version": 1,
                "snapshot": str((tmp_path / "jobs.json").resolve()),
                "resource_class": "parallel",
                "master_indices": values,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid Slurm chunk map"):
        load_chunk_map(chunk_map)


def test_chunk_map_rejects_wrong_snapshot_and_resource_class(tmp_path):
    snapshot = tmp_path / "jobs.json"
    chunk_map = write_chunk_map(snapshot, "parallel", (3, 7), 0)
    with pytest.raises(ValueError, match="different snapshot"):
        load_chunk_map(
            chunk_map, snapshot_path=tmp_path / "other.json"
        )
    with pytest.raises(ValueError, match="different resource class"):
        load_chunk_map(
            chunk_map,
            snapshot_path=snapshot,
            resource_class="serial",
        )


def test_worker_rejects_chunk_local_index_out_of_bounds(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(snapshot, _jobs_for_models(tmp_path, ("ols",)))
    chunk_map = write_chunk_map(snapshot, "parallel", (0,), 0)
    monkeypatch.setattr(
        slurm_jobs,
        "run_nk_grid",
        lambda *args, **kwargs: pytest.fail("worker must not run"),
    )

    with pytest.raises(SystemExit) as error:
        slurm_jobs.main(
            [
                "run",
                "--snapshot",
                str(snapshot),
                "--chunk-map",
                str(chunk_map),
                "--index",
                "1",
            ]
        )
    assert error.value.code == 2


def test_dependency_never_satisfied_targets_only_current_snapshot_receipts(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "jobs.json"
    snapshot.write_text("{}", encoding="utf-8")
    other_snapshot = tmp_path / "other.json"
    other_snapshot.write_text("{}", encoding="utf-8")
    for job_id, owner in (
        ("101", snapshot),
        ("102", snapshot),
        ("999", other_snapshot),
    ):
        (tmp_path / f"slurm-{job_id}.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "snapshot": str(owner.resolve()),
                    "slurm_job_id": job_id,
                }
            ),
            encoding="utf-8",
        )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "101|DependencyNeverSatisfied|PENDING\n"
                    "102|Resources|PENDING\n"
                    "999|DependencyNeverSatisfied|PENDING\n"
                    "777|DependencyNeverSatisfied|PENDING\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(slurm_jobs.subprocess, "run", fake_run)
    payload = dependency_never_satisfied_diagnostics(
        snapshot, cleanup=True
    )

    assert payload == {
        "snapshot": str(snapshot.resolve()),
        "receipt_job_ids": ["101", "102"],
        "dependency_never_satisfied_job_ids": ["101"],
        "cleaned_job_ids": ["101"],
    }
    assert calls[0][0] == [
        "squeue",
        "--noheader",
        "--jobs",
        "101,102",
        "--format",
        "%A|%r|%T",
    ]
    assert calls[1][0] == ["scancel", "101"]


@pytest.mark.parametrize(
    ("array_spec", "job_count", "expected"),
    [
        ("0", 1, (0,)),
        ("0-4", 5, (0, 1, 2, 3, 4)),
        ("0-6:2", 7, (0, 2, 4, 6)),
        ("0,2,4%2", 5, (0, 2, 4)),
        ("1-3,5", 6, (1, 2, 3, 5)),
    ],
)
def test_array_spec_expansion_accepts_supported_slurm_forms(
    array_spec,
    job_count,
    expected,
):
    assert _expand_array_indices(array_spec, job_count) == expected


@pytest.mark.parametrize(
    ("array_spec", "job_count", "match"),
    [
        ("", 3, "must not be empty"),
        ("0,,2", 3, "empty token"),
        ("0-2%0", 3, "concurrency suffix"),
        ("0-2%many", 3, "concurrency suffix"),
        ("0-2:0", 3, "step must be"),
        ("0:2", 3, "step requires a range"),
        ("2-1", 3, "start must not exceed"),
        ("zero", 3, "index must be an integer"),
        ("0-2,2", 3, "duplicate indices"),
        ("3", 3, "between 0 and 2"),
        ("0-3", 3, "between 0 and 2"),
    ],
)
def test_array_spec_expansion_rejects_ambiguous_or_out_of_bounds_forms(
    array_spec,
    job_count,
    match,
):
    with pytest.raises(ValueError, match=match):
        _expand_array_indices(array_spec, job_count)


@pytest.mark.parametrize(
    ("extra_args", "expected_rerun"),
    [([], False), (["--rerun-completed"], True)],
)
def test_run_command_applies_safe_worker_policy_and_runtime_cpus(
    tmp_path,
    monkeypatch,
    extra_args,
    expected_rerun,
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        [SlurmJob("panel", "ols", 123, (0,), _config(tmp_path, rerun_completed=True), tmp_path / "final.csv")],
    )
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return config.out

    monkeypatch.setattr(slurm_jobs, "run_nk_grid", fake_run)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    slurm_jobs.main(
        [
            "run",
            "--snapshot",
            str(snapshot),
            "--index",
            "0",
            *extra_args,
        ]
    )

    assert captured["config"].n_jobs == 8
    assert captured["config"].rerun_completed is expected_rerun
    assert captured["kwargs"]["stop_after_batch"] is None
    assert captured["kwargs"]["defer_materialization_on_stop"] is False


@pytest.mark.parametrize(
    ("completion_status", "marker_remains"),
    [("incomplete", True), ("complete", False)],
)
def test_run_command_handles_stop_marker_after_engine_returns(
    tmp_path,
    monkeypatch,
    completion_status,
    marker_remains,
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        [SlurmJob("panel", "ols", 123, (0,), _config(tmp_path), tmp_path / "final.csv")],
    )
    stop_marker = tmp_path / "stop-requested"
    stop_marker.touch()
    result = tmp_path / "result.csv"
    result.with_suffix(".manifest.json").write_text(
        json.dumps({"completion": {"status": completion_status}}),
        encoding="utf-8",
    )
    observed = {}

    def fake_run(config, **kwargs):
        observed["stop_requested"] = kwargs["stop_after_batch"]()
        observed["defer_materialization_on_stop"] = kwargs[
            "defer_materialization_on_stop"
        ]
        return result

    monkeypatch.setattr(slurm_jobs, "run_nk_grid", fake_run)
    monkeypatch.setenv("NK_GRID_STOP_REQUEST_FILE", str(stop_marker))

    slurm_jobs.main(
        [
            "run",
            "--snapshot",
            str(snapshot),
            "--index",
            "0",
        ]
    )

    assert observed["stop_requested"] is True
    assert observed["defer_materialization_on_stop"] is True
    assert stop_marker.exists() is marker_remains


def test_receipt_records_runtime_policy_resources_and_reproducible_rerun(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(snapshot, [SlurmJob("panel", "ols", 123, (0,), _config(tmp_path), tmp_path / "final.csv")])
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")
    engine_dir = worker.parent.parent
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    monkeypatch.setenv("ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("VENV", str(venv))
    monkeypatch.setenv("PYTHON", str(python))

    receipt = write_submission_receipt(
        snapshot,
        slurm_job_id="12345",
        array_spec="0%3",
        worker_script=worker,
        allow_large_run=True,
        rerun_completed=True,
        cpus_per_task=8,
        memory="64G",
        time_limit="2-00:00:00",
        max_restarts=4,
        resource_class="parallel",
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    command = payload["rerun_command_template"]
    assert "snapshot_sha256" not in payload
    assert "worker_script_sha256" not in payload
    assert payload["execution_paths"]["engine_dir"] == str(engine_dir)
    assert payload["execution_policy"] == {
        "rerun_completed": True,
        "max_restarts": 4,
        "max_concurrent_per_class": 3,
        "advance_signal_seconds": 300,
        "requeue_watchdog_seconds": 240,
    }
    assert payload["resources"] == {
        "class": "parallel",
        "cpus_per_task": 8,
        "memory": "64G",
        "time_limit": "2-00:00:00",
    }
    assert f"--chdir={engine_dir}" in command
    assert "ENGINE_DIR=" in command
    assert "--requeue" in command
    assert "--signal=B:USR1@300" in command
    assert "--open-mode=append" in command
    assert "--job-name=al-nk-grid-parallel" in command
    assert "--export=ALL" in command
    assert "--cpus-per-task=8" in command
    assert "--mem=64G" in command
    assert "--time=2-00:00:00" in command
    assert command.endswith(" 1 1")
    assert "--array=0" in payload["jobs"][0]["rerun_command"]
    assert "<task-index>" not in payload["jobs"][0]["rerun_command"]


def test_chunk_receipt_rerun_uses_local_not_sparse_master_index(tmp_path):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot, _jobs_for_models(tmp_path, ("ols", "ridge"))
    )
    chunk_map = write_chunk_map(snapshot, "parallel", (1,), 0)
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    receipt = write_submission_receipt(
        snapshot,
        slurm_job_id="12345",
        array_spec="0",
        worker_script=worker,
        allow_large_run=False,
        resource_class="parallel",
        chunk_map=chunk_map,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    assert payload["chunk"]["master_indices"] == [1]
    assert payload["jobs"][0]["index"] == 1
    assert "--array=0" in payload["jobs"][0]["rerun_command"]
    assert "--array=1" not in payload["jobs"][0]["rerun_command"]


def test_receipt_freezes_present_and_missing_model_environment_and_signal_policy(
    tmp_path,
    monkeypatch,
):
    for key in slurm_jobs.OPTIONAL_SLURM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RF_N_ESTIMATORS", "17")
    monkeypatch.setenv("ENGINE_DIR", str(tmp_path / "engine"))
    monkeypatch.setenv("VENV", str(tmp_path / "venv"))
    monkeypatch.setenv("PYTHON", str(tmp_path / "venv" / "bin" / "python"))
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(snapshot, [SlurmJob("panel", "ols", 123, (0,), _config(tmp_path), tmp_path / "final.csv")])
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    receipt = write_submission_receipt(
        snapshot,
        slurm_job_id="12345",
        array_spec="0",
        worker_script=worker,
        allow_large_run=False,
        resource_class="parallel",
        requeue_watchdog_seconds=0,
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    submission_environment = payload["submission_environment"]
    assert set(submission_environment) == set(
        slurm_jobs.OPTIONAL_SLURM_ENV_KEYS
    )
    assert submission_environment["RF_N_ESTIMATORS"] == "17"
    assert submission_environment["XGB_MAX_ROUNDS"] is None
    assert payload["execution_policy"]["advance_signal_seconds"] == 300
    assert payload["execution_policy"]["requeue_watchdog_seconds"] == 0

    for command in (
        payload["rerun_command_template"],
        payload["jobs"][0]["rerun_command"],
    ):
        arguments = shlex.split(command)
        sbatch_position = arguments.index("sbatch")
        frozen_environment = arguments[:sbatch_position]
        assert "RF_N_ESTIMATORS=17" in frozen_environment
        assert any(
            frozen_environment[index : index + 2]
            == ["-u", "XGB_MAX_ROUNDS"]
            for index in range(len(frozen_environment) - 1)
        )
        assert "REQUEUE_WATCHDOG_SECONDS=0" in frozen_environment
        assert "--requeue" in arguments
        assert "--signal=B:USR1@300" in arguments
        assert "--open-mode=append" in arguments


@pytest.mark.parametrize(
    ("resource_class", "array_spec", "match"),
    [
        ("serial", "0", "do not belong"),
        # The retired BART class must now be rejected as unknown, not silently
        # accepted as an empty array.
        ("bart", "0", "Unknown Slurm resource class"),
        ("gpu", "0", "Unknown Slurm resource class"),
    ],
)
def test_receipt_rejects_unknown_or_mismatched_resource_class(
    tmp_path,
    resource_class,
    array_spec,
    match,
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        _jobs_for_models(tmp_path, ("ols", "lightgbm", "super_learner")),
    )
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        write_submission_receipt(
            snapshot,
            slurm_job_id="12345",
            array_spec=array_spec,
            worker_script=worker,
            allow_large_run=False,
            resource_class=resource_class,
        )


def test_receipt_rejects_array_indices_outside_snapshot(tmp_path):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        _jobs_for_models(tmp_path, ("ols", "lightgbm", "super_learner")),
    )
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 0 and 2"):
        write_submission_receipt(
            snapshot,
            slurm_job_id="12345",
            array_spec="0,3%2",
            worker_script=worker,
            allow_large_run=False,
            resource_class="parallel",
        )


@pytest.mark.parametrize("watchdog_seconds", [-1, 241])
def test_receipt_rejects_watchdog_without_requeue_safety_margin(
    tmp_path,
    watchdog_seconds,
):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(snapshot, _jobs_for_models(tmp_path, ("ols",)))
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 0 and 240"):
        write_submission_receipt(
            snapshot,
            slurm_job_id="12345",
            array_spec="0",
            worker_script=worker,
            allow_large_run=False,
            resource_class="parallel",
            requeue_watchdog_seconds=watchdog_seconds,
        )


def test_receipt_accepts_only_indices_from_declared_resource_class(tmp_path):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        _jobs_for_models(
            tmp_path,
            ("ols", "lightgbm", "extra_trees", "ridge", "super_learner"),
        ),
    )
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    receipt = write_submission_receipt(
        snapshot,
        slurm_job_id="12345",
        array_spec="1%1",
        worker_script=worker,
        allow_large_run=False,
        resource_class="serial",
        cpus_per_task=1,
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["array"] == "1%1"
    assert payload["job_count"] == 1
    assert payload["snapshot_job_count"] == 5
    assert payload["resources"]["class"] == "serial"
    assert payload["resources"]["cpus_per_task"] == 1
    assert [(job["index"], job["model"]) for job in payload["jobs"]] == [(1, "lightgbm")]


def test_super_learner_receipt_records_independent_class_and_cpu(tmp_path):
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(
        snapshot,
        _jobs_for_models(
            tmp_path,
            ("ols", "lightgbm", "extra_trees", "ridge", "super_learner"),
        ),
    )
    worker = tmp_path / "engine" / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")

    receipt = write_submission_receipt(
        snapshot,
        slurm_job_id="12345",
        array_spec="4%1",
        worker_script=worker,
        allow_large_run=False,
        resource_class="super_learner",
        cpus_per_task=4,
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["resources"]["class"] == "super_learner"
    assert payload["resources"]["cpus_per_task"] == 4
    assert [(job["index"], job["model"]) for job in payload["jobs"]] == [
        (4, "super_learner")
    ]


def test_large_run_authorization_estimates_each_job_once(tmp_path, monkeypatch):
    jobs = [
        SlurmJob(
            f"panel-{index}",
            "ols",
            123,
            (0,),
            _config(tmp_path, out=tmp_path / f"result-{index}.csv"),
            tmp_path / f"final-{index}.csv",
        )
        for index in range(3)
    ]
    calls = []

    def fake_estimate(config):
        calls.append(config.out)
        return {"top_level_model_cells": 1}

    monkeypatch.setattr(slurm_jobs, "estimate_run_size", fake_estimate)

    require_large_run_authorization(jobs, allow_large_run=False)

    assert calls == [job.config.out for job in jobs]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("explicit_maximum", "scontrol_output", "expected_code"),
    [
        (None, "MaxArraySize = 2\n", 0),
        (None, "ClusterName = synthetic\n", 1),
        ("2", "must-not-be-read\n", 0),
    ],
    ids=("auto-detect", "auto-detect-fails", "explicit-override"),
)
def test_submitter_max_array_size_detection_and_override(
    tmp_path,
    explicit_maximum,
    scontrol_output,
    expected_code,
):
    engine = tmp_path / "engine"
    (engine / "slurm").mkdir(parents=True)
    (engine / "slurm" / "run_nk_grid.sbatch").write_text(
        "#!/bin/bash\n", encoding="utf-8"
    )
    manifest = tmp_path / "panels.yaml"
    manifest.write_text("panels: []\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_log = tmp_path / "python.log"
    sbatch_log = tmp_path / "sbatch.log"
    scontrol_log = tmp_path / "scontrol.log"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_PYTHON_LOG"
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$3" = "snapshot" ]; then echo 3; exit 0; fi
if [ "$3" = "indices" ]; then
  case "$*" in
    *"--resource-class parallel"*) echo "0,1,2" ;;
    *) echo "" ;;
  esac
  exit 0
fi
if [ "$3" = "chunk-map" ]; then
  case "$*" in
    *"--chunk-ordinal 0"*) echo /synthetic/chunk-0.json ;;
    *"--chunk-ordinal 1"*) echo /synthetic/chunk-1.json ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "receipt" ]; then echo /synthetic/receipt.json; exit 0; fi
if [ "$3" = "build-map" ]; then echo "/synthetic/$*.json"; exit 0; fi
if [ "$3" = "map-count" ]; then echo 1; exit 0; fi
exit 9
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf 'called\n' >> "$FAKE_SCONTROL_LOG"
printf '%s' "$FAKE_SCONTROL_OUTPUT"
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_SBATCH_LOG"
LINES=0
if [ -f "$FAKE_SBATCH_LOG" ]; then LINES=$(wc -l < "$FAKE_SBATCH_LOG"); fi
echo "$((50000 + LINES));cluster"
""",
    )
    command = [
        "bash",
        str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
        "--manifest",
        str(manifest),
    ]
    if explicit_maximum is not None:
        command.extend(["--max-array-size", explicit_maximum])
    completed = subprocess.run(
        command,
        env={
            **os.environ,
            "ENGINE_DIR": str(engine),
            "VENV": str(tmp_path / "venv"),
            "PYTHON": str(fake_python),
            "FAKE_PYTHON_LOG": str(python_log),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SCONTROL_LOG": str(scontrol_log),
            "FAKE_SCONTROL_OUTPUT": scontrol_output,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == expected_code, completed.stderr
    if expected_code:
        assert "Unable to determine Slurm MaxArraySize" in completed.stderr
        assert not sbatch_log.exists()
        assert scontrol_log.exists()
    else:
        python_calls = python_log.read_text(encoding="utf-8")
        chunk_calls = [
            call
            for call in python_calls.splitlines()
            if " aleatoric_nk_grid.slurm_jobs chunk-map " in f" {call} "
        ]
        assert len(chunk_calls) == 2
        assert "--max-array-size 2 --chunk-ordinal 0" in python_calls
        assert "--max-array-size 2 --chunk-ordinal 1" in python_calls
        sbatch_calls = sbatch_log.read_text(encoding="utf-8").splitlines()
        assert len(sbatch_calls) == 4
        assert "--array=0-1" in sbatch_calls[0]
        assert "--array=0" in sbatch_calls[1]
        assert "--dependency=afterany:50001" in sbatch_calls[1]
        assert "--dependency=afterany:50002" in sbatch_calls[2]
        assert "--dependency=afterany:50003" in sbatch_calls[3]
        if explicit_maximum is None:
            assert scontrol_log.exists()
        else:
            assert not scontrol_log.exists()


@pytest.mark.parametrize("receipt_fails", [False, True])
def test_submitter_splits_mutually_exclusive_resource_class_arrays(
    tmp_path,
    receipt_fails,
):
    engine = tmp_path / "engine"
    worker = engine / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest = tmp_path / "article" / "panels.yaml"
    manifest.parent.mkdir()
    manifest.write_text("panels: []\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_log = tmp_path / "python.log"
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_counter = tmp_path / "sbatch-counter"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_PYTHON_LOG"
if [ "$1" = "-c" ]; then
  exit 0
fi
if [ "$3" = "snapshot" ]; then
  echo 5
  exit 0
fi
if [ "$3" = "indices" ]; then
  case "$*" in
    *"--resource-class parallel"*) echo "0,2,3" ;;
    *"--resource-class serial"*) echo "1" ;;
    *"--resource-class super_learner"*) echo "4" ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "chunk-map" ]; then echo "/synthetic/$*.chunk.json"; exit 0; fi
if [ "$3" = "build-map" ]; then
  case "$*" in
    *"--kind finalizer"*) echo "${FAKE_FINALIZER_MAP}" ;;
    *"--kind publish"*) echo "${FAKE_PUBLISH_MAP}" ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "map-count" ]; then
  case "$*" in
    *"${FAKE_FINALIZER_MAP}"*) echo 2 ;;
    *"${FAKE_PUBLISH_MAP}"*) echo 1 ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "receipt" ]; then
  if [ "${FAIL_RECEIPT:-0}" = "1" ]; then
    echo synthetic-receipt-failure >&2
    exit 9
  fi
  echo /synthetic/receipt.json
  exit 0
fi
exit 9
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
COUNT=0
if [ -f "$FAKE_SBATCH_COUNTER" ]; then
  COUNT=$(<"$FAKE_SBATCH_COUNTER")
fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > "$FAKE_SBATCH_COUNTER"
{
  printf 'CALL %s\n' "$COUNT"
  printf '%s\n' "$@"
} >> "$FAKE_SBATCH_LOG"
echo "$((98764 + COUNT));cluster-a"
""",
    )
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    environment = {
        **os.environ,
        "ENGINE_DIR": str(engine),
        "VENV": str(tmp_path / "venv"),
        "PYTHON": str(fake_python),
        "FAKE_PYTHON_LOG": str(python_log),
        "FAKE_SBATCH_LOG": str(sbatch_log),
        "FAKE_SBATCH_COUNTER": str(sbatch_counter),
        "FAKE_FINALIZER_MAP": str(tmp_path / "finalizers.json"),
        "FAKE_PUBLISH_MAP": str(tmp_path / "publish.json"),
        "FAIL_RECEIPT": "1" if receipt_fails else "0",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
        "--manifest",
        str(manifest),
        "--max-array-size",
        "10",
            "--rerun-completed",
            "--max-concurrent-per-class",
            "2",
            "--cpus-per-task",
            "6",
            "--super-learner-cpus-per-task",
            "4",
            "--mem",
            "32G",
            "--time",
            "2-00:00:00",
            "--finalizer-mem",
            "12G",
            "--finalizer-time",
            "08:00:00",
            "--publish-mem",
            "24G",
            "--publish-time",
            "1-12:00:00",
            "--max-restarts",
            "3",
            "--requeue-watchdog-seconds",
            "120",
            "--max-array-size",
            "10",
        ],
        cwd=unrelated,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "Submitted Slurm parallel array 98765 with 3 tasks "
        "(array=0-2%2, chunk=0)"
        in completed.stdout
    )
    assert (
        "Submitted Slurm serial array 98766 with 1 tasks "
        "(array=0%2, chunk=0)"
        in completed.stdout
    )
    assert (
        "Submitted Slurm super_learner array 98767 with 1 tasks "
        "(array=0%2, chunk=0)"
        in completed.stdout
    )
    assert "Submitted seed finalizer array 98768 with 2 tasks" in completed.stdout
    assert "Submitted panel publish array 98769 with 1 tasks" in completed.stdout
    if receipt_fails:
        assert "Receipt:" not in completed.stdout
        for job_id in ("98765", "98766", "98767"):
            assert f"job {job_id} was submitted" in completed.stderr
        assert completed.stderr.count("synthetic-receipt-failure") == 3
    else:
        assert completed.stdout.count("Receipt: /synthetic/receipt.json") == 3
    assert (engine / "logs").is_dir()
    sbatch_lines = sbatch_log.read_text(encoding="utf-8").splitlines()
    call_starts = [
        index for index, line in enumerate(sbatch_lines) if line.startswith("CALL ")
    ]
    assert len(call_starts) == 5
    sbatch_calls = []
    for position, start in enumerate(call_starts):
        end = (
            call_starts[position + 1]
            if position + 1 < len(call_starts)
            else len(sbatch_lines)
        )
        sbatch_calls.append(sbatch_lines[start + 1 : end])

    expected_resources = [
        ("--cpus-per-task=6", "--array=0-2%2"),
        ("--cpus-per-task=1", "--array=0%2"),
        ("--cpus-per-task=4", "--array=0%2"),
    ]
    snapshot_paths = set()
    for sbatch_args, resource_args in zip(sbatch_calls[:3], expected_resources):
        assert f"--chdir={engine}" in sbatch_args
        assert "--requeue" in sbatch_args
        assert "--signal=B:USR1@300" in sbatch_args
        assert "--open-mode=append" in sbatch_args
        assert "--export=ALL" in sbatch_args
        for argument in (*resource_args, "--mem=32G", "--time=2-00:00:00"):
            assert argument in sbatch_args
        assert sbatch_args[-5] == str(worker)
        assert sbatch_args[-4].startswith(
            str(engine / "logs" / "slurm-specs" / "jobs-")
        )
        assert sbatch_args[-4].endswith(".json")
        snapshot_paths.add(sbatch_args[-4])
        assert sbatch_args[-3:-1] == ["0", "1"]
        assert sbatch_args[-1].endswith(".chunk.json")
    assert len(snapshot_paths) == 1
    finalizer_call, publish_call = sbatch_calls[3:]
    for call in (finalizer_call, publish_call):
        assert "--requeue" not in call
        assert "--signal=B:USR1@300" not in call
        assert "--cpus-per-task=1" in call
        assert "--mem=32G" not in call
        assert "--time=2-00:00:00" not in call
    assert "--mem=12G" in finalizer_call
    assert "--time=08:00:00" in finalizer_call
    assert "--mem=24G" in publish_call
    assert "--time=1-12:00:00" in publish_call
    assert "--dependency=afterany:98765:98766:98767" in finalizer_call
    assert "--array=0-1" in finalizer_call
    assert finalizer_call[-1] == "finalize"
    assert "--dependency=afterany:98768" in publish_call
    assert "--array=0" in publish_call
    assert publish_call[-1] == "publish"

    python_calls = python_log.read_text(encoding="utf-8").splitlines()
    assert sum(" snapshot " in f" {call} " for call in python_calls) == 1
    for resource_class, array_spec, cpus in (
        ("parallel", "0-2%2", "6"),
        ("serial", "0%2", "1"),
        ("super_learner", "0%2", "4"),
    ):
        assert any(
            " indices " in f" {call} "
            and f"--resource-class {resource_class}" in call
            for call in python_calls
        )
        receipt_call = next(
            call
            for call in python_calls
            if " receipt " in f" {call} "
            and f"--resource-class {resource_class}" in call
        )
        assert f"--array-spec {array_spec}" in receipt_call
        assert f"--cpus-per-task {cpus}" in receipt_call
        assert "--memory 32G" in receipt_call
        assert "--time-limit 2-00:00:00" in receipt_call
        assert "--chunk-map-path /synthetic/" in receipt_call
        assert "--max-restarts 3" in receipt_call
        assert "--requeue-watchdog-seconds 120" in receipt_call
        assert "--rerun-completed" in receipt_call
    assert all(
        "finalization-receipt" not in call
        and "finalization-status" not in call
        for call in python_calls
    )


def test_submitter_rejects_legacy_max_concurrent_option():
    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
            "--max-concurrent",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "--max-concurrent is ambiguous after resource-class splitting; "
        "use --max-concurrent-per-class"
    ) in completed.stderr


def test_submitter_reports_prior_job_when_second_resource_class_fails(
    tmp_path,
):
    engine = tmp_path / "engine"
    worker = engine / "slurm" / "run_nk_grid.sbatch"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest = tmp_path / "article" / "panels.yaml"
    manifest.parent.mkdir()
    manifest.write_text("panels: []\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_log = tmp_path / "python.log"
    sbatch_counter = tmp_path / "sbatch-counter"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_PYTHON_LOG"
if [ "$1" = "-c" ]; then
  exit 0
fi
if [ "$3" = "snapshot" ]; then
  PREVIOUS=""
  for ARGUMENT in "$@"; do
    if [ "$PREVIOUS" = "--snapshot" ]; then
      : > "$ARGUMENT"
    fi
    PREVIOUS="$ARGUMENT"
  done
  echo 2
  exit 0
fi
if [ "$3" = "count" ]; then
  echo 2
  exit 0
fi
if [ "$3" = "indices" ]; then
  case "$*" in
    *"--resource-class parallel"*) echo "0" ;;
    *"--resource-class serial"*) echo "1" ;;
    *"--resource-class super_learner"*) echo "" ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "recovery-indices" ]; then echo "1"; exit 0; fi
if [ "$3" = "dependency-diagnostics" ]; then
  echo '{"dependency_never_satisfied_job_ids":[]}'
  exit 0
fi
if [ "$3" = "chunk-map" ]; then echo /synthetic/chunk.json; exit 0; fi
if [ "$3" = "build-map" ]; then
  case "$*" in
    *"--kind finalizer"*) echo /synthetic/finalizers.json ;;
    *"--kind publish"*) echo /synthetic/publish.json ;;
    *) exit 8 ;;
  esac
  exit 0
fi
if [ "$3" = "map-count" ]; then echo 1; exit 0; fi
if [ "$3" = "receipt" ]; then
  echo /synthetic/parallel-receipt.json
  exit 0
fi
exit 9
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
COUNT=0
if [ -f "$FAKE_SBATCH_COUNTER" ]; then
  COUNT=$(<"$FAKE_SBATCH_COUNTER")
fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > "$FAKE_SBATCH_COUNTER"
if [ "$COUNT" -eq 2 ]; then
  echo synthetic-serial-submit-failure >&2
  exit 9
fi
echo '41001;cluster-a'
""",
    )
    environment = {
        **os.environ,
        "ENGINE_DIR": str(engine),
        "VENV": str(tmp_path / "venv"),
        "PYTHON": str(fake_python),
        "FAKE_PYTHON_LOG": str(python_log),
        "FAKE_SBATCH_COUNTER": str(sbatch_counter),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
            "--manifest",
            str(manifest),
            "--max-array-size",
            "10",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert (
        "Submitted Slurm parallel array 41001 with 1 tasks (array=0, chunk=0)"
        in completed.stdout
    )
    assert "Receipt: /synthetic/parallel-receipt.json" in completed.stdout
    assert "Submitted Slurm serial" not in completed.stdout
    assert "Failed to submit Slurm serial array." in completed.stderr
    assert "already submitted: parallel=41001" in completed.stderr
    assert sbatch_counter.read_text(encoding="utf-8").strip() == "2"
    receipt_calls = [
        call
        for call in python_log.read_text(encoding="utf-8").splitlines()
        if " receipt " in f" {call} "
    ]
    assert len(receipt_calls) == 1
    assert "--resource-class parallel" in receipt_calls[0]
    snapshot_path = next(
        (engine / "logs" / "slurm-specs").glob("jobs-*.json")
    )

    recovery = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
            "--snapshot",
            str(snapshot_path),
            "--resource-class",
            "serial",
            "--max-array-size",
            "10",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert recovery.returncode == 0, recovery.stderr
    assert (
        'Dependency diagnostics: '
        '{"dependency_never_satisfied_job_ids":[]}'
    ) in recovery.stdout
    assert (
        "Submitted Slurm serial array 41001 with 1 tasks "
        "(array=0, chunk=0)"
    ) in recovery.stdout
    assert "Submitted Slurm parallel" not in recovery.stdout
    assert "Submitted seed finalizer array 41001" in recovery.stdout
    assert "Submitted panel publish array 41001" in recovery.stdout
    assert sbatch_counter.read_text(encoding="utf-8").strip() == "5"
    recovery_receipts = [
        call
        for call in python_log.read_text(encoding="utf-8").splitlines()
        if " receipt " in f" {call} "
    ]
    assert len(recovery_receipts) == 2
    assert "--resource-class serial" in recovery_receipts[-1]
    assert "--master-indices 1" in "\n".join(
        python_log.read_text(encoding="utf-8").splitlines()
    )
    assert " dependency-diagnostics " in (
        f" {' '.join(python_log.read_text(encoding='utf-8').splitlines())} "
    )


def test_finalizer_submission_failure_blocks_panel_publish(tmp_path):
    engine = tmp_path / "engine"
    (engine / "slurm").mkdir(parents=True)
    (engine / "slurm" / "run_nk_grid.sbatch").write_text(
        "#!/bin/bash\n", encoding="utf-8"
    )
    manifest = tmp_path / "panels.yaml"
    manifest.write_text("panels: []\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$3" = "snapshot" ]; then echo 1; exit 0; fi
if [ "$3" = "indices" ]; then
  case "$*" in
    *"--resource-class parallel"*) echo 0 ;;
    *) echo "" ;;
  esac
  exit 0
fi
if [ "$3" = "chunk-map" ]; then echo /synthetic/chunk.json; exit 0; fi
if [ "$3" = "receipt" ]; then echo /synthetic/receipt.json; exit 0; fi
if [ "$3" = "build-map" ]; then echo "/synthetic/$*.json"; exit 0; fi
if [ "$3" = "map-count" ]; then echo 1; exit 0; fi
exit 9
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_SBATCH_LOG"
case "$*" in
  *"--job-name=al-nk-finalize"*)
    echo synthetic-finalizer-failure >&2
    exit 9
    ;;
esac
echo '60001;cluster'
""",
    )
    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
            "--manifest",
            str(manifest),
            "--max-array-size",
            "10",
        ],
        env={
            **os.environ,
            "ENGINE_DIR": str(engine),
            "VENV": str(tmp_path / "venv"),
            "PYTHON": str(fake_python),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Seed arrays submitted but finalizer submission failed." in completed.stderr
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--job-name=al-nk-finalize" in calls[1]
    assert not any("--job-name=al-nk-publish" in call for call in calls)


def test_recovery_with_no_missing_or_incomplete_jobs_submits_nothing(
    tmp_path,
):
    engine = tmp_path / "engine"
    (engine / "slurm").mkdir(parents=True)
    snapshot = tmp_path / "jobs.json"
    snapshot.write_text("{}", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$3" = "count" ]; then echo 1; exit 0; fi
if [ "$3" = "indices" ]; then
  case "$*" in
    *"--resource-class parallel"*) echo 0 ;;
    *) echo "" ;;
  esac
  exit 0
fi
if [ "$3" = "dependency-diagnostics" ]; then echo '{}'; exit 0; fi
if [ "$3" = "recovery-indices" ]; then echo ""; exit 0; fi
exit 9
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_SBATCH_LOG"
echo '70001;cluster'
""",
    )

    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "submit_nk_grid.sh"),
            "--snapshot",
            str(snapshot),
            "--resource-class",
            "parallel",
            "--max-array-size",
            "10",
        ],
        env={
            **os.environ,
            "ENGINE_DIR": str(engine),
            "VENV": str(tmp_path / "venv"),
            "PYTHON": str(fake_python),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert (
        "No missing/incomplete matching resource-class tasks were submitted."
        in completed.stderr
    )
    assert not sbatch_log.exists()


def test_worker_rejects_restart_count_above_limit_before_starting_python(
    tmp_path,
):
    environment = {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "VENV": str(tmp_path / "venv"),
        "PYTHON": str(tmp_path / "python-must-not-run"),
        "SLURM_ARRAY_TASK_ID": "2",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_JOB_ID": "456",
        "SLURM_RESTART_COUNT": "6",
        "MAX_SLURM_RESTARTS": "5",
        "REQUEUE_WATCHDOG_SECONDS": "0",
    }

    completed = subprocess.run(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "run_nk_grid.sbatch"),
            str(tmp_path / "jobs.json"),
            "0",
            "0",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75
    assert (
        "Slurm restart count 6 exceeds configured maximum 5"
        in completed.stderr
    )
    assert "Python not found" not in completed.stderr


@pytest.mark.parametrize(
    ("restart_count", "expected_returncode", "expect_requeue"),
    [("0", 0, True), ("5", 75, False)],
)
def test_worker_usr1_stops_at_checkpoint_boundary_and_requeues_array_element(
    tmp_path,
    restart_count,
    expected_returncode,
    expect_requeue,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ready = tmp_path / "ready"
    scontrol_log = tmp_path / "scontrol.log"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
if [ "$1" = "-c" ]; then
  exit 0
fi
: > "$FAKE_READY"
while [ ! -f "$NK_GRID_STOP_REQUEST_FILE" ]; do
  sleep 0.02
done
exit 0
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf '%s\n' "$*" > "$FAKE_SCONTROL_LOG"
""",
    )
    environment = {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "VENV": str(tmp_path / "venv"),
        "PYTHON": str(fake_python),
        "FAKE_READY": str(ready),
        "FAKE_SCONTROL_LOG": str(scontrol_log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SLURM_ARRAY_TASK_ID": "4",
        "SLURM_ARRAY_JOB_ID": "123",
        "SLURM_JOB_ID": "456",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_RESTART_COUNT": restart_count,
        "MAX_SLURM_RESTARTS": "5",
    }
    process = subprocess.Popen(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "run_nk_grid.sbatch"),
            str(tmp_path / "jobs.json"),
            "0",
            "0",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("worker did not reach the synthetic engine process")
        time.sleep(0.02)
    os.kill(process.pid, signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == expected_returncode, (
        f"stdout={stdout}\nstderr={stderr}"
    )
    if expect_requeue:
        assert scontrol_log.read_text(encoding="utf-8").strip() == "requeue 123_4"
    else:
        assert not scontrol_log.exists()
        assert "Maximum Slurm restart count reached" in stderr
    assert "checkpoint-boundary stop" in stderr


def test_worker_watchdog_zero_forces_requeue_before_delayed_python_exit(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ready = tmp_path / "ready"
    python_finished = tmp_path / "python-finished"
    scontrol_log = tmp_path / "scontrol.log"
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        """#!/bin/bash
if [ "$1" = "-c" ]; then
  exit 0
fi
: > "$FAKE_READY"
while [ ! -f "$NK_GRID_STOP_REQUEST_FILE" ]; do
  sleep 0.02
done
sleep 1
: > "$FAKE_PYTHON_FINISHED"
exit 0
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf '%s\n' "$*" > "$FAKE_SCONTROL_LOG"
if [ -f "$FAKE_PYTHON_FINISHED" ]; then
  echo after-python-exit >> "$FAKE_SCONTROL_LOG"
else
  echo before-python-exit >> "$FAKE_SCONTROL_LOG"
fi
""",
    )
    environment = {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "VENV": str(tmp_path / "venv"),
        "PYTHON": str(fake_python),
        "FAKE_READY": str(ready),
        "FAKE_PYTHON_FINISHED": str(python_finished),
        "FAKE_SCONTROL_LOG": str(scontrol_log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SLURM_ARRAY_TASK_ID": "7",
        "SLURM_ARRAY_JOB_ID": "321",
        "SLURM_JOB_ID": "654",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_RESTART_COUNT": "0",
        "MAX_SLURM_RESTARTS": "5",
        "REQUEUE_WATCHDOG_SECONDS": "0",
    }
    process = subprocess.Popen(
        [
            "bash",
            str(ENGINE_DIR / "slurm" / "run_nk_grid.sbatch"),
            str(tmp_path / "jobs.json"),
            "0",
            "0",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("worker did not reach the delayed synthetic engine process")
        time.sleep(0.02)

    os.kill(process.pid, signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert python_finished.exists()
    assert scontrol_log.read_text(encoding="utf-8").splitlines() == [
        "requeue 321_7",
        "before-python-exit",
    ]
    assert (
        "Checkpoint-boundary stop exceeded 0s; forcing requeue from the last "
        "complete checkpoint"
    ) in stderr
    assert "Requeueing 321_7 (watchdog fallback before wall timeout)" in stderr
    assert "Watchdog already requested requeue" in stderr


def test_panel_declared_large_run_authorization_is_honored(tmp_path):
    """A panel that authorizes its own oversized grid must not need the CLI flag."""

    oversized = _config(
        tmp_path,
        n_seeds=100,
        n_draws=50,
        n_sizes_n=20,
        n_sizes_k=20,
        allow_large_run=True,
    )
    jobs = [SlurmJob(panel="big", model="ols", seed=123, draws=(0,), config=oversized, final_out=tmp_path / "final.csv")]

    # An absent CLI flag arrives as None and must defer to the panel.
    require_large_run_authorization(jobs, allow_large_run=None)

    unauthorized = [
        SlurmJob(
            panel="big",
            model="ols",
            seed=123,
            draws=(0,),
            config=replace(oversized, allow_large_run=False),
            final_out=tmp_path / "final.csv",
        )
    ]
    with pytest.raises(ValueError, match="requires --allow-large-run"):
        require_large_run_authorization(unauthorized, allow_large_run=None)


def test_receipt_separates_cli_and_panel_large_run_authorization(tmp_path):
    """Auditing must distinguish CLI-level from panel-level authorization."""

    jobs = [
        SlurmJob(
            panel="panel-cli",
            model="ols",
            seed=123,
            draws=(0,),
            config=_config(tmp_path, out=tmp_path / "a.csv"),
            final_out=tmp_path / "final-a.csv",
        ),
        SlurmJob(
            panel="panel-self",
            model="ridge",
            seed=123,
            draws=(0,),
            config=_config(
                tmp_path, out=tmp_path / "b.csv", allow_large_run=True
            ),
            final_out=tmp_path / "final-b.csv",
        ),
    ]
    snapshot = tmp_path / "jobs.json"
    _write_snapshot(snapshot, jobs)
    receipt_path = write_submission_receipt(
        snapshot,
        slurm_job_id="4242",
        array_spec="0-1",
        worker_script=ENGINE_DIR / "slurm" / "run_nk_grid.sbatch",
        allow_large_run=False,
        receipt_path=tmp_path / "receipt.json",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["cli_allow_large_run"] is False
    assert "allow_large_run" not in receipt
    by_panel = {entry["panel"]: entry for entry in receipt["jobs"]}
    assert by_panel["panel-cli"]["panel_allow_large_run"] is False
    assert by_panel["panel-cli"]["effective_allow_large_run"] is False
    assert by_panel["panel-self"]["panel_allow_large_run"] is True
    assert by_panel["panel-self"]["effective_allow_large_run"] is True

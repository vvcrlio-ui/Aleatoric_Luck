"""Portable tests of actual local policy functions with explicit I/O boundaries.

AST loading avoids importing POSIX engine dependencies. These tests check
decisions and parameter routing, NOT POSIX locks, durability or full runs.
The companion integration file exercises the ordinary imports on Linux.
"""
from __future__ import annotations

import ast
import argparse
from dataclasses import dataclass, fields, replace
import json
import os
import sys
import shutil
import types
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "aleatoric_nk_grid"


def _functions(filename, names, **bindings):
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = dict(bindings)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SRC / filename), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def _manifest(policy="delete", status="complete", **counts):
    return {"experiment_id": "x", "output": {"checkpoint_retention": policy},
            "completion": {"status": status, "expected_rows": 1,
                           "materialized_rows": 1, "completed_rows": 1,
                           "failed_rows": 0, **counts}}


def _local(tmp_path):
    out = tmp_path / "result.csv"
    out.write_text("final table", encoding="utf-8")
    parts = tmp_path / "result.parts"
    parts.mkdir()
    (parts / "part.csv").write_text("checkpoint", encoding="utf-8")
    verify = Mock()
    def retire(_):
        retired = tmp_path / "retired"
        parts.rename(retired)
        return retired
    module = _functions("nk_grid.py", ["_prune_checkpoint_parts", "_apply_completed_checkpoint_retention"],
        checkpoint_parts_dir=lambda _: parts,
        checkpoint_parts=lambda _: list(parts.glob("*.csv")),
        verify_materialized_checkpoint=verify, retire_checkpoint_parts=Mock(side_effect=retire),
        shutil=shutil, log_progress=Mock())
    return module, out, parts, verify


@pytest.mark.parametrize("keep", [True, False])
def test_writer_preserves_each_checkpoint_only_when_requested(keep):
    publish, compact = Mock(return_value=Path("part.csv")), Mock(return_value=Path("compact.csv"))
    module = _functions("experiment.py", ["write_checkpoint_part"], pd=pd,
        _write_checkpoint_frame_atomic=publish, checkpoint_loose_parts_dir=lambda _: Path("parts"),
        compact_checkpoint_parts=compact)
    for draw in range(101):
        result = module.write_checkpoint_part([{"draw": draw}], Path("out.csv"), keep_all=keep)
        assert result == Path("part.csv" if keep else "compact.csv")
    assert publish.call_count == 101
    assert compact.call_count == (0 if keep else 101)


@pytest.mark.parametrize("policy", ["default", "keep", "delete"])
def test_complete_cleanup_policy_and_final_table_preservation(tmp_path, policy):
    module, out, parts, verify = _local(tmp_path)
    assert module._prune_checkpoint_parts(out, _manifest(policy)) is (policy != "keep")
    assert parts.exists() is (policy == "keep")
    assert out.read_text(encoding="utf-8") == "final table"
    assert verify.call_count == (0 if policy == "keep" else 1)


@pytest.mark.parametrize("change", [{"status": "incomplete"}, {"status": "complete_with_failures"},
    {"failed_rows": 1}, {"completed_rows": 0}, {"materialized_rows": 0}])
def test_unsuccessful_run_never_cleans(tmp_path, change):
    module, out, parts, verify = _local(tmp_path)
    assert not module._prune_checkpoint_parts(out, _manifest(**change))
    assert parts.is_dir()
    verify.assert_not_called()


def test_failed_final_verification_retains_all_shards(tmp_path):
    module, out, parts, verify = _local(tmp_path)
    verify.side_effect = ValueError("bad CSV")
    with pytest.raises(RuntimeError, match="retained"):
        module._prune_checkpoint_parts(out, _manifest())
    assert (parts / "part.csv").is_file()
    module.retire_checkpoint_parts.assert_not_called()


@pytest.mark.parametrize("policy", [None, True, 1, [], "yes", "DELETE"])
def test_invalid_retention_rejected_before_mutation(tmp_path, policy):
    module, out, parts, verify = _local(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_retention"):
        module._prune_checkpoint_parts(out, _manifest(policy))
    assert parts.is_dir()
    verify.assert_not_called()


def test_completed_reuse_applies_explicit_delete_and_cannot_reconstruct_keep(tmp_path):
    module, out, parts, verify = _local(tmp_path)
    manifest = _manifest("keep")
    module._apply_completed_checkpoint_retention(out, manifest, "default")
    assert parts.is_dir() and manifest["output"]["checkpoint_retention"] == "keep"
    module._apply_completed_checkpoint_retention(out, manifest, "delete")
    assert manifest["output"]["checkpoint_parts_deleted"] is True
    assert not parts.exists()
    with pytest.raises(ValueError, match="cannot reconstruct"):
        module._apply_completed_checkpoint_retention(out, manifest, "keep")


def test_engine_batch_flush_routes_resolved_policy():
    tree = ast.parse((SRC / "nk_grid.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "write_checkpoint_part"]
    assert len(calls) == 1
    arg = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "keep_all")
    for policy in ("default", "keep", "delete"):
        actual = eval(compile(ast.Expression(arg), "flush", "eval"), {"config": types.SimpleNamespace(checkpoint_retention=policy)})
        assert actual is (policy == "keep")


@pytest.mark.parametrize("previous", ["default", "keep", "delete"])
@pytest.mark.parametrize("requested", ["default", "keep", "delete"])
def test_resume_without_flag_inherits_previous_explicit_policy(previous, requested):
    @dataclass(frozen=True)
    class Config:
        checkpoint_retention: str
    module = _functions("nk_grid.py", ["_resumed_checkpoint_config"], replace=replace)
    config = module._resumed_checkpoint_config(Config(requested), _manifest(previous))
    assert config.checkpoint_retention == (previous if requested == "default" else requested)


@pytest.mark.parametrize("policy", [None, "keep", "delete"])
def test_actual_config_core_cli_and_dynamic_snapshot_codec(monkeypatch, policy):
    module = _functions("nk_grid.py", ["NKGridConfig", "parse_args"],
        __name__=__name__, dataclass=dataclass, argparse=argparse, Path=Path, os=os,
        SUPPORTED_MODEL_NAMES=("ols",),
        DEFAULT_MODEL_PARAMS_PATH=SRC.parents[1] / "model_params.yaml")
    argv = ["nk-grid", "--schema", "input.json", "--out", "out.csv", "--outcome", "y", "--models", "ols"]
    if policy is not None:
        argv += ["--checkpoints", policy]
    monkeypatch.setattr(sys, "argv", argv)
    config = module.parse_args()
    assert config.checkpoint_retention == (policy or "default")
    codec = _functions("flat_task_table.py", ["_config_to_json", "_config_from_json"],
        fields=fields, Path=Path, NKGridConfig=module.NKGridConfig)
    payload = json.loads(json.dumps(codec._config_to_json(config)))
    assert codec._config_from_json(payload).checkpoint_retention == (policy or "default")
    del payload["checkpoint_retention"]
    assert codec._config_from_json(payload).checkpoint_retention == "default"


def _selected_output_reuse_condition():
    """Load the actual completion-route condition, without importing fcntl."""
    tree = ast.parse((SRC / "nk_grid.py").read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "_run_nk_grid_locked")
    matches = [node for node in run.body if isinstance(node, ast.If) and any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        and child.func.id == "_apply_completed_checkpoint_retention"
        for child in ast.walk(node))]
    assert len(matches) == 1
    return compile(ast.Expression(matches[0].test), "selected-output-reuse", "eval")


def _should_reuse_selected_output(out, *, rerun_completed, complete):
    return eval(_selected_output_reuse_condition(), {
        "config": types.SimpleNamespace(rerun_completed=rerun_completed),
        "out_path": out, "metadata": {"experiment_id": "x"}, "expected_rows": 1,
        "jobs": [("ols", 1, 0, 10, 1)], "completed_statuses": {},
        "_verified_complete_artifacts": lambda *_: complete,
        "_prediction_export_is_complete": lambda *_: True,
    })


@pytest.mark.parametrize("rerun_completed", [True, False])
def test_completed_direct_output_cannot_bypass_keep_after_delete(tmp_path, rerun_completed):
    module, out, parts, verify = _local(tmp_path)
    manifest = _manifest("keep")
    module._apply_completed_checkpoint_retention(out, manifest, "delete")
    assert not parts.exists()
    with pytest.raises(ValueError, match="cannot reconstruct"):
        if _should_reuse_selected_output(out, rerun_completed=rerun_completed, complete=True):
            module._apply_completed_checkpoint_retention(out, manifest, "keep")


def test_real_output_selector_new_timestamp_does_not_apply_old_output_retention(tmp_path):
    old = tmp_path / "result_x_20000101-000000.csv"
    old.write_text("previous complete output", encoding="utf-8")
    fresh = tmp_path / "result_x_20990101-000000.csv"
    jobs = [("ols", 1, 0, 10, 1)]
    module = _functions("nk_grid.py", ["_select_output_path"], json=json,
        _identity_path_segment=lambda value: value,
        manifest_path=lambda path: path.with_suffix(".manifest.json"),
        load_checkpoint_index=lambda _: pd.DataFrame([{"experiment_id": "x"}]),
        rows_for_experiment=lambda frame, _: frame,
        _completed_jobs_for_experiment=lambda *_: set(jobs),
        _timestamped_out_path=Mock(return_value=fresh), log_progress=Mock())
    selected = module._select_output_path(tmp_path / "result.csv", preset="pilot",
        experiment_id="x", jobs=jobs, rerun_completed=True)
    assert selected == fresh
    assert not _should_reuse_selected_output(selected, rerun_completed=True, complete=selected == old)
    assert old.read_text(encoding="utf-8") == "previous complete output"

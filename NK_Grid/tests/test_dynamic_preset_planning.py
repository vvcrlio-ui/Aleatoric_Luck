from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aleatoric_nk_grid import run_panels
from aleatoric_nk_grid.chunk_planning import (
    DYNAMIC_PRESETS,
    _cluster_from_payload,
    build_dynamic_plan,
    main as planning_main,
    request_from_preset,
)
from aleatoric_nk_grid.flat_task_table import (
    _config_from_json,
    close_generation,
    execution_groups,
    finalize_snapshot,
    prepare_round,
    run_slice,
    verify_rounds,
)


ENGINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_DIR.parent
MANIFEST = REPO_ROOT / "FFCWS" / "panels.yaml"
PANEL = "ffc_median_mode_gpa"
SUBMITTER = ENGINE_DIR / "slurm" / "submit_flat_task_table.sh"


def _tree_state(root: Path) -> tuple[tuple[str, int, bytes | None], ...]:
    entries = []
    for path in sorted(root.rglob("*")):
        entries.append((
            str(path.relative_to(root)),
            path.stat().st_ino,
            path.read_bytes() if path.is_file() else None,
        ))
    return tuple(entries)


def _payload(root: Path, *, preset: str = "dev-dynamic", constraint: str = "none") -> dict[str, object]:
    return request_from_preset(
        MANIFEST, panel=PANEL, preset=preset, root=root,
        account="project", partition="short", constraint=constraint,
        models=("ols", "lightgbm"),
    )


def _legacy_plan(payload: dict[str, object]) -> dict[str, object]:
    """The pre-preset CLI path, retained here as the byte-compatibility oracle."""

    return build_dynamic_plan(
        _config_from_json(payload["config"]),
        n_grid=[int(value) for value in payload["n_grid"]],
        k_grid=[int(value) for value in payload["k_grid"]],
        cluster=_cluster_from_payload(payload["cluster"]),
        table_path=payload["task_table"], snapshot_path=payload["snapshot"],
        output_dir=payload["output_dir"], panel=str(payload["panel"]),
    )


def _plan_from_cli(tmp_path: Path, *, preset: str = "dev-dynamic", constraint: str = "none") -> tuple[Path, dict[str, object]]:
    root = tmp_path / "dynamic"
    plan_path = root / "plan.json"
    planning_main([
        "--manifest", str(MANIFEST), "--panel", PANEL, "--preset", preset,
        "--account", "project", "--partition", "short", "--constraint", constraint,
        "--root", str(root), "--models", "ols,lightgbm",
    ])
    return root, json.loads(plan_path.read_text(encoding="utf-8"))


def test_request_cli_remains_byte_equivalent_to_the_legacy_payload_path(tmp_path, capsys):
    root = tmp_path / "run"
    payload = _payload(root)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = _legacy_plan(payload)
    expected_bytes = json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    expected_stdout = json.dumps(expected["memory"], sort_keys=True) + "\n"
    shutil.rmtree(root)

    plan_path = tmp_path / "plan.json"
    planning_main(["--request", str(request), "--plan-out", str(plan_path)])

    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == hashlib.sha256(expected_bytes).hexdigest()
    assert hashlib.sha256(capsys.readouterr().out.encode("utf-8")).hexdigest() == hashlib.sha256(expected_stdout.encode("utf-8")).hexdigest()


def test_manifest_entry_requires_every_site_value_and_is_exclusive(tmp_path):
    common = ["--manifest", str(MANIFEST), "--panel", PANEL, "--preset", "dev-dynamic"]
    for flag in ("--account", "--partition", "--constraint", "--root"):
        args = [*common, "--account", "project", "--partition", "short", "--constraint", "none", "--root", str(tmp_path / "root")]
        del args[args.index(flag):args.index(flag) + 2]
        with pytest.raises(SystemExit) as exc_info:
            planning_main(args)
        assert exc_info.value.code == 2
    with pytest.raises(SystemExit) as exc_info:
        planning_main(["--request", str(tmp_path / "request.json"), "--manifest", str(MANIFEST)])
    assert exc_info.value.code == 2
    with pytest.raises(SystemExit) as exc_info:
        planning_main([])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("preset, expected_rows, expected_workers, expected_rounds", [
    ("pilot", 1008, 32, 2),
    ("dev-dynamic", 6, 2, 2),
])
def test_dynamic_presets_produce_their_declared_task_grid(
    tmp_path, preset, expected_rows, expected_workers, expected_rounds,
):
    root, plan = _plan_from_cli(tmp_path, preset=preset)
    payload = _payload(root=tmp_path / "math", preset=preset)
    config = payload["config"]
    assert isinstance(config, dict)
    # The assertion derives the row count from the resolved panel repeats,
    # dynamic grids, and generic execution groups; it does not encode input shape.
    groups = len(execution_groups(config["models"], k_features=int(payload["k_grid"][0])))
    expected_by_math = (
        int(config["n_seeds"])
        * len(payload["n_grid"])
        * len(payload["k_grid"])
        * groups
    )
    assert plan["row_count"] == expected_by_math == expected_rows
    assert plan["workers"] == expected_workers
    assert plan["rounds"] == expected_rounds
    assert root.joinpath("tasks.parquet").is_file()


def test_unknown_preset_panel_and_preset_layer_mismatch_list_available_names(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="available dynamic presets: dev-dynamic, pilot"):
        _payload(tmp_path / "unknown", preset="missing")
    with pytest.raises(ValueError, match="available panels: .*ffc_median_mode_gpa"):
        request_from_preset(
            MANIFEST, panel="missing", preset="dev-dynamic", root=tmp_path / "panel",
            account="project", partition="short", constraint="none",
        )
    monkeypatch.setitem(run_panels.PRESETS, "static-only", {})
    with pytest.raises(ValueError, match="available dynamic presets: dev-dynamic, pilot"):
        _payload(tmp_path / "static-only", preset="static-only")
    monkeypatch.setitem(DYNAMIC_PRESETS, "dynamic-only", {
        "n_grid": [1], "k_grid": [1], "workers": 1, "rounds": 1, "time_limit": "00:01:00",
    })
    with pytest.raises(ValueError, match="available dynamic presets: dev-dynamic, pilot"):
        _payload(tmp_path / "dynamic-only", preset="dynamic-only")


@pytest.mark.parametrize("occupied", ["snapshot.json", "tasks.parquet", "out"])
def test_reused_dynamic_root_is_rejected_without_any_write(tmp_path, occupied):
    root = tmp_path / "existing"
    root.mkdir()
    target = root / occupied
    if occupied == "out":
        target.mkdir(); (target / "marker").write_bytes(b"keep")
    else:
        target.write_bytes(b"keep")
    before = _tree_state(root)

    with pytest.raises(ValueError, match="new empty directory"):
        _payload(root)

    assert _tree_state(root) == before


@pytest.mark.parametrize("constraint", ["none", "skl-compat"])
def test_dynamic_plan_dry_run_preserves_constraint_semantics(tmp_path, constraint):
    root, plan = _plan_from_cli(tmp_path, constraint=constraint)
    completed = subprocess.run(
        ["bash", str(SUBMITTER), str(root / "plan.json")], cwd=tmp_path,
        env={**os.environ, "ENGINE_DIR": str(ENGINE_DIR), "PYTHON": sys.executable},
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 8
    assert "--dependency=afterany:dry-prep-1" in lines[1]
    assert "--dependency=afterany:dry-work-1" in lines[2]
    assert "--dependency=afterany:dry-close-1" in lines[3]
    assert "--dependency=afterok:dry-verify" in lines[-1]
    all_args = [
        *plan["submission"]["sbatch_args"],
        *plan["preparation"]["sbatch_args"],
        *plan["verification"]["sbatch_args"],
        *plan["finalization"]["sbatch_args"],
    ]
    if constraint == "none":
        assert "--constraint" not in completed.stdout
        assert not any(argument.startswith("--constraint=") for argument in all_args)
    else:
        assert all(f"--constraint={constraint}" in arguments for arguments in (
            plan["submission"]["sbatch_args"], plan["preparation"]["sbatch_args"],
            plan["verification"]["sbatch_args"], plan["finalization"]["sbatch_args"],
        ))


@pytest.mark.slow
def test_dev_dynamic_preset_runs_the_complete_local_queue(tmp_path):
    root, plan = _plan_from_cli(tmp_path)
    snapshot = root / "snapshot.json"
    prepared = prepare_round(
        snapshot, round_index=1, prep_token="job-1", prep_job_id="job-1",
        submission_generation="g1", expected_pointer_version=0,
    )
    assert prepared["todo_rows"] == plan["row_count"]
    for worker_index in range(int(plan["workers"])):
        run_slice(
            snapshot, round_index=1, worker_index=worker_index, expected_prep_token="job-1",
            prep_job_id="job-1", submission_generation="g1", expected_pointer_version=0,
        )
    close_generation(
        snapshot, round_index=1, submission_generation="g1",
        expected_prep_token="job-1", prep_job_id="job-1", expected_pointer_version=0,
    )
    no_work = prepare_round(
        snapshot, round_index=2, prep_token="job-2", prep_job_id="job-2",
        submission_generation="g2", expected_previous_generation="g1", expected_pointer_version=1,
    )
    assert no_work["no_generation"] is True
    verification = verify_rounds(
        snapshot, round_index=2, submission_generation="g2", expected_prep_token="job-2",
        prep_job_id="job-2", expected_previous_generation="g1", expected_pointer_version=1,
    )
    assert verification["exit_code"] == 0
    finalized = finalize_snapshot(
        snapshot, round_index=2, submission_generation="g2", expected_prep_token="job-2",
        prep_job_id="job-2", expected_previous_generation="g1", expected_pointer_version=1,
    )
    assert finalized["final_rows"] == plan["model_row_count"] == 6
    assert root.joinpath("final.csv").is_file()

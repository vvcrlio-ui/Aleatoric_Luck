from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parents[1]
SUBMITTER = ENGINE_DIR / "slurm" / "submit_flat_task_table.sh"
WORKER = ENGINE_DIR / "slurm" / "run_flat_task_table.sbatch"
PREP = ENGINE_DIR / "slurm" / "prep_dynamic_queue.sbatch"
VERIFY = ENGINE_DIR / "slurm" / "verify_dynamic_queue.sbatch"
CALIBRATE = ENGINE_DIR / "slurm" / "calibrate.sbatch"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _plan(path: Path) -> Path:
    path.write_text(json.dumps({
        "snapshot": "/frozen/snapshot.json", "workers": 3, "rounds": 2,
        "submission": {"array": "0-2%3", "sbatch_args": [
            "--partition=long", "--cpus-per-task=1", "--mem=8G", "--time=12:00:00",
            "--account=test-account", "--constraint=test-arch",
        ], "account": "test-account", "constraint": "test-arch"},
    }), encoding="utf-8")
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(parents=True)
    _write_executable(fake_bin / "sbatch", "#!/bin/bash\n[ -d logs ] || { echo 'logs directory missing before sbatch' >&2; exit 87; }\nprintf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\necho 12345\n")
    return {**os.environ, "ENGINE_DIR": str(ENGINE_DIR), "PYTHON": sys.executable, "FAKE_SBATCH_LOG": str(tmp_path / "sbatch.log"), "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}


def test_dynamic_submitter_prints_the_entire_afterany_chain(tmp_path):
    completed = subprocess.run(["bash", str(SUBMITTER), str(_plan(tmp_path / "plan.json"))], env=_environment(tmp_path), check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 5  # prep/work for each round, then verify
    assert all("--cpus-per-task=1" in line for line in lines)
    assert "--dependency=afterany:dry-prep-1" in lines[1]
    assert "--dependency=afterany:dry-work-1" in lines[2]
    assert "--dependency=afterany:dry-work-2" in lines[-1]
    assert not any("afterok" in line for line in lines)


def test_dynamic_submitter_submits_every_link_and_writes_receipt(tmp_path):
    environment = _environment(tmp_path); plan = _plan(tmp_path / "plan.json")
    completed = subprocess.run(["bash", str(SUBMITTER), "--submit", str(plan)], cwd=tmp_path, env=environment, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    calls = (tmp_path / "sbatch.log").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5
    assert all("--parsable" in call for call in calls)
    assert all("afterany" in call for call in calls[1:])
    receipt = next(tmp_path.glob("plan.submission-receipt-*.json"))
    receipt_payload = json.loads(receipt.read_text())
    assert [entry["label"] for entry in receipt_payload["jobs"]] == ["prep-1", "work-1", "prep-2", "work-2", "verify"]
    assert receipt_payload["sbatch_account"] == "test-account"
    assert receipt_payload["sbatch_constraint"] == "test-arch"


def test_dynamic_submitter_rejects_non_single_core_plan_before_sbatch(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    payload = json.loads(plan.read_text()); payload["submission"]["sbatch_args"][1] = "--cpus-per-task=8"; plan.write_text(json.dumps(payload))
    environment = _environment(tmp_path)
    completed = subprocess.run(["bash", str(SUBMITTER), str(plan)], env=environment, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "one CPU" in completed.stderr
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_dynamic_submitter_rejects_missing_plan_fields_before_submission(tmp_path):
    required_paths = (
        ("snapshot",),
        ("workers",),
        ("rounds",),
        ("submission", "array"),
        ("submission", "sbatch_args"),
        ("submission", "account"),
        ("submission", "constraint"),
    )
    for field_path in required_paths:
        case_dir = tmp_path.joinpath(*field_path)
        case_dir.mkdir(parents=True)
        plan = _plan(case_dir / "plan.json")
        payload = json.loads(plan.read_text(encoding="utf-8"))
        owner = payload
        for field in field_path[:-1]:
            owner = owner[field]
        del owner[field_path[-1]]
        plan.write_text(json.dumps(payload), encoding="utf-8")
        environment = _environment(case_dir)
        completed = subprocess.run(
            ["bash", str(SUBMITTER), str(plan)], env=environment,
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode != 0
        assert "invalid dynamic plan JSON" in completed.stderr
        assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def _worker_environment(tmp_path: Path, python: Path) -> dict[str, str]:
    return {**os.environ, "ENGINE_DIR": str(ENGINE_DIR), "PYTHON": str(python), "VENV": str(tmp_path / "venv"), "SLURM_ARRAY_TASK_ID": "7"}


def test_dynamic_worker_uses_venv_python_and_skips_module_when_unset(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/bin/bash\nif [ \"$1\" = \"-c\" ]; then exit 0; fi\nprintf '%s\\n' \"$*\" > \"$WORKER_ARGS\"\n")
    environment = _worker_environment(tmp_path, fake_python); environment["WORKER_ARGS"] = str(tmp_path / "args.txt")
    completed = subprocess.run(["bash", str(WORKER), "/frozen/snapshot.json", "3"], env=environment, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert f"python={fake_python}" in completed.stdout
    assert "--round 3 --worker-index 7" in (tmp_path / "args.txt").read_text()


def test_dynamic_worker_reports_architecture_without_mislabeling_install(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/bin/bash\necho 'Illegal instruction' >&2\nexit 132\n")
    environment = _worker_environment(tmp_path, fake_python); environment.update({"BMRC_GCC_ARCH_NATIVE": "native", "MODULE_CPU_TYPE": "cpu"})
    completed = subprocess.run(["bash", str(WORKER), "/frozen/snapshot.json", "1"], env=environment, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "CPU architecture" in completed.stderr
    assert "not installed" not in completed.stderr


def test_all_compute_scripts_share_the_module_and_architecture_guards():
    module_block = '''if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi'''
    block_md5s = set()
    for script in (WORKER, PREP, VERIFY):
        text = script.read_text(encoding="utf-8")
        assert text.count(module_block) == 1
        start = text.index(module_block)
        actual_block = text[start:start + len(module_block)]
        block_md5s.add(hashlib.md5(actual_block.encode("utf-8")).hexdigest())
        assert 'venv is incompatible with this node CPU architecture' in text
        assert 'BMRC_GCC_ARCH_NATIVE=${BMRC_GCC_ARCH_NATIVE:-?}' in text
        assert 'grep -qi "Illegal instruction"' not in text
        assert "python=$PYTHON" in text
    assert len(block_md5s) == 1


def test_calibrate_script_is_generic_and_uses_the_shared_module_block():
    calibrate = CALIBRATE.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    start = worker.index('if [ -n "${PYTHON_MODULE:-}" ]; then')
    end = worker.index("\nfi", start) + len("\nfi")
    assert worker[start:end] in calibrate
    assert not any(token in calibrate.lower() for token in ("smr", "5970", "497"))
    assert "#SBATCH --cpus-per-task=8" in calibrate
    assert "#SBATCH --mem=48G" in calibrate
    assert "#SBATCH --time=04:00:00" in calibrate
    assert '--memory-cells "$CALIBRATION_MEMORY_CELLS"' in calibrate
    assert not any(token in calibrate for token in ("--stage-a", "--stage-b"))


def _calibrate_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_python = tmp_path / "python"
    args_path = tmp_path / "calibrate-args.txt"
    _write_executable(fake_python, '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$CALIBRATE_ARGS"\n')
    environment = {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "PYTHON": str(fake_python),
        "VENV": str(tmp_path / "venv"),
        "CALIBRATION_JOB_NAME": "test-calibration",
        "SBATCH_ACCOUNT": "test-account",
        "SBATCH_CONSTRAINT": "none",
        "CALIBRATION_SHAPE_SCHEMA": str(tmp_path / "schema.yaml"),
        "CALIBRATION_N_TRAIN": "100",
        "CALIBRATION_MEMORY_CELLS": "ols:10:5",
        "CALIBRATION_OUT_DIR": str(tmp_path / "output"),
        "CALIBRATION_ASSUME_FEATURE_DTYPE": "float64",
        "CALIBRATE_ARGS": str(args_path),
    }
    environment.pop("PYTHON_MODULE", None)
    environment.pop("CALIBRATION_TASK_MEMORY_CELLS", None)
    return environment, args_path


def test_calibrate_script_omits_optional_task_memory_cells_when_unset(tmp_path):
    environment, args_path = _calibrate_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(CALIBRATE)], env=environment,
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--task-memory-cells" not in args_path.read_text(encoding="utf-8").splitlines()


def test_calibrate_script_passes_optional_task_memory_cells_unchanged(tmp_path):
    environment, args_path = _calibrate_environment(tmp_path)
    task_cells = ",".join(f"ols:{n}:5" for n in range(10, 18))
    environment["CALIBRATION_TASK_MEMORY_CELLS"] = task_cells
    completed = subprocess.run(
        ["bash", str(CALIBRATE)], env=environment,
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    args = args_path.read_text(encoding="utf-8").splitlines()
    option_index = args.index("--task-memory-cells")
    assert args[option_index + 1] == task_cells

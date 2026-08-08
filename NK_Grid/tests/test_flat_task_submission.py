from __future__ import annotations

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


def test_worker_module_block_is_local_development_safe_and_illegal_instruction_grep_is_absent():
    text = WORKER.read_text(encoding="utf-8")
    assert 'if [ -n "${PYTHON_MODULE:-}" ]; then' in text
    assert 'PYTHON_MODULE is set but the module command is unavailable' in text
    assert 'grep -qi "Illegal instruction"' not in text
    assert "python=$PYTHON" in text

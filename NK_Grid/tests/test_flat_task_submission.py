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
CALIBRATE = ENGINE_DIR / "slurm" / "calibrate.sbatch"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _plan(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "snapshot": "/frozen/snapshot.json",
                "submissions": [
                    {
                        "resource_class": "serial",
                        "array": "0-1%1",
                        "sbatch_args": [
                            "--partition=long",
                            "--cpus-per-task=1",
                            "--mem=8G",
                            "--time=01:00:00",
                        ],
                    },
                    {
                        "resource_class": "super_learner",
                        "array": "2%1",
                        "sbatch_args": [
                            "--partition=long",
                            "--cpus-per-task=8",
                            "--mem=32G",
                            "--time=02:00:00",
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    _write_executable(
        fake_bin / "sbatch",
        "#!/bin/bash\n[ -d logs ] || { echo 'logs directory missing before sbatch' >&2; exit 87; }\nprintf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\necho \"Submitted batch job 12345\"\n",
    )
    return {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "PYTHON": sys.executable,
        "SBATCH_ACCOUNT": "test-account",
        "SBATCH_CONSTRAINT": "test-arch",
        "FAKE_SBATCH_LOG": str(tmp_path / "sbatch.log"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }


def test_flat_task_submitter_dry_run_is_exact_and_never_calls_sbatch(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    environment = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SUBMITTER), str(plan)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    quoted_worker = str(WORKER).replace(" ", "\\ ")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "DRY RUN: sbatch --partition=long --cpus-per-task=1 --mem=8G --time=01:00:00 "
        "--account=test-account --constraint=test-arch "
        f"--array=0-1%1 {quoted_worker} /frozen/snapshot.json",
        "DRY RUN: sbatch --partition=long --cpus-per-task=8 --mem=32G --time=02:00:00 "
        "--account=test-account --constraint=test-arch "
        f"--array=2%1 {quoted_worker} /frozen/snapshot.json",
    ]
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_flat_task_submitter_rejects_missing_plan_fields_before_submission(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    payload = json.loads(plan.read_text(encoding="utf-8"))
    del payload["submissions"][0]["array"]
    plan.write_text(json.dumps(payload), encoding="utf-8")
    environment = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SUBMITTER), str(plan)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "submissions[0] missing non-empty array" in completed.stderr
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_flat_task_submitter_rejects_an_array_over_the_explicit_cluster_limit(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    environment = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SUBMITTER), "--submit", "--max-array-size", "1", str(plan)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "exceeding MaxArraySize 1" in completed.stderr
    assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_flat_task_submitter_requires_submit_and_writes_a_receipt(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    environment = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SUBMITTER), "--submit", "--max-array-size", "10", str(plan)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt_paths = list(tmp_path.glob("plan.submission-receipt-*.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    assert receipt["plan"] == str(plan.resolve())
    assert receipt["snapshot"] == "/frozen/snapshot.json"
    assert receipt["submitted_at"].endswith("Z")
    assert [entry["slurm_job_id"] for entry in receipt["jobs"]] == ["12345", "12345"]
    assert [entry["array"] for entry in receipt["jobs"]] == ["0-1%1", "2%1"]
    assert [entry["sbatch_account"] for entry in receipt["jobs"]] == ["test-account", "test-account"]
    assert [entry["sbatch_constraint"] for entry in receipt["jobs"]] == ["test-arch", "test-arch"]
    sbatch_calls = Path(environment["FAKE_SBATCH_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(sbatch_calls) == 2
    assert all("run_flat_task_table.sbatch /frozen/snapshot.json" in call for call in sbatch_calls)


def test_flat_task_submitter_requires_an_explicit_account_and_constraint(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    for missing in ("SBATCH_ACCOUNT", "SBATCH_CONSTRAINT"):
        environment = _environment(tmp_path / missing)
        del environment[missing]
        completed = subprocess.run(
            ["bash", str(SUBMITTER), str(plan)],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert missing in completed.stderr
        assert not Path(environment["FAKE_SBATCH_LOG"]).exists()


def test_flat_task_submitter_allows_an_explicit_constraint_opt_out(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    environment = _environment(tmp_path)
    environment["SBATCH_CONSTRAINT"] = "none"

    completed = subprocess.run(
        ["bash", str(SUBMITTER), str(plan)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--account=test-account" in completed.stdout
    assert "--constraint=" not in completed.stdout


def test_flat_task_submitter_creates_logs_before_submission_and_is_idempotent(tmp_path):
    plan = _plan(tmp_path / "plan.json")
    environment = _environment(tmp_path)

    for _ in range(2):
        completed = subprocess.run(
            ["bash", str(SUBMITTER), "--submit", "--max-array-size", "10", str(plan)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "logs").is_dir()


def _worker_environment(tmp_path: Path, python: Path) -> dict[str, str]:
    return {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "PYTHON": str(python),
        "VENV": str(tmp_path / "venv"),
        "SLURM_ARRAY_TASK_ID": "7",
    }


def test_flat_worker_skips_module_handling_when_module_is_unset(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(
        fake_python,
        "#!/bin/bash\nif [ \"$1\" = \"-c\" ]; then exit 0; fi\nprintf '%s\\n' \"$*\" > \"$WORKER_ARGS\"\n",
    )
    environment = _worker_environment(tmp_path, fake_python)
    environment["WORKER_ARGS"] = str(tmp_path / "worker-args.txt")

    completed = subprocess.run(
        ["bash", str(WORKER), "/frozen/snapshot.json"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "chunk=7" in completed.stdout
    assert "-m aleatoric_nk_grid.flat_task_table run" in (tmp_path / "worker-args.txt").read_text(encoding="utf-8")


def test_flat_worker_rejects_a_module_request_when_module_is_unavailable(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/bin/bash\nexit 0\n")
    environment = _worker_environment(tmp_path, fake_python)
    environment["PYTHON_MODULE"] = "a-module-that-is-not-present"

    completed = subprocess.run(
        ["bash", str(WORKER), "/frozen/snapshot.json"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "PYTHON_MODULE is set but the module command is unavailable" in completed.stderr


def test_flat_worker_reports_cpu_architecture_incompatibility(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/bin/bash\necho 'Illegal instruction' >&2\nexit 132\n")
    environment = _worker_environment(tmp_path, fake_python)
    environment.update({
        "BMRC_GCC_ARCH_NATIVE": "test-native",
        "MODULE_CPU_TYPE": "test-cpu",
        "VENV": "/venvs/test-native",
    })

    completed = subprocess.run(
        ["bash", str(WORKER), "/frozen/snapshot.json"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "CPU architecture" in completed.stderr
    assert "BMRC_GCC_ARCH_NATIVE=test-native" in completed.stderr
    assert "MODULE_CPU_TYPE=test-cpu" in completed.stderr
    assert "VENV=/venvs/test-native" in completed.stderr
    assert "aleatoric_nk_grid is not installed" not in completed.stderr


def test_calibrate_script_is_generic_and_uses_the_shared_module_block():
    worker_text = WORKER.read_text(encoding="utf-8")
    calibrate_text = CALIBRATE.read_text(encoding="utf-8")
    module_block = '''if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi'''
    assert module_block in worker_text
    assert module_block in calibrate_text
    assert not any(token in calibrate_text.lower() for token in ("smr", "5970", "497"))

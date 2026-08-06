from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parents[1]
SUBMITTER = ENGINE_DIR / "slurm" / "submit_flat_task_table.sh"


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
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sbatch",
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\necho \"Submitted batch job 12345\"\n",
    )
    return {
        **os.environ,
        "ENGINE_DIR": str(ENGINE_DIR),
        "PYTHON": sys.executable,
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

    worker = ENGINE_DIR / "slurm" / "run_flat_task_table.sbatch"
    quoted_worker = str(worker).replace(" ", "\\ ")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "DRY RUN: sbatch --partition=long --cpus-per-task=1 --mem=8G --time=01:00:00 "
        f"--array=0-1%1 {quoted_worker} /frozen/snapshot.json",
        "DRY RUN: sbatch --partition=long --cpus-per-task=8 --mem=32G --time=02:00:00 "
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
    sbatch_calls = Path(environment["FAKE_SBATCH_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(sbatch_calls) == 2
    assert all("run_flat_task_table.sbatch /frozen/snapshot.json" in call for call in sbatch_calls)

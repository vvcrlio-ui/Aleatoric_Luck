from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from aleatoric_nk_grid.dynamic_exit_monitor import classify_dynamic_exit, main


@pytest.mark.parametrize(
    "exit_code,classification,retry,alarm,terminal",
    [
        (0, "success", False, False, True),
        (3, "verification-incomplete", False, False, True),
        (6, "protocol-or-corruption", False, True, True),
        (7, "busy-or-recovery-required", True, False, False),
        (8, "superseded", False, False, True),
        (9, "unexpected-exit", False, True, True),
    ],
)
def test_monitor_retries_only_code_7(
    exit_code: int, classification: str, retry: bool, alarm: bool,
    terminal: bool,
):
    decision = classify_dynamic_exit(exit_code)
    assert decision.classification == classification
    assert decision.retry is retry
    assert decision.production_integrity_alarm is alarm
    assert decision.terminal is terminal


def test_code_8_is_structured_superseded_without_retry_or_alarm(capsys):
    main(["--phase", "close", "8"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "classification": "superseded",
        "exit_code": 8,
        "phase": "close",
        "production_integrity_alarm": False,
        "retry": False,
        "terminal": True,
    }


@pytest.mark.parametrize(
    "exit_code,classification",
    [
        (0, "success"),
        (3, "verification-incomplete"),
        (6, "protocol-or-corruption"),
        (7, "busy-or-recovery-required"),
        (8, "superseded"),
    ],
)
def test_shell_monitor_uses_the_same_exit_protocol(exit_code: int, classification: str):
    script = Path(__file__).resolve().parents[1] / "slurm" / "monitor_flat_task_exit.sh"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [str(script), "verify", str(exit_code)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["exit_code"] == exit_code
    assert payload["classification"] == classification
    assert payload["retry"] is (exit_code == 7)
    assert payload["production_integrity_alarm"] is (exit_code == 6)

from __future__ import annotations

import json

import pytest

from aleatoric_nk_grid.dynamic_exit_monitor import classify_dynamic_exit, main


@pytest.mark.parametrize(
    "exit_code,classification,retry,alarm,terminal",
    [
        (0, "success", False, False, True),
        (6, "protocol-or-corruption", False, True, True),
        (7, "busy-or-recovery-required", True, False, False),
        (8, "superseded", False, False, True),
        (3, "unexpected-exit", False, True, True),
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

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid.model_registry import load_model_params
from aleatoric_nk_grid.native_process import (
    IsolatedProcessRunner,
    NativeProcessCrashed,
    NativeProcessRemoteError,
    NativeProcessTimedOut,
)
from aleatoric_nk_grid.nk_grid import _fit_predict_model_cell


MODEL_PARAMS = Path(__file__).resolve().parents[1] / "model_params.yaml"


def _terminate_child_abruptly() -> None:
    os._exit(23)


def _sleep_for(seconds: float) -> None:
    time.sleep(seconds)


def _raise_value_error() -> None:
    raise ValueError("expected child exception")


def _raise_system_exit() -> None:
    raise SystemExit("child requested exit")


def _start_blocking_grandchild(pid_file: str) -> None:
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import os,sys,time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)", pid_file]
    )
    while grandchild.poll() is None:
        time.sleep(0.05)


def test_native_process_crash_is_retried_and_does_not_kill_parent():
    attempts: list[int] = []
    parent_pid = os.getpid()
    with IsolatedProcessRunner(max_attempts=2) as runner:
        try:
            with pytest.raises(NativeProcessCrashed, match="2 time"):
                runner.run(
                    _terminate_child_abruptly,
                    on_native_crash=lambda attempt, _error: attempts.append(attempt),
                )
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"process pools unavailable in this sandbox: {exc}")
        recovered_child_pid = runner.run(os.getpid)

    assert attempts == [1, 2]
    assert recovered_child_pid != parent_pid


def test_native_process_timeout_kills_worker_and_parent_can_continue():
    attempts: list[int] = []
    parent_pid = os.getpid()
    with IsolatedProcessRunner(
        max_attempts=1,
        timeout_seconds=5.0,
        shutdown_grace_seconds=0.5,
    ) as runner:
        try:
            first_child_pid = runner.run(os.getpid)
            runner.timeout_seconds = 0.1
            with pytest.raises(NativeProcessTimedOut, match="timed out 1 time"):
                runner.run(
                    _sleep_for,
                    60.0,
                    on_native_timeout=lambda attempt, _error: attempts.append(attempt),
                )
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")
        runner.timeout_seconds = 5.0
        recovered_child_pid = runner.run(os.getpid)

    assert attempts == [1]
    assert recovered_child_pid != first_child_pid
    assert recovered_child_pid != parent_pid


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are unavailable")
def test_timeout_kills_native_worker_descendants_and_runner_recovers(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    with IsolatedProcessRunner(max_attempts=1, timeout_seconds=1.0, shutdown_grace_seconds=0.2) as runner:
        try:
            with pytest.raises(NativeProcessTimedOut):
                runner.run(_start_blocking_grandchild, str(pid_file))
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")
        deadline = time.monotonic() + 3.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(grandchild_pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if state.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("timeout leaked a native worker grandchild")
        runner.timeout_seconds = 5.0
        assert runner.run(os.getpid) != os.getpid()


def test_python_exception_crosses_boundary_without_destroying_worker():
    with IsolatedProcessRunner(max_attempts=1, timeout_seconds=5.0) as runner:
        try:
            child_pid = runner.run(os.getpid)
            with pytest.raises(ValueError, match="expected child exception"):
                runner.run(_raise_value_error)
            assert runner.run(os.getpid) == child_pid
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")


def test_child_base_exception_cannot_exit_parent():
    parent_pid = os.getpid()
    with IsolatedProcessRunner(max_attempts=1, timeout_seconds=5.0) as runner:
        try:
            child_pid = runner.run(os.getpid)
            with pytest.raises(NativeProcessRemoteError, match="SystemExit"):
                runner.run(_raise_system_exit)
            assert runner.run(os.getpid) == child_pid
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")

    assert os.getpid() == parent_pid


def test_lightgbm_fit_and_prediction_cross_the_isolated_process_boundary():
    pytest.importorskip("lightgbm")
    X = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 30),
            "x2": np.arange(30) % 4,
        }
    )
    y = pd.Series(2.0 * X["x1"] - 0.2 * X["x2"])
    params = load_model_params(
        MODEL_PARAMS,
        task="regression",
        models=["lightgbm"],
    )["lightgbm"]
    with IsolatedProcessRunner(max_attempts=1) as runner:
        try:
            result = runner.run(
                _fit_predict_model_cell,
                model_name="lightgbm",
                model_seed=123,
                task="regression",
                params=params,
                X_train=X.iloc[:24],
                y_train=y.iloc[:24],
                X_test=X.iloc[24:],
            )
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"process pools unavailable in this sandbox: {exc}")

    assert np.asarray(result["predictions"]).shape == (6,)
    assert result["fit_seconds"] >= 0.0
    assert result["best_rounds"] >= 1

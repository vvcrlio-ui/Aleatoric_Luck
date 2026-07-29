from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import ctypes
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aleatoric_nk_grid import native_process
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


def _start_grandchild(pid_file: str) -> int:
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import os,sys,time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)", pid_file]
    )
    deadline = time.monotonic() + 5.0
    while not Path(pid_file).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not Path(pid_file).exists():
        raise RuntimeError("grandchild did not publish its PID")
    return grandchild.pid


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        return proc_stat.read_text(encoding="utf-8").split()[2] != "Z"
    if sys.platform == "darwin":
        buffer = ctypes.create_string_buffer(136)
        result = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pidinfo(
            pid, 3, 0, buffer, len(buffer)
        )
        if result <= 0:
            return False
        status = struct.unpack_from("I", buffer.raw, 4)[0]
        return status != 5
    return True


def _wait_not_running(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} is still running")


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
    with IsolatedProcessRunner(max_attempts=1, timeout_seconds=5.0, shutdown_grace_seconds=0.2) as runner:
        try:
            worker_pid = runner.run(os.getpid)
            grandchild_pid = runner.run(_start_grandchild, str(pid_file))
            assert int(pid_file.read_text(encoding="utf-8")) == grandchild_pid
            assert _process_is_running(grandchild_pid)
            runner.timeout_seconds = 0.1
            with pytest.raises(NativeProcessTimedOut):
                runner.run(_sleep_for, 60.0)
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")
        _wait_not_running(worker_pid)
        _wait_not_running(grandchild_pid)
        runner.timeout_seconds = 5.0
        assert runner.run(os.getpid) != os.getpid()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are unavailable")
def test_crash_kills_native_worker_grandchild_and_runner_recovers(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    with IsolatedProcessRunner(max_attempts=1, timeout_seconds=5.0) as runner:
        try:
            worker_pid = runner.run(os.getpid)
            grandchild_pid = runner.run(_start_grandchild, str(pid_file))
            assert _process_is_running(grandchild_pid)
            with pytest.raises(NativeProcessCrashed):
                runner.run(_terminate_child_abruptly)
        except (NotImplementedError, PermissionError) as exc:
            pytest.skip(f"processes unavailable in this sandbox: {exc}")
        _wait_not_running(worker_pid)
        _wait_not_running(grandchild_pid)
        assert runner.run(os.getpid) != os.getpid()


def test_timeout_escalates_from_sigterm_to_sigkill(tmp_path, monkeypatch):
    signals: list[int] = []
    real_killpg = os.killpg

    def delayed_killpg(pid: int, requested_signal: int) -> None:
        signals.append(requested_signal)
        if requested_signal == signal.SIGKILL:
            real_killpg(pid, requested_signal)

    monkeypatch.setattr(native_process.os, "killpg", delayed_killpg)
    with IsolatedProcessRunner(
        max_attempts=1,
        timeout_seconds=0.1,
        shutdown_grace_seconds=0.1,
    ) as runner:
        with pytest.raises(NativeProcessTimedOut):
            runner.run(_sleep_for, 60.0)

    assert signals[:2] == [signal.SIGTERM, signal.SIGKILL]


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

"""Reusable subprocess isolation for models backed by native libraries."""

from __future__ import annotations

import multiprocessing
import os
import signal
import traceback
from multiprocessing.connection import Connection, wait
from typing import Any, Callable


class NativeProcessCrashed(RuntimeError):
    """Raised when a native-model worker terminates outside Python control."""


class NativeProcessTimedOut(TimeoutError):
    """Raised after an isolated native-model cell exceeds its deadline."""


class NativeProcessRemoteError(RuntimeError):
    """Fallback for a child exception that cannot cross the process boundary."""


def _worker_loop(connection: Connection) -> None:
    """Execute sequential requests until the parent closes the worker."""

    # Loky/joblib descendants inherit this group, so timeout/crash cleanup can
    # terminate the complete model tree rather than leaking grandchildren.
    os.setsid()
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if request is None:
                return
            request_id, function, args, kwargs = request
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                try:
                    connection.send((request_id, "error", exc))
                except BaseException:
                    connection.send(
                        (
                            request_id,
                            "remote_error",
                            type(exc).__name__,
                            str(exc),
                            traceback.format_exc(),
                        )
                    )
            else:
                try:
                    connection.send((request_id, "ok", result))
                except BaseException as exc:
                    connection.send(
                        (
                            request_id,
                            "remote_error",
                            type(exc).__name__,
                            f"failed to serialize native result: {exc}",
                            traceback.format_exc(),
                        )
                    )
    finally:
        connection.close()


class IsolatedProcessRunner:
    """Run one callable at a time in a reusable spawn-based subprocess.

    A segmentation fault, native abort, or timeout is confined to the child.
    The parent terminates and discards that worker, optionally retries with a
    fresh interpreter, and ultimately raises a normal Python exception that the
    checkpoint loop can persist as a failed cell.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 2,
        timeout_seconds: float = 21_600.0,
        shutdown_grace_seconds: float = 2.0,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if shutdown_grace_seconds <= 0:
            raise ValueError("shutdown_grace_seconds must be greater than zero")
        self.max_attempts = int(max_attempts)
        self.timeout_seconds = float(timeout_seconds)
        self.shutdown_grace_seconds = float(shutdown_grace_seconds)
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._request_id = 0

    def _start_worker(self) -> tuple[multiprocessing.Process, Connection]:
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_loop,
            args=(child_connection,),
            name="aleatoric-native-model",
        )
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            raise
        child_connection.close()
        self._process = process
        self._connection = parent_connection
        return process, parent_connection

    def _worker(self) -> tuple[multiprocessing.Process, Connection]:
        process = self._process
        connection = self._connection
        if process is None or connection is None:
            return self._start_worker()
        if not process.is_alive():
            self._discard_worker(force=False)
            return self._start_worker()
        return process, connection

    def _discard_worker(self, *, force: bool) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            connection.close()
        if process is None:
            return
        if force:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        process.join(timeout=self.shutdown_grace_seconds)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join(timeout=self.shutdown_grace_seconds)
        # A just-spawned worker can time out before it has completed setsid().
        # Keep the process-group kill as the descendant-cleanup primitive, but
        # use multiprocessing's direct fallback so a stubborn parent never
        # escapes and makes ``Process.close`` raise while still alive.
        if process.is_alive():
            process.kill()
            process.join(timeout=self.shutdown_grace_seconds)
        if not process.is_alive():
            process.close()

    def _receive(
        self,
        process: multiprocessing.Process,
        connection: Connection,
        request_id: int,
    ) -> tuple[Any, ...]:
        ready = wait(
            (connection, process.sentinel),
            timeout=self.timeout_seconds,
        )
        if not ready:
            raise NativeProcessTimedOut(
                "native model subprocess exceeded "
                f"{self.timeout_seconds:g} seconds"
            )
        if connection in ready:
            try:
                response = connection.recv()
            except EOFError as exc:
                raise NativeProcessCrashed(
                    "native model subprocess closed its result pipe "
                    f"(exitcode={process.exitcode})"
                ) from exc
            if not response or response[0] != request_id:
                raise NativeProcessCrashed(
                    "native model subprocess returned an invalid response"
                )
            return tuple(response)
        process.join(timeout=self.shutdown_grace_seconds)
        raise NativeProcessCrashed(
            "native model subprocess terminated without a result "
            f"(exitcode={process.exitcode})"
        )

    def run(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        on_native_crash: Callable[[int, BaseException], None] | None = None,
        on_native_timeout: Callable[[int, BaseException], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            process, connection = self._worker()
            self._request_id += 1
            request_id = self._request_id
            try:
                connection.send((request_id, function, args, kwargs))
                response = self._receive(process, connection, request_id)
            except NativeProcessTimedOut as exc:
                last_error = exc
                self._discard_worker(force=True)
                if on_native_timeout is not None:
                    on_native_timeout(attempt, exc)
                continue
            except (
                BrokenPipeError,
                ConnectionError,
                EOFError,
                NativeProcessCrashed,
                OSError,
            ) as exc:
                last_error = exc
                self._discard_worker(force=True)
                if on_native_crash is not None:
                    on_native_crash(attempt, exc)
                continue

            status = response[1]
            if status == "ok":
                return response[2]
            if status == "error":
                error = response[2]
                if isinstance(error, Exception):
                    raise error
                if isinstance(error, BaseException):
                    raise NativeProcessRemoteError(
                        "native model subprocess raised "
                        f"{type(error).__name__}: {error}"
                    )
                raise NativeProcessRemoteError(str(error))
            if status == "remote_error":
                _, _, error_type, message, remote_traceback = response
                raise NativeProcessRemoteError(
                    f"{error_type}: {message}\nRemote traceback:\n{remote_traceback}"
                )
            raise NativeProcessCrashed(
                f"native model subprocess returned unknown status: {status!r}"
            )

        if isinstance(last_error, NativeProcessTimedOut):
            raise NativeProcessTimedOut(
                "native model subprocess timed out "
                f"{self.max_attempts} time(s); the worker was terminated and "
                "the parent checkpoint process remained alive"
            ) from last_error
        raise NativeProcessCrashed(
            "native model subprocess terminated abruptly "
            f"{self.max_attempts} time(s); the cell was isolated and the parent "
            "checkpoint process remained alive"
        ) from last_error

    def close(self) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            return
        if connection is not None and process.is_alive():
            try:
                connection.send(None)
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        self._discard_worker(force=False)

    def __enter__(self) -> IsolatedProcessRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

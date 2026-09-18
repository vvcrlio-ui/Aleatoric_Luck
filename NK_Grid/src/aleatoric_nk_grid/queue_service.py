"""Authenticated opt-in worker RPC. No submission, pause or import RPC exposed.

Default binding is loopback. Cluster networking/TLS or an authenticated tunnel
must be provisioned and validated separately before a production deployment.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
import random
import socket
import ssl
import sys
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

from .shared_queue import (Dispatcher, HEARTBEAT_SECONDS, LeaseLostError, MAX_BATCH_TASKS,
                           MAX_SUBMISSIONS, QueueError, TARGET_BATCH_SECONDS, atomic_json,
                           canonical, digest, file_lock, sync_directory)


class TransientServiceError(OSError):
    """A retryable unavailable response, not evidence of a revoked lease."""


class PayloadTooLargeError(QueueError):
    """The original result is retained, but cannot use the bounded RPC."""


class SafeBoundaryDrain(BaseException):
    """A persisted training boundary, not a scientific failure or a result ACK."""


class Client:
    def __init__(self, url, token, queue_id, *, ca_file=None, timeout=30):
        self.url = url.rstrip("/"); self.token = token; self.queue_id = queue_id
        self.context = ssl.create_default_context(cafile=ca_file) if ca_file else None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be positive and finite")
        self.timeout = timeout

    def call(self, operation, **arguments):
        raw = canonical({"queue_id": self.queue_id, **arguments})
        if len(raw) > 1024**2:
            raise PayloadTooLargeError("Request exceeds the 1 MiB limit")
        request = urllib.request.Request(self.url + "/" + operation,
            data=raw,
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # HTTPError owns the response socket too, including retryable errors.
            with exc:
                if exc.code == 429 or exc.code >= 500:
                    raise TransientServiceError("Dispatcher temporarily unavailable") from exc
                raw = exc.read().decode()
            try:
                error = json.loads(raw)
            except ValueError:
                error = {}
            if exc.code == 409 and isinstance(error, dict) and error.get("code") == "lease_lost":
                raise LeaseLostError(error.get("error", "Lease lost")) from exc
            if exc.code == 413 and isinstance(error, dict) and error.get("code") == "payload_too_large":
                raise PayloadTooLargeError(error.get("error", "Payload too large")) from exc
            raise QueueError(raw) from exc


def connection_budget(requested, *, fd_reserve=64):
    """Leave descriptors for the journal, progress, subprocesses and shutdown."""
    if type(requested) is not int or requested < 1:
        raise ValueError("max_connections must be a positive integer")
    try:
        import resource
    except ImportError:  # Windows has no RLIMIT_NOFILE.
        return requested
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return requested
    try:
        used = len(os.listdir('/proc/self/fd'))
    except OSError:
        used = 32
    available = int(soft) - used - fd_reserve - 1  # Listening socket.
    if available < 1:
        raise QueueError("Insufficient file descriptors for dispatcher and control reserve")
    return min(requested, available)


def retry_delay(failures, base=2., cap=30.):
    """Capped exponential backoff with jitter, below the 300s GPA lease."""
    return random.uniform(.5, 1.) * min(cap, base * 2 ** min(failures, 10))


def make_server(dispatcher, *, token, host="127.0.0.1", port=0, tls_context=None,
                threaded_tls_handshake=False, max_connections=128, request_timeout=10.,
                max_submissions=MAX_SUBMISSIONS):
    # Keep threaded_tls_handshake for existing launchers; all TLS now uses the
    # bounded request threads, never a handshake on the single accept thread.
    if len(token) < 32:
        raise ValueError("Use a private random token of at least 32 characters")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("Request timeout must be positive and finite")
    capacity = connection_budget(max_connections)
    if type(max_submissions) is not int or max_submissions < 1:
        raise ValueError("max_submissions must be a positive integer")
    submit_capacity = min(max_submissions, max(1, capacity // 2))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log bearer tokens or payloads.

        def reply(self, status, value):
            raw = canonical(value)
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    self.reply(413, {"error": "Request exceeds the 1 MiB limit", "code": "payload_too_large"})
                    return
                if length <= 0:
                    raise QueueError("Invalid request size")
                raw = self.rfile.read(length)
                # Consume the bounded body before closing even a rejected
                # connection, otherwise Windows can send RST instead of 401.
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                    self.reply(401, {"error": "Unauthorized"}); return
                value = json.loads(raw)
                if value.pop("queue_id", None) != dispatcher.queue_id:
                    raise QueueError("Wrong queue identity")
                methods = {"/claim": dispatcher.claim, "/heartbeat": dispatcher.heartbeat,
                           "/submit": dispatcher.submit, "/stats": dispatcher.stats,
                           "/claim_batch": getattr(dispatcher, "claim_batch", None),
                           "/protocol": getattr(dispatcher, "protocol", None),
                           "/heartbeat_batch": getattr(dispatcher, "heartbeat_batch", None),
                           "/worker_fault": getattr(dispatcher, "worker_fault", None),
                           "/submit_batch": getattr(dispatcher, "submit_batch", None),
                           "/identity": lambda: {"queue_id": dispatcher.queue_id, "epoch": dispatcher.epoch}}
                if methods.get(self.path) is None:
                    self.reply(404, {"error": "Unknown operation"}); return
                if self.path in ("/submit", "/submit_batch"):
                    # Do not let requests waiting on durable journal I/O occupy
                    # every connection and starve lease renewals.
                    if not self.server.submit_slots.acquire(blocking=False):
                        with self.server.metrics_lock:
                            self.server.submit_backpressure += 1
                        self.reply(503, {"error": "Result submission busy; retain receipt and retry"})
                        return
                    with self.server.metrics_lock:
                        self.server.active_submissions += 1
                        self.server.peak_submissions = max(self.server.peak_submissions,
                                                           self.server.active_submissions)
                    try:
                        self.reply(200, methods[self.path](**value))
                    finally:
                        with self.server.metrics_lock:
                            self.server.active_submissions -= 1
                        self.server.submit_slots.release()
                else:
                    self.reply(200, methods[self.path](**value))
            except LeaseLostError as exc:
                self.reply(409, {"error": str(exc), "code": "lease_lost"})
            except (QueueError, ValueError, TypeError, KeyError) as exc:
                self.reply(409, {"error": str(exc)})
            except (ConnectionError, ssl.SSLError, TimeoutError):
                # The peer is gone: a second reply only creates another traceback.
                raise
            except Exception:
                self.reply(503, {"error": "Dispatcher unavailable; retain result and retry"})

    class QueueHTTPServer(ThreadingHTTPServer):
        request_queue_size = 1024
        daemon_threads = False
        block_on_close = True

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(capacity)
            self.submit_slots = threading.BoundedSemaphore(submit_capacity)
            self.submit_backpressure = self.active_submissions = self.peak_submissions = 0
            self.metrics_lock = threading.Lock()
            self.active_connections = self.peak_connections = self.transport_errors = 0
            super().__init__(*args)

        def connection_stats(self):
            with self.metrics_lock:
                value = dict(max_connections=capacity, active_connections=self.active_connections,
                             peak_connections=self.peak_connections, transport_errors=self.transport_errors,
                             submit_capacity=submit_capacity, submit_backpressure=self.submit_backpressure,
                             active_submissions=self.active_submissions,
                             peak_submissions=self.peak_submissions)
            try:
                import resource
                value['nofile_soft_limit'] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
                value['open_fds'] = len(os.listdir('/proc/self/fd'))
            except (ImportError, OSError):
                pass
            return value

        def get_request(self):
            # Acquire BEFORE accept: queued peers stay in the kernel backlog and
            # consume neither a process FD nor a Python thread. A short wait lets
            # serve_forever observe shutdown even when every slot is occupied.
            if not self.slots.acquire(timeout=.05):
                raise BlockingIOError("Dispatcher connection capacity reached")
            connection = None
            try:
                connection, address = super().get_request()
                connection.settimeout(request_timeout)  # Includes HTTP headers.
                with self.metrics_lock:
                    self.active_connections += 1
                    self.peak_connections = max(self.peak_connections, self.active_connections)
                return connection, address
            except BaseException:
                if connection is not None:
                    connection.close()
                self.slots.release()
                raise

        def release_slot(self):
            with self.metrics_lock:
                self.active_connections -= 1
            self.slots.release()

        def process_request(self, request, address):
            try:
                super().process_request(request, address)
            except BaseException:
                self.release_slot()  # Thread creation failed; base server closes FD.
                # ThreadingMixIn registers the thread before start(). Remove an
                # unstarted thread so server_close() can still join the others.
                if hasattr(self._threads, 'reap'):
                    self._threads.reap()
                raise

        def process_request_thread(self, request, address):
            try:
                if tls_context is not None:
                    request = tls_context.wrap_socket(request, server_side=True,
                                                      do_handshake_on_connect=False)
                    request.do_handshake()
                self.finish_request(request, address)
            except Exception:
                self.handle_error(request, address)
            finally:
                try:
                    self.shutdown_request(request)
                finally:
                    self.release_slot()

        def handle_error(self, request, address):
            if isinstance(sys.exc_info()[1], (ConnectionError, ssl.SSLError, TimeoutError)):
                with self.metrics_lock:
                    self.transport_errors += 1
                return
            super().handle_error(request, address)

    server = QueueHTTPServer((host, port), Handler)
    server.dispatcher = dispatcher
    return server


@contextmanager
def worker_slot(spool, queue_id):
    """Persist one logical worker identity and exclusively own its durable spool.

    Reuse the same path when restarting a slot, including on another node. Each
    concurrent slot needs a distinct shared-storage path. Never copy a live slot.
    """
    root = Path(spool); root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".worker.lock"):
        path = root / ".worker-identity"
        if path.exists():
            value = json.loads(path.read_bytes())
            if (value.get("queue_id") != queue_id or
                    not isinstance(value.get("worker"), str) or
                    not 0 < len(value["worker"]) <= 256):
                raise QueueError("Worker spool identity does not match this queue")
        else:
            value = {"queue_id": queue_id,
                     "worker": socket.gethostname()[:200] + ":" + uuid.uuid4().hex}
            atomic_json(path, value)
        yield value["worker"]


def execute_worker(client, worker, execute, *, spool, cached_cells=lambda: (),
                   heartbeat_seconds=20, idle_seconds=2., stop=lambda: False,
                   deadline_seconds=3600, recover_stale_leases=False):
    """Keep one task in flight. Local spool survives a lost submit response.

    Stop is a drain: finish and acknowledge current model, then stop claiming.
    A lost lease prevents stale submission; a numerical exception becomes an
    explicit failed result. Server lease duration must exceed heartbeat cadence.
    """
    root = Path(spool); root.mkdir(parents=True, exist_ok=True)
    until = time.monotonic() + deadline_seconds
    accepted = 0
    recovered = 0

    def report(state, **extra):
        return {"state": state, "accepted": accepted,
                **({"recovered_leases": recovered} if recovered else {}), **extra}

    def quarantine(receipt, reason):
        nonlocal recovered
        # Keep every token's evidence outside the replay glob. Never retag a
        # stale result with a new lease, or overwrite it on the next execution.
        previous = json.loads(receipt.read_bytes())
        rejected = root / "rejected"; rejected.mkdir(exist_ok=True)
        target = rejected / (receipt.stem + "-" + digest(previous["token"]) + ".json")
        if target.exists() and target.read_bytes() != receipt.read_bytes():
            raise QueueError("Conflicting quarantined receipt")
        os.replace(receipt, target)
        sync_directory(rejected); sync_directory(root)
        recovered += 1
        print(json.dumps({"worker_event": "lease_recovered", "worker": worker,
                          "reason": str(reason), "receipt": str(target)}),
              file=sys.stderr, flush=True)
        time.sleep(idle_seconds * random.uniform(.8, 1.2))
    # Recover before claiming: an accepted result whose reply was lost will no
    # longer be returned by claim. The server also checks live/stale lease tokens.
    for receipt in sorted(root.glob("*.json")):
        previous = json.loads(receipt.read_bytes())
        if previous.get("queue_id") != client.queue_id:
            raise QueueError("Result spool belongs to another queue")
        failures = 0
        while True:
            try:
                client.call("submit", task_id=receipt.stem, token=previous["token"],
                            worker=worker, result=previous["result"])
                receipt.unlink(); accepted += 1
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= until:
                    return report("unacknowledged", receipt=str(receipt))
                time.sleep(retry_delay(failures, idle_seconds)); failures += 1
            except QueueError as exc:
                # Retain evidence. A reclaimed task gets a new token and must be
                # recomputed; an obsolete result may never override its winner.
                if recover_stale_leases:
                    if not isinstance(exc, LeaseLostError):
                        raise
                    quarantine(receipt, exc)
                break
    failures = 0
    while not stop() and time.monotonic() < until:
        try:
            lease = client.call("claim", worker=worker, cached_cells=cached_cells())
        except (OSError, urllib.error.URLError):
            time.sleep(retry_delay(failures, idle_seconds)); failures += 1; continue
        failures = 0
        if lease["state"] in {"complete", "blocked", "paused"}:
            return report(lease["state"])
        if lease["state"] == "wait":
            time.sleep(idle_seconds * random.uniform(.8, 1.2)); continue
        done = threading.Event(); lost = []

        def heartbeat(lease=lease, done=done, lost=lost):
            heartbeat_failures = 0
            delay = heartbeat_seconds * random.uniform(.8, 1.2)
            while not done.wait(delay):
                try:
                    client.call("heartbeat", task_id=lease["id"], token=lease["token"], worker=worker)
                except QueueError as exc:
                    lost.append(exc); return
                except (OSError, urllib.error.URLError):
                    # Retry promptly instead of waiting another full heartbeat
                    # period. The server still fences revoked lease tokens.
                    delay = min(heartbeat_seconds, retry_delay(heartbeat_failures, base=2., cap=30.))
                    heartbeat_failures += 1
                else:
                    heartbeat_failures = 0
                    delay = heartbeat_seconds * random.uniform(.8, 1.2)

        thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
        try:
            receipt = root / (lease["id"] + ".json")
            previous = json.loads(receipt.read_bytes()) if receipt.exists() else None
            if previous and previous.get("queue_id") == client.queue_id and previous.get("token") == lease["token"]:
                result = previous["result"]
            else:
                try:
                    result = execute(lease["task"])
                except Exception as exc:
                    result = {**lease["task"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                atomic_json(receipt, {"queue_id": client.queue_id, "token": lease["token"], "result": result})
            if lost:
                if recover_stale_leases:
                    if not isinstance(lost[0], LeaseLostError):
                        raise lost[0]
                    quarantine(receipt, lost[0])
                    continue
                return report("lease_lost", receipt=str(receipt))
            rejected = False
            while True:
                try:
                    client.call("submit", task_id=lease["id"], token=lease["token"], worker=worker, result=result)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= until:
                        return report("unacknowledged", receipt=str(receipt))
                    time.sleep(retry_delay(failures, idle_seconds)); failures += 1
                except QueueError as exc:
                    if recover_stale_leases:
                        if not isinstance(exc, LeaseLostError):
                            raise
                        quarantine(receipt, exc)
                        rejected = True
                        break
                    return report("submission_rejected", receipt=str(receipt), error=str(exc))
            if rejected:
                continue
            receipt.unlink(); accepted += 1
        finally:
            done.set(); thread.join(timeout=5)
    return report("drained")


def execute_batch_worker(client, worker, execute, *, spool, cached_cells=lambda: (),
                         heartbeat_seconds=HEARTBEAT_SECONDS, idle_seconds=2.,
                         stop=lambda: False, deadline_seconds=3600, recover_stale_leases=False,
                         target_batch_seconds=TARGET_BATCH_SECONDS, max_batch=MAX_BATCH_TASKS,
                         protocol_version=1, deadline_epoch=None,
                         flush_seconds=None, submit_bytes=None,
                         max_buffer_bytes=4 * 1024**2, max_spool_bytes=16 * 1024**2,
                         fault_dir=None):
    """Run explicit v2 incremental transport, or the compatible v1 worker.

    Production opts into v2: server-priced leases, token heartbeats and byte /
    time bounded result chunks. The v1 path retains its old adaptive batching
    for compatibility tests and explicitly selected legacy callers.
    """
    if protocol_version == 2:
        return _execute_incremental_worker(client, worker, execute, spool=spool,
            cached_cells=cached_cells, heartbeat_seconds=heartbeat_seconds,
            idle_seconds=idle_seconds, stop=stop, deadline_seconds=deadline_seconds,
            deadline_epoch=deadline_epoch, recover_stale_leases=recover_stale_leases,
            max_batch=max_batch, flush_seconds=flush_seconds, submit_bytes=submit_bytes,
            max_buffer_bytes=max_buffer_bytes, max_spool_bytes=max_spool_bytes, fault_dir=fault_dir)
    if protocol_version != 1:
        raise QueueError("Unsupported worker protocol version")
    if deadline_epoch is not None:
        deadline_seconds = min(deadline_seconds, max(0., float(deadline_epoch) - time.time()))
    root = Path(spool); root.mkdir(parents=True, exist_ok=True)
    until = time.monotonic() + deadline_seconds
    accepted = 0
    recovered = 0
    count = 1

    def report(state, **extra):
        return {"state": state, "accepted": accepted, "cells_per_batch": count,
                **({"recovered_leases": recovered} if recovered else {}), **extra}

    def quarantine(receipt, reason):
        nonlocal recovered
        previous = json.loads(receipt.read_bytes())
        rejected = root / "rejected"; rejected.mkdir(exist_ok=True)
        target = rejected / (receipt.stem + "-" + digest(previous["token"]) + ".json")
        if target.exists() and target.read_bytes() != receipt.read_bytes():
            raise QueueError("Conflicting quarantined receipt")
        os.replace(receipt, target)
        sync_directory(rejected); sync_directory(root)
        recovered += 1
        print(json.dumps({"worker_event": "lease_recovered", "worker": worker,
                          "reason": str(reason), "receipt": str(target)}),
              file=sys.stderr, flush=True)
        time.sleep(idle_seconds * random.uniform(.8, 1.2))

    def deliver(receipt, previous):
        """Resend one durable batch until it is accepted, lost or out of time."""
        failures = 0
        while True:
            try:
                client.call("submit_batch", task_ids=previous["task_ids"], token=previous["token"],
                            worker=worker, results=previous["results"])
                return len(previous["task_ids"])
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= until:
                    return None
                time.sleep(retry_delay(failures, idle_seconds)); failures += 1
            except QueueError as exc:
                if recover_stale_leases:
                    if not isinstance(exc, LeaseLostError):
                        raise
                    quarantine(receipt, exc)
                    return 0
                raise

    # Recover before claiming: an accepted batch whose reply was lost will no
    # longer be returned by claim. The server also fences stale lease tokens.
    for receipt in sorted(root.glob("*.json")):
        previous = json.loads(receipt.read_bytes())
        if previous.get("queue_id") != client.queue_id:
            raise QueueError("Result spool belongs to another queue")
        if "task_ids" not in previous:
            raise QueueError("Result spool holds single-cell receipts; use execute_worker")
        delivered = deliver(receipt, previous)
        if delivered is None:
            return report("unacknowledged", receipt=str(receipt))
        if delivered:
            receipt.unlink(); accepted += delivered

    failures = 0
    while not stop() and time.monotonic() < until:
        try:
            lease = client.call("claim_batch", worker=worker, count=count,
                                cached_cells=cached_cells())
        except (OSError, urllib.error.URLError):
            time.sleep(retry_delay(failures, idle_seconds)); failures += 1; continue
        failures = 0
        if lease["state"] in {"complete", "blocked", "paused"}:
            return report(lease["state"])
        if lease["state"] == "wait":
            time.sleep(idle_seconds * random.uniform(.8, 1.2)); continue
        tasks = lease["tasks"]
        done = threading.Event(); lost = []

        def heartbeat(lease=lease, done=done, lost=lost):
            # One renewal covers the whole batch; they share a deadline.
            heartbeat_failures = 0
            delay = heartbeat_seconds * random.uniform(.8, 1.2)
            while not done.wait(delay):
                try:
                    client.call("heartbeat", task_id=lease["tasks"][0]["id"],
                                token=lease["token"], worker=worker)
                except QueueError as exc:
                    lost.append(exc); return
                except (OSError, urllib.error.URLError):
                    delay = min(heartbeat_seconds, retry_delay(heartbeat_failures, base=2., cap=30.))
                    heartbeat_failures += 1
                else:
                    heartbeat_failures = 0
                    delay = heartbeat_seconds * random.uniform(.8, 1.2)

        thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
        try:
            receipt = root / ("batch-" + digest(lease["token"]) + ".json")
            previous = json.loads(receipt.read_bytes()) if receipt.exists() else None
            if previous and previous.get("queue_id") == client.queue_id and previous.get("token") == lease["token"]:
                results = previous["results"]
            else:
                started = time.monotonic()
                results = []
                for task in tasks:
                    try:
                        results.append(execute(task["task"]))
                    except Exception as exc:
                        # One unusable cell is a failed result, not a failed batch.
                        results.append({**task["task"], "status": "failed",
                                        "error": f"{type(exc).__name__}: {exc}"})
                elapsed = time.monotonic() - started
                atomic_json(receipt, {"queue_id": client.queue_id, "token": lease["token"],
                                      "task_ids": [task["id"] for task in tasks], "results": results})
                # Size the next request from what this batch actually cost, and
                # never start one that cannot finish before the drain deadline.
                seconds = elapsed / len(tasks)
                room = max(0., until - time.monotonic())
                count = max(1, min(max_batch,
                                   int(target_batch_seconds / seconds) if seconds > 0 else max_batch,
                                   int(room / seconds) if seconds > 0 else max_batch))
            if lost:
                if recover_stale_leases:
                    if not isinstance(lost[0], LeaseLostError):
                        raise lost[0]
                    quarantine(receipt, lost[0])
                    continue
                return report("lease_lost", receipt=str(receipt))
            delivered = deliver(receipt, json.loads(receipt.read_bytes()))
            if delivered is None:
                return report("unacknowledged", receipt=str(receipt))
            if not delivered:
                continue
            receipt.unlink(); accepted += delivered
        finally:
            done.set(); thread.join(timeout=5)
    return report("drained")


def _batch_request(queue_id, worker, token, task_ids, results, chunk_id=None):
    value = dict(queue_id=queue_id, worker=worker, token=token,
                 task_ids=task_ids, results=results)
    if chunk_id is not None:
        value["chunk_id"] = chunk_id
    return value


class _ChunkSender:
    """One bounded producer buffer and an independent durable, timed sender.

    The producer reserves one maximum request before starting a cell. A single
    oversized result is preserved as blocked evidence and stops the worker;
    there is no truncation, repeated poison replay, or unbounded follow-on work.
    """

    def __init__(self, *, client, worker, token, root, until, flush_seconds,
                 soft_bytes, hard_bytes, max_buffer_bytes, max_spool_bytes,
                 idle_seconds, recover_stale_leases):
        self.client = client; self.worker = worker; self.token = token
        self.root = root; self.until = until; self.flush_seconds = flush_seconds
        self.soft_bytes = soft_bytes; self.hard_bytes = hard_bytes
        self.max_buffer_bytes = max_buffer_bytes; self.max_spool_bytes = max_spool_bytes
        self.idle_seconds = idle_seconds; self.recover_stale_leases = recover_stale_leases
        self.cv = threading.Condition(); self.pending = deque()
        self.pending_bytes = self.inflight_bytes = 0
        self.envelope_bytes = len(canonical(_batch_request(client.queue_id, worker, token, [], [], "0" * 32)))
        self.closed = False; self.force = False; self.error = None; self.receipt = None
        self.accepted = self.recovered = self.chunks = 0
        self.peak_buffer_bytes = self.peak_spool_bytes = 0
        self.spool_seconds = self.submit_seconds = 0.
        self.thread = threading.Thread(target=self._run, name="result-chunks", daemon=False)

    def start(self):
        self.thread.start()

    def fail(self, exc):
        with self.cv:
            if self.error is None:
                self.error = exc
            self.cv.notify_all()

    def capacity(self, stopping):
        with self.cv:
            while not self.error and not stopping():
                if self.pending_bytes + self.inflight_bytes <= self.max_buffer_bytes - self.hard_bytes:
                    return True
                self.force = True; self.cv.notify_all(); self.cv.wait(.1)
            return False

    def add(self, task_id, result):
        # Encoded pair sizes include both array separators. The envelope is
        # accounted for separately when constructing the actual request.
        size = len(canonical(task_id)) + len(canonical(result)) + 2
        with self.cv:
            self.pending.append((task_id, result, size, time.monotonic()))
            self.pending_bytes += size
            self.peak_buffer_bytes = max(self.peak_buffer_bytes,
                                         self.pending_bytes + self.inflight_bytes)
            if self.envelope_bytes + size - 2 > self.hard_bytes and self.error is None:
                self.error = PayloadTooLargeError("One result exceeds the request limit")
            self.cv.notify_all()

    def finish(self):
        with self.cv:
            self.closed = True; self.cv.notify_all()
        self.thread.join()

    def status(self):
        with self.cv:
            return {"unacknowledged_chunks": int(bool(self.pending)) + int(bool(self.inflight_bytes)),
                    "pending_result_bytes": self.pending_bytes + self.inflight_bytes}

    def _preserve(self, path, reason):
        folder = self.root / reason; folder.mkdir(exist_ok=True)
        target = folder / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise QueueError("Conflicting preserved chunk receipt")
            path.unlink()
        else:
            os.replace(path, target)
        sync_directory(folder); sync_directory(self.root)
        self.receipt = target
        return target

    def _deliver(self, path, value):
        failures = 0
        while True:
            with self.cv:
                failure = self.error
            if failure is not None:
                if isinstance(failure, LeaseLostError) and self.recover_stale_leases:
                    self._preserve(path, "rejected"); self.recovered += 1
                return
            try:
                started = time.monotonic()
                self.client.call("submit_batch", **{k: v for k, v in value.items() if k != "queue_id"})
                self.submit_seconds += time.monotonic() - started
                path.unlink(); sync_directory(self.root)
                self.accepted += len(value["task_ids"]); self.chunks += 1
                return
            except (OSError, urllib.error.URLError) as exc:
                if time.monotonic() >= self.until:
                    self.fail(exc); return
                delay = min(retry_delay(failures, self.idle_seconds),
                            max(0., self.until - time.monotonic()))
                with self.cv:
                    if self.error is None:
                        self.cv.wait(delay)
                failures += 1
            except QueueError as exc:
                self.fail(exc)
                if isinstance(exc, PayloadTooLargeError):
                    self._preserve(path, "blocked")
                elif isinstance(exc, LeaseLostError) and self.recover_stale_leases:
                    self._preserve(path, "rejected"); self.recovered += 1
                return

    def _run(self):
        try:
            while True:
                with self.cv:
                    while True:
                        if not self.pending:
                            if self.closed:
                                return
                            self.cv.wait(); continue
                        due = self.pending[0][3] + self.flush_seconds - time.monotonic()
                        if (self.closed or self.force or self.error or due <= 0 or
                                self.pending_bytes + self.envelope_bytes - 2 >= self.soft_bytes):
                            break
                        self.cv.wait(due)
                    chunk_id = uuid.uuid4().hex
                    value = _batch_request(self.client.queue_id, self.worker, self.token, [], [], chunk_id)
                    size = len(canonical(value))
                    while self.pending:
                        item = self.pending[0]
                        added = item[2] - (2 if not value["task_ids"] else 0)
                        if value["task_ids"] and size + added > self.soft_bytes:
                            break
                        self.pending.popleft(); self.pending_bytes -= item[2]
                        value["task_ids"].append(item[0]); value["results"].append(item[1])
                        size += added
                        if size >= self.soft_bytes:
                            break
                    self.force = False; self.inflight_bytes = size
                    self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.pending_bytes + size)
                path = self.root / ("chunk-" + chunk_id + ".json")
                # Receipt JSON has the exact HTTP fields (atomic_json adds a
                # newline). Replay preserves canonical bytes after ACK loss.
                on_disk = sum(p.stat().st_size for p in self.root.rglob("*.json"))
                if on_disk + size + 1 > self.max_spool_bytes:
                    self.fail(QueueError("Result spool byte limit reached"))
                started = time.monotonic()
                atomic_json(path, value)
                self.spool_seconds += time.monotonic() - started
                self.peak_spool_bytes = max(self.peak_spool_bytes, on_disk + size + 1)
                self.receipt = path
                if size > self.hard_bytes:
                    self.fail(PayloadTooLargeError("One result exceeds the request limit"))
                    self._preserve(path, "blocked")
                else:
                    self._deliver(path, value)
                with self.cv:
                    self.inflight_bytes = 0; self.cv.notify_all()
        except Exception as exc:
            # Filesystem failures must wake a producer waiting for capacity.
            # A previously saved receipt remains replayable.
            self.fail(exc)


def _execute_incremental_worker(client, worker, execute, *, spool, cached_cells,
                                heartbeat_seconds, idle_seconds, stop, deadline_seconds,
                                deadline_epoch, recover_stale_leases, max_batch,
                                flush_seconds, submit_bytes, max_buffer_bytes, max_spool_bytes,
                                fault_dir):
    root = Path(spool); root.mkdir(parents=True, exist_ok=True)
    remaining = float(deadline_seconds)
    if deadline_epoch is not None:
        remaining = min(remaining, float(deadline_epoch) - time.time())
    if not math.isfinite(remaining):
        raise QueueError("Worker deadline must be finite")
    until = time.monotonic() + max(0., remaining)
    accepted = recovered = chunks = 0
    peak_buffer = peak_spool = 0
    spool_seconds = submit_seconds = 0.

    def report(state, **extra):
        value = dict(state=state, accepted=accepted, protocol_version=2,
                    chunks=chunks, recovered_leases=recovered,
                    peak_buffer_bytes=peak_buffer, peak_spool_bytes=peak_spool,
                    spool_seconds=spool_seconds, submit_seconds=submit_seconds, **extra)
        if state in {"payload_too_large", "spool_limit", "submission_rejected"}:
            # A terminal slot error must drain the whole allocation; otherwise
            # thousands of healthy idle slots can keep billing for this lease.
            # The controller scans this durable marker if the RPC is unavailable.
            fault = {"queue_id": client.queue_id, "worker": worker, "code": state,
                     "message": str(extra.get("error", state))[:1000],
                     "receipt": extra.get("receipt"), "created_at": time.time()}
            try:
                atomic_json(root / "worker-fault", fault)
            except OSError as exc:
                value["fault_marker_error"] = str(exc)
            if fault_dir is not None:
                try:
                    shared_faults = Path(fault_dir); shared_faults.mkdir(parents=True, exist_ok=True)
                    atomic_json(shared_faults / (digest(worker) + ".json"), fault)
                except OSError as exc:
                    value["shared_fault_marker_error"] = str(exc)
            try:
                client.call("worker_fault", worker=worker, code=state, message=fault["message"])
            except (QueueError, OSError, urllib.error.URLError) as exc:
                value["fault_notification_error"] = str(exc)
        return value

    failures = 0
    while True:
        try:
            policy = client.call("protocol", worker=worker, version=2, hostname=socket.gethostname())
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= until:
                return report("unavailable")
            time.sleep(min(retry_delay(failures, idle_seconds), max(0., until - time.monotonic())))
            failures += 1
    if policy.get("version") != 2:
        raise QueueError("Dispatcher did not acknowledge worker protocol 2")
    fault_path = root / "worker-fault"
    if fault_path.exists():
        previous_fault = json.loads(fault_path.read_bytes())
        if previous_fault.get("queue_id") != client.queue_id or previous_fault.get("worker") != worker:
            raise QueueError("Worker fault marker belongs to another queue or worker")
        return report(previous_fault["code"], error=previous_fault.get("message"),
                      receipt=previous_fault.get("receipt"))
    hard_bytes = min(1024**2, int(policy["max_request_bytes"]))
    soft_bytes = min(512 * 1024, int(policy["submit_bytes"]),
                     int(submit_bytes) if submit_bytes is not None else 512 * 1024)
    cadence = min(float(policy["status_seconds"]), float(heartbeat_seconds))
    flush = min(float(policy["flush_seconds"]),
                float(flush_seconds) if flush_seconds is not None else 10.)
    count = min(int(policy["max_batch"]), max_batch)
    if (type(max_batch) is not int or count < 1 or not 0 < soft_bytes <= hard_bytes or
            not math.isfinite(cadence) or cadence <= 0 or not math.isfinite(flush) or flush <= 0 or
            max_buffer_bytes < 2 * hard_bytes or max_spool_bytes < max_buffer_bytes + hard_bytes):
        raise QueueError("Invalid incremental worker limits")
    blocked = sorted((root / "blocked").glob("*.json"))
    if blocked:
        return report("payload_too_large", receipt=str(blocked[0]))

    def sender_for(token):
        return _ChunkSender(client=client, worker=worker, token=token, root=root,
            until=until, flush_seconds=flush, soft_bytes=soft_bytes, hard_bytes=hard_bytes,
            max_buffer_bytes=max_buffer_bytes, max_spool_bytes=max_spool_bytes,
            idle_seconds=idle_seconds, recover_stale_leases=recover_stale_leases)

    def sender_report(sender):
        nonlocal accepted, recovered, chunks, peak_buffer, peak_spool, spool_seconds, submit_seconds
        accepted += sender.accepted; recovered += sender.recovered; chunks += sender.chunks
        peak_buffer = max(peak_buffer, sender.peak_buffer_bytes)
        peak_spool = max(peak_spool, sender.peak_spool_bytes)
        spool_seconds += sender.spool_seconds; submit_seconds += sender.submit_seconds

    def error_report(sender):
        exc = sender.error
        state = ("payload_too_large" if isinstance(exc, PayloadTooLargeError) else
                 "lease_lost" if isinstance(exc, LeaseLostError) else
                 "unacknowledged" if isinstance(exc, (OSError, urllib.error.URLError)) else
                 "submission_rejected")
        return report(state, error=str(exc), receipt=str(sender.receipt) if sender.receipt else None)

    # Existing immutable receipts are acknowledged before a new batch is
    # acquired. A duplicate accepted task remains legal even after lease expiry.
    for path in sorted(root.glob("*.json")):
        previous = json.loads(path.read_bytes())
        if previous.get("queue_id") != client.queue_id or previous.get("worker", worker) != worker:
            raise QueueError("Result spool belongs to another queue or worker")
        if "task_ids" not in previous:
            raise QueueError("Result spool holds single-cell receipts; use execute_worker")
        value = _batch_request(client.queue_id, worker, previous["token"],
                               previous["task_ids"], previous["results"], previous.get("chunk_id"))
        sender = sender_for(previous["token"]); sender.receipt = path
        if len(canonical(value)) > hard_bytes:
            sender.fail(PayloadTooLargeError("Saved result exceeds the request limit"))
            sender._preserve(path, "blocked")
        else:
            replay_done = threading.Event()

            def replay_heartbeat(token=previous["token"]):
                delay = 0.
                while not replay_done.wait(delay):
                    try:
                        client.call("heartbeat_batch", worker=worker, token=token,
                            status={"state": "submitting", "unacknowledged_chunks": 1})
                    except LeaseLostError:
                        # A committed receipt is still a valid duplicate after
                        # expiry. Let submit perform the authoritative check.
                        return
                    except (QueueError, OSError, urllib.error.URLError):
                        pass
                    delay = cadence * random.uniform(.8, 1.2)

            replay_thread = threading.Thread(target=replay_heartbeat, name="replay-heartbeat", daemon=False)
            replay_thread.start()
            try:
                sender._deliver(path, value)
            finally:
                replay_done.set(); replay_thread.join()
        sender_report(sender)
        if sender.error and not (recover_stale_leases and isinstance(sender.error, LeaseLostError)):
            return error_report(sender)

    draining = bool(policy.get("drain"))
    last_completed = None
    failures = 0
    while not draining and not stop() and time.monotonic() < until:
        # Reserve enough disk for all bounded buffered results if this batch
        # loses its lease. Quarantined evidence also counts against the limit.
        on_disk = sum(p.stat().st_size for p in root.rglob("*.json"))
        if on_disk + max_buffer_bytes + hard_bytes > max_spool_bytes:
            return report("spool_limit", error="Result spool cannot reserve one bounded batch")
        try:
            lease = client.call("claim_batch", worker=worker, count=count,
                remaining_seconds=max(0., until - time.monotonic()), cached_cells=cached_cells(),
                status={"state": "idle", "last_completed_at": last_completed})
        except (OSError, urllib.error.URLError):
            time.sleep(min(retry_delay(failures, idle_seconds), max(0., until - time.monotonic())))
            failures += 1; continue
        failures = 0
        if lease["state"] in {"complete", "blocked", "paused", "draining", "drained"}:
            return report("drained" if lease["state"] in {"draining", "drained"} else lease["state"])
        if lease["state"] == "wait":
            delay = max(idle_seconds, min(30., float(lease.get("retry_after_seconds", 5.))))
            time.sleep(min(delay * random.uniform(.8, 1.2), max(0., until - time.monotonic())))
            continue
        tasks = lease["tasks"]
        sender = sender_for(lease["token"])
        done = threading.Event(); drain = threading.Event()
        current = {"task_id": None, "started": None, "remaining": len(tasks)}
        state_lock = threading.Lock()

        def status():
            with state_lock:
                value = {"state": "computing" if current["task_id"] is not None else "submitting",
                         "task_id": current["task_id"],
                         "cell_elapsed_seconds": max(0., time.monotonic() - current["started"]) if current["started"] is not None else 0.,
                         "last_completed_at": last_completed, "prefetched_tasks": current["remaining"]}
            return {**value, **sender.status()}

        def heartbeat():
            failures = 0
            delay = 0.  # Register computing state promptly, then report long cells.
            while not done.wait(delay):
                try:
                    reply = client.call("heartbeat_batch", worker=worker, token=lease["token"], status=status())
                    if reply.get("drain"):
                        drain.set()
                        if hasattr(execute, 'request_drain'):
                            execute.request_drain()
                except QueueError as exc:
                    sender.fail(exc); return
                except (OSError, urllib.error.URLError):
                    delay = min(cadence, retry_delay(failures, base=2., cap=30.)); failures += 1
                else:
                    failures = 0; delay = cadence * random.uniform(.8, 1.2)

        sender.start()
        heartbeat_thread = threading.Thread(target=heartbeat, name="batch-heartbeat", daemon=False)
        heartbeat_started = False
        try:
            for task in tasks:
                stopping = lambda: drain.is_set() or stop() or time.monotonic() >= until
                if not sender.capacity(stopping):
                    break
                with state_lock:
                    current.update(task_id=task["id"], started=time.monotonic(), remaining=current["remaining"] - 1)
                if not heartbeat_started:
                    heartbeat_thread.start(); heartbeat_started = True
                started = time.monotonic()
                try:
                    result = execute(task["task"])
                except SafeBoundaryDrain:
                    drain.set()
                    break
                except Exception as exc:
                    result = {**task["task"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                result = {**result, "_worker_wall_seconds": time.monotonic() - started}
                with state_lock:
                    current.update(task_id=None, started=None)
                    last_completed = time.time()
                sender.add(task["id"], result)
        finally:
            sender.finish()
            done.set()
            if heartbeat_started:
                heartbeat_thread.join()
        sender_report(sender)
        if sender.error:
            if recover_stale_leases and isinstance(sender.error, LeaseLostError):
                continue
            return error_report(sender)
        draining = drain.is_set()
    return report("drained")


def serve_with_drain(server, *, drain_file=None, stop=lambda: False):
    """Optional local operator file is authoritative: present=pause claims.

    Current leases may heartbeat/submit while paused. Removing the file resumes
    claims; without this option the journal's pause setting stays authoritative.
    """
    server.timeout = .2
    while not stop():
        if drain_file is not None:
            requested = Path(drain_file).exists()
            if requested != server.dispatcher.paused:
                server.dispatcher.pause(requested)
        server.handle_request()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--drain-file", type=Path, help="Local control file: create to drain; remove to resume")
    parser.add_argument("--ready-file", type=Path, help="Publish only after complete index rebuild and journal replay")
    parser.add_argument("--generation", help="Unique controller-assigned launch generation for --ready-file")
    parser.add_argument("--advertise-host", help="Resolvable TLS certificate hostname for the readiness endpoint")
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error('Both --tls-cert and --tls-key are required')
    if args.host not in ('127.0.0.1', '::1', 'localhost') and not args.tls_cert:
        parser.error('Non-loopback service requires TLS')
    if bool(args.ready_file) != bool(args.generation):
        parser.error('--ready-file and --generation must be supplied together')
    if args.advertise_host and not args.ready_file:
        parser.error('--advertise-host requires --ready-file')
    tls_context = None
    if args.tls_cert:
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
    token = args.token_file.read_text().strip()
    with Dispatcher(args.root, scratch=args.scratch) as dispatcher:
        server = make_server(dispatcher, token=token, host=args.host, port=args.port, tls_context=tls_context)
        ready = None
        try:
            if args.ready_file:
                from .queue_readiness import publish_ready
                host = args.advertise_host or (socket.getfqdn() if args.host in ('0.0.0.0', '::') else args.host)
                ready = publish_ready(args.ready_file, dispatcher, host=host, port=server.server_port,
                                      tls=tls_context is not None, generation=args.generation)
            serve_with_drain(server, drain_file=args.drain_file)
        finally:
            server.server_close()
            if ready:
                from .queue_readiness import remove_owned_ready
                remove_owned_ready(args.ready_file, ready)


if __name__ == "__main__":
    main()

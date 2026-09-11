"""Authenticated opt-in worker RPC. No submission, pause or import RPC exposed.

Default binding is loopback. Cluster networking/TLS or an authenticated tunnel
must be provisioned and validated separately before a production deployment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
import random
import socket
import ssl
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

from .shared_queue import Dispatcher, QueueError, atomic_json, canonical, file_lock


class TransientServiceError(OSError):
    """A retryable unavailable response, not evidence of a revoked lease."""


class Client:
    def __init__(self, url, token, queue_id, *, ca_file=None, timeout=30):
        self.url = url.rstrip("/"); self.token = token; self.queue_id = queue_id
        self.context = ssl.create_default_context(cafile=ca_file) if ca_file else None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be positive and finite")
        self.timeout = timeout

    def call(self, operation, **arguments):
        request = urllib.request.Request(self.url + "/" + operation,
            data=canonical({"queue_id": self.queue_id, **arguments}),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.context) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientServiceError("Dispatcher temporarily unavailable") from exc
            raise QueueError(exc.read().decode()) from exc


def make_server(dispatcher, *, token, host="127.0.0.1", port=0, tls_context=None):
    if len(token) < 32:
        raise ValueError("Use a private random token of at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log bearer tokens or payloads.

        def reply(self, status, value):
            raw = canonical(value)
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_POST(self):
            try:
                self.connection.settimeout(10)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
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
                           "/identity": lambda: {"queue_id": dispatcher.queue_id, "epoch": dispatcher.epoch}}
                if self.path not in methods:
                    self.reply(404, {"error": "Unknown operation"}); return
                self.reply(200, methods[self.path](**value))
            except (QueueError, ValueError, TypeError, KeyError) as exc:
                self.reply(409, {"error": str(exc)})
            except Exception:
                self.reply(503, {"error": "Dispatcher unavailable; retain result and retry"})

    class QueueHTTPServer(ThreadingHTTPServer):
        request_queue_size = 1024
    server = QueueHTTPServer((host, port), Handler)
    if tls_context is not None:
        server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    server.daemon_threads = True
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
                   deadline_seconds=3600):
    """Keep one task in flight. Local spool survives a lost submit response.

    Stop is a drain: finish and acknowledge current model, then stop claiming.
    A lost lease prevents stale submission; a numerical exception becomes an
    explicit failed result. Server lease duration must exceed heartbeat cadence.
    """
    root = Path(spool); root.mkdir(parents=True, exist_ok=True)
    until = time.monotonic() + deadline_seconds
    accepted = 0
    # Recover before claiming: an accepted result whose reply was lost will no
    # longer be returned by claim. The server also checks live/stale lease tokens.
    for receipt in sorted(root.glob("*.json")):
        previous = json.loads(receipt.read_bytes())
        if previous.get("queue_id") != client.queue_id:
            raise QueueError("Result spool belongs to another queue")
        while True:
            try:
                client.call("submit", task_id=receipt.stem, token=previous["token"],
                            worker=worker, result=previous["result"])
                receipt.unlink(); accepted += 1
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= until:
                    return {"state": "unacknowledged", "accepted": accepted, "receipt": str(receipt)}
                time.sleep(idle_seconds)
            except QueueError:
                # Retain evidence. A reclaimed task gets a new token and must be
                # recomputed; an obsolete result may never override its winner.
                break
    while not stop() and time.monotonic() < until:
        try:
            lease = client.call("claim", worker=worker, cached_cells=cached_cells())
        except (OSError, urllib.error.URLError):
            time.sleep(idle_seconds * random.uniform(.8, 1.2)); continue
        if lease["state"] in {"complete", "blocked", "paused"}:
            return {"state": lease["state"], "accepted": accepted}
        if lease["state"] == "wait":
            time.sleep(idle_seconds * random.uniform(.8, 1.2)); continue
        done = threading.Event(); lost = []

        def heartbeat(lease=lease, done=done, lost=lost):
            while not done.wait(heartbeat_seconds):
                try:
                    client.call("heartbeat", task_id=lease["id"], token=lease["token"], worker=worker)
                except QueueError as exc:
                    lost.append(str(exc)); return
                except (OSError, urllib.error.URLError):
                    # A network error is not proof of revoked ownership. The
                    # server's token/expiry check is authoritative at submit.
                    continue

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
                return {"state": "lease_lost", "accepted": accepted, "receipt": str(receipt)}
            while True:
                try:
                    client.call("submit", task_id=lease["id"], token=lease["token"], worker=worker, result=result)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= until:
                        return {"state": "unacknowledged", "accepted": accepted, "receipt": str(receipt)}
                    time.sleep(idle_seconds)
                except QueueError as exc:
                    return {"state": "submission_rejected", "accepted": accepted,
                            "receipt": str(receipt), "error": str(exc)}
            receipt.unlink(); accepted += 1
        finally:
            done.set(); thread.join(timeout=5)
    return {"state": "drained", "accepted": accepted}


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

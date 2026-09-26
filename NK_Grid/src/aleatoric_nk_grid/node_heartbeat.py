"""Node-local batching of actual heartbeat calls, never autonomous renewal.

Each forwarded item has a waiting worker request. No cached lease is periodically
renewed after its worker exits. A relay loss falls back to the original RPC.
"""
import json
import multiprocessing
import os
from pathlib import Path
import queue
import secrets
import socket
import socketserver
import struct
import tempfile
import threading
import time

from .shared_queue import canonical, atomic_json, LeaseLostError, QueueError

LIMIT = 64 * 1024


def _read(sock):
    def exact(n):
        chunks = bytearray()
        while len(chunks) < n:
            part = sock.recv(n - len(chunks))
            if not part:
                raise OSError('Heartbeat relay disconnected')
            chunks.extend(part)
        return bytes(chunks)
    size, = struct.unpack('!I', exact(4))
    if not 0 < size <= LIMIT:
        raise OSError('Heartbeat relay frame too large')
    return json.loads(exact(size))


def _write(sock, value):
    raw = canonical(value)
    if len(raw) > LIMIT:
        raise OSError('Heartbeat relay frame too large')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def _serve(path, url, token, queue_id, ca_file, window, keepalive, stop):
    from .queue_service import Client
    remote = Client(url, token, queue_id, ca_file=ca_file, timeout=10, keepalive=keepalive)
    from .protocol_metrics import write_import_proof
    write_import_proof('heartbeat-relay')
    secret = secrets.token_hex(24)
    pending = queue.Queue(maxsize=256)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(window + 12.)
            try:
                item = _read(self.request)
                if item.pop('secret', None) != secret:
                    raise OSError('Wrong heartbeat relay identity')
                event = threading.Event()
                reply = {}
                pending.put_nowait((item, event, reply))
                if not event.wait(window + 11.):
                    raise OSError('Heartbeat relay timed out')
                _write(self.request, reply)
            except (OSError, ValueError, queue.Full):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        request_queue_size = 128

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(128)
            super().__init__(*args)

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()

    server = Server(('127.0.0.1', 0), Handler)
    accepting = threading.Thread(target=server.serve_forever, daemon=True)
    accepting.start()
    atomic_json(path, {'queue_id': queue_id, 'port': server.server_address[1], 'secret': secret})
    try:
        while not stop.is_set():
            try:
                first = pending.get(timeout=.2)
            except queue.Empty:
                continue
            batch = [first]
            until = time.monotonic() + window
            size = len(canonical(first[0]))
            while len(batch) < 128 and size < 384 * 1024 and not stop.is_set():
                try:
                    # Coalesce already queued calls only. Waiting to fill a
                    # window stalls the legacy worker's heartbeat-thread join
                    # at every cheap-cell boundary.
                    more = pending.get_nowait()
                except queue.Empty:
                    break
                batch.append(more); size += len(canonical(more[0]))
                if time.monotonic() >= until:
                    break
            try:
                response = remote.call('heartbeat_many', hostname=socket.gethostname(),
                                       items=[item[0] for item in batch])
                answers = response['items']
                if len(answers) != len(batch):
                    raise OSError('Heartbeat batch response cardinality changed')
            except Exception:
                answers = [{'retry_direct': True} for _ in batch]
            for (_, event, reply), answer in zip(batch, answers):
                reply.update(answer); event.set()
    finally:
        server.shutdown(); server.server_close(); accepting.join(timeout=2)
        remote.close()


class HeartbeatRelay:
    def __init__(self, client, *, ca_file, window, leader=False, endpoint=None):
        self.client = client
        self.window = min(5., max(.01, window))
        if endpoint is None:
            # This directory is node-local, job/queue-specific and private to uid.
            uid = getattr(os, 'getuid', lambda: 0)()
            folder = Path(tempfile.gettempdir()) / ('nk-heartbeat-%s-%s-%s' %
                (uid, os.environ.get('SLURM_JOB_ID', 'local'), client.queue_id[:20]))
            folder.mkdir(mode=0o700, exist_ok=True)
            if folder.is_symlink() or (hasattr(os, 'getuid') and folder.stat().st_uid != os.getuid()):
                raise QueueError('Unsafe heartbeat relay directory')
            os.chmod(folder, 0o700)
            endpoint = folder / 'endpoint.json'
        self.endpoint = Path(endpoint)
        self.process = self.stop = None
        if leader:
            context = multiprocessing.get_context('spawn')
            self.stop = context.Event()
            self.process = context.Process(target=_serve,
                args=(self.endpoint, client.url, client.token, client.queue_id, ca_file,
                      self.window, client.keepalive, self.stop), daemon=True)
            self.process.start()

    def call(self, arguments):
        try:
            endpoint = json.loads(self.endpoint.read_bytes())
            if endpoint['queue_id'] != self.client.queue_id:
                raise OSError('Wrong heartbeat relay queue')
            with socket.create_connection(('127.0.0.1', endpoint['port']), timeout=.5) as sock:
                sock.settimeout(self.window + 12.)
                _write(sock, {'secret': endpoint['secret'], **arguments})
                answer = _read(sock)
            if answer.get('retry_direct'):
                raise OSError('Heartbeat relay requested fallback')
        except (OSError, ValueError, KeyError):
            self.client.metrics.add('heartbeat_relay_fallback')
            return self.client._call('heartbeat_batch', **arguments)
        if answer.get('code') == 'lease_lost':
            raise LeaseLostError(answer['error'])
        if 'error' in answer:
            raise QueueError(answer['error'])
        self.client.metrics.add('heartbeat_relay_forwarded')
        return answer

    def close(self):
        if self.process is not None:
            self.stop.set(); self.process.join(timeout=1.)
            if self.process.is_alive():
                self.process.terminate(); self.process.join(timeout=2.)
            self.process = None

"""Node-local transparent relay: the workers of one node share a few persistent connections.

Each worker used to open a fresh TLS connection for almost every request (about 0.95
connections per request at production scale), and the handshakes cost the dispatcher
more than half of its CPU. A relay holds a small pool of persistent upstream
connections and forwards the exact requests the workers already send.

The relay is deliberately stateless and deliberately dumb:

* it forwards ``(operation, body, headers)`` and returns ``(status, body)`` unchanged,
  so lease tokens, worker names, error mapping and the 409/503/413 replies keep their
  meaning; the server state is keyed by worker and lease token, never by connection;
* it never caches or prefetches a lease, never renews one on a worker's behalf, never
  acknowledges or replays a submission, and never retries an upstream failure;
* a request is sent direct only when the relay was provably never reached. Once a
  request may have been forwarded, any failure surfaces as ``OSError`` and follows the
  worker's existing idempotent retry path (a claim is not idempotent, so replaying it
  through a second route could lease twice);
* control traffic (claim, heartbeat) and result submission use separate upstream
  connections, so a slow submission cannot delay a heartbeat.

Local transport is length-prefixed frames on 127.0.0.1 guarded by a random secret kept
in a private directory, as in ``node_heartbeat``. The relay runs as a child of the
node's rank-0 worker; if it exits, workers fall back to the direct route.
"""
import hmac
import json
import multiprocessing
import os
from pathlib import Path
import socket
import socketserver
import ssl
import struct
import tempfile
import threading
import time

from .protocol_metrics import ProtocolMetrics
from .queue_transport import PooledTransport
from .shared_queue import QueueError, atomic_json

# Requests are at most 1 MiB (Client limit) and responses 4 MiB (Client read limit).
FRAME_MAX = 5 * 1024 ** 2
HEADER_MAX = 64 * 1024
CONTROL_CONNECTIONS = 1
SUBMIT_CONNECTIONS = 2
MAX_HANDLERS = 256
REPLY_TIMEOUT = 120.
SUBMIT_OPERATIONS = frozenset({'submit', 'submit_batch'})
FORWARDED_OPERATIONS = frozenset({
    'claim', 'claim_batch', 'heartbeat', 'heartbeat_batch', 'heartbeat_many', 'submit', 'submit_batch',
    'protocol', 'worker_binding', 'worker_fault', 'identity', 'stats'})
FORWARDED_HEADERS = ('Authorization', 'Content-Type')


def _exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            raise OSError('Node relay peer closed the connection')
        chunks.extend(part)
    return bytes(chunks)


def send_frame(sock, header, body):
    head = json.dumps(header, separators=(',', ':')).encode()
    if len(head) > HEADER_MAX or len(body) > FRAME_MAX:
        raise OSError('Node relay frame too large')
    sock.sendall(struct.pack('!II', len(head), len(body)) + head + body)


def recv_frame(sock):
    head_size, body_size = struct.unpack('!II', _exact(sock, 8))
    if not 0 < head_size <= HEADER_MAX or body_size > FRAME_MAX:
        raise OSError('Node relay frame too large')
    header = json.loads(_exact(sock, head_size))
    if not isinstance(header, dict):
        raise ValueError('Node relay header is not an object')
    return header, _exact(sock, body_size)


def endpoint_path(queue_id):
    """Private, node-local, job- and queue-specific rendezvous file."""
    uid = getattr(os, 'getuid', lambda: 0)()
    folder = Path(tempfile.gettempdir()) / ('nk-relay-%s-%s-%s' % (uid, os.environ.get('SLURM_JOB_ID', 'local'), queue_id[:20]))
    folder.mkdir(mode=0o700, exist_ok=True)
    if folder.is_symlink() or (hasattr(os, 'getuid') and folder.stat().st_uid != os.getuid()):
        raise QueueError('Unsafe node relay directory')
    os.chmod(folder, 0o700)
    return folder / 'endpoint.json'


def relay_stats_path(spool) -> Path:
    """Where a node's relay publishes its running counters: next to the protocol metrics, one file per node."""
    path = Path(spool).parent / "relay-stats" / (socket.gethostname() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def relay_main(endpoint_file, url, ca_file, queue_id, idle_seconds, stop, stats_file=None,
               control_connections=CONTROL_CONNECTIONS, submit_connections=SUBMIT_CONNECTIONS, stats_interval=30.):
    """Serve until `stop` is set. Runs in its own process.

    With `stats_file` the counters are rewritten atomically every `stats_interval` seconds (and at exit), so a
    running relay can be observed, not just a finished one.
    """
    context = ssl.create_default_context(cafile=ca_file) if str(url).startswith('https') else None
    metrics = ProtocolMetrics(stats_file is not None)
    make = lambda: PooledTransport(url, context=context, timeout=30, metrics=metrics, idle_seconds=idle_seconds)
    pools = {'control': [make() for _ in range(control_connections)],
             'submit': [make() for _ in range(submit_connections)]}
    secret = os.urandom(24).hex()
    counters = {'requests': 0, 'errors': 0, 'rejected': 0, 'by_operation': {}}
    lock = threading.Lock()
    turn = {'control': 0, 'submit': 0}
    slots = threading.BoundedSemaphore(MAX_HANDLERS)

    def choose(channel):
        pool = pools[channel]
        with lock:
            turn[channel] += 1
            start = turn[channel]
        order = [(start + k) % len(pool) for k in range(len(pool))]
        # Prefer a connection nobody is using: a stalled submission then blocks only its own connection.
        return pool[next((i for i in order if not pool[i].locks[channel].locked()), order[0])]

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                self.request.settimeout(60)
                header, body = recv_frame(self.request)
                if not hmac.compare_digest(str(header.get('secret', '')), secret):
                    with lock: counters['rejected'] += 1
                    return
                operation = header.get('operation')
                if operation not in FORWARDED_OPERATIONS:
                    with lock: counters['rejected'] += 1
                    return
                headers = {k: v for k, v in (header.get('headers') or {}).items()
                           if k in FORWARDED_HEADERS and isinstance(v, str)}
                channel = 'submit' if operation in SUBMIT_OPERATIONS else 'control'
                status, reply = choose(channel).request(operation, body, headers)
                send_frame(self.request, {'status': status}, reply)
                with lock:
                    counters['requests'] += 1
                    counters['by_operation'][operation] = counters['by_operation'].get(operation, 0) + 1
            except Exception:
                # A network-facing handler must never take the relay down: any malformed frame,
                # bad header type or upstream failure closes the connection unanswered and the
                # caller follows its retry path.
                with lock: counters['errors'] += 1

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        request_queue_size = 512
        allow_reuse_address = True

        def process_request(self, request, address):
            if not slots.acquire(blocking=False):
                self.shutdown_request(request)         # over capacity: the caller retries
                return
            try:
                super().process_request(request, address)
            except BaseException:
                slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                slots.release()

    server = Server(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cpu0, wall0 = time.process_time(), time.monotonic()
    atomic_json(endpoint_file, {'queue_id': queue_id, 'port': server.server_address[1], 'secret': secret,
                                'pid': os.getpid()})
    def write_stats(final=False):
        counts = metrics.snapshot().get('counts', {})
        with lock:
            snapshot = {**counters, 'by_operation': dict(counters['by_operation'])}
        atomic_json(stats_file, {**snapshot, 'cpu_seconds': time.process_time() - cpu0,
                                 'wall_seconds': time.monotonic() - wall0, 'final': final, 'pid': os.getpid(),
                                 'upstream_tls_handshakes': counts.get('tls_handshake', 0),
                                 'upstream_connections_reused': counts.get('connection_reused', 0)})
    try:
        next_stats = time.monotonic() + stats_interval
        while not stop.wait(min(1.0, stats_interval)):
            if stats_file is not None and time.monotonic() >= next_stats:
                write_stats()
                next_stats = time.monotonic() + stats_interval
    finally:
        server.shutdown()
        server.server_close()
        try:
            Path(endpoint_file).unlink()
        except OSError:
            pass
        if stats_file is not None:
            write_stats(final=True)
        for pool in pools.values():
            for transport in pool:
                transport.close()


class RelayTransport:
    """Same ``request`` contract as ``PooledTransport``; goes through the node relay."""

    def __init__(self, endpoint_file, direct, metrics, queue_id):
        self.endpoint_file, self.direct, self.metrics, self.queue_id = Path(endpoint_file), direct, metrics, queue_id
        self.endpoint = None

    def _connect(self):
        """Return a connected socket, or None if the relay was provably never reached."""
        try:
            if self.endpoint is None:
                endpoint = json.loads(self.endpoint_file.read_bytes())
                if endpoint['queue_id'] != self.queue_id:
                    raise KeyError('queue_id')
                self.endpoint = endpoint
            return socket.create_connection(('127.0.0.1', int(self.endpoint['port'])), timeout=1.0)
        except (OSError, ValueError, KeyError, TypeError):
            self.endpoint = None
            return None

    def request(self, operation, raw, headers):
        sock = self._connect()
        if sock is None:
            self.metrics.add('node_relay_fallback')
            return self.direct.request(operation, raw, headers)
        try:
            with sock:
                sock.settimeout(REPLY_TIMEOUT)
                send_frame(sock, {'secret': self.endpoint['secret'], 'operation': operation,
                                  'headers': {k: v for k, v in headers.items() if k in FORWARDED_HEADERS}}, raw)
                header, body = recv_frame(sock)
            status = header['status']
            if type(status) is not int:
                raise ValueError('Node relay returned no status')
        except (OSError, ValueError, KeyError, struct.error) as exc:
            # The request may already have reached the dispatcher: never replay it another way.
            self.metrics.add('node_relay_error')
            raise OSError('Node relay failed after the request was sent') from exc
        self.metrics.add('node_relay_forwarded')
        return status, body

    def close(self):
        self.direct.close()


class NodeRelay:
    """Owns the relay process on the leader worker; other workers only use it."""

    def __init__(self, client, *, ca_file, idle_seconds, leader, endpoint=None, stats_file=None):
        self.endpoint = Path(endpoint) if endpoint is not None else endpoint_path(client.queue_id)
        self.process = self.stop = None
        if leader:
            context = multiprocessing.get_context('spawn')
            self.stop = context.Event()
            self.process = context.Process(
                target=relay_main, daemon=True,
                args=(str(self.endpoint), client.url, None if ca_file is None else str(ca_file), client.queue_id,
                      float(idle_seconds), self.stop, None if stats_file is None else str(stats_file)))
            self.process.start()

    def close(self):
        if self.process is not None:
            self.stop.set()
            self.process.join(timeout=3.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.)
            self.process = None


def attach_node_relay(client, *, ca_file, idle_seconds=.5, leader=False, endpoint=None, stats_file=None):
    """Route `client` through the node relay. The relay never being reachable means direct."""
    if not 0 < float(idle_seconds) < 30:
        raise QueueError('Node relay idle limit must be within 0..30 seconds')
    direct = client.transport
    if direct is None:
        direct = PooledTransport(client.url, context=client.context, timeout=client.timeout,
                                 metrics=client.metrics, idle_seconds=0.)
    relay = NodeRelay(client, ca_file=ca_file, idle_seconds=idle_seconds, leader=leader,
                      endpoint=endpoint, stats_file=stats_file)
    client.node_relay = relay
    client.transport = RelayTransport(relay.endpoint, direct, client.metrics, client.queue_id)
    return relay

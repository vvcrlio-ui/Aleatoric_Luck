"""Bounded HTTP/1.1 channels, with separate control and submission locks.

No transparent replay: a broken connection is returned to the caller's existing
idempotent retry path. Connections expire before the server idle deadline.
"""
import http.client
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit


class _HTTP(http.client.HTTPConnection):
    def __init__(self, *args, metrics, **kwargs):
        self.metrics = metrics
        super().__init__(*args, **kwargs)

    def connect(self):
        with self.metrics.span('tcp_connect'):
            super().connect()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class _HTTPS(_HTTP):
    def __init__(self, *args, context, **kwargs):
        self.context = context or ssl.create_default_context()
        super().__init__(*args, **kwargs)

    def connect(self):
        super().connect()
        try:
            with self.metrics.span('tls_handshake'):
                self.sock = self.context.wrap_socket(self.sock, server_hostname=self.host)
        except BaseException:
            self.close()
            raise


class PooledTransport:
    def __init__(self, url, *, context, timeout, metrics, idle_seconds=.5):
        self.url = urlsplit(url)
        if self.url.scheme not in ('http', 'https') or not self.url.hostname or self.url.username:
            raise ValueError('Expected an HTTP(S) dispatcher URL without user info')
        self.context, self.timeout, self.metrics = context, timeout, metrics
        self.idle_seconds = idle_seconds
        self.locks = {k: threading.Lock() for k in ('control', 'submit')}
        self.connections = {}
        self.closed = False

    def request(self, operation, raw, headers):
        channel = 'submit' if operation in ('submit', 'submit_batch') else 'control'
        with self.metrics.span('connection_pool_wait_' + channel):
            self.locks[channel].acquire()
        try:
            if self.closed:
                raise OSError('Dispatcher transport closed')
            old = self.connections.pop(channel, None)
            connection = None
            if old:
                connection, last_used = old
                if time.monotonic() - last_used > self.idle_seconds:
                    connection.close(); connection = None
            if connection is None:
                common = dict(timeout=self.timeout, metrics=self.metrics)
                port = self.url.port or (443 if self.url.scheme == 'https' else 80)
                connection = (_HTTPS(self.url.hostname, port, context=self.context, **common)
                              if self.url.scheme == 'https' else _HTTP(self.url.hostname, port, **common))
            else:
                self.metrics.add('connection_reused')
            try:
                path = self.url.path.rstrip('/') + '/' + operation
                if not self.idle_seconds:
                    headers = {**headers, 'Connection': 'close'}
                connection.request('POST', path, body=raw, headers=headers)
                response = connection.getresponse()
                # The server emits bounded JSON, not an arbitrary streaming body.
                body = response.read(4 * 1024**2 + 1)
                if len(body) > 4 * 1024**2:
                    raise OSError('Dispatcher response exceeds limit')
                if response.will_close:
                    connection.close()
                else:
                    self.connections[channel] = (connection, time.monotonic())
                return response.status, body
            except http.client.HTTPException as exc:
                connection.close()
                raise OSError('Dispatcher connection failed') from exc
            except BaseException:
                connection.close()
                raise
        finally:
            self.locks[channel].release()

    def close(self):
        for channel, lock in self.locks.items():
            with lock:
                self.closed = True
                old = self.connections.pop(channel, None)
                if old:
                    old[0].close()

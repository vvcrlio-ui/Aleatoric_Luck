"""Opt-in, bounded operational counters; never part of scientific rows."""
from collections import defaultdict
from contextlib import contextmanager
import threading
import time


def write_import_proof(role):
    """Optional deployment evidence from modules actually loaded in this process."""
    import json
    import os
    from pathlib import Path
    import sys
    import socket
    folder = os.environ.get('NKGRID_IMPORT_AUDIT_DIR')
    if not folder:
        return
    root = Path(folder); root.mkdir(parents=True, exist_ok=True)
    origins = {name: str(Path(module.__file__).resolve()) for name, module in tuple(sys.modules.items())
               if name.startswith('aleatoric_nk_grid.') and getattr(module, '__file__', None)}
    hostname = socket.gethostname()
    value = {'pid': os.getpid(), 'hostname': hostname, 'role': role, 'origins': origins,
             'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
             'revision': os.environ.get('NKGRID_OPERATIONAL_REVISION')}
    (root / ('%s-%s-%s.json' % (role, hostname, os.getpid()))).write_text(json.dumps(value, indent=2), encoding='utf-8')


class ProtocolMetrics:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.windows = {}
        self.event_times = {}

    def add(self, name, seconds=0., count=1):
        if not self.enabled:
            return
        epoch = int(time.time())
        with self.lock:
            self.totals[name] += max(0., seconds)
            self.counts[name] += count
            if count:
                elapsed = time.monotonic() - self.started
                span = self.event_times.setdefault(name, {'first_elapsed': elapsed, 'last_elapsed': elapsed})
                span['last_elapsed'] = elapsed
            bucket = self.windows.setdefault(epoch, defaultdict(int))
            bucket[name] += count
            for old in tuple(self.windows):
                if old < epoch - 119:
                    del self.windows[old]

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        start = time.monotonic()
        cpu = time.thread_time()
        try:
            yield
        finally:
            self.add(name, time.monotonic() - start)
            self.add(name + '_thread_cpu', time.thread_time() - cpu)

    def snapshot(self):
        if not self.enabled:
            return {}
        with self.lock:
            import os
            import socket
            return {'observed_epoch': time.time(), 'pid': os.getpid(), 'hostname': socket.gethostname(),
                    'rank': os.environ.get('SLURM_PROCID'), 'local_rank': os.environ.get('SLURM_LOCALID'),
                    'wall_seconds': time.monotonic() - self.started,
                    'process_cpu_seconds': time.process_time() - self.cpu_started,
                    'seconds': dict(self.totals), 'counts': dict(self.counts),
                    'event_times': {k: dict(v) for k, v in self.event_times.items()},
                    'one_second_counts': {str(k): dict(v) for k, v in sorted(self.windows.items())},
                    'note': 'Background RPC spans overlap compute; do not sum them as idle time.'}


class MetricsPublisher:
    """Low-frequency sidecar outside the immutable receipt spool."""
    def __init__(self, metrics, path, interval=30.):
        self.metrics, self.path, self.interval = metrics, path, interval
        self.stop = threading.Event()
        self.thread = None

    def publish(self):
        import json
        from .shared_queue import atomic_json
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            value = self.metrics.snapshot()
            # Low-frequency history retains earlier rate windows. It is
            # best-effort telemetry, not a scientific durability receipt.
            with self.path.with_suffix('.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, separators=(',', ':')) + '\n')
            atomic_json(self.path, value)
        except OSError:
            self.metrics.add('telemetry_write_error')

    def __enter__(self):
        if self.metrics.enabled:
            def loop():
                while not self.stop.wait(self.interval):
                    self.publish()
            self.thread = threading.Thread(target=loop, name='protocol-metrics', daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        if self.thread is not None:
            self.stop.set(); self.thread.join()
            self.publish()

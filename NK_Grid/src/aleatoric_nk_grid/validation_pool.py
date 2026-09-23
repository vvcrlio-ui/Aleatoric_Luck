"""Bounded spawn pool for read-only scientific validation; parent owns commits."""
from concurrent.futures import TimeoutError
import importlib
import multiprocessing
import os
import threading
import time
from .shared_queue import QueueError

_identity = _validator = _design = _warm_barrier = None


def _initialize(identity, cpu_ids=None, ordinal=None, barrier=None, fast_reads=False):
    global _identity, _validator, _design, _warm_barrier
    _warm_barrier = barrier
    if cpu_ids is not None:
        with ordinal.get_lock():
            index = ordinal.value; ordinal.value += 1
        if index >= len(cpu_ids):
            raise QueueError('Unexpected replacement validator; stop before changing affinity')
        os.sched_setaffinity(0, {cpu_ids[index]})
        if os.sched_getaffinity(0) != {cpu_ids[index]}:
            raise QueueError('Validator CPU binding failed')
    from .prediction_workflow import ResultCacheValidator, design_for
    _identity = identity
    _validator = ResultCacheValidator(identity, fast_reads=fast_reads)
    _design = design_for(identity)
    from .protocol_metrics import write_import_proof
    write_import_proof('validator')


def _warm():
    _warm_barrier.wait(timeout=90.)
    return {'pid': os.getpid(), 'cpus': sorted(os.sched_getaffinity(0))}


def validate_rows(identity, validator, design, results):
    from .result_migration import validate_scientific_result
    from .prediction_workflow import task_kind
    for row in results:
        if row.get('status') == 'failed':
            continue
        validate_scientific_result(row, task_kind=task_kind(identity, row))
        spec = design.panels[row['panel_id']][2]['cell_spec'] if hasattr(design, 'panels') else identity['cell_spec']
        if row.get('algorithm_version') != spec['algorithm_version']:
            raise QueueError('Scientific identity changed')
        validator(row)


def _validate_encoded(payloads):
    """Same as _validate, for rows already serialized by the dispatcher (it needs those bytes for the journal anyway)."""
    import json
    started, cpu = time.monotonic(), time.process_time()
    results = [json.loads(payload) for payload in payloads]
    validate_rows(_identity, _validator, _design, results)
    return {'records': len(results), 'cpu_seconds': time.process_time() - cpu,
            'wall_seconds': time.monotonic() - started,
            'fast_reads': getattr(_validator, 'context', None) is not None}


def _validate(results):
    started, cpu = time.monotonic(), time.process_time()
    validate_rows(_identity, _validator, _design, results)
    return {'records': len(results), 'cpu_seconds': time.process_time() - cpu,
            'wall_seconds': time.monotonic() - started,
            'fast_reads': getattr(_validator, 'context', None) is not None}


class ValidationPool:
    def __init__(self, identity, workers, timeout=120., cpu_ids=None, fast_reads=False):
        self.timeout = timeout
        self.stats_lock = threading.Lock()
        self.stats = {'successful_batches': 0, 'records': 0, 'cpu_seconds': 0., 'wall_seconds': 0., 'fast_reads': False}
        self.slots = threading.BoundedSemaphore(2 * workers)
        # Resolve at construction: controller isolation tests restore
        # sys.modules after mocks, which can invalidate an eagerly held pool
        # class and its spawn target's pickle identity.
        process_module = importlib.import_module('concurrent.futures.process')
        self.broken_pool_error = process_module.BrokenProcessPool
        context = multiprocessing.get_context('spawn')
        ordinal = context.Value('i', 0) if cpu_ids is not None else None
        barrier = context.Barrier(workers) if cpu_ids is not None else None
        if cpu_ids is not None and (len(cpu_ids) != workers or len(set(cpu_ids)) != workers):
            raise QueueError('Validator CPU reservation differs from pool size')
        self.pool = process_module.ProcessPoolExecutor(max_workers=workers,
            mp_context=context, initializer=_initialize, initargs=(identity, cpu_ids, ordinal, barrier, bool(fast_reads)))
        self.failed = False
        self.binding_proof = []
        if cpu_ids is not None:
            try:
                futures = [self.pool.submit(_warm) for _ in cpu_ids]
                self.binding_proof = [f.result(timeout=100.) for f in futures]
                if {r['cpus'][0] for r in self.binding_proof} != set(cpu_ids):
                    raise QueueError('Validator CPU proof mismatch')
            except BaseException:
                self.failed = True; self.close()
                raise

    def validate(self, results):
        return self._run(_validate, results)

    def validate_encoded(self, payloads):
        return self._run(_validate_encoded, payloads)

    def _run(self, function, argument):
        if self.failed:
            raise OSError('Validation pool unavailable')
        if not self.slots.acquire(timeout=self.timeout):
            raise OSError('Validation queue busy')
        future = None
        try:
            future = self.pool.submit(function, argument)
            # Keep the slot until actual completion, even when the caller times out.
            future.add_done_callback(lambda _: self.slots.release())
            result = future.result(timeout=self.timeout)
            with self.stats_lock:
                self.stats['successful_batches'] += 1
                for name in ('records', 'cpu_seconds', 'wall_seconds'):
                    self.stats[name] += result[name]
                self.stats['fast_reads'] = bool(result.get('fast_reads'))       # what the child actually runs
            return True
        except (TimeoutError, self.broken_pool_error) as exc:
            self.failed = True
            raise OSError('Validation timeout; no result accepted') from exc
        finally:
            if future is None:
                self.slots.release()

    def snapshot(self):
        with self.stats_lock:
            return {**self.stats, 'scope': 'Successful validator calls only; excludes process startup.'}

    def close(self):
        if self.failed:
            # Python 3.12 lacks public terminate_workers(). Bound shutdown of
            # a failed/hung pool; its unvalidated rows were never accepted.
            processes = list((getattr(self.pool, '_processes', None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=2.)
                if process.is_alive():
                    process.kill(); process.join(timeout=2.)
            self.pool.shutdown(wait=False, cancel_futures=True)
        else:
            self.pool.shutdown(wait=True, cancel_futures=True)

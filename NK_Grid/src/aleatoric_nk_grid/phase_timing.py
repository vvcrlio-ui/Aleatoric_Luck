"""Stage timing goes to stderr, never into authoritative recovery artifacts."""
import functools
import json
import sys
import time


def timed_phase(name):
    def decorate(function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            status = 'failed'
            try:
                result = function(*args, **kwargs)
                status = 'ok'
                return result
            finally:
                print(json.dumps({'nkgrid_phase': name, 'seconds': time.perf_counter()-start,
                                  'status': status}), file=sys.stderr, flush=True)
        return measured
    return decorate

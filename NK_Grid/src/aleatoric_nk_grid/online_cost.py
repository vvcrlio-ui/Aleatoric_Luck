"""Bounded, conservative batch prices learned only after durable result acceptance.

Each generation/shard publishes cumulative counts and bounded histograms, never task
payloads or a replacement scientific contract. Peer snapshots replace, not add to,
the last snapshot from that shard. Empirical p99 is rounded upwards (at most 9.1%);
observed long tasks remain single-task leases. Neither is a future fit-time guarantee.
"""
import json
import math
from pathlib import Path
import threading
import time

from .cost_profile import observation_kind, positive_integer, positive_number
from .shared_queue import atomic_json, digest

FORMAT = 'online-cold-cost-histogram-v2'
MAX_GROUPS = 20000
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


def _bucket(seconds):
    seconds = max(seconds, 2.**-20)
    index = math.ceil(8 * math.log2(seconds))
    if 2.**(index / 8) < seconds: index += 1
    return index


class OnlineCosts:
    def __init__(self, queue_id, *, source=0, sources=1, minimum=100,
                 safety_factor=1.25, refresh_seconds=15., max_groups=MAX_GROUPS, long_task_seconds=30.):
        self.queue_id, self.source, self.sources = queue_id, source, sources
        self.minimum, self.safety_factor = minimum, safety_factor
        self.refresh_seconds, self.max_groups = refresh_seconds, max_groups
        self.long_task_seconds = long_task_seconds
        self.lock = threading.Lock()
        self.groups = {}; self.prices = {}; self.identity_keys = {}
        self.accepted = self.used = self.ignored = self.errors = self.dropped_groups = 0
        self.peer_accepted = {}; self.directory = None; self.last_refresh = float('-inf')
        self.disabled = False

    def _identity(self, identity):
        # Workflow identities contain scalar fields; canonicalize once per pipeline.
        canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if canonical not in self.identity_keys:
            self.identity_keys[canonical] = digest(identity)
        return self.identity_keys[canonical]

    def observe(self, row, expected_identity):
        """Caller must invoke once for a newly fsynced result, never on replay."""
        with self.lock:
            self.accepted += 1
            if self.disabled:
                self.ignored += 1; return
            if (row.get('status') != 'ok' or row.get('error')
                    or row.get('_cost_identity') != expected_identity
                    or observation_kind(row, expected_identity) not in ('cold', 'legacy')):
                self.ignored += 1; return
            n, k = positive_integer(row.get('N')), positive_integer(row.get('K'))
            width = positive_integer(row.get('K_expanded', k))
            model = row.get('model')
            if not n or not k or not width or not isinstance(model, str) or not model:
                self.ignored += 1; return
            key = (self._identity(expected_identity), model, n, k, width)
            base = key[:4]
            if key not in self.groups:
                if len(self.groups) >= self.max_groups:
                    self.dropped_groups += 1; self.disabled = True; self.prices = {}; return
                self.groups[key] = [0, 0, 0., {}]  # rows, timings, maximum, upper-rounded histogram
                self.prices.pop(base, None)   # a newly observed width must also be measured
            group = self.groups[key]; group[0] += 1
            seconds = positive_number(row.get('_worker_wall_seconds'))
            if (seconds is None or not math.isfinite(seconds * self.safety_factor)
                    or seconds > 2.**20):
                self.ignored += 1; self.prices.pop(base, None); return
            group[1] += 1; group[2] = max(group[2], seconds); self.used += 1
            bucket = _bucket(seconds); group[3][bucket] = group[3].get(bucket, 0) + 1
            if base in self.prices:
                self.prices[base] = max(self.prices[base], seconds * self.safety_factor)

    def price(self, model, n, k, identity):
        with self.lock:
            return self.prices.get((self._identity(identity), model, int(n), int(k)))

    def _snapshot(self, directory):
        with self.lock:
            return {'format': FORMAT, 'queue_id': self.queue_id, 'generation_directory': str(directory),
                    'source': self.source, 'sources': self.sources, 'accepted': self.accepted,
                    'groups': [[*key, *values[:3], sorted(values[3].items())]
                               for key, values in sorted(self.groups.items())]}

    def _checked(self, value, source, directory):
        if (not isinstance(value, dict) or value.get('format') != FORMAT
                or value.get('queue_id') != self.queue_id
                or value.get('generation_directory') != str(directory)
                or type(value.get('source')) is not int or value['source'] != source
                or type(value.get('sources')) is not int or value['sources'] != self.sources
                or type(value.get('accepted')) is not int or value['accepted'] < 0
                or value['accepted'] < self.peer_accepted.get(source, 0)
                or not isinstance(value.get('groups'), list) or len(value['groups']) > self.max_groups):
            raise ValueError('Online timing snapshot identity/counter mismatch')
        seen = set(); rows = 0
        for group in value['groups']:
            if not isinstance(group, list) or len(group) != 9:
                raise ValueError('Malformed online timing group')
            ident, model, n, k, width, count, complete, maximum, histogram = group
            if (not isinstance(ident, str) or len(ident) != 64
                    or any(c not in '0123456789abcdef' for c in ident)
                    or not isinstance(model, str) or not 0 < len(model) <= 128
                    or any(type(x) is not int or x <= 0 for x in (n, k, width, count))
                    or type(complete) is not int or not 0 <= complete <= count
                    or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                    or not math.isfinite(maximum) or maximum < 0 or (complete > 0) != (maximum > 0)):
                raise ValueError('Invalid online timing statistics')
            key = tuple(group[:5])
            if key in seen: raise ValueError('Duplicate online timing group')
            seen.add(key); rows += count
            if not isinstance(histogram, list) or len(histogram) > 321:
                raise ValueError('Invalid online histogram')
            bins = {}
            for entry in histogram:
                if (not isinstance(entry, list) or len(entry) != 2
                        or type(entry[0]) is not int or not -160 <= entry[0] <= 160
                        or type(entry[1]) is not int or entry[1] <= 0 or entry[0] in bins):
                    raise ValueError('Malformed online histogram bin')
                bins[entry[0]] = entry[1]
            if sum(bins.values()) != complete or (complete and _bucket(maximum) != max(bins)):
                raise ValueError('Online histogram does not match its observations')
        if rows > value['accepted']: raise ValueError('Online observations exceed accepted results')
        return value

    def refresh(self, directory, *, force=False):
        """Low-frequency shared summaries; all failures fall back to unpriced tasks."""
        now = time.monotonic()
        if not force and now - self.last_refresh < self.refresh_seconds: return
        self.last_refresh = now
        directory = Path(directory).resolve()
        if self.directory is not None and directory != self.directory:
            raise ValueError('Online timing generation directory changed')
        self.directory = directory
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            local = self._snapshot(directory)
            if len(json.dumps(local, separators=(',', ':')).encode()) > MAX_SNAPSHOT_BYTES:
                raise ValueError('Online timing snapshot is oversized')
            atomic_json(directory / ('s%d.json' % self.source), local)
            snapshots = [local]
            for source in range(self.sources):
                if source == self.source: continue
                path = directory / ('s%d.json' % source)
                try:
                    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                        raise ValueError('Online timing snapshot is oversized')
                    value = self._checked(json.loads(path.read_bytes()), source, directory)
                    snapshots.append(value)
                    self.peer_accepted[source] = value['accepted']
                except FileNotFoundError:
                    pass
                except (OSError, ValueError, TypeError):
                    with self.lock: self.errors += 1
            # A missing/rejected shard cannot silently change the observed population.
            if len(snapshots) != self.sources:
                with self.lock: self.prices = {}
                return
            merged = {}
            for snapshot in snapshots[1:]:
                for group in snapshot['groups']:
                    key = tuple(group[:5]); value = merged.setdefault(key, [0, 0, 0., {}])
                    value[0] += group[5]; value[1] += group[6]; value[2] = max(value[2], group[7])
                    for index, count in group[8]:value[3][index] = value[3].get(index, 0) + count
            with self.lock:
                # Include local observations that arrived while files were read.
                for key, group in self.groups.items():
                    value = merged.setdefault(key, [0, 0, 0., {}])
                    value[0] += group[0]; value[1] += group[1]; value[2] = max(value[2], group[2])
                    for index,count in group[3].items():value[3][index] = value[3].get(index, 0) + count
                prices = {}; incomplete = set()
                for key, (rows, count, maximum, histogram) in merged.items():
                    base = key[:4]
                    if rows != count or count < self.minimum:
                        incomplete.add(base); continue
                    rank = math.ceil(.99 * count); seen = 0
                    for index, amount in sorted(histogram.items()):
                        seen += amount
                        if seen >= rank:
                            quantile = 2.**(index / 8); break
                    # Rare known long fits must not be packed into short batches.
                    if maximum >= self.long_task_seconds:quantile = max(quantile, maximum)
                    prices[base] = max(prices.get(base, 0.), quantile * self.safety_factor)
                for key in incomplete:prices.pop(key,None)
                self.prices = {} if self.disabled else {k:v for k,v in prices.items() if math.isfinite(v)}
        except (OSError, ValueError, TypeError):
            with self.lock:
                self.errors += 1; self.prices = {}

    def stats(self):
        with self.lock:
            return {'accepted_seen': self.accepted, 'cold_timings': self.used, 'ignored': self.ignored,
                    'local_groups': len(self.groups), 'priced_groups': len(self.prices),
                    'errors': self.errors, 'dropped_groups': self.dropped_groups,
                    'disabled': self.disabled,
                    'bound': 'upper-rounded empirical p99 with long-fit maximum guard; not a future runtime guarantee'}

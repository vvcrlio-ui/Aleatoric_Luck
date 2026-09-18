"""Streaming timing evidence for scheduling; never changes scientific identity.

Means use all valid observations. Quantiles use nearest ranks of deterministic
uniform reservoirs of float32 values, rounded upwards. Each status/source holds
at most reservoir_size values per (model,N,original K,expanded K). Below the cap,
only float32 rounding (<1.2e-7 relative) remains; above it, the reported DKW bound
is a per-reservoir 99% confidence rank error, not a seconds error or an upper
bound on future runtimes. Sorting uses one bounded reservoir-sized Python list.
"""
from __future__ import annotations
from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct

PROFILE_FORMAT = 'model-cell-cost-v2'
TIMING_SOURCES = ('_worker_wall_seconds', '_cell_wall_seconds')
DEFAULT_RESERVOIR_SIZE = 8192
DEFAULT_MIN_BATCH_OBSERVATIONS = 100


def observation_kind(result, identity):
    """Classify actual work without changing scientific status or task identity.

    Cache hits and resumed work remain useful diagnostics but cannot lower the
    price of an uncached task. Old workflow rows need positive cold-work proof;
    untagged legacy non-workflow profiles retain their historical behavior.
    """
    if result.get('_prediction_cache_hit') is True:
        return 'cache_hit'
    if identity is None:
        return 'legacy'
    if result.get('_prediction_cache_hit') is not False:
        return 'unknown_workflow'
    phase = identity.get('phase')
    if phase == 'base':
        folds = positive_integer(identity.get('oof_folds'))
        if folds is None:
            return 'unknown_workflow'
        expected = folds + 1
        if result.get('oof_complete') is False:
            return 'incomplete_oof'
        if result.get('oof_complete') is not True:
            return 'unknown_workflow'
        declared = result.get('_expected_base_fit_count', expected)
        count = result.get('_base_fit_count')
        resumed = result.get('_resumed_fit_count', 0)
        if (type(count) is not int or count < 0 or type(resumed) is not int or resumed < 0
                or type(declared) is not int or declared != expected):
            return 'unknown_workflow'
        if resumed or count < expected:
            return 'partial_resume'
        return 'cold' if count == expected else 'unknown_workflow'
    if phase == 'sl':
        # A genuine combiner fit has zero base fits by design. That zero alone
        # cannot prove cold SL work; cached meta-results also report zero.
        if (type(result.get('_combiner_fit_count')) is int and result['_combiner_fit_count'] == 1
                and type(result.get('_base_fit_count')) is int and result['_base_fit_count'] == 0
                and result.get('oof_complete') is True):
            return 'cold'
        return 'unknown_workflow'
    return 'unknown_workflow'


def positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def positive_integer(value):
    number = positive_number(value)
    return int(number) if number is not None and number.is_integer() else None


def _float32_up(value):
    try:
        packed = struct.pack('!f', value)
    except (OverflowError, struct.error):
        return None
    rounded = struct.unpack('!f', packed)[0]
    if not math.isfinite(rounded) or rounded <= 0:
        return None
    if rounded < value:
        bits = struct.unpack('!I', packed)[0]
        rounded = struct.unpack('!f', struct.pack('!I', bits + 1))[0]
    return rounded if math.isfinite(rounded) else None


class _Observations:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.values = array('f')
        self.count = 0
        self.mean = 0.0
        self.maximum = 0.0
        self.rng = random.Random(seed)

    def add(self, value):
        rounded = _float32_up(value)
        if rounded is None:
            return False
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.maximum = max(self.maximum, value)
        if len(self.values) < self.capacity:
            self.values.append(rounded)
        else:
            index = self.rng.randrange(self.count)
            if index < self.capacity:
                self.values[index] = rounded
        return True

    def summary(self):
        ordered = sorted(self.values)
        quantiles = {f'p{p}': ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]
                     for p in (50, 90, 99)}
        sampled = self.count > len(ordered)
        return dict(observations=self.count, mean_seconds=self.mean,
                    max_seconds=max(self.maximum, quantiles['p99']), **quantiles,
                    quantile_samples=len(ordered), quantiles_exact=not sampled,
                    rank_error_99confidence=(min(1.0, math.sqrt(math.log(200) / (2 * len(ordered))))
                                             if sampled else 0.0))


class CostProfileBuilder:
    """Bounded accumulator with failed/skipped/retry timing separated.

    Only status ok without error prices successful work. Internal retries remain
    included in elapsed wall time; without retry metadata their separate cost
    cannot be reconstructed. Explicit retry rows stay in separate status groups.
    """
    def __init__(self, *, reservoir_size=DEFAULT_RESERVOIR_SIZE, max_groups=100000, seed=0):
        if positive_integer(reservoir_size) is None or positive_integer(max_groups) is None:
            raise ValueError('reservoir_size and max_groups must be positive integers')
        self.reservoir_size, self.max_groups = int(reservoir_size), int(max_groups)
        self.seed = seed
        self.groups = {}
        self.rows_seen = self.rows_used = self.rows_skipped = 0
        self.status_counts = Counter()
        self.observation_counts = Counter()

    @property
    def retained_values(self):
        return sum(len(values.values) for group in self.groups.values()
                   for values in group['sources'].values())

    def add(self, result):
        self.rows_seen += 1
        if not isinstance(result, dict):
            self.rows_skipped += 1
            return False
        model = result.get('model')
        n, k = positive_integer(result.get('N')), positive_integer(result.get('K'))
        width = positive_integer(result.get('K_expanded', k))
        status = result.get('status', 'unknown')
        if not isinstance(status, str) or not status or len(status) > 64:
            status = 'unknown'
        if status == 'ok' and result.get('error'):
            status = 'error'
        self.status_counts[status] += 1
        if not isinstance(model, str) or not model or not n or not k or not width:
            self.rows_skipped += 1
            return False
        identity = result.get('_cost_identity')
        if identity is not None and (not isinstance(identity, dict) or not identity):
            raise ValueError('Workflow cost identity must be a nonempty mapping')
        identity_key = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False) if identity is not None else ''
        kind = observation_kind(result, identity)
        self.observation_counts[kind] += 1
        key = (model, n, k, width, status, identity_key, kind)
        if key not in self.groups:
            if len(self.groups) >= self.max_groups:
                raise ValueError('Cost profile exceeds max_groups; check journal grouping')
            self.groups[key] = {'rows': 0, 'sources': {}}
        group = self.groups[key]
        group['rows'] += 1
        used = False
        for field in TIMING_SOURCES:
            value = positive_number(result.get(field))
            if value is None or _float32_up(value) is None:
                continue
            if field not in group['sources']:
                seed = hashlib.sha256(repr((self.seed, key, field)).encode()).digest()
                group['sources'][field] = _Observations(self.reservoir_size, seed)
            used = group['sources'][field].add(value) or used
        self.rows_used += int(used)
        self.rows_skipped += int(not used)
        return used

    def profile(self, *, evidence=None):
        samples, status_groups, widths = [], [], {}
        for (model, n, k, width, status, identity_key, kind), group in sorted(self.groups.items()):
            base = dict(model=model, N=n, K=k, K_expanded=width, status=status,
                        observation_kind=kind, rows=group['rows'])
            if identity_key:
                base['cost_identity'] = json.loads(identity_key)
            sources = {field: values.summary() for field, values in group['sources'].items()}
            status_groups.append({**base, 'timings': sources})
            if status != 'ok' or not sources or kind not in ('cold', 'legacy'):
                continue
            complete = [field for field in TIMING_SOURCES if field in sources and
                        sources[field]['observations'] == group['rows']]
            # Never combine timings from different sources into one distribution.
            source = (complete or [field for field in TIMING_SOURCES if field in sources])[0]
            sample = {**base, **sources[source], 'timing_source': source,
                      'timing_complete': source in complete}
            sample['seconds'] = sample['mean_seconds']
            samples.append(sample)
            widths.setdefault(str(k), set()).add(width)
        metadata = dict(evidence or {})
        metadata.update(rows_seen=self.rows_seen, rows_used=self.rows_used,
            rows_skipped=self.rows_skipped, groups=len(samples), status_counts=dict(self.status_counts),
            observation_counts=dict(self.observation_counts), workflow_observation_filter='cold-only-v1',
            measured='successful cold workflow or legacy elapsed wall seconds, one timing source per group',
            internal_retry_accounting='included in elapsed time; no separate attribution without journal metadata',
            sampling={'method': 'uniform-reservoir-float32-up',
                'capacity_per_status_source_group': self.reservoir_size, 'max_status_groups': self.max_groups,
                'quantile_method': 'nearest-rank', 'seed': self.seed,
                'retained_float32_bytes': self.retained_values * array('f').itemsize,
                'sort_peak_note': 'one reservoir-sized list of Python floats; object/group overhead additional',
                'error_note': 'rank bounds are per group at 99% confidence, not family-wise; no elapsed-time hard bound'})
        return dict(format=PROFILE_FORMAT, evidence=metadata,
            expanded_by_k={k: next(iter(v)) for k, v in widths.items() if len(v) == 1},
            samples=samples, status_groups=status_groups)


def build(source, stride=1, limit=None, *, reservoir_size=DEFAULT_RESERVOIR_SIZE,
          max_groups=100000, seed=0):
    """Stream a result journal, ignoring and reporting an unfinished last line.

    Legacy stride is periodic sampling with no random-sample guarantee; such
    profiles cannot authorize batch prices. Limit counts all valid timed rows,
    including failures whose timings are stored separately from successes.
    """
    if positive_integer(stride) is None or (limit is not None and positive_integer(limit) is None):
        raise ValueError('stride and limit must be positive integers')
    builder = CostProfileBuilder(reservoir_size=reservoir_size, max_groups=max_groups, seed=seed)
    path, rows, incomplete_tail = Path(source), 0, False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for index, line in enumerate(handle):
            if not line.endswith(b'\n'):
                incomplete_tail = True
                break
            digest.update(line)
            rows += 1
            if index % stride:
                continue
            try:
                record = json.loads(line)
                result = record.get('result') if isinstance(record, dict) else None
            except (ValueError, TypeError):
                result = None
            builder.add(result)
            if limit is not None and builder.rows_used >= limit:
                break
    profile = builder.profile(evidence=dict(source=str(path.resolve()), source_bytes=path.stat().st_size,
        scanned_prefix_sha256=digest.hexdigest(), journal_rows_seen=rows, stride=int(stride),
        limit=limit, incomplete_tail=incomplete_tail))
    if not profile['samples']:
        raise ValueError('No successful cold/legacy timed results found; hits, resumes, failed/skipped rows cannot price cold work')
    return profile


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reservoir-size', type=int, default=DEFAULT_RESERVOIR_SIZE)
    parser.add_argument('--max-groups', type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        profile = build(args.source, args.stride, args.limit,
                        reservoir_size=args.reservoir_size, max_groups=args.max_groups)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(profile, sort_keys=True, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    evidence = profile['evidence']
    seconds = [sample['seconds'] for sample in profile['samples']]
    print(json.dumps(dict(groups=evidence['groups'], rows_used=evidence['rows_used'],
        rows_skipped=evidence['rows_skipped'], fastest_group_seconds=min(seconds),
        slowest_group_seconds=max(seconds), output=str(args.output),
        retained_float32_bytes=evidence['sampling']['retained_float32_bytes'])))


if __name__ == '__main__':
    main()

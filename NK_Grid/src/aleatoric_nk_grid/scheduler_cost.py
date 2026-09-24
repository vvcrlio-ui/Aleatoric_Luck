"""Mean work estimates and separately validated conservative batch prices.

Nearest measured log-size cell anchors a positive N*sqrt(K_expanded) scaling.
This is a transparent initial estimate, not a calibrated production-time claim.
All workers can claim all classes. No estimate changes numerical training.
"""
from __future__ import annotations
import math
from collections import Counter
from .cost_profile import (PROFILE_FORMAT, TIMING_SOURCES,
                           DEFAULT_MIN_BATCH_OBSERVATIONS, positive_integer, positive_number)
from .shared_queue import QueueError

CATEGORY = {'super_learner': 'sl', 'shallow_neural_network': 'nn',
            'xgboost': 'boost', 'lightgbm': 'boost',
            **{model: 'five' for model in ('ols', 'ridge', 'lasso', 'random_forest', 'extra_trees')}}
DEFAULT_COST_WEIGHTS = {'ols': 1, 'ridge': 5, 'lasso': 5, 'random_forest': 1,
    'shallow_neural_network': 15, 'extra_trees': 1, 'super_learner': 40, 'xgboost': 5, 'lightgbm': 5}


class EmptyDurationProfile(QueueError):
    """A structurally valid profile has no observations usable for pricing."""


class CostEstimator:
    def __init__(self, *, profile=None, weights=None,
                 min_batch_observations=DEFAULT_MIN_BATCH_OBSERVATIONS,
                 batch_safety_factor=1.0):
        self.weights = {**DEFAULT_COST_WEIGHTS, **(weights or {})}
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in self.weights.values()):
            raise QueueError('Cost weights must be positive and finite')
        self.profile = profile
        self.by_model = {}; self.memo = {}; self.exact = {}; self.unpriced_exact = set()
        self.identity_estimators = {}; self.identity_samples = {}; self.untrusted_identity_samples = 0
        self.batch_memo = {}
        if positive_integer(min_batch_observations) is None:
            raise QueueError('Minimum batch observations must be a positive integer')
        if positive_number(batch_safety_factor) is None or float(batch_safety_factor) < 1:
            raise QueueError('Batch safety factor must be finite and at least one')
        self.min_batch_observations = int(min_batch_observations)
        self.batch_safety_factor = float(batch_safety_factor)
        if profile is not None:
            if (not isinstance(profile, dict) or
                    profile.get('format') not in ('model-cell-cost-v1', PROFILE_FORMAT) or
                    not isinstance(profile.get('evidence'), dict) or not profile['evidence']):
                raise QueueError('Duration profile must declare its evidence')
            if not isinstance(profile.get('samples'), list):
                raise QueueError('Duration profile samples must be a list')
            seen = set()
            for sample in profile['samples']:
                if not isinstance(sample, dict) or not isinstance(sample.get('model'), str) or not sample['model']:
                    raise QueueError('Duration sample must declare a model')
                for key in ('N', 'K_expanded', 'seconds'):
                    if positive_number(sample.get(key)) is None:
                        raise QueueError('Duration sample must be positive and finite')
                if profile['format'] == PROFILE_FORMAT:
                    self._validate_measured_sample(sample)
                    key = sample['model'], int(sample['N']), int(sample['K'])
                    from .shared_queue import digest
                    cost_id = sample.get('cost_identity')
                    if cost_id is not None and (not isinstance(cost_id, dict) or not cost_id):
                        raise QueueError('Invalid workflow cost identity')
                    identity = (*key, int(sample['K_expanded']), digest(cost_id) if cost_id else None)
                    if identity in seen:
                        raise QueueError('Duplicate measured duration group')
                    seen.add(identity)
                    if cost_id:
                        if sample.get('observation_kind') != 'cold':
                            # Pre-filter workflow profiles may mix near-free
                            # cache hits into training means/p99. Do not reuse
                            # them, even when their scientific identity matches.
                            self.untrusted_identity_samples += 1
                            continue
                        self.identity_samples.setdefault(digest(cost_id), []).append(sample)
                        continue
                    self.exact.setdefault(key, []).append(sample)
                self.by_model.setdefault(sample['model'], []).append(sample)
            if (not self.by_model and not self.identity_samples and not self.untrusted_identity_samples
                    and not any(group.get('cost_identity') for group in profile.get('status_groups', ())
                                if isinstance(group, dict))):
                raise EmptyDurationProfile('Empty duration profile')
            for group in profile.get('status_groups', ()):
                if not isinstance(group, dict):
                    raise QueueError('Malformed status timing group')
                if group.get('cost_identity') is not None:
                    continue
                key = group.get('model'), group.get('N'), group.get('K')
                if group.get('status') == 'ok' and not any(
                        sample['K_expanded'] == group.get('K_expanded')
                        for sample in self.exact.get(key, ())):
                    self.unpriced_exact.add(key)

    @staticmethod
    def _validate_measured_sample(sample):
        if sample.get('observation_kind', 'legacy') not in ('cold', 'legacy'):
            raise QueueError('Non-cold observations cannot supply duration samples')
        if sample.get('status') != 'ok':
            raise QueueError('Only successful samples may estimate successful work')
        if sample.get('timing_source') not in TIMING_SOURCES:
            raise QueueError('Unknown duration timing source')
        if not isinstance(sample.get('timing_complete'), bool):
            raise QueueError('Duration sample must declare timing completeness')
        for field in ('N', 'K', 'K_expanded', 'observations', 'rows', 'quantile_samples'):
            if positive_integer(sample.get(field)) is None:
                raise QueueError('Duration sample counts must be positive integers')
        for field in ('mean_seconds', 'p50', 'p90', 'p99', 'max_seconds'):
            if not isinstance(sample.get(field), (int, float)) or positive_number(sample.get(field)) is None:
                raise QueueError('Duration statistics must be positive and finite')
        if sample['seconds'] != sample['mean_seconds']:
            raise QueueError('Duration seconds must be the mean, never a quantile')
        if not sample['p50'] <= sample['p90'] <= sample['p99'] <= sample['max_seconds']:
            raise QueueError('Duration quantiles must be ordered')
        if sample['mean_seconds'] > sample['max_seconds']:
            raise QueueError('Duration mean exceeds maximum')
        if not sample['quantile_samples'] <= sample['observations'] <= sample['rows']:
            raise QueueError('Duration observation counts are inconsistent')
        if sample['timing_complete'] != (sample['observations'] == sample['rows']):
            raise QueueError('Duration timing completeness is inconsistent')
        if sample.get('quantiles_exact') is not (sample['quantile_samples'] == sample['observations']):
            raise QueueError('Duration quantile sampling flag is inconsistent')
        error = sample.get('rank_error_99confidence')
        if (isinstance(error, bool) or not isinstance(error, (int, float)) or
                not math.isfinite(error) or not 0 <= error <= 1):
            raise QueueError('Duration quantile rank error is invalid')
        if sample['quantiles_exact'] and error != 0:
            raise QueueError('Exact duration quantiles cannot have sampling error')
        if not sample['quantiles_exact'] and error <= 0:
            raise QueueError('Sampled duration quantiles must report rank error')

    def _for_identity(self, identity):
        from .shared_queue import digest
        key = digest(identity)
        if key not in self.identity_estimators:
            samples = self.identity_samples.get(key)
            if not samples:
                self.identity_estimators[key] = None
            else:
                profile = {**self.profile,
                    'samples': [{k: v for k, v in sample.items() if k != 'cost_identity'} for sample in samples],
                    'status_groups': [{k: v for k, v in group.items() if k != 'cost_identity'}
                        for group in self.profile.get('status_groups', ())
                        if group.get('cost_identity') is not None and digest(group['cost_identity']) == key]}
                self.identity_estimators[key] = CostEstimator(profile=profile,
                    min_batch_observations=self.min_batch_observations, batch_safety_factor=self.batch_safety_factor)
        return self.identity_estimators[key]

    def estimate(self, model, n, k, *, identity=None):
        if identity is not None:
            measured = self._for_identity(identity)
            if measured is not None and model in measured.by_model:
                return measured.estimate(model, n, k)
            # Unknown workflow costs only sort work; they never inherit an old
            # standalone/full-SL timing, or authorize a multi-cell lease.
            return 1. if identity.get('phase') == 'sl' else float(n) * math.sqrt(k)
        key = model, n, k
        if key in self.memo:
            return self.memo[key]
        exact = self.exact.get(key)
        if exact:
            # Original K can expand to different widths across repeats. Preserve
            # those distributions and weight means by observation count.
            total = sum(sample['observations'] for sample in exact)
            cost = sum(sample['mean_seconds'] * (sample['observations'] / total) for sample in exact)
            self.memo[key] = cost
            return cost
        expanded = float((self.profile or {}).get('expanded_by_k', {}).get(str(k), k))
        if not math.isfinite(expanded) or expanded <= 0:
            raise QueueError('Invalid expanded feature count')
        samples = self.by_model.get(model, ())
        if self.profile is not None and not samples:
            raise QueueError('Measured profile lacks model: ' + model)
        if samples:
            reference = min(samples, key=lambda r: abs(math.log(n / r['N'])) +
                            abs(math.log(expanded / r['K_expanded'])))
            cost = reference['seconds'] * n / reference['N'] * math.sqrt(expanded / reference['K_expanded'])
        else:
            cost = float(self.weights.get(model, 1)) * n * math.sqrt(expanded)
        if not math.isfinite(cost) or cost <= 0:
            raise QueueError('Invalid estimated task duration')
        self.memo[key] = cost
        return cost

    def batch_seconds(self, model, n, k, *, identity=None):
        """Exact measured successful p99, or None when safe pricing is unknown.

        No interpolation, weight heuristics, skipped rows, or v1 means can price
        a batch. Width subgroups need complete timing coverage and sufficient
        observations; their maximum p99 is used conservatively. Historical cell
        timings are explicitly tagged and still omit outer worker overhead.
        The safety factor is operational and never changes mean work sizing.
        """
        if identity is not None:
            measured = self._for_identity(identity)
            return measured.batch_seconds(model, n, k) if measured is not None else None
        key = model, n, k
        if key in self.batch_memo:
            return self.batch_memo[key]
        samples = self.exact.get(key, ())
        evidence = (self.profile or {}).get('evidence', {})
        valid = samples and evidence.get('stride', 1) == 1 and all(
            sample['timing_complete'] and
            sample['observations'] >= self.min_batch_observations and
            sample['quantile_samples'] >= self.min_batch_observations for sample in samples)
        # A width subgroup with no usable successful timings must also leave the
        # original-K group unknown, rather than silently price only its easy part.
        if key in self.unpriced_exact:
            valid = False
        price = max(sample['p99'] for sample in samples) * self.batch_safety_factor if valid else None
        if price is not None and not math.isfinite(price):
            raise QueueError('Invalid conservative batch duration')
        self.batch_memo[key] = price
        return price

    def mean_seconds(self, model, n, k, *, identity=None):
        """Measured mean estimate for sizing, or None without a model anchor.

        Unlike estimate(), this never returns dimensionless analytic weights.
        Callers may use a separate default estimator to sort an unpriced group,
        but must report unknown total seconds when any remaining group lacks an
        anchor. V1 interpolation remains available for legacy sizing evidence.
        """
        if identity is not None:
            measured = self._for_identity(identity)
            return measured.mean_seconds(model, n, k) if measured is not None else None
        if self.profile is None or model not in self.by_model:
            return None
        return self.estimate(model, n, k)

    def coverage(self, groups):
        """Report exact batch coverage; unknown groups remain valid single claims."""
        unique = set(tuple(group) for group in groups)
        missing, sources = [], Counter()
        for model, n, k in sorted(unique):
            if self.batch_seconds(model, n, k) is None:
                missing.append(dict(model=model, N=n, K=k))
            else:
                for source in {sample['timing_source'] for sample in self.exact[(model, n, k)]}:
                    sources[source] += 1
        total, unknown = len(unique), len(missing)
        return dict(total_groups=total, priced_groups=total - unknown, unknown_groups=unknown,
                    coverage_fraction=(total - unknown) / total if total else 1.0,
                    missing=missing, timing_sources=dict(sources),
                    min_batch_observations=self.min_batch_observations,
                    batch_safety_factor=self.batch_safety_factor)

    def validate_coverage(self, groups, *, require_complete=False):
        report = self.coverage(groups)
        if require_complete and report['unknown_groups']:
            raise QueueError(f"Duration profile lacks exact batch prices for {report['unknown_groups']} groups")
        return report

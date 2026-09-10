"""Measured model/cell duration estimates used only for task ordering.

Nearest measured log-size cell anchors a positive N*sqrt(K_expanded) scaling.
This is a transparent initial estimate, not a calibrated production-time claim.
All workers can claim all classes. No estimate changes numerical training.
"""
from __future__ import annotations
import math
from .shared_queue import QueueError

CATEGORY = {'super_learner': 'sl', 'shallow_neural_network': 'nn',
            'xgboost': 'boost', 'lightgbm': 'boost',
            **{model: 'five' for model in ('ols', 'ridge', 'lasso', 'random_forest', 'extra_trees')}}
DEFAULT_COST_WEIGHTS = {'ols': 1, 'ridge': 5, 'lasso': 5, 'random_forest': 1,
    'shallow_neural_network': 15, 'extra_trees': 1, 'super_learner': 40, 'xgboost': 5, 'lightgbm': 5}


class CostEstimator:
    def __init__(self, *, profile=None, weights=None):
        self.weights = {**DEFAULT_COST_WEIGHTS, **(weights or {})}
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in self.weights.values()):
            raise QueueError('Cost weights must be positive and finite')
        self.profile = profile
        self.by_model = {}; self.memo = {}
        if profile is not None:
            if profile.get('format') != 'model-cell-cost-v1' or not profile.get('evidence'):
                raise QueueError('Duration profile must declare its evidence')
            for sample in profile['samples']:
                for key in ('N', 'K_expanded', 'seconds'):
                    if not math.isfinite(float(sample[key])) or float(sample[key]) <= 0:
                        raise QueueError('Duration sample must be positive and finite')
                self.by_model.setdefault(sample['model'], []).append(sample)
            if not self.by_model:
                raise QueueError('Empty duration profile')

    def estimate(self, model, n, k):
        key = model, n, k
        if key in self.memo:
            return self.memo[key]
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

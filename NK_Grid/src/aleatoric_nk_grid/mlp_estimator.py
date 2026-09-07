"""Shared MLP estimator construction, independent of CV and model dispatch."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class FitBatchMLPRegressor(MLPRegressor):
    """Resolve full at each actual fit; retain constructor policy for clone."""

    def fit(self, X, y, sample_weight=None):
        policy = self.batch_size
        self.fit_n_ = len(y)
        # sklearn removes the early-stopping holdout before clipping batch size.
        # This is diagnostic only; retain the historical constructor/fit policy.
        optimizer_n = len(y)
        if self.early_stopping and self.solver in {"adam", "sgd"}:
            optimizer_n -= int(np.ceil(self.validation_fraction * len(y)))
        if self.solver == "lbfgs":
            # L-BFGS evaluates the full training objective; batch_size is ignored.
            self.effective_batch_size_ = len(y)
        else:
            self.effective_batch_size_ = min(200, optimizer_n) if policy == "auto" else (
                optimizer_n if policy == "full" else min(policy, optimizer_n))
        self.batch_size = len(y) if policy == "full" else policy
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                result = super().fit(X, y, sample_weight=sample_weight)
            self.convergence_warnings_ = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
            for warning in captured:
                warnings.warn(warning.message, warning.category, stacklevel=2)
            return result
        finally:
            self.batch_size = policy


def build_mlp_regressor(
    *, seed: int, alpha: float, params: Mapping[str, Any],
) -> FitBatchMLPRegressor:
    """Build the same base learner for legacy and complete-pipeline CV."""

    return FitBatchMLPRegressor(
        batch_size=params.get("mlp_batch_size", "auto"),
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        solver=params["solver"],
        alpha=alpha,
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        early_stopping=params["early_stopping"],
        validation_fraction=params.get("validation_fraction", 0.1),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        random_state=seed,
    )

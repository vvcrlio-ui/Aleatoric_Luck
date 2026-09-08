"""Shared MLP estimator construction, independent of CV and model dispatch."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class FitBatchMLPRegressor(MLPRegressor):
    """Resolve policies at each actual fit; retain constructor values for clone.

    ``effective_batch`` fixes the regularizer denominator to the actual fit's
    clipped nominal batch size, leaving native ordinary batches unchanged.
    ``fit_samples`` retains the earlier candidate's incoming-fit-N denominator.
    Native backprop still computes data loss and gradients by batch; alpha is
    scaled temporarily so both penalty terms use the selected denominator.
    The initialization, Adam updates, shuffling and stopping loop are unchanged.
    """

    def __init__(self, loss="squared_error", hidden_layer_sizes=(100,), activation="relu", *,
                 solver="adam", alpha=0.0001, batch_size="auto", learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=0.0001, verbose=False, warm_start=False,
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 n_iter_no_change=10, max_fun=15000, l2_normalization="batch"):
        self.l2_normalization = l2_normalization
        super().__init__(loss=loss, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
            shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping, validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def _fit(self, X, y, sample_weight=None, incremental=False):
        if self.l2_normalization not in {"batch", "fit_samples", "effective_batch"}:
            raise ValueError("MLP l2_normalization must be batch, fit_samples or effective_batch")
        if self.l2_normalization == "effective_batch" and sample_weight is not None:
            raise ValueError("effective_batch L2 currently requires unweighted fitting")
        self.fit_n_ = len(y)
        self.effective_batch_size_ = self._resolve_effective_batch_size(len(y))
        return super()._fit(X, y, sample_weight=sample_weight, incremental=incremental)

    def _backprop(self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads):
        alpha = self.alpha
        if self.l2_normalization == "fit_samples":
            # sklearn 1.8 divides both penalty terms by this same sw_sum.
            # With weights, data terms retain native weighted normalization,
            # while the declared penalty denominator remains the fit row count.
            batch_denominator = len(y) if sample_weight is None else sample_weight.sum()
            self.alpha = alpha * (batch_denominator / self.fit_n_)
        elif self.l2_normalization == "effective_batch":
            if sample_weight is not None:
                raise ValueError("effective_batch L2 currently requires unweighted fitting")
            # The ordinary-batch and alpha=0 code paths remain bitwise native.
            # Only short tails change; loss and gradient share native backprop.
            if alpha != 0 and len(y) != self.effective_batch_size_:
                self.alpha = alpha * (len(y) / self.effective_batch_size_)
        try:
            return super()._backprop(X, y, sample_weight, activations, deltas, coef_grads, intercept_grads)
        finally:
            self.alpha = alpha

    def _resolve_effective_batch_size(self, n):
        # sklearn removes the early-stopping holdout before clipping batch size.
        optimizer_n = n
        if self.early_stopping and self.solver in {"adam", "sgd"}:
            optimizer_n -= int(np.ceil(self.validation_fraction * n))
        if self.solver == "lbfgs":
            # L-BFGS evaluates the full training objective; batch_size is ignored.
            return n
        return min(200, optimizer_n) if self.batch_size == "auto" else (
            optimizer_n if self.batch_size == "full" else min(self.batch_size, optimizer_n))

    def fit(self, X, y, sample_weight=None):
        policy = self.batch_size
        self.fit_n_ = len(y)
        self.effective_batch_size_ = self._resolve_effective_batch_size(len(y))
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
        loss="squared_error",
        batch_size=params.get("mlp_batch_size", "auto"),
        l2_normalization=params.get("mlp_l2_normalization", "batch"),
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        solver=params["solver"],
        alpha=alpha,
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        early_stopping=params["early_stopping"],
        validation_fraction=params.get("validation_fraction", 0.1),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        tol=params.get("tol", 1e-4),
        shuffle=True,
        warm_start=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        random_state=seed,
    )

"""Contracts for ridge's analytic leave-one-out alpha selection."""

from __future__ import annotations

from functools import partial
from time import perf_counter

import numpy as np
import pytest
from sklearn.linear_model import Ridge, RidgeCV as SklearnRidgeCV

from aleatoric_nk_grid import model_registry
from aleatoric_nk_grid.model_registry import AdaptiveRidgeCV


def _synthetic(n: int, p: int, seed: int = 20260808):
    rng = np.random.default_rng(seed + n * 1000 + p)
    X = rng.normal(size=(n, p))
    coefficients = rng.normal(size=p)
    y = X @ coefficients + rng.normal(scale=0.2, size=n)
    return X, y


@pytest.mark.parametrize(
    ("n", "p", "alpha"),
    [
        (12, 6, 1e-4),  # n > p, lower end of the declared alpha grid
        (6, 12, 1.0),  # n < p
        (9, 9, 1e4),  # n == p, upper end of the declared alpha grid
    ],
)
def test_adaptive_ridge_uses_exact_leave_one_out_errors(
    monkeypatch, n, p, alpha
):
    X, y = _synthetic(n, p)
    # Production avoids retaining this n × n_alphas diagnostic array.  Turn it
    # on only in this test so the estimator's analytic LOO values are visible.
    monkeypatch.setattr(
        model_registry,
        "RidgeCV",
        partial(SklearnRidgeCV, store_cv_results=True),
    )
    model = AdaptiveRidgeCV(
        alpha_log10_min=np.log10(alpha),
        alpha_log10_max=np.log10(alpha),
        n_alphas=1,
        scoring=None,
    ).fit(X, y)

    brute_force = np.mean(
        [
            (y[index] - Ridge(alpha=alpha).fit(
                np.delete(X, index, axis=0), np.delete(y, index)
            ).predict(X[index : index + 1])[0]) ** 2
            for index in range(n)
        ]
    )
    assert model.model_.cv is None
    assert float(np.mean(model.model_.cv_results_[:, 0])) == pytest.approx(
        brute_force, rel=1e-9
    )


def test_adaptive_ridge_keeps_configured_scoring():
    X, y = _synthetic(12, 6)
    model = AdaptiveRidgeCV(
        alpha_log10_min=-2,
        alpha_log10_max=2,
        n_alphas=3,
        scoring="neg_mean_squared_error",
    ).fit(X, y)
    assert model.model_.scoring == "neg_mean_squared_error"


def test_analytic_leave_one_out_is_at_least_five_times_faster_than_five_fold():
    X, y = _synthetic(300, 150)
    alphas = np.logspace(-4, 4, 50)
    started = perf_counter()
    SklearnRidgeCV(
        alphas=alphas, cv=5, scoring="neg_mean_squared_error"
    ).fit(X, y)
    five_fold_seconds = perf_counter() - started
    started = perf_counter()
    AdaptiveRidgeCV(
        alpha_log10_min=-4,
        alpha_log10_max=4,
        n_alphas=50,
        scoring="neg_mean_squared_error",
    ).fit(X, y)
    loo_seconds = perf_counter() - started
    assert loo_seconds <= five_fold_seconds / 5, (
        f"analytic LOO={loo_seconds:.6f}s, five-fold={five_fold_seconds:.6f}s"
    )


def test_ridge_requires_two_rows_but_accepts_two_rows():
    with pytest.raises(ValueError, match="Ridge requires at least two training rows."):
        AdaptiveRidgeCV(
            alpha_log10_min=-4,
            alpha_log10_max=4,
            n_alphas=2,
            scoring="neg_mean_squared_error",
        ).fit([[1.0]], [1.0])
    X, y = _synthetic(2, 1)
    assert AdaptiveRidgeCV(
        alpha_log10_min=-4,
        alpha_log10_max=4,
        n_alphas=2,
        scoring="neg_mean_squared_error",
    ).fit(X, y).predict(X).shape == (2,)

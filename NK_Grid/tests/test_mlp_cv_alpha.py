"""shallow_neural_network 的 L2 强度改为 cell 内 CV 选择后的行为契约。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aleatoric_nk_grid.model_registry import (
    AdaptiveMLPRegressor,
    load_model_params,
    make_model,
)


FAST = dict(
    hidden_layer_sizes=[4],
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=40,
    early_stopping=False,
    alpha_log10_min=-2,
    alpha_log10_max=2,
    n_alphas=3,
    max_cv_folds=2,
)


def _toy(n: int, p: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    return X, y


def test_same_seed_gives_bitwise_identical_predictions():
    X, y = _toy(30, 6)
    pred = [
        AdaptiveMLPRegressor(seed=123, **FAST).fit(X, y).predict(X)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(pred[0], pred[1])


def test_fits_at_grid_minimum_n_without_validation_split():
    # 网格最小 N=10 曾让 early_stopping 的内部验证集直接崩溃；
    # CV 选择不切验证集，必须在 N=10 正常工作。
    X, y = _toy(10, 5)
    model = AdaptiveMLPRegressor(seed=1, **FAST).fit(X, y)
    assert model.predict(X).shape == (10,)


def test_selected_alpha_comes_from_the_declared_grid():
    X, y = _toy(25, 4)
    model = AdaptiveMLPRegressor(seed=5, **FAST).fit(X, y)
    grid = np.logspace(FAST["alpha_log10_min"], FAST["alpha_log10_max"], FAST["n_alphas"])
    assert model.alpha_ in grid
    assert len(model.cv_mse_) == FAST["n_alphas"]


def test_pure_noise_with_p_over_n_prefers_strong_penalty():
    # y 与 X 独立且 p>n：任何拟合都是过拟合，验证误差应把重惩罚选出来。
    # FAST 的 4 神经元 × 40 轮没有过拟合能力，选择会退化成平票——
    # 这里必须用足够容量（32 神经元 × 300 轮）让最弱 alpha 真的输。
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 80))
    y = rng.normal(size=30)
    capable = dict(FAST, hidden_layer_sizes=[32], max_iter=300)
    model = AdaptiveMLPRegressor(seed=2, **capable).fit(X, y)
    assert model.alpha_ >= 1.0
    # 最弱的 alpha 一定不是赢家
    assert model.cv_mse_[0] > min(model.cv_mse_)


def test_fewer_than_two_rows_raises():
    with pytest.raises(ValueError, match="at least two training rows"):
        AdaptiveMLPRegressor(seed=0, **FAST).fit([[1.0]], [1.0])


def test_make_model_builds_cv_pipeline_from_locked_params():
    params = load_model_params(
        Path(__file__).resolve().parents[1] / "model_params.yaml",
        task="regression",
        models=("shallow_neural_network",),
    )["shallow_neural_network"]
    fast = dict(params)
    fast.update(hidden_layer_sizes=[4], max_iter=40, n_alphas=2, max_cv_folds=2)
    model = make_model(
        "shallow_neural_network", seed=42, n_jobs=1, task="regression", params=fast
    )
    X, y = _toy(12, 3)
    model.fit(X, y)
    inner = model[-1].regressor_
    assert isinstance(inner, AdaptiveMLPRegressor)
    assert inner.alpha_ in np.logspace(
        fast["alpha_log10_min"], fast["alpha_log10_max"], fast["n_alphas"]
    )


def test_converged_large_n_predictions_are_bitwise_invariant_to_iteration_cap():
    X, y = _toy(1000, 3)
    params = dict(
        FAST, hidden_layer_sizes=[4], n_alphas=2, max_cv_folds=2,
        learning_rate_init=0.01,
    )
    low = AdaptiveMLPRegressor(seed=91, **dict(params, max_iter=500)).fit(X, y)
    high = AdaptiveMLPRegressor(seed=91, **dict(params, max_iter=2000)).fit(X, y)
    np.testing.assert_array_equal(low.predict(X), high.predict(X))
    assert low.alpha_ == high.alpha_


def test_small_p_over_n_mlp_no_longer_hits_2000_iteration_cap():
    X, y = _toy(10, 80, seed=19)
    params = dict(FAST, hidden_layer_sizes=[32], max_iter=2000, n_alphas=2)
    model = AdaptiveMLPRegressor(seed=2, **params).fit(X, y)
    assert model.model_.n_iter_ < params["max_iter"]

"""Check the declared epoch budget reaches actual fold and final MLP fits."""
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

from aleatoric_nk_grid.model_registry import load_model_params, make_model

ROOT = Path(__file__).resolve().parents[2]


def test_budget_reaches_actual_fold_and_final_fits(monkeypatch):
    params = load_model_params(ROOT / "FFCWS/model_params.yaml", task="regression", models=["super_learner"])["super_learner"]
    params.update(mlp_batch_size="auto", n_estimators=2, lgbm_n_estimators=2, ridge_n_alphas=2, hidden_layer_sizes=[2])
    seen = []
    original = MLPRegressor.fit

    def fit(self, X, y, **kwargs):
        # Observe the effective estimator before reducing runtime for this wiring test.
        seen.append((len(y), self.max_iter))
        configured = self.max_iter
        self.max_iter = 1
        try:
            return original(self, X, y, **kwargs)
        finally:
            self.max_iter = configured

    monkeypatch.setattr(MLPRegressor, "fit", fit)
    rng = np.random.default_rng(29)
    X = rng.normal(size=(15, 2))
    make_model("super_learner", seed=1, n_jobs=1, task="regression", params=params).fit(X, X[:, 0])
    assert sorted(seen) == [(12, 2000)] * 5 + [(15, 2000)]

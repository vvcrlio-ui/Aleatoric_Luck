"""Observe real inner-fold fits, so accepting a config without using it fails."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from aleatoric_nk_grid.model_registry import load_model_params, make_model, DEFAULT_MODEL_PARAMS_PATH


@pytest.mark.parametrize('batch', [32, 64, 128, 'auto', 'full'])
@pytest.mark.parametrize('n', [10, 201])
@pytest.mark.parametrize('fold_local', [False, True])
def test_batch_reaches_each_cv_and_final_fit(monkeypatch, batch, n, fold_local):
    params = load_model_params(DEFAULT_MODEL_PARAMS_PATH, task='regression', models=['shallow_neural_network'])['shallow_neural_network']
    params.update(mlp_batch_size=batch, n_alphas=2, hidden_layer_sizes=[2])
    seen = []
    original = MLPRegressor.fit

    def fit(self, X, y, **kwargs):
        seen.append((len(y), self.batch_size, self.max_iter))
        budget = self.max_iter
        self.max_iter = 1
        try:
            return original(self, X, y, **kwargs)
        finally:
            self.max_iter = budget

    monkeypatch.setattr(MLPRegressor, 'fit', fit)
    X = pd.DataFrame(np.random.default_rng(41).normal(size=(n, 2)))
    model = make_model('shallow_neural_network', seed=7, n_jobs=1, params=params,
                       preprocessor=StandardScaler() if fold_local else None)
    model = clone(model).fit(X, X[0])
    fold_sizes = [n - len(v) for v in np.array_split(np.arange(n), 3)]
    expected = [(s, s if batch == 'full' else batch, 2000) for s in fold_sizes * 2 + [n]]
    assert seen == expected
    np.testing.assert_array_equal(model.predict(X), pickle.loads(pickle.dumps(model)).predict(X))


@pytest.mark.parametrize('bad', [True, False, 0, -1, 1.5, '64', 'bad', None])
def test_reject_invalid_standalone_policy(bad):
    params = load_model_params(DEFAULT_MODEL_PARAMS_PATH, task='regression', models=['shallow_neural_network'])['shallow_neural_network']
    with pytest.raises(ValueError, match='mlp_batch_size'):
        make_model('shallow_neural_network', seed=7, n_jobs=1, params={**params, 'mlp_batch_size':bad})

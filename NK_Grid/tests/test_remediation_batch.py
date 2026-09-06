import pickle
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from aleatoric_nk_grid.model_registry import FitBatchMLPRegressor, _validated_params, load_model_params, DEFAULT_MODEL_PARAMS_PATH


@pytest.mark.parametrize('n', [199, 200, 201, 250, 251, 252])
@pytest.mark.parametrize('policy', ['auto', 'full'])
def test_c1_c2_actual_fit_and_clone(n, policy):
    for fit_n in (n, n - (n + 4) // 5):
        estimator = FitBatchMLPRegressor(hidden_layer_sizes=(2,), max_iter=1, random_state=8, batch_size=policy)
        model = clone(estimator).fit(np.arange(fit_n).reshape(-1, 1) / fit_n, np.zeros(fit_n))
        assert model.fit_n_ == fit_n
        assert model.effective_batch_size_ == (fit_n if policy == 'full' else min(200, fit_n))
        assert clone(model).batch_size == policy
        assert pickle.loads(pickle.dumps(model)).batch_size == policy


def test_c3_auto_compatibility():
    X = np.random.RandomState(1).normal(size=(201, 2)); y = X[:, 0]
    predictions = []
    for cls, policy in ((MLPRegressor, 'auto'), (FitBatchMLPRegressor, 'auto'), (FitBatchMLPRegressor, 200)):
        model = cls(hidden_layer_sizes=(2,), max_iter=2, random_state=8, batch_size=policy).fit(X, y)
        predictions.append(model.predict(X))
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_array_equal(predictions[0], predictions[2])


@pytest.mark.parametrize('bad', [True, 0, -1, 1.5, '200', 'bad'])
def test_batch_contract_rejects(bad):
    params = load_model_params(DEFAULT_MODEL_PARAMS_PATH, task='regression', models=['super_learner'])['super_learner']
    with pytest.raises(ValueError, match='mlp_batch_size'):
        _validated_params('regression', 'super_learner', {**params, 'mlp_batch_size': bad})

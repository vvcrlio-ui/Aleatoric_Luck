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
        reference = MLPRegressor(hidden_layer_sizes=(2,), max_iter=1, random_state=8,
                                 batch_size=fit_n if policy == 'full' else 'auto')
        reference.fit(np.arange(fit_n).reshape(-1, 1) / fit_n, np.zeros(fit_n))
        np.testing.assert_array_equal(model.predict([[.25]]), reference.predict([[.25]]))


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


@pytest.mark.parametrize('signal', ['zero', 'strong'])
def test_c4_c5_small_n_wide_diagnostics_and_external_prediction_isolation(signal):
    import pandas as pd
    from aleatoric_nk_grid.model_registry import make_model
    from aleatoric_nk_grid.preprocessing import FoldPreprocessor, SourceGroup
    rng = np.random.default_rng(238)
    X = pd.DataFrame(rng.normal(size=(10, 30)), columns=[f'x{i}' for i in range(30)])
    y = np.zeros(10) if signal == 'zero' else 100 * X['x0'].to_numpy()
    groups = tuple(SourceGroup(c, (c,), i, 'continuous', source_prior=0.) for i, c in enumerate(X))
    preprocessor = FoldPreprocessor(groups,
        {'continuous':'median','ordinal':'most_frequent','onehot_group':'atomic_mode','model_overrides':{}},
        'super_learner')
    params = load_model_params(DEFAULT_MODEL_PARAMS_PATH, task='regression', models=['super_learner'])['super_learner']
    params = {**params, 'n_estimators':2, 'lgbm_n_estimators':2, 'max_iter':2,
              'ridge_n_alphas':3, 'hidden_layer_sizes':[2], 'diagnostics':True, 'mlp_batch_size':'full'}
    model = make_model('super_learner', seed=11, n_jobs=1, params=params, preprocessor=preprocessor).fit(X, y)
    before = pickle.dumps(model)
    assert np.isfinite(model.predict(X)).all()
    assert np.isfinite(model.predict(X * 1e8)).all()
    assert pickle.dumps(model) == before
    fits = model.diagnostics_['fits']
    assert len([f for f in fits if f['phase'] == 'oof']) == 20
    for fit in fits:
        assert fit['N'] == (8 if fit['phase'] == 'oof' else 10)
        if fit['model'] == 'shallow_nn':
            assert fit['batch'] == fit['N']
            assert fit['iterations'] == 2
            assert fit['convergence_warnings']
        if 'mse' in fit:
            assert np.isfinite(fit['mse'])

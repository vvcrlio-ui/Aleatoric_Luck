"""M2 native old-CV oracle, with tolerances declared before candidate execution."""
import numpy as np
import pytest
from aleatoric_nk_grid.model_registry import (
    XGBoostCVRegressor, LightGBMCVRegressor, _select_cv_round,
)


@pytest.mark.parametrize('curve,patience,expected', [
    ([1, 2, 3], None, 1), ([3, 2, 1], None, 3),
    ([1, 1, 0], 1, 1), ([3, 2, 2, 1], 2, 4), ([2, 1, 1], None, 2),
])
def test_b2_round_oracle(curve, patience, expected):
    assert _select_cv_round(curve, patience) == expected


@pytest.mark.parametrize('kind', ['dense', 'wide', 'nan', 'constant', 'minimum'])
@pytest.mark.parametrize('library', ['xgboost', 'lightgbm'])
def test_b1_b3_old_cv_oracle(library, kind):
    random = np.random.RandomState(43)
    n, width = (5 if kind == 'minimum' else 23), (101 if kind == 'wide' else 6)
    X = random.normal(size=(n, width))
    y = random.normal(size=n) * np.arange(1, n + 1)
    if kind == 'nan':
        X[::3, ::2] = np.nan
    if kind == 'constant':
        y[:] = 3
    rounds, seed = 9, 11
    if library == 'xgboost':
        import xgboost as lib
        candidate = XGBoostCVRegressor(seed, 1, objective='reg:squarederror', eval_metric='rmse',
            max_depth=2, eta=.3, max_rounds=rounds, cv_folds=5).fit(X, y)
        expected_params = {'objective':'reg:squarederror','eval_metric':'rmse','max_depth':2,'eta':.3,'nthread':1,'seed':seed}
        assert candidate.params_ == expected_params
        test = np.array_split(np.random.RandomState(seed).permutation(n), 5)
        folds = [(np.concatenate([test[j] for j in range(5) if j != i]), test[i]) for i in range(5)]
        data = lib.DMatrix(X, label=y)
        old = lib.cv(expected_params, data, num_boost_round=rounds, folds=folds, seed=seed,
                     verbose_eval=False)['test-rmse-mean'].to_numpy()
        selected = int(np.argmin(old)) + 1
        oracle = lib.train(expected_params, data, num_boost_round=selected).predict(data)
    else:
        import lightgbm as lib
        candidate = LightGBMCVRegressor(seed, 1, objective='regression', metric='rmse',
            learning_rate=.1, num_leaves=7, min_data_in_leaf=1, verbosity=-1,
            max_rounds=rounds, cv_folds=5, early_stopping_rounds=3).fit(X, y)
        expected_params = {'objective':'regression','metric':'rmse','learning_rate':.1,'num_leaves':7,
                           'min_data_in_leaf':1,'verbosity':-1,'num_threads':1,'seed':seed}
        assert candidate.params_ == expected_params
        old = lib.cv(expected_params, lib.Dataset(X, label=y), num_boost_round=rounds,
            nfold=5, stratified=False, seed=seed,
            callbacks=[lib.early_stopping(3, verbose=False)])['valid rmse-mean']
        selected = int(np.argmin(old)) + 1
        oracle = lib.train(expected_params, lib.Dataset(X, label=y), num_boost_round=selected).predict(X)
    np.testing.assert_allclose(candidate.cv_curve_[:len(old)], old, rtol=1e-7, atol=1e-9)
    assert candidate.best_rounds_ == selected
    np.testing.assert_allclose(candidate.predict(X), oracle, rtol=1e-7, atol=1e-9)


def test_b5_third_fold_exception(monkeypatch):
    import xgboost as lib
    original = lib.train
    count = 0
    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError('injected third fold failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(lib, 'train', fail)
    model = XGBoostCVRegressor(1, 1, objective='reg:squarederror', eval_metric='rmse',
                              max_depth=2, eta=.3, max_rounds=3, cv_folds=5)
    with pytest.raises(RuntimeError, match='third fold'):
        model.fit(np.arange(60).reshape(20, 3), np.arange(20))
    assert not hasattr(model, 'model_')
    assert not hasattr(model, 'best_rounds_')


def test_b4_xgb_python_context_lifetime(monkeypatch):
    """Weak references certify lifetime only, not native allocator/RSS release."""
    import weakref
    import xgboost as lib
    original = lib.train
    references = []
    def observed(*args, **kwargs):
        assert not any(ref() is not None for ref in references), 'previous fold context remains live'
        model = original(*args, **kwargs)
        references.append(weakref.ref(model))
        return model
    monkeypatch.setattr(lib, 'train', observed)
    model = XGBoostCVRegressor(1, 1, objective='reg:squarederror', eval_metric='rmse',
                              max_depth=2, eta=.3, max_rounds=3, cv_folds=5)
    model.fit(np.arange(60).reshape(20, 3), np.arange(20))
    assert len(references) == 6

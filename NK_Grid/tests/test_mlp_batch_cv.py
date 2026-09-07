"""Behavioral tests: real fit boundaries, numerical oracle, failure policies."""
import json
import ast
import pickle
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aleatoric_nk_grid.mlp_batch_cv import BatchSearchMLP, SerialBatchStack
from aleatoric_nk_grid.model_registry import FitBatchMLPRegressor, load_model_params, make_model

ROOT = Path(__file__).resolve().parents[2]


def params(name='shallow_neural_network', **updates):
    p = load_model_params(ROOT/'NK_Grid/model_params.yaml', task='regression', models=[name])[name]
    p.update(updates)
    return p


def data(n=18, k=3):
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(n, k)))
    return X, 10 + 3 * X[0].to_numpy() + rng.normal(size=n)


@pytest.fixture
def fast_fits(monkeypatch):
    calls = []
    owner_thread = threading.get_ident()
    original = MLPRegressor.fit
    def fit(self, X, y, **kwargs):
        assert threading.get_ident() == owner_thread
        calls.append((len(y), self.alpha, self.batch_size, self.max_iter))
        budget = self.max_iter
        self.max_iter = 1
        try:
            return original(self, X, y, **kwargs)
        finally:
            self.max_iter = budget
    monkeypatch.setattr(MLPRegressor, 'fit', fit)
    return calls


@pytest.mark.parametrize('typed', [False, True])
def test_cartesian_actual_fits_and_winner(fast_fits, typed):
    X, y = data()
    p = params()
    model = clone(make_model('shallow_neural_network', seed=17, params=p,
                            preprocessor=SimpleImputer() if typed else None)).fit(X, y)
    expected = [(12, float(a), b, 2000) for a in np.logspace(-2, 2, 5)
                for b in [32, 64, 128, 256] for _ in range(3)]
    assert fast_fits[:-1] == expected
    scores = model.diagnostics_['candidates']
    winner = min(scores, key=lambda r: r['mean_mse'])
    assert fast_fits[-1] == (18, winner['alpha'], winner['batch'], 2000)
    assert model.model_[-1].regressor_.batch_size == model.batch_size_ == winner['batch']
    assert model.diagnostics_['fit_count'] == 61
    assert len(scores) == 20
    json.dumps(model.diagnostics_, allow_nan=False)
    np.testing.assert_array_equal(model.predict(X), pickle.loads(pickle.dumps(model)).predict(X))


def test_real_numerical_oracle():
    X, y = data(43, 5)  # unequal folds distinguish fold means from pooled MSE
    p = params(n_alphas=2, mlp_batch_candidates=[4, 16], hidden_layer_sizes=[4], max_iter=65)
    model = make_model('shallow_neural_network', seed=7, params=p).fit(X, y)
    scores = []
    for alpha in [0.01, 100.0]:
        for batch in [4, 16]:
            losses = []
            for t, v in KFold(3).split(X):
                estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                    StandardScaler(), TransformedTargetRegressor(transformer=StandardScaler(),
                    regressor=MLPRegressor(hidden_layer_sizes=(4,), alpha=alpha, batch_size=batch,
                        max_iter=65, random_state=7)))
                pred = estimator.fit(X.iloc[t], y[t]).predict(X.iloc[v])
                losses.append(np.mean((pred-y[v])**2))
            scores.append(np.mean(losses))
    np.testing.assert_allclose(model.cv_mse_, scores, rtol=1e-13)
    best = [(a,b) for a in [.01,100.] for b in [4,16]][np.argmin(scores)]
    assert (model.alpha_, model.batch_size_) == best


class AuditPreprocessor(TransformerMixin, BaseEstimator):
    seen = []
    def fit(self, X, y=None):
        self.seen.append(tuple(X.index))
        return self
    def transform(self, X):
        return np.asarray(X)


def test_stack_oof_boundaries_and_cost(monkeypatch, fast_fits):
    X, y = data(20)
    p = params('super_learner', n_estimators=2, lgbm_n_estimators=2, ridge_n_alphas=2,
               hidden_layer_sizes=[2])
    seen, audit_windows = [], []
    original = BatchSearchMLP.fit
    def fit(self, X, y):
        seen.append((tuple(X.index), np.asarray(y).copy()))
        start = len(AuditPreprocessor.seen)
        result = original(self, X, y)
        audit_windows.append(AuditPreprocessor.seen[start:])
        return result
    monkeypatch.setattr(BatchSearchMLP, 'fit', fit)
    model = clone(make_model('super_learner', seed=3, n_jobs=8, params=p,
                             preprocessor=AuditPreprocessor())).fit(X, y)
    expected = [tuple(t) for t,v in KFold(5).split(X)] + [tuple(range(20))]
    assert [r[0] for r in seen] == expected
    for (indices, labels) in seen:
        np.testing.assert_array_equal(labels, y[list(indices)])
    for indices, window in zip(expected, audit_windows):
        own_rows = np.asarray(indices)
        assert window == [tuple(own_rows[t]) for t,v in KFold(3).split(own_rows)]*4 + [indices]
    assert len(fast_fits) == model.diagnostics_['fit_count'] == 78
    assert all(c[1] == .01 and c[3] == 2000 for c in fast_fits)
    fits = model.diagnostics_['fits']
    assert [f['phase'] for f in fits] == ['oof']*5 + ['full']
    assert all(f['fit_count'] == 13 for f in fits)
    assert len(model.model_.estimators_) == 4
    assert all(not hasattr(m, 'estimators_') for m in [model.model_.named_estimators_.shallow_nn])
    np.testing.assert_array_equal(model.predict(X), pickle.loads(pickle.dumps(model)).predict(X))


def test_oof_label_perturbation_does_not_change_own_predictions(monkeypatch, fast_fits):
    X, y = data(15)
    learner = BatchSearchMLP(FitBatchMLPRegressor(hidden_layer_sizes=(2,), random_state=1),
                             [.01], [2, 8], 3)
    captured = []
    from sklearn.linear_model import LinearRegression
    original = LinearRegression.fit
    def fit(self, X, y, **kw):
        captured.append(np.asarray(X).copy())
        return original(self, X, y, **kw)
    monkeypatch.setattr(LinearRegression, 'fit', fit)
    for target in [y, np.r_[y[:3] + 10000, y[3:]]]:
        SerialBatchStack([('shallow_nn', learner)], 5, True, False).fit(X, target)
    np.testing.assert_array_equal(captured[0][:3], captured[1][:3])


def test_serial_stack_matches_sklearn_with_same_base_learners(fast_fits):
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    X,y = data(15)
    p = params('super_learner', n_estimators=2, lgbm_n_estimators=2,
        ridge_n_alphas=2, hidden_layer_sizes=[2], mlp_batch_candidates=[4,8])
    model = make_model('super_learner', seed=7, params=p).fit(X,y)
    oracle = StackingRegressor(estimators=model.model_.estimators,
        final_estimator=LinearRegression(positive=True), cv=5, n_jobs=1).fit(X,y)
    np.testing.assert_array_equal(model.predict(X), oracle.predict(X))
    np.testing.assert_array_equal(model.model_.final_estimator_.coef_, oracle.final_estimator_.coef_)


def test_inner_preprocessing_stats_and_training_rows(monkeypatch, fast_fits):
    X, y = data(12)
    X.iloc[0, 0] = 1e9
    X.iloc[4, 1] = np.nan
    scaler_stats, imputer_stats = [], []
    sf, imf = StandardScaler.fit, SimpleImputer.fit
    def scale(self, X, y=None, **kw):
        result = sf(self, X, y, **kw)
        scaler_stats.append((np.asarray(X).shape, self.mean_.copy()))
        return result
    def impute(self, X, y=None, **kw):
        result = imf(self, X, y, **kw)
        imputer_stats.append(self.statistics_.copy())
        return result
    monkeypatch.setattr(StandardScaler, 'fit', scale)
    monkeypatch.setattr(SimpleImputer, 'fit', impute)
    model = BatchSearchMLP(FitBatchMLPRegressor(hidden_layer_sizes=(2,)), [.01], [4], 3).fit(X, y)
    t, v = next(KFold(3).split(X))
    median = np.nanmedian(X.iloc[t], axis=0)
    train = np.where(np.isnan(X.iloc[t]), median, X.iloc[t])
    np.testing.assert_allclose(imputer_stats[0], median)
    np.testing.assert_allclose(scaler_stats[0][1], train.mean(axis=0))
    np.testing.assert_allclose(scaler_stats[1][1], [y[t].mean()])
    assert scaler_stats[0][0][0] == len(t)
    assert model.diagnostics_['fit_count'] == 4


def test_typed_preprocessor_inner_rows(fast_fits):
    X, y = data(12)
    AuditPreprocessor.seen = []
    BatchSearchMLP(FitBatchMLPRegressor(hidden_layer_sizes=(2,)), [.01], [4, 8], 3,
                   AuditPreprocessor()).fit(X, y)
    assert AuditPreprocessor.seen == [tuple(t) for t,v in KFold(3).split(X)]*2 + [tuple(X.index)]


@pytest.mark.parametrize('bad', [[], [32,32], [True], [False], [0], [-1], [1.5], ['32'], None, '32', {32}])
@pytest.mark.parametrize('name', ['shallow_neural_network', 'super_learner'])
def test_invalid_candidates(bad, name):
    with pytest.raises(ValueError, match='mlp_batch_candidates'):
        make_model(name, seed=1, params=params(name, mlp_batch_candidates=bad))


@pytest.mark.parametrize('bad', [True, False, 0, 1, -2, 2.5, '3', None])
@pytest.mark.parametrize('name,field', [('shallow_neural_network','max_cv_folds'),
    ('super_learner','mlp_batch_cv_folds'), ('super_learner','cv')])
def test_invalid_folds(bad, name, field):
    with pytest.raises(ValueError, match='folds|stacking cv'):
        make_model(name, seed=1, params=params(name, **{field:bad}))


@pytest.mark.parametrize('n', [1,2,3,10])
def test_small_constant_missing(n, fast_fits):
    X = pd.DataFrame(np.full((n, 2), np.nan))
    model = make_model('shallow_neural_network', seed=1, params=params()).fit(X, np.ones(n))
    assert np.isfinite(model.predict(X)).all()
    assert bool(model.diagnostics_['fallback_reason']) == (n < 3)
    assert len(fast_fits) == (1 if n < 3 else 61)
    if n < 3:
        assert (model.alpha_, model.batch_size_) == (.01,32)


def test_ties_failures_nonfinite(monkeypatch, fast_fits):
    X,y = data()
    original = FitBatchMLPRegressor.fit
    def fit(self, X, y, **kw):
        if self.batch_size == 4:
            raise ValueError('injected candidate failure')
        return original(self, X, y, **kw)
    monkeypatch.setattr(FitBatchMLPRegressor, 'fit', fit)
    monkeypatch.setattr(FitBatchMLPRegressor, 'predict', lambda self,X: np.zeros(len(X)))
    m = BatchSearchMLP(FitBatchMLPRegressor(), [.1,1], [4,8,16], 3).fit(X,y)
    assert (m.alpha_,m.batch_size_) == (.1,8)
    assert m.candidate_scores_[0]['error'] == 'ValueError: injected candidate failure'
    assert m.candidate_scores_[0]['mean_mse'] is None
    assert m.model_[-1].regressor_.batch_size == 8
    monkeypatch.setattr(FitBatchMLPRegressor, 'predict', lambda self,X: np.full(len(X), np.nan))
    with pytest.raises(ValueError, match='All MLP.*failed'):
        BatchSearchMLP(FitBatchMLPRegressor(), [.1], [4,8], 3).fit(X,y)


def test_clone_preserves_search_contract():
    for name in ['shallow_neural_network', 'super_learner']:
        p = params(name, mlp_batch_candidates=[7,19])
        model = make_model(name, seed=17, params=p)
        copied = clone(model)
        actual = copied.params if hasattr(copied, 'params') else copied.get_params()
        assert actual['mlp_batch_size'] == 'cv'
        assert actual['mlp_batch_candidates'] == [7,19]
    search = clone(BatchSearchMLP(FitBatchMLPRegressor(), [.01], [7,19], 2))
    assert search.mlp_batch_candidates == [7,19]
    assert search.max_cv_folds == 2


@pytest.mark.parametrize('name', ['shallow_neural_network', 'super_learner'])
def test_retry_diagnostic_payload_excludes_invocation_timing(name, monkeypatch, fast_fits):
    from aleatoric_nk_grid import mlp_batch_cv
    clock = iter(range(1000))
    monkeypatch.setattr(mlp_batch_cv.time, 'perf_counter', lambda: float(next(clock) ** 2))
    X,y = data(12)
    p = params(name, hidden_layer_sizes=[2], mlp_batch_candidates=[4,8])
    if name == 'super_learner':
        p.update(n_estimators=2, lgbm_n_estimators=2, ridge_n_alphas=2)
    models = [make_model(name, seed=17, params=p).fit(X,y) for _ in range(2)]
    payloads = [json.dumps(m.diagnostics_, sort_keys=True, allow_nan=False) for m in models]
    assert payloads[0] == payloads[1]
    assert 'seconds' not in payloads[0]


def test_selected_batch_is_used_not_just_reported(monkeypatch, fast_fits):
    monkeypatch.setattr(FitBatchMLPRegressor, 'predict',
        lambda self,X: np.full(len(X), 1.0 if self.batch_size == 4 else 0.0))
    X, _ = data()
    model = BatchSearchMLP(FitBatchMLPRegressor(), [.01], [4,8], 3).fit(X, np.ones(len(X)))
    assert model.batch_size_ == 8
    assert fast_fits[-1][2] == 8
    np.testing.assert_array_equal(model.predict(X), np.ones(len(X)))


def test_final_failure_propagates(monkeypatch, fast_fits):
    original = FitBatchMLPRegressor.fit
    def fit(self, X, y, **kw):
        if len(y) == 18:
            raise RuntimeError('final refit failed')
        return original(self, X, y, **kw)
    monkeypatch.setattr(FitBatchMLPRegressor, 'fit', fit)
    X,y = data()
    with pytest.raises(RuntimeError, match='final refit failed'):
        BatchSearchMLP(FitBatchMLPRegressor(), [.01], [4,8], 3).fit(X,y)


def test_finite_fold_losses_overflowing_mean_are_invalid(monkeypatch, fast_fits):
    monkeypatch.setattr(FitBatchMLPRegressor, 'predict',
        lambda self,X: np.full(len(X), 1e154 if self.batch_size == 4 else 0.0))
    X,_ = data(3)  # one validation row per fold: overflow only across fold means
    model = BatchSearchMLP(FitBatchMLPRegressor(), [.01], [4,8], 3).fit(X,np.zeros(len(X)))
    assert model.batch_size_ == 8
    assert model.candidate_scores_[0]['mean_mse'] is None
    assert model.candidate_scores_[0]['error'] == 'ValueError: nonfinite mean validation MSE'
    json.dumps(model.diagnostics_, allow_nan=False)


def test_twofold_three_rows_fallback(fast_fits):
    X,y = data(3)
    m = BatchSearchMLP(FitBatchMLPRegressor(), [.01], [4,8], 2).fit(X,y)
    assert m.fallback_reason_
    assert len(fast_fits) == 1


def test_stack_small_n_fallback_and_existing_ridge_limit(fast_fits):
    X,y = data(3)
    p = params('super_learner', n_estimators=2, lgbm_n_estimators=2,
        ridge_n_alphas=2, hidden_layer_sizes=[2])
    model = make_model('super_learner', seed=3, params=p, preprocessor=SimpleImputer()).fit(X,y)
    assert model.diagnostics_['fit_count'] == 16  # 3 single-fit OOF + 12+1 full
    assert all(f['fallback_reason'] for f in model.diagnostics_['fits'][:3])
    assert model.diagnostics_['fits'][-1]['fallback_reason'] is None
    with pytest.raises(ValueError, match='LeaveOneOut'):
        make_model('super_learner', seed=3, params=p, preprocessor=SimpleImputer()).fit(X.iloc[:2],y[:2])
    with pytest.raises(ValueError, match='at least two training rows'):
        make_model('super_learner', seed=3, params=p).fit(X.iloc[:1],y[:1])


@pytest.mark.parametrize('policy', ['auto', 'full', 64])
def test_effective_batch_accounts_for_early_stopping(policy):
    X,y = data(20)
    model = FitBatchMLPRegressor(batch_size=policy, early_stopping=True,
        validation_fraction=.2, hidden_layer_sizes=(2,), max_iter=2, random_state=1).fit(X,y)
    assert model.fit_n_ == 20
    assert model.effective_batch_size_ == 16


@pytest.mark.parametrize('solver', ['adam', 'sgd', 'lbfgs'])
@pytest.mark.parametrize('policy', ['auto', 'full', 2])
def test_effective_batch_solver_diagnostic_preserves_predictions(solver, policy):
    X, y = data(15)
    settings = dict(solver=solver, hidden_layer_sizes=(2,), max_iter=3, random_state=1)
    model = FitBatchMLPRegressor(batch_size=policy, **settings).fit(X, y)
    oracle = MLPRegressor(batch_size=len(y) if policy == 'full' else policy,
                          **settings).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), oracle.predict(X))
    assert model.n_iter_ == oracle.n_iter_
    assert model.batch_size == policy
    assert model.effective_batch_size_ == (2 if policy == 2 and solver != 'lbfgs' else len(y))


def test_lbfgs_search_reports_full_objective_without_changing_search(monkeypatch):
    X, y = data(15)
    calls = []
    original = MLPRegressor.fit

    def fit(self, X, y, **kwargs):
        calls.append((len(y), self.batch_size))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(MLPRegressor, 'fit', fit)
    model = make_model('shallow_neural_network', seed=1, params=params(
        solver='lbfgs', n_alphas=1, mlp_batch_candidates=[2, 7],
        hidden_layer_sizes=[2], max_iter=10)).fit(X, y)
    assert calls == [(10, 2)] * 3 + [(10, 7)] * 3 + [(15, 2)]
    assert model.batch_size_ == 2
    assert model.cv_mse_[0] == model.cv_mse_[1]
    assert model.diagnostics_['fit_count'] == 7
    assert model.diagnostics_['effective_batch'] == 15
    assert model.diagnostics_['solver'] == 'lbfgs'
    assert model.diagnostics_['batch_size_applicable'] is False
    for candidate in model.diagnostics_['candidates']:
        for fold in candidate['folds']:
            assert fold['effective_batch'] == fold['N'] == 10
            assert fold['solver'] == 'lbfgs'
            assert fold['batch_size_applicable'] is False
    json.dumps(model.diagnostics_, allow_nan=False)


@pytest.mark.parametrize('solver', ['adam', 'sgd'])
def test_stochastic_solver_keeps_existing_diagnostic_fields(solver):
    X, y = data(9)
    model = make_model('shallow_neural_network', seed=1, params=params(
        solver=solver, n_alphas=1, mlp_batch_candidates=[2],
        hidden_layer_sizes=[2], max_iter=2)).fit(X, y)
    assert model.diagnostics_['effective_batch'] == 2
    assert 'solver' not in model.diagnostics_
    assert 'batch_size_applicable' not in model.diagnostics_
    for fold in model.diagnostics_['candidates'][0]['folds']:
        assert fold['effective_batch'] == 2
        assert 'solver' not in fold
        assert 'batch_size_applicable' not in fold


def test_configs_and_version():
    from aleatoric_nk_grid.model_registry import load_algorithm_version
    for folder in ['NK_Grid', 'FFCWS', 'SMR']:
        path = ROOT/folder/'model_params.yaml'
        assert load_algorithm_version(path) == 'nk-grid-v7-mlp-batch-cv-1'
        p = load_model_params(path, task='regression', models=['shallow_neural_network','super_learner'])
        assert all(v['mlp_batch_size'] == 'cv' for v in p.values())
        assert p['super_learner']['cv'] == 5
        assert p['super_learner']['mlp_batch_cv_folds'] == 3


def test_serial_and_candidate_models_released(monkeypatch, fast_fits):
    references, threads = [], []
    original = FitBatchMLPRegressor.fit
    def fit(self, X, y, **kw):
        references.append(weakref.ref(self))
        threads.append(threading.get_ident())
        return original(self, X, y, **kw)
    monkeypatch.setattr(FitBatchMLPRegressor, 'fit', fit)
    X,y = data()
    model = make_model('shallow_neural_network', seed=2, params=params()).fit(X,y)
    assert threads == [threading.get_ident()]*61
    assert sum(r() is not None for r in references) == 1
    assert references[-1]() is model.model_[-1].regressor_


@pytest.mark.parametrize('change', ['algorithm_version', 'mlp_batch_candidates', 'mlp_batch_cv_folds', 'rule'])
def test_existing_spec_rejects_changed_search_contract(change, monkeypatch):
    # Execute the existing validator unchanged, without importing POSIX locks.
    # This checks the rejection logic, not Linux checkpoint/Slurm execution.
    from aleatoric_nk_grid.execution_contract import ContractError
    from aleatoric_nk_grid.model_registry import resolved_model_params
    tree = ast.parse((ROOT/'NK_Grid/src/aleatoric_nk_grid/nk_grid.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NKGridExecutionSession')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_spec')
    scope = dict(ContractError=ContractError, resolved_model_params=resolved_model_params)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_validate_spec', 'exec'), scope)
    current = params('super_learner')
    prior = dict(current)
    if change == 'mlp_batch_candidates':
        prior[change] = [32,64]
    if change == 'mlp_batch_cv_folds':
        prior[change] = 2
    spec = dict(resolved_n_grid=[20], resolved_k_grid=[3], resolved_repeat_plan=[[1,0]],
        model_n_jobs=1, algorithm_version='nk-grid-v6-fold-local-1' if change == 'algorithm_version'
        else 'nk-grid-v7-mlp-batch-cv-1', resolved_model_params=resolved_model_params({'super_learner':prior}))
    if change == 'rule':
        from aleatoric_nk_grid import mlp_batch_cv
        monkeypatch.setattr(mlp_batch_cv, 'BATCH_CV_RULE', 'changed-rule')
    session = SimpleNamespace(spec=SimpleNamespace(payload=spec), n_grid=[20], k_grid=[3],
        repeat_pairs=((1,0),), config=SimpleNamespace(n_jobs=1),
        algorithm_version='nk-grid-v7-mlp-batch-cv-1', selected_model_params={'super_learner':current})
    with pytest.raises(ContractError, match='algorithm version mismatch|model parameter mismatch'):
        scope['_validate_spec'](session)

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer
from aleatoric_nk_grid import svd_fallback as sf
from aleatoric_nk_grid.fold_local import FoldLocalRidge


def test_normal_path_bitwise_and_does_not_call_scipy(monkeypatch):
    X = np.random.default_rng(45).normal(size=(30, 40))
    expected = np.linalg.svd(X, full_matrices=False)
    monkeypatch.setattr(sf.linalg, "svd", lambda *a, **kw: pytest.fail("Fallback on normal path"))
    actual, used = sf.ridge_svd(X)
    assert not used
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_fallback_reconstruction_and_input_unchanged(monkeypatch):
    X = np.random.default_rng(123).normal(size=(12, 30)); X[:, 10:] = 0
    original = X.copy()
    def fail(*a, **kw):
        raise np.linalg.LinAlgError("SVD did not converge")
    monkeypatch.setattr(sf.np.linalg, "svd", fail)
    (u, s, vt), used = sf.ridge_svd(X)
    assert used
    np.testing.assert_allclose((u*s)@vt, X, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_input_not_rescued(monkeypatch, value):
    def fail(*a, **kw):
        raise np.linalg.LinAlgError("original failure")
    monkeypatch.setattr(sf.np.linalg, "svd", fail)
    monkeypatch.setattr(sf.linalg, "svd", lambda *a, **kw: pytest.fail("Nonfinite fallback"))
    with pytest.raises(np.linalg.LinAlgError, match="original"):
        sf.ridge_svd(np.array([[value]]))


def test_unrelated_errors_not_caught(monkeypatch):
    def fail(*a, **kw):
        raise ValueError("invalid shape")
    monkeypatch.setattr(sf.np.linalg, "svd", fail)
    with pytest.raises(ValueError):
        sf.ridge_svd(np.ones((3, 3)))


def test_fallback_failure_propagates(monkeypatch):
    def fail(*a, **kw):
        raise np.linalg.LinAlgError("still failed")
    monkeypatch.setattr(sf.np.linalg, "svd", fail)
    monkeypatch.setattr(sf.linalg, "svd", fail)
    with pytest.raises(np.linalg.LinAlgError):
        sf.ridge_svd(np.ones((3, 3)))


def test_real_ridge_cv_fallback_preserves_alpha_and_predictions(monkeypatch):
    X = pd.DataFrame(np.random.default_rng(812).normal(size=(32, 15)))
    y = np.random.default_rng(17).normal(size=32)
    args = dict(preprocessor=FunctionTransformer(), alpha_log10_min=-4,
                alpha_log10_max=6, n_alphas=63, scoring="neg_mean_squared_error")
    baseline = FoldLocalRidge(**args).fit(X, y)
    def fail(*a, **kw):
        raise np.linalg.LinAlgError("forced same failure gate")
    monkeypatch.setattr(sf.np.linalg, "svd", fail)
    fixed = FoldLocalRidge(**args).fit(X, y)
    assert fixed.svd_fallback_count_ == 5
    assert fixed.alpha_ == baseline.alpha_
    np.testing.assert_allclose(fixed.cv_predictions_, baseline.cv_predictions_, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(fixed.predict(X), baseline.predict(X))


def test_super_learner_nested_ridge_uses_fallback(monkeypatch):
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from aleatoric_nk_grid.model_registry import make_model, load_model_params
    path=Path(__file__).resolve().parents[1]/"FFCWS/model_params.yaml"
    params=load_model_params(path,task="regression",models=["super_learner"])["super_learner"]
    # Small integration fixture; production hyperparameters stay unchanged.
    params={**params,"cv":2,"n_estimators":3,"lgbm_n_estimators":3,"max_iter":5,"ridge_n_alphas":7}
    X=pd.DataFrame(np.random.default_rng(99).normal(size=(24,5)))
    y=np.random.default_rng(100).normal(size=24)
    def build():
        return make_model("super_learner",seed=17,n_jobs=1,task="regression",
                          params=params,preprocessor=FunctionTransformer())
    calls=[];original=sf.np.linalg.svd
    def fail(*a,**kw):
        calls.append(1);raise np.linalg.LinAlgError("forced nested SVD failure")
    with threadpool_limits(1):
        baseline=build().fit(X,y)
        monkeypatch.setattr(sf.np.linalg,"svd",fail)
        fixed=build().fit(X,y)
    assert len(calls)==15  # full Ridge and two OOF Ridge fits, five SVDs each
    assert fixed.model_.named_estimators_["ridge"].svd_fallback_count_==5
    np.testing.assert_allclose(fixed.predict(X),baseline.predict(X),rtol=1e-10,atol=1e-10)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from aleatoric_nk_grid.preprocessing import FoldPreprocessor, SourceGroup, preprocess_cell
from aleatoric_nk_grid.fold_local import FoldLocalRidge

IMPUTATION = {'continuous': 'median', 'ordinal': 'most_frequent', 'onehot_group': 'atomic_mode', 'model_overrides': {}}


def process(columns):
    return FoldPreprocessor(tuple(SourceGroup(c, (c,), i, 'continuous', source_prior=0.) for i,c in enumerate(columns)), IMPUTATION, 'ridge')


def test_d1_d3_d5_train_statistics_and_external_extremes():
    train = pd.DataFrame({'a': [1., 3., np.nan], 'b': [1., 2., 8.]})
    model = process(train.columns).fit(train)
    expected = model.transform(train)
    for extreme in (10., 1e20):
        model.transform(pd.DataFrame({'a': [extreme, np.nan], 'b': [extreme, np.nan]}))
        pd.testing.assert_frame_equal(model.transform(train), expected)
        assert model.fill_['a'] == 2.
    # A false full-N fill learns 3 instead of the independently known 2.
    contaminated = process(train.columns).fit(pd.concat([train, pd.DataFrame({'a': [1e20], 'b': [1e20]})]))
    assert contaminated.fill_['a'] != model.fill_['a']


def test_d4_all_missing_overrides_validation_values():
    train = pd.DataFrame({'a': [np.nan, np.nan], 'b': [2., 4.]})
    valid = pd.DataFrame({'a': [99.], 'b': [np.nan]})
    transform = process(train.columns).fit(train)
    assert transform.transform(valid).to_numpy().tolist() == [[0., 3.]]
    old = preprocess_cell(train, valid, transform.groups, IMPUTATION, model_name='ridge')
    pd.testing.assert_frame_equal(old.X_test, transform.transform(valid), check_flags=False)


def test_d6_d7_complete_pipeline_loo_independent_oracle():
    X = pd.DataFrame({'a': [np.nan, 1., 4., 10., 100.], 'b': [2., 4., 1., 5., 1000.]})
    y = np.array([1., 2., -1., 3., 20.])
    model = FoldLocalRidge(process(X.columns), -1, 1, 3, 'neg_mean_squared_error').fit(X, y)
    oracle = np.empty((5, 3))
    for valid in range(5):
        rows = [i for i in range(5) if i != valid]
        train = X.to_numpy()[rows].copy(); test = X.to_numpy()[[valid]].copy()
        fills = np.nanmedian(train, axis=0)
        train = np.where(np.isnan(train), fills, train); test = np.where(np.isnan(test), fills, test)
        mean = train.mean(axis=0); scale = train.std(axis=0); scale[scale == 0] = 1
        train = (train - mean) / scale; test = (test - mean) / scale
        for j, alpha in enumerate([.1, 1., 10.]):
            oracle[valid, j] = Ridge(alpha=alpha).fit(train, y[rows]).predict(test)[0]
    np.testing.assert_allclose(model.cv_predictions_, oracle, rtol=1e-10, atol=1e-12)
    losses = np.mean((oracle - y[:, None]) ** 2, axis=0)
    np.testing.assert_allclose(model.cv_mse_, losses, rtol=1e-10, atol=1e-12)
    assert model.alpha_ == [.1, 1., 10.][np.argmin(losses)]
    assert clone(model).preprocessor.groups == model.preprocessor.groups

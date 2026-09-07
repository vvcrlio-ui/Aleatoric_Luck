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


def test_d4_typed_transform_matches_original_rules():
    from aleatoric_nk_grid.preprocessing import _preprocess_cell_reference
    groups = (SourceGroup('a',('a',),0,'continuous',source_prior=7.),
              SourceGroup('category',('c0','c1'),1,'onehot_group',reference_feature='c0',reference_level=0,level_values=(0,1)),
              SourceGroup('ordinal',('o',),2,'ordinal',ordinal_levels=(0,1,2),source_prior=0))
    rng = np.random.default_rng(382)
    for absent in (False,True):
        for passthrough in (False,True):
            train = pd.DataFrame({'a':rng.normal(size=12), 'c0':[1.,0.]*6, 'c1':[0.,1.]*6, 'o':[0.,1.,2.]*4})
            train.loc[[0,3,5],:]=np.nan
            if absent: train.loc[:,['a','c0','c1']]=np.nan
            valid = pd.DataFrame({'a':[np.nan,20.], 'c0':[np.nan,0.], 'c1':[np.nan,1.], 'o':[np.nan,2.]})
            imputation={**IMPUTATION,'model_overrides':{'ridge':'passthrough'} if passthrough else {}}
            oracle=_preprocess_cell_reference(train,valid,groups,imputation,model_name='ridge')
            model=FoldPreprocessor(groups,imputation,'ridge')
            pd.testing.assert_frame_equal(model.fit_transform(train),oracle.X_train)
            pd.testing.assert_frame_equal(model.transform(valid),oracle.X_test)


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


def test_d1_d2_actual_native_fold_model_invariance(monkeypatch):
    import lightgbm as lgb
    from aleatoric_nk_grid.model_registry import LightGBMCVRegressor
    X = pd.DataFrame({'a': [np.nan, 1., 4., 10., 100., 3., 6., 11., 2., 9.],
                      'b': np.arange(10, dtype=float)})
    y = np.array([1., 2., -1., 3., 20., 2., 5., 10., 0., 8.])
    original = lgb.train; trees = []
    def recording(*args, **kwargs):
        model = original(*args, **kwargs)
        if kwargs.get('valid_sets'):
            trees.append(model.dump_model()['tree_info'])
        return model
    monkeypatch.setattr(lgb, 'train', recording)
    params = dict(seed=11, n_jobs=1, objective='regression', metric='rmse', learning_rate=.1,
                  num_leaves=3, min_data_in_leaf=1, verbosity=-1, max_rounds=3,
                  cv_folds=5, early_stopping_rounds=2, preprocessor=process(X.columns))
    LightGBMCVRegressor(**params).fit(X, y)
    first = trees[0]; trees.clear()
    valid = np.random.RandomState(11).permutation(10)[:2]
    perturbed = X.copy(); perturbed.iloc[valid, :] = 1e8
    target = y.copy(); target[valid] = -1e8
    LightGBMCVRegressor(**params).fit(perturbed, target)
    assert trees[0] == first


def test_d2_d7_mlp_target_and_scale_are_fold_local(monkeypatch):
    from sklearn.neural_network import MLPRegressor
    from aleatoric_nk_grid.model_registry import make_model, load_model_params, DEFAULT_MODEL_PARAMS_PATH
    X = pd.DataFrame({'a': np.arange(9, dtype=float), 'b': np.arange(9, dtype=float)**2 * 1e6})
    y = np.array([1., 2., 1000., 3., 5., 9., 2., 8., 4.])
    params = load_model_params(DEFAULT_MODEL_PARAMS_PATH, task='regression', models=['shallow_neural_network'])['shallow_neural_network']
    params = {**params, 'hidden_layer_sizes':[2], 'max_iter':2, 'n_alphas':2}
    original = MLPRegressor.fit; fits = []
    def recording(self, X, y, **kwargs):
        result = original(self, X, y, **kwargs)
        fits.append((np.asarray(X).copy(), np.asarray(y).copy(), self.coefs_[0].copy()))
        return result
    monkeypatch.setattr(MLPRegressor, 'fit', recording)
    make_model('shallow_neural_network', seed=7, params=params, preprocessor=process(X.columns)).fit(X,y)
    first = fits[0]; fits.clear()
    expected_y = (y[3:] - y[3:].mean()) / y[3:].std()
    np.testing.assert_allclose(first[1], expected_y, rtol=1e-12, atol=1e-12)
    changed = X.copy(); changed.iloc[:3,:] = 1e12
    target = y.copy(); target[:3] = -1e12
    make_model('shallow_neural_network', seed=7, params=params, preprocessor=process(X.columns)).fit(changed,target)
    for old, new in zip(first, fits[0]):
        np.testing.assert_array_equal(old,new)


def test_lasso_one_feature_soft_threshold_oracle():
    from aleatoric_nk_grid.fold_local import FoldLocalLasso
    X = pd.DataFrame({'a':np.arange(7,dtype=float)})
    y = np.array([1.,4.,3.,10.,8.,9.,11.])
    model = FoldLocalLasso(process(X.columns),7,1,-2,0,3,5,20000).fit(X,y)
    validation = ([0,1],[2,3],[4],[5],[6])
    losses=[]
    for valid in validation:
        train=[i for i in range(7) if i not in valid]
        raw=X['a'].to_numpy(); mean=raw[train].mean(); scale=raw[train].std()
        tx=(raw[train]-mean)/scale; vx=(raw[valid]-mean)/scale; target_mean=y[train].mean()
        covariance=np.mean(tx*(y[train]-target_mean))
        fold=[]
        for alpha in (1.,.1,.01):
            coefficient=np.sign(covariance)*max(abs(covariance)-alpha,0)/np.mean(tx**2)
            fold.append(np.mean((vx*coefficient+target_mean-y[valid])**2))
        losses.append(fold)
    np.testing.assert_allclose(model.cv_mse_,np.mean(losses,axis=0),rtol=1e-10,atol=1e-12)


def test_super_oof_columns_are_held_out(monkeypatch):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from aleatoric_nk_grid.model_registry import make_model, load_model_params, DEFAULT_MODEL_PARAMS_PATH
    X=pd.DataFrame({'a':np.arange(10,dtype=float),'b':np.arange(10,dtype=float)**2})
    y=np.random.default_rng(319).normal(size=10)
    params=load_model_params(DEFAULT_MODEL_PARAMS_PATH,task='regression',models=['super_learner'])['super_learner']
    params={**params,'n_estimators':2,'lgbm_n_estimators':2,'max_iter':2,'ridge_n_alphas':3,'hidden_layer_sizes':[2]}
    original=LinearRegression.fit; received=[]
    def capture(self,X,y,**kwargs):
        received.append(np.asarray(X).copy())
        return original(self,X,y,**kwargs)
    monkeypatch.setattr(LinearRegression,'fit',capture)
    model=make_model('super_learner',seed=19,n_jobs=1,params=params,preprocessor=process(X.columns)).fit(X,y)
    assert len(received)==1
    expected=np.empty((10,4))
    for train,valid in KFold(5).split(X):
        for column,(_,estimator) in enumerate(model.model_.estimators):
            expected[valid,column]=clone(estimator).fit(X.iloc[train],y[train]).predict(X.iloc[valid])
    np.testing.assert_allclose(received[0],expected,rtol=1e-10,atol=1e-12)

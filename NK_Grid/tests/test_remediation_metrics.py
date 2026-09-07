import math
import pytest
from aleatoric_nk_grid.evaluation import regression_denominators


@pytest.mark.parametrize('pred', [[0,2], [2,2], [1,1], [10,-10]])
def test_e1_e2_direct_sum_oracle(pred):
    test, train = [0,2], [2,2]
    got = regression_denominators(test,pred,train)
    mse = sum((a-b)**2 for a,b in zip(test,pred)) / 2
    assert got == {'mse':mse, 'null_mse_train_mean':2., 'test_target_variance':1.,
                   'skill_train_mean':1-mse/2, 'r2_test_mean':1-mse}


def test_e3_zero_denominators():
    got = regression_denominators([2],[3],[2])
    assert got['mse'] == 1
    assert math.isnan(got['skill_train_mean']) and math.isnan(got['r2_test_mean'])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_e3_nonfinite_rejected(bad):
    with pytest.raises(ValueError,match='finite'):
        regression_denominators([0,2],[0,bad],[1,2])


def test_e4_affine_and_e5_train_mean():
    a = regression_denominators([0,2],[2,2],[2])
    b = regression_denominators([7,1],[1,1],[1])
    assert b['mse'] == 9*a['mse']
    assert b['skill_train_mean'] == a['skill_train_mean']
    assert b['r2_test_mean'] == a['r2_test_mean']
    c = regression_denominators([0,2],[2,2],[1])
    assert c['mse'] == a['mse'] and c['r2_test_mean'] == a['r2_test_mean']
    assert c['skill_train_mean'] != a['skill_train_mean']

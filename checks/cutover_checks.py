from datetime import datetime, timezone
import copy
import pytest
from aleatoric_nk_grid.cutover_guard import blockers


def values():
    now=datetime(2026,9,11,7,tzinfo=timezone.utc)
    state=dict(timezone='Europe/Paris',not_before_local_date='2026-09-11',
        deployment_ready=True,production_switched=False,old_run_id='r',old_analysis_id='a',expected_per_booster=2_000_000)
    common=dict(run_id='r',analysis_id='a',captured_at=now.isoformat())
    coverage=dict(**common,verified_full_unique_design=True,
        valid_by_model=dict(xgboost=2_000_000,lightgbm=2_000_000),failed_by_model=dict(xgboost=0,lightgbm=0))
    jobs=dict(**common,query_ok=True,exhaustive_old_run_scope=True,control_lock_held=True,
        jobs=[dict(job_id='controller',state='PENDING',never_started=True)])
    return state,coverage,jobs,now


@pytest.mark.parametrize('status',['RUNNING','COMPLETING','CONFIGURING','SUSPENDED','STAGE_OUT','UNKNOWN','PENDING'])
def test_even_one_active_or_uncertain_job_blocks(status):
    s,c,j,n=values();j['jobs'].append(dict(job_id='worker',state=status))
    assert 'active_or_unknown_job:worker' in blockers(s,c,j,now=n)


def test_full_verified_and_quiescent_required():
    s,c,j,n=values();assert blockers(s,c,j,now=n)==[]
    changes=[('state','deployment_ready',False),('state','production_switched',True),
        ('coverage','verified_full_unique_design',False),('coverage','run_id','other'),
        ('jobs','query_ok',False),('jobs','exhaustive_old_run_scope',False),('jobs','control_lock_held',False),
        ('jobs','jobs',None),('jobs','captured_at','2026-09-11T06:50:00+00:00'),
        ('coverage','failed_by_model',{'xgboost':0,'lightgbm':1}),
        ('coverage','valid_by_model',{'xgboost':2_000_000,'lightgbm':1_999_999})]
    for target,key,value in changes:
        ss,cc,jj=copy.deepcopy((s,c,j));dict(state=ss,coverage=cc,jobs=jj)[target][key]=value
        assert blockers(ss,cc,jj,now=n),(target,key)
    s['not_before_local_date']='2026-09-12'
    assert 'not_before_date' in blockers(s,c,j,now=n)

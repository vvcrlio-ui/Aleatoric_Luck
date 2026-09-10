import csv
from dataclasses import asdict
import json
import pytest
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, atomic_json, canonical, digest, file_digest
from aleatoric_nk_grid.result_migration import OLD_ALGORITHM, NEW_ALGORITHM
from aleatoric_nk_grid.pending_resume import Design, prepare, merge
from aleatoric_nk_grid.scheduler_cost import CostEstimator


def fixture(tmp_path, *, duplicate=False, conflict=False, complete=False):
    old = dict(algorithm_version=OLD_ALGORITHM, git_commit='old', resolved_n_grid=[10,20],
        resolved_k_grid=[1], resolved_repeat_plan=[[1,0]], models=['ols','ridge'])
    new = {**old, 'algorithm_version': NEW_ALGORITHM, 'git_commit': 'new'}
    cert = {'policy':'exact-scheduler-plus-conditional-gesvd-v1', 'files':{'fold_local.py':{'old':'a','new':'b'}}}
    rows = []
    for n in (10,20):
        for model in old['models']:
            row = dict(seed=1,draw=0,N=n,K=1,model=model,status='ok',error='',mse='.25',rmse='.5',mae='.4',algorithm_version=OLD_ALGORITHM)
            if n==20 and not complete:
                if model=='ridge':
                    continue
                row['status']='failed'; row['error']='SVD'
            rows.append(row)
    if duplicate or conflict:
        rows.append({**rows[0], **({'mse':'.3'} if conflict else {})})
    bundle = tmp_path/'bundle';bundle.mkdir()
    (bundle/'results.jsonl').write_bytes(b''.join(canonical({'result':row,'origin':{
        'analysis_id':'old-analysis','payload_sha256':digest(row)}})+b'\n' for row in rows))
    atomic_json(bundle/'manifest.json', dict(format='sealed-legacy-export-v1',sealed=True,
        source_analysis_id='old-analysis', cell_spec=old, public_columns=list(rows[0]), rows=len(rows),
        results_sha256=file_digest(bundle/'results.jsonl')))
    args = dict(new_spec=new,certificate=cert,expected_manifest_sha256=file_digest(bundle/'manifest.json'),scratch=tmp_path/'scratch')
    return bundle,args


def finish_queue(tmp_path, resumed):
    path=tmp_path/'new.jsonl'
    with Dispatcher(resumed/'queue',scratch=tmp_path/'scratch') as q:
        while True:
            lease=q.claim('w')
            if lease['state']=='complete':break
            assert lease['task']['N']==20
            q.submit(lease['id'],lease['token'],'w',{**lease['task'],'status':'ok','error':'',
                'mse':.25,'rmse':.5,'mae':.4,'algorithm_version':NEW_ALGORITHM,'_fit_seconds':.1})
        q.export_results(path)
    return path


def test_missing_and_failed_only_resume_then_standard_csv(tmp_path):
    bundle,args=fixture(tmp_path,duplicate=True)
    resumed=tmp_path/'resume'
    receipt=prepare(bundle,resumed,**args)
    assert (receipt['old_valid_unique'],receipt['pending'],receipt['identical_old_duplicates'])==(2,2,1)
    with Dispatcher(resumed/'queue',scratch=tmp_path/'scratch') as q:
        assert q.stats()['total']==2
        assert not (resumed/'queue/events.jsonl').read_text().count('"kind":"import"')
    new_results=finish_queue(tmp_path,resumed)
    output=tmp_path/'results.csv'
    report=merge(output,resumed=resumed,new_results=new_results,scratch=tmp_path/'scratch')
    with output.open(newline='') as f:
        reader=csv.DictReader(f); rows=list(reader)
        assert reader.fieldnames==json.loads((bundle/'manifest.json').read_bytes())['public_columns']
    assert len(rows)==4 and report['validated_complete']
    assert not {'origin','source','_fit_seconds'} & rows[0].keys()
    assert report['algorithm_versions']=={OLD_ALGORITHM:2,NEW_ALGORITHM:2}


def test_conflicting_old_rows_never_publish_ready(tmp_path):
    bundle,args=fixture(tmp_path,conflict=True)
    with pytest.raises(QueueError,match='Conflicting'):
        prepare(bundle,tmp_path/'resume',**args)
    assert not (tmp_path/'resume/ready.json').exists()


@pytest.mark.parametrize('variant',['missing','duplicate','failed','wrong_origin','version','metric','columns'])
def test_bad_new_results_never_publish_csv(tmp_path,variant):
    bundle,args=fixture(tmp_path)
    resumed=tmp_path/'resume';prepare(bundle,resumed,**args)
    path=finish_queue(tmp_path,resumed)
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    if variant=='missing':rows.pop()
    elif variant=='duplicate':rows.append(rows[0])
    elif variant=='failed':rows[0]['result']['status']='failed'
    elif variant=='wrong_origin':rows[0]['origin']['queue_id']='wrong'
    elif variant=='version':rows[0]['result']['algorithm_version']=OLD_ALGORITHM
    elif variant=='metric':rows[0]['result']['mse']='nan'
    elif variant=='columns':del rows[0]['result']['error']
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
    with pytest.raises(QueueError):
        merge(tmp_path/'final.csv',resumed=resumed,new_results=path,scratch=tmp_path/'scratch')
    assert not (tmp_path/'final.csv').exists()


def test_complete_old_bundle_needs_no_new_queue(tmp_path):
    bundle,args=fixture(tmp_path,complete=True)
    resumed=tmp_path/'resume';receipt=prepare(bundle,resumed,**args)
    assert receipt['queue_id'] is None and not (resumed/'queue').exists()
    assert merge(tmp_path/'final.csv',resumed=resumed,scratch=tmp_path/'scratch')['validated_complete']


def test_design_compact_and_strict_keys():
    d=Design(dict(resolved_n_grid=list(range(1,21)),resolved_k_grid=list(range(1,21)),
        resolved_repeat_plan=[[seed,draw] for seed in range(100) for draw in range(50)],models=list('abcdefghi')))
    assert d.count==18_000_000 and len(d.bits)==2_250_000
    assert d.ordinal(dict(seed=99,draw=49,N=20,K=20,model='i'))==17_999_999
    for value in (20.5,True,'20.5'):
        with pytest.raises(QueueError):d.ordinal(dict(seed=99,draw=49,N=value,K=20,model='i'))
    with pytest.raises(QueueError):d.ordinal(dict(seed=999,draw=49,N=20,K=20,model='i'))


def test_measured_costs_can_rank_ridge_ahead_of_sl_and_validate_profile():
    profile={'format':'model-cell-cost-v1','evidence':{'kind':'synthetic test'},'expanded_by_k':{'10':40},
        'samples':[{'model':'ridge','N':100,'K_expanded':40,'seconds':60},
                   {'model':'super_learner','N':100,'K_expanded':40,'seconds':30}]}
    cost=CostEstimator(profile=profile)
    assert cost.estimate('ridge',100,10)==60
    assert cost.estimate('ridge',200,10)==120
    assert cost.estimate('ridge',100,10)>cost.estimate('super_learner',100,10)
    with pytest.raises(QueueError):cost.estimate('ols',100,10)
    with pytest.raises(QueueError):CostEstimator(profile={**profile,'evidence':None})

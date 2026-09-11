"""Full read-only audit of every captured WAL prefix on a compute node."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import json, math, hashlib, time, traceback
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from array import array
sys.path.insert(0, '/valhalla/projects/ehpc-dev-2026d08-299/aleatoric_luck-goal-20260908/runs/ffc-gpa-production-20260909')
import monitor as m
from aleatoric_nk_grid.execution_contract import task_row_digest, sha256_file, resolve_repo_locator
from aleatoric_nk_grid.task_table import read_row_group, _row_id
from aleatoric_nk_grid.config import execution_groups_for_models

OUT=Path(sys.argv[1]).resolve()
CTX={}
def now(): return datetime.now(timezone.utc).isoformat()
def save(name,value):
    target=OUT/name
    temp=target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,target)

class PrefixReader:
    def __init__(self,handle,limit): self.handle,self.limit=handle,limit; self.sha=hashlib.sha256()
    def read(self,n=-1):
        remaining=max(0,self.limit-self.handle.tell())
        data=self.handle.read(remaining if n<0 else min(n,remaining)); self.sha.update(data); return data
    def tell(self): return self.handle.tell()
    def seek(self,n): return self.handle.seek(n)

def diagnostic_errors(value):
    if isinstance(value,dict):
        if value.get('error'): yield str(value['error'])
        for child in value.values(): yield from diagnostic_errors(child)
    elif isinstance(value,list):
        for child in value: yield from diagnostic_errors(child)

def audit_worker(item):
    path=Path(item['path']); worker=item['worker']
    stats=dict(worker=worker,captured_bytes=item['size'],checked_bytes=0,started=0,completed_tasks=0,
               valid_results=0,failed_results=0,aborted=0,interrupted_starts=0,errors=[],by_model={},
               assigned_tasks=0,uncommitted_tail_bytes=0,last_completed_cell=None)
    began=time.monotonic(); valid_keys=array('I')
    try:
        assigned=read_row_group(Path(CTX['assignment']),worker)
        own=CTX['index'][worker]
        assert len(assigned)==own['row_count']
        assert task_row_digest(assigned)==own['canonical_task_rows_sha256']
        stats['assigned_tasks']=len(assigned)
        ids={}
        canonical_indices=set()
        for row in assigned:
            assert row.models==CTX['groups'][row.group]
            assert row.row_id==_row_id(row.seed,row.draw,row.n_samples,row.k_features,row.group,row.models)
            assert CTX['seed0']<=row.seed<CTX['seed0']+100 and 0<=row.draw<50
            idx=(((CTX['ks'].index(row.k_features)*20+CTX['ns'].index(row.n_samples))*5000+(row.seed-CTX['seed0'])*50+row.draw)*2+CTX['group_names'].index(row.group))
            assert idx%CTX['workers']==worker, 'Assignment differs from first-round canonical modulo distribution'
            assert idx not in canonical_indices and row.row_id not in ids, 'Duplicate assigned task'
            canonical_indices.add(idx);ids[row.row_id]=row
        counters=Counter();terminal=set();prior=-1;active=None
        with path.open('rb') as raw_handle:
            handle=PrefixReader(raw_handle,item['size'])
            first=m.frame(handle)
            assert first and first[0]['event_type']=='IDENTITY'
            identity=json.loads(first[1])
            expected=dict(wal_format=m.wal.WAL_FORMAT,analysis_id=CTX['analysis_id'],execution_plan_id=CTX['round']['execution_plan_id'],
                execution_contract_sha256=CTX['execution_sha'],round=CTX['round']['round'],submission_generation=CTX['round']['generation'],
                worker=worker,workers=CTX['workers'],assignment_path=CTX['assignment'],assignment_sha256=CTX['assignment_sha'],
                assignment_index_path=CTX['index_path'],assignment_index_sha256=CTX['index_sha'],assignment_row_group=worker,
                assignment_row_count=len(assigned),assignment_row_group_digest=own['canonical_task_rows_sha256'])
            assert identity==expected,'WAL identity mismatch'
            stats['checked_bytes']=handle.tell()
            while True:
                record=m.frame(handle)
                if record is None: break
                header,payload=record
                kind,seq,rowid=header['event_type'],header['sequence'],header['row_id']
                assert rowid in ids,'Unassigned WAL task'
                if kind=='TASK_STARTED':
                    assert seq==prior+1,'Sequence gap or duplicate'
                    assert rowid not in terminal,'Repeated completed task'
                    if active is not None: stats['interrupted_starts']+=1
                    active=(seq,rowid);prior=seq;stats['started']+=1
                else:
                    assert kind in ('TASK_RESULT','TASK_ABORTED') and active==(seq,rowid),'Invalid result/start transition'
                    assert rowid not in terminal,'Duplicate terminal task'
                    terminal.add(rowid);active=None
                    if kind=='TASK_ABORTED':
                        stats['aborted']+=1
                        stats['errors'].append({'row_id':rowid,'aborted':payload.decode()})
                    else:
                        columns,results=m.wal.decode_public_rows(payload)
                        assert list(columns)==CTX['header'],'Result header mismatch'
                        row=ids[rowid]
                        assert len(results)==len(row.models) and {r['model'] for r in results}==set(row.models),'Model keys mismatch'
                        for result in results:
                            assert tuple(int(result[k]) for k in ('seed','draw','N','K'))==(row.seed,row.draw,row.n_samples,row.k_features),'Result cell differs from assignment'
                            assert result['algorithm_version']==m.ALGORITHM,'Wrong algorithm'
                            problems=[]
                            if result['status']!='ok' or result.get('error'): problems.append(result.get('error') or result['status'])
                            for metric in ('mse','rmse','mae'):
                                value=float(result[metric])
                                if not math.isfinite(value) or value<0: problems.append('Invalid '+metric)
                            for metric in ('r2_test','r2_test_mean'):
                                if result.get(metric) and not math.isfinite(float(result[metric])): problems.append('Nonfinite '+metric)
                            if result.get('mlp_diagnostics_json'):
                                problems.extend(diagnostic_errors(json.loads(result['mlp_diagnostics_json'])))
                            if problems:
                                stats['failed_results']+=1
                                if len(stats['errors'])<20: stats['errors'].append({'row_id':rowid,'model':result['model'],'problems':problems})
                            else:
                                counters[result['model']]+=1;stats['valid_results']+=1
                                ordinal=(((CTX['ks'].index(row.k_features)*20+CTX['ns'].index(row.n_samples))*5000+(row.seed-CTX['seed0'])*50+row.draw)*9+CTX['models'].index(result['model']))
                                valid_keys.append(ordinal)
                        stats['completed_tasks']+=1
                        stats['last_completed_cell']={'N':row.n_samples,'K':row.k_features,'seed':row.seed,'draw':row.draw,'group':row.group}
                stats['checked_bytes']=handle.tell()
        stats['uncommitted_tail_bytes']=item['size']-stats['checked_bytes']
        stats['active_task']=active
        stats['by_model']=dict(counters)
        stats['post_scan_bytes']=path.stat().st_size
        assert stats['post_scan_bytes']==item['size'] and stats['uncommitted_tail_bytes']==0, 'Stopped WAL changed or contains a partial tail'
        stats['wal_sha256']=handle.sha.hexdigest()
        stats['wal_path']=str(path)
        with (OUT/('keys-'+str(worker)+'.bin')).open('wb') as keys_file: valid_keys.tofile(keys_file)
        stats['full_captured_prefix_checked']=True
    except Exception as exc:
        stats['errors'].append({'exception':repr(exc),'traceback':traceback.format_exc()})
        stats['full_captured_prefix_checked']=False
    stats['seconds']=time.monotonic()-began
    return stats

def main():
    assert 'al-17820c4142ba439b-' not in m.command(['squeue','-r','-u','xwan','-h','-o','%j']), 'Old GPA jobs must remain stopped'
    OUT.mkdir(parents=True,exist_ok=False)
    assert not (OUT/'report.json').exists(),'Already completed; do not overwrite audit'
    state=m.read(m.RUN/'continuation.json');plan=m.read(m.RUN/'plan.json');snapshot=m.read(plan['snapshot'])
    assert len(state['rounds'])==1 and state['run_id']=='17820c4142ba439b82911477a3b4b958'
    assert sha256_file(m.RUN/'plan.json')==state['plan_sha256']
    assert m.command(['git','-C',str(m.REPO),'rev-parse','HEAD'])==m.COMMIT
    assert not m.command(['git','-C',str(m.REPO),'status','--porcelain'])
    analysis=m.AnalysisContract.from_payload(m.read(snapshot['analysis_contract']))
    assert analysis.analysis_id==plan['analysis_id']
    cell=analysis.payload['cell_execution_spec']
    save('scientific-identity.json', {'cell_spec':cell,'public_columns':analysis.payload['public_result_schema']['columns']})
    assert cell['git_commit']==m.COMMIT and cell['algorithm_version']==m.ALGORITHM
    for locator,digest in ((cell['schema_locator'],cell['schema_file_sha256']),(cell['model_params_locator'],cell['model_params_sha256'])):
        resolve_repo_locator(locator,digest,repo_root=m.REPO)
    for binding in cell['input_provenance'].values(): resolve_repo_locator(binding['path'],binding['sha256'],repo_root=m.REPO)
    rnd=state['rounds'][0]
    directory=Path(snapshot['output_dir'])/'executions'/rnd['execution_plan_id']/('round-'+str(rnd['round']))/('generation-'+rnd['generation'])
    paths=sorted(directory.glob('worker-*.events.wal'))
    assert len(paths)==rnd['resources']['workers']==698
    with paths[0].open('rb') as handle: identity=json.loads(m.frame(handle)[1])
    assert sha256_file(identity['assignment_path'])==identity['assignment_sha256']
    assert sha256_file(identity['assignment_index_path'])==identity['assignment_index_sha256']
    from aleatoric_nk_grid.execution_contract import DynamicExecutionContract
    execution=DynamicExecutionContract.from_payload(m.read(Path(snapshot['output_dir'])/'execution-contracts'/(rnd['execution_plan_id']+'.json')))
    assert execution.execution_plan_id==rnd['execution_plan_id']
    assert sha256_file(plan['task_table'])==execution.payload['task_table_file_sha256']
    groups=dict(execution_groups_for_models(snapshot['config']['models']))
    CTX.update(assignment=identity['assignment_path'],assignment_sha=identity['assignment_sha256'],index_path=identity['assignment_index_path'],
        index_sha=identity['assignment_index_sha256'],index={r['worker']:r for r in m.read(identity['assignment_index_path'])['row_groups']},
        groups=groups,group_names=list(groups),models=cell['models'],workers=len(paths),seed0=snapshot['config']['seed'],ns=cell['resolved_n_grid'],ks=cell['resolved_k_grid'],
        analysis_id=analysis.analysis_id,round=rnd,execution_sha=execution.sha256,header=analysis.payload['public_result_schema']['columns'])
    frontier_start=now()
    items=[dict(path=str(p),worker=int(p.name.split('-')[1].split('.')[0]),size=p.stat().st_size) for p in paths]
    frontier_end=now()
    save('frontier.json',dict(captured_from=frontier_start,captured_to=frontier_end,run_id=state['run_id'],analysis_id=analysis.analysis_id,files=items))
    print(json.dumps({'phase':'captured','files':len(items),'bytes':sum(i['size'] for i in items),'time':frontier_end}),flush=True)
    totals=Counter();by_model=Counter();results=[];began=time.monotonic()
    import multiprocessing
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(audit_worker,item) for item in items]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            for key in ('valid_results','failed_results','aborted','completed_tasks','assigned_tasks','checked_bytes','uncommitted_tail_bytes','interrupted_starts'):
                totals[key]+=result[key]
            by_model.update(result['by_model'])
            with (OUT/'workers.jsonl').open('a') as handle: handle.write(json.dumps(result)+'\n')
            if len(results)%20==0 or len(results)==len(items):
                progress=dict(time=now(),workers_checked=len(results),workers_expected=len(items),totals=dict(totals),by_model=dict(by_model),
                    errors=sum(bool(r['errors']) for r in results),elapsed_seconds=time.monotonic()-began)
                save('progress.json',progress);print(json.dumps(progress),flush=True)
    final=dict(started_at_utc=frontier_start,frontier_captured_to_utc=frontier_end,finished_at_utc=now(),run_id=state['run_id'],analysis_id=analysis.analysis_id,
        scope='Every byte/record in all captured first-round WAL prefixes; newly appended records after frontier excluded',
        captured_bytes=sum(i['size'] for i in items),workers_checked=len(results),workers_expected=len(items),totals=dict(totals),by_model=dict(by_model),
        all_prefixes_checked=all(r['full_captured_prefix_checked'] for r in results),errors=[r for r in results if r['errors']],
        complete_experiment=False,seconds=time.monotonic()-began)
    assert totals['assigned_tasks']==4000000,'Assignment coverage differs from full design'
    final['audit_passed']=final['all_prefixes_checked'] and not final['errors'] and totals['failed_results']==totals['aborted']==0
    save('report.json',final)
    print(json.dumps(final),flush=True)

if __name__=='__main__': main()

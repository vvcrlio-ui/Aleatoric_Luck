"""Bounded local GPA throughput study, three physical cores, two rotated repeats.

Uses original full model/CV budgets. Windows numerical calls are in-process in
ALL arms. Real queue RPC, per-model journal fsync, raw inputs and outer diagnostic
cache are included. Linux native isolation, Lustre/network and Slurm excluded.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parents[2]
OUT=ROOT.parent/'gpa-benchmark-01'
TMP=OUT/'tmp'
os.environ['TEMP']=os.environ['TMP']=str(TMP)
import sys,time,json,hashlib,threading,ctypes,struct,types
from ctypes import wintypes as wt
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
sys.path.insert(0,str(ROOT/'NK_Grid/src'))
from aleatoric_nk_grid.shared_queue import Dispatcher,ModelTask,canonical,file_digest
from aleatoric_nk_grid.queue_service import Client,make_server,execute_worker

MODELS=('ols','ridge','lasso','random_forest','shallow_neural_network','extra_trees','super_learner')
CELLS=((122,47),(122,400),(907,261),(122,261))
WEIGHTS=dict(zip(MODELS,(1,5,5,1,15,1,40)))
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
psapi=ctypes.WinDLL('psapi',use_last_error=True)
kernel.GetCurrentProcess.restype=wt.HANDLE
kernel.SetProcessAffinityMask.argtypes=[wt.HANDLE,ctypes.c_size_t]
kernel.OpenProcess.argtypes=[wt.DWORD,wt.BOOL,wt.DWORD];kernel.OpenProcess.restype=wt.HANDLE
kernel.CloseHandle.argtypes=[wt.HANDLE]
class Counters(ctypes.Structure):
    _fields_=[('cb',wt.DWORD),('faults',wt.DWORD)]+[(x,ctypes.c_size_t) for x in ('peak_ws','ws','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile','private')]
psapi.GetProcessMemoryInfo.argtypes=[wt.HANDLE,ctypes.POINTER(Counters),wt.DWORD]
def core_masks():
    n=wt.DWORD(0);kernel.GetLogicalProcessorInformation(None,ctypes.byref(n))
    buf=ctypes.create_string_buffer(n.value)
    if not kernel.GetLogicalProcessorInformation(buf,ctypes.byref(n)):raise ctypes.WinError(ctypes.get_last_error())
    masks=[]
    for offset in range(0,n.value,32):
        mask,relationship=struct.unpack_from('QI',buf.raw,offset)
        if relationship==0:masks.append(mask & -mask)
    assert len(masks)>=3
    return masks[:3]
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(data)+b'\n')
def clean(value):
    import numpy as np
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def initialize(mask):
    global session,nk,predictions,cache
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),mask):raise ctypes.WinError(ctypes.get_last_error())
    def forbid(*a,**kw):raise RuntimeError('POSIX persistence not part of Windows numerical benchmark')
    sys.modules['fcntl']=types.SimpleNamespace(flock=forbid,LOCK_EX=2,LOCK_SH=1,LOCK_NB=4,LOCK_UN=8)
    sys.modules['resource']=types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:types.SimpleNamespace(ru_maxrss=0))
    import numpy as np
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.config import NKGridConfig
    from aleatoric_nk_grid.cell_cache import session_namespace
    from threadpoolctl import threadpool_limits
    global limits
    limits=threadpool_limits(1)
    nk.SERIAL_OUTER_MODELS=frozenset()
    original_fit=nk._fit_predict_model_cell
    predictions={};cache=None
    def trace(**kwargs):
        value=original_fit(**kwargs)
        predictions[kwargs['model_name']]=np.asarray(value['predictions']).copy()
        return value
    nk._fit_predict_model_cell=trace
    config=NKGridConfig(schema=ORIGINAL/'FFCWS/data/remediation_20260907/schema/ffc_median_mode_gpa.json',
        out=OUT/'unused.csv',outcome='gpa',models=MODELS,seed=12345,test_size=.2,n_seeds=1,n_draws=1,
        n_sizes_n=4,n_sizes_k=5,max_n=1165,max_k=3400,batch_size=1,n_jobs=1,
        n_grid=(10,122,907,1165),k_grid=(1,47,261,400,3400),model_params=ROOT/'FFCWS/model_params.yaml')
    t=time.perf_counter();session=nk.NKGridExecutionSession.open_from_config(config)
    session_namespace(session)
    session.run_cell_group(seed=12345,draw=0,n_samples=10,k_features=1,models=MODELS)
    return {'pid':os.getpid(),'mask':mask,'load_and_warmup_seconds':time.perf_counter()-t}
def configure(arm,run):
    global cache
    from aleatoric_nk_grid.cell_cache import CachedSession,NodeInputStore,session_namespace
    if cache:cache.close()
    cache=(CachedSession(session,max_bytes=256*1024**2,store=NodeInputStore(OUT/run/'node-cache',
        namespace=session_namespace(session),max_bytes=1024**3)) if arm=='queue_cache' else None)
    import gc;gc.collect()
    return True
def execute(values):
    global predictions
    predictions={};start=time.perf_counter();cpu=time.process_time()
    task=ModelTask(**values[0]);models=tuple(v['model'] for v in values)
    if cache:
        assert len(values)==1
        rows=[cache.run(task)]
    else:
        rows=session.run_cell_group(seed=task.seed,draw=task.draw,n_samples=task.N,k_features=task.K,models=models)
    for row in rows:
        assert row['status']=='ok',(row['model'],row['error'])
        pred=predictions[row['model']]
        row['_prediction_sha256']=hashlib.sha256(pred.tobytes()).hexdigest()
        if task.N==122 and task.K==400:
            import numpy as np
            ref=ORIGINAL/'runs/seven-model-parallel-20260910-corrected/predictions'/f'N122-K400-serial-0-{row["model"]}.npy'
            if not np.array_equal(pred,np.load(ref)):
                raise AssertionError('Mismatch against saved ORIGINAL production-version prediction: '+row['model'])
    return {'rows':clean(rows),'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
        'cache_stats':dict(cache.stats) if cache else {},'cache_bytes':cache.bytes if cache else 0,
        'cached_cells':cache.cached_cells if cache else []}
def main():
    if OUT.exists():raise RuntimeError('Fresh output required')
    TMP.mkdir(parents=True)
    started=time.monotonic();deadline=started+2400
    masks=core_masks()
    save(OUT/'protocol.json',{'cells':CELLS,'models':MODELS,'workers':3,'physical_core_masks':masks,'repeats':2,
        'max_wall_seconds':2400,'seed':12345,'draw':0,'algorithm':'conditional-gesvd-new-version in all arms',
        'arms':['fixed_groups','queue_single','queue_cache'],'baseline':'four imputed7 cell groups statically modulo-assigned to three workers',
        'scope':__doc__,'cost_weights':WEIGHTS,'memory_cache_bytes_per_worker':256*1024**2,'shared_disk_cap':1024**3,
        'source_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in (ROOT/'NK_Grid/src/aleatoric_nk_grid').glob('*.py')}})
    pools=[ProcessPoolExecutor(max_workers=1) for _ in masks]
    try:
        ready=[pool.submit(initialize,mask) for pool,mask in zip(pools,masks)]
        ready=[future.result(timeout=240) for future in ready];save(OUT/'startup.json',ready)
        handles=[kernel.OpenProcess(0x410,False,r['pid']) for r in ready]
        handles.append(kernel.OpenProcess(0x410,False,os.getpid()))
        all_records=[];reference={}
        for repeat in range(2):
            arms=('fixed_groups','queue_single','queue_cache') if repeat==0 else ('queue_cache','queue_single','fixed_groups')
            for arm in arms:
                if time.monotonic()>deadline:raise TimeoutError('Benchmark budget')
                run=f'{repeat}-{arm}'
                for pool in pools:pool.submit(configure,arm,run).result(timeout=30)
                root=OUT/run;root.mkdir(exist_ok=True)
                done=threading.Event();samples=[]
                def sample():
                    while not done.is_set():
                        ws=private=0
                        for handle in handles:
                            c=Counters();c.cb=ctypes.sizeof(c)
                            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
                            ws+=int(c.ws);private+=int(c.private)
                        samples.append({'elapsed':time.perf_counter()-t,'working_set':ws,'private_commit':private})
                        done.wait(.2)
                records=[];lock=threading.Lock();worker_busy=[0.]*3
                q=server=service_thread=None
                tasks=[ModelTask(12345,0,n,k,m) for n,k in CELLS for m in MODELS]
                if arm!='fixed_groups':
                    Dispatcher.create(root/'queue',((task,WEIGHTS[task.model]*task.N*(task.K**.5)) for task in tasks),identity={'benchmark':run},lease_seconds=120)
                    q=Dispatcher(root/'queue',scratch=root/'scratch')
                    server=make_server(q,token='local-benchmark-token-'+ 'x'*32)
                    service_thread=threading.Thread(target=server.serve_forever);service_thread.start()
                    client=Client(f'http://127.0.0.1:{server.server_port}','local-benchmark-token-'+'x'*32,q.queue_id)
                t=time.perf_counter();sampler=threading.Thread(target=sample);sampler.start()
                def collect(i,values):
                    packet=pools[i].submit(execute,values).result(timeout=max(1,deadline-time.monotonic()))
                    with lock:records.append(packet);worker_busy[i]+=packet['seconds']
                    return packet
                def worker(i):
                    if arm=='fixed_groups':
                        for pos,(n,k) in enumerate(CELLS):
                            if pos%3==i:
                                packet=collect(i,[{'seed':12345,'draw':0,'N':n,'K':k,'model':m} for m in MODELS])
                                with (root/f'worker-{i}.jsonl').open('ab') as f:
                                    f.write(canonical(packet)+b'\n');f.flush();os.fsync(f.fileno())
                    else:
                        cells=[]
                        def numerical(value):
                            packet=collect(i,[value]);cells[:]=packet['cached_cells'];return packet['rows'][0]
                        return execute_worker(client,f'w{i}',numerical,spool=root/f'spool-{i}',cached_cells=lambda:cells,
                            heartbeat_seconds=20,deadline_seconds=max(1,deadline-time.monotonic()))
                try:
                    with ThreadPoolExecutor(max_workers=3) as threads:
                        reports=list(threads.map(worker,range(3)))
                    wall=time.perf_counter()-t
                finally:
                    done.set();sampler.join()
                    if server:server.shutdown();service_thread.join();server.server_close()
                if q:
                    assert q.stats()['done']==len(tasks);q.close()
                flat=[row for packet in records for row in packet['rows']]
                for row in flat:
                    key=(row['seed'],row['draw'],row['N'],row['K'],row['model'])
                    fingerprint=row['_prediction_sha256']
                    if key in reference:assert reference[key]==fingerprint,('prediction mismatch',key,run)
                    else:reference[key]=fingerprint
                assert len(flat)==28
                record={'run':run,'arm':arm,'repeat':repeat,'wall_seconds':wall,'worker_busy_seconds':worker_busy,
                    'reserved_core_seconds':3*wall,'model_task_cpu_seconds':sum(p['cpu_seconds'] for p in records),
                    'peak_sum_working_set':max(s['working_set'] for s in samples),
                    'peak_sum_private_commit':max(s['private_commit'] for s in samples),
                    'records':records,'reports':reports,'prediction_bitwise_equal':True}
                save(root/'result.json',record);save(root/'memory.json',samples);all_records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in {'records','reports'}}),flush=True)
                save(OUT/'results.json',all_records)
        save(OUT/'completion.json',{'complete':True,'model_runs':168,'distinct_model_cells':28,'all_bitwise_equal':True,
            'original_version_anchor_comparisons':42,'total_seconds':time.monotonic()-started})
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)
if __name__=='__main__':main()

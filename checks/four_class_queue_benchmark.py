"""Four-class queue study: nine models, four physical cores, independent tasks.

The category router is a bounded benchmark adapter, not a production service.
All arms retain one model per lease/result. Windows in-process numerical fits,
full model/CV budgets, real loopback RPC and fsynced result acknowledgments.
"""
import os
from pathlib import Path
import sys
import time
import json
import hashlib
import threading
import ctypes
import struct
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import gpa_queue_benchmark as base
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, digest, file_digest
from aleatoric_nk_grid.queue_service import Client, make_server, execute_worker

ROOT=base.ROOT
OUT=ROOT.parent/'four-class-benchmark-01'
SEVEN=base.MODELS
MODELS=SEVEN+('xgboost','lightgbm')
CELLS=base.CELLS
CLASSES=('sl','nn','boost','five')
CATEGORY={'super_learner':'sl','shallow_neural_network':'nn','xgboost':'boost','lightgbm':'boost',
          **{model:'five' for model in ('ols','ridge','lasso','random_forest','extra_trees')}}
WEIGHTS={**base.WEIGHTS,'xgboost':5,'lightgbm':5}
ARMS=('global_single','four_dedicated','four_shared','four_shared_cache')


class CategoryRouter:
    """Retain independent durable queues and route claims by remaining cost.

    Shared policy picks the greatest (pending+leased estimated work)/(leased+1),
    then takes that category's highest-cost task. No task group is indivisible.
    Dedicated policy gives each of four workers exactly one category.
    """
    def __init__(self,root,tasks,*,dedicated=False):
        self.mutex=threading.RLock();self.dedicated=dedicated;self.queues={}
        self.category_by_id={task.id:CATEGORY[task.model] for task in tasks}
        try:
            for category in CLASSES:
                path=root/category
                Dispatcher.create(path,((task,WEIGHTS[task.model]*task.N*task.K**.5)
                    for task in tasks if CATEGORY[task.model]==category),
                    identity={'benchmark':'four-classes','category':category},lease_seconds=120)
                self.queues[category]=Dispatcher(path,scratch=root/'scratch')
        except BaseException:
            self.close();raise
        self.queue_id=digest({'queue_ids':{c:q.queue_id for c,q in self.queues.items()},'dedicated':dedicated})

    def claim(self,worker,*,cached_cells=()):
        with self.mutex:
            for q in self.queues.values():
                q._reap()
                active=q.db.execute("SELECT * FROM tasks WHERE state='leased' AND worker=?",(worker,)).fetchone()
                if active is not None:return q._lease_payload(active)
            if self.dedicated:
                return self.queues[CLASSES[int(worker[1:])%4]].claim(worker,cached_cells=cached_cells)
            candidates=[]
            for category,q in self.queues.items():
                # Bounded 36-task adapter only. A production implementation
                # would maintain these totals incrementally, not scan tasks.
                if not q.stats().get('pending',0):continue
                cost=q.db.execute("SELECT SUM(cost) FROM tasks WHERE state IN ('pending','leased')").fetchone()[0]
                score=cost/(q.stats().get('leased',0)+1)
                candidates.append((score,category))
            if candidates:
                _,category=max(candidates)
                return self.queues[category].claim(worker,cached_cells=cached_cells)
            stats=self.stats()
            return {'state':'wait' if stats.get('leased',0) else
                ('blocked' if stats.get('failed',0) or stats.get('exhausted',0) else 'complete')}

    def heartbeat(self,task_id,token,worker):
        with self.mutex:return self.queues[self.category_by_id[task_id]].heartbeat(task_id,token,worker)

    def submit(self,task_id,token,worker,result):
        with self.mutex:return self.queues[self.category_by_id[task_id]].submit(task_id,token,worker,result)

    def stats(self):
        with self.mutex:
            values=[q.stats() for q in self.queues.values()]
            return {**{s:sum(v.get(s,0) for v in values) for s in ('pending','leased','done','failed','exhausted','total')},
                    'queue_id':self.queue_id,'paused':False}

    def close(self):
        for q in self.queues.values():q.close()


def masks():
    length=base.wt.DWORD(0);base.kernel.GetLogicalProcessorInformation(None,ctypes.byref(length))
    buffer=ctypes.create_string_buffer(length.value)
    if not base.kernel.GetLogicalProcessorInformation(buffer,ctypes.byref(length)):raise ctypes.WinError(ctypes.get_last_error())
    values=[]
    for offset in range(0,length.value,32):
        mask,relationship=struct.unpack_from('QI',buffer.raw,offset)
        if relationship==0:values.append(mask & -mask)
    if len(values)<4:raise RuntimeError('Four physical cores required')
    return values[:4]


def initialize(mask):
    base.MODELS=MODELS;base.OUT=OUT;base.TMP=OUT/'tmp'
    os.environ['TEMP']=os.environ['TMP']=str(base.TMP)
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib')
    ready=base.initialize(mask)
    global anchors
    previous=json.loads((ROOT.parent/'gpa-benchmark-01/results.json').read_bytes())[0]
    anchors={(r['N'],r['K'],r['model']):r['_prediction_sha256']
        for packet in previous['records'] for r in packet['rows']}
    return ready


def configure(arm,run):
    return base.configure('queue_cache' if arm=='four_shared_cache' else 'queue_single',run)


def numerical(value):
    base.predictions={};task=ModelTask(**value)
    start=time.perf_counter();cpu=time.process_time()
    if base.cache:row=base.cache.run(task)
    else:row=base.session.run_cell_group(seed=task.seed,draw=task.draw,n_samples=task.N,k_features=task.K,models=(task.model,))[0]
    assert row['status']=='ok',(value,row['error'])
    prediction=base.predictions[task.model]
    row['_prediction_sha256']=hashlib.sha256(prediction.tobytes()).hexdigest()
    key=(task.N,task.K,task.model)
    if key in anchors:
        assert row['_prediction_sha256']==anchors[key],('prior seven-model prediction mismatch',value)
    return {'row':base.clean(row),'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'cache_stats':dict(base.cache.stats) if base.cache else {},
            'cache_bytes':base.cache.bytes if base.cache else 0,
            'cached_cells':base.cache.cached_cells if base.cache else []}


def main():
    if OUT.exists():raise RuntimeError('Fresh output directory required')
    (OUT/'tmp').mkdir(parents=True)
    os.environ['TEMP']=os.environ['TMP']=str(OUT/'tmp')
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib')
    started=time.monotonic();deadline=started+1800
    physical=masks()
    base.save(OUT/'protocol.json',{'cells':CELLS,'models':MODELS,'classes':CATEGORY,'workers':4,
        'masks':physical,'arms':ARMS,'repeats':2,'seed':12345,'draw':0,'max_seconds':1800,
        'task_granularity':'one model per lease, execution and durable acknowledgment in all arms',
        'queue_policy':CategoryRouter.__doc__,'cost_weights':WEIGHTS,'scope':__doc__,
        'memory_cache_bytes_per_worker':256*1024**2,'shared_cache_disk_bytes':1024**3,
        'source_sha256':{p.name:file_digest(p) for p in (ROOT/'NK_Grid/src/aleatoric_nk_grid').glob('*.py')},
        'benchmark_sha256':file_digest(__file__)})
    pools=[ProcessPoolExecutor(max_workers=1) for _ in physical]
    handles=[]
    try:
        futures=[pool.submit(initialize,mask) for pool,mask in zip(pools,physical)]
        ready=[f.result(timeout=240) for f in futures];base.save(OUT/'startup.json',ready)
        handles=[base.kernel.OpenProcess(0x410,False,r['pid']) for r in ready]
        handles.append(base.kernel.OpenProcess(0x410,False,os.getpid()))
        if not all(handles):raise ctypes.WinError(ctypes.get_last_error())
        all_runs=[];reference={}
        for repeat in range(2):
            for arm in (ARMS if repeat==0 else tuple(reversed(ARMS))):
                if time.monotonic()>deadline:raise TimeoutError('Benchmark time budget')
                run=f'{repeat}-{arm}';root=OUT/run;root.mkdir()
                for pool in pools:pool.submit(configure,arm,run).result(timeout=30)
                tasks=[ModelTask(12345,0,n,k,m) for n,k in CELLS for m in MODELS]
                if arm=='global_single':
                    Dispatcher.create(root/'queue',((task,WEIGHTS[task.model]*task.N*task.K**.5) for task in tasks),
                        identity={'benchmark':run},lease_seconds=120)
                    dispatcher=Dispatcher(root/'queue',scratch=root/'scratch')
                else:dispatcher=CategoryRouter(root/'queues',tasks,dedicated=arm=='four_dedicated')
                token='benchmark-four-classes-'+'x'*32
                server=make_server(dispatcher,token=token)
                service=threading.Thread(target=server.serve_forever);service.start()
                client=Client(f'http://127.0.0.1:{server.server_port}',token,dispatcher.queue_id)
                records=[];lock=threading.Lock();stop=threading.Event();samples=[];sample_errors=[]
                worker_busy=[0.]*4;worker_finished=[0.]*4
                start=time.perf_counter()
                def sample():
                    try:
                        while not stop.is_set():
                            working=private=0
                            for handle in handles:
                                value=base.Counters();value.cb=ctypes.sizeof(value)
                                if not base.psapi.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
                                working+=int(value.ws);private+=int(value.private)
                            samples.append({'elapsed':time.perf_counter()-start,'working_set':working,'private_commit':private})
                            stop.wait(.2)
                    except BaseException as exc:sample_errors.append(str(exc))
                sampler=threading.Thread(target=sample);sampler.start()
                def worker(i):
                    cells=[]
                    def execute(value):
                        dispatch_start=time.perf_counter()-start
                        packet=pools[i].submit(numerical,value).result(timeout=max(1,deadline-time.monotonic()))
                        packet.update(worker=i,category=CATEGORY[value['model']],start=dispatch_start,finish=time.perf_counter()-start)
                        with lock:
                            records.append(packet);worker_busy[i]+=packet['seconds']
                        cells[:]=packet['cached_cells']
                        return packet['row']
                    report=execute_worker(client,f'w{i}',execute,spool=root/f'spool-{i}',cached_cells=lambda:cells,
                        heartbeat_seconds=20,deadline_seconds=max(1,deadline-time.monotonic()))
                    worker_finished[i]=time.perf_counter()-start
                    return report
                print(json.dumps({'starting':run}),flush=True)
                try:
                    with ThreadPoolExecutor(max_workers=4) as threads:reports=list(threads.map(worker,range(4)))
                    wall=time.perf_counter()-start
                    assert dispatcher.stats()['done']==36,dispatcher.stats()
                    assert not sample_errors,sample_errors
                finally:
                    stop.set();sampler.join();server.shutdown();service.join();server.server_close();dispatcher.close()
                keys=set()
                for packet in records:
                    row=packet['row'];key=(row['seed'],row['draw'],row['N'],row['K'],row['model'])
                    assert key not in keys;keys.add(key)
                    if key in reference:assert reference[key]==row['_prediction_sha256'],('cross-arm prediction mismatch',key,run)
                    else:reference[key]=row['_prediction_sha256']
                assert len(keys)==36
                result={'run':run,'arm':arm,'repeat':repeat,'wall_seconds':wall,'worker_busy_seconds':worker_busy,
                    'worker_finished_seconds':worker_finished,'reserved_core_seconds':4*wall,
                    'model_cpu_seconds':sum(p['cpu_seconds'] for p in records),'reports':reports,'records':records,
                    'peak_sum_working_set':max(s['working_set'] for s in samples),
                    'peak_sum_private_commit':max(s['private_commit'] for s in samples),'prediction_bitwise_equal':True}
                base.save(root/'result.json',result);base.save(root/'memory.json',samples)
                all_runs.append(result);base.save(OUT/'results.json',all_runs)
                print(json.dumps({k:v for k,v in result.items() if k not in {'records','reports'}}),flush=True)
        base.save(OUT/'completion.json',{'complete':True,'model_runs':288,'distinct_model_cells':36,
            'all_bitwise_equal':True,'prior_seven_model_hash_comparisons':224,'wall_seconds':time.monotonic()-started})
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)
        for handle in handles:base.kernel.CloseHandle(handle)


if __name__=='__main__':main()

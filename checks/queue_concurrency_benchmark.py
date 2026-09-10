"""Compare four versus eight physical-core workers using all nine GPA models.

Full production model parameters; same four-cell workload; single-model tasks,
loopback RPC, durable result commits, reversed repeat order. No cross-task cache.
"""
from pathlib import Path
import os
import time
import json
import threading
import ctypes
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import four_class_queue_benchmark as b
from local_cpu_inventory import inventory

ROOT=b.ROOT
OUT=ROOT.parent/'queue-concurrency-01'
ARMS=((4,'global_single'),(4,'four_shared'),(8,'global_single'),(8,'four_shared'))


def initialize(mask):
    b.OUT=OUT
    return b.initialize(mask)


def main():
    if OUT.exists():raise RuntimeError('Fresh output required')
    hardware=inventory()
    if hardware['physical_cores']<8:raise RuntimeError('Eight physical cores required')
    if hardware['available_physical_memory_bytes']<8*1024**3:raise RuntimeError('Insufficient free physical memory for bounded probe')
    (OUT/'tmp').mkdir(parents=True)
    os.environ['TEMP']=os.environ['TMP']=str(OUT/'tmp')
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib')
    start_all=time.monotonic();deadline=start_all+1800
    b.base.save(OUT/'protocol.json',{'hardware':hardware,'arms':ARMS,'cells':b.CELLS,'models':b.MODELS,
        'repeats':2,'seed':12345,'draw':0,'scope':__doc__,'cost_weights':b.WEIGHTS,
        'category_policy':b.CategoryRouter.__doc__,'benchmark_sha256':b.file_digest(__file__),
        'router_benchmark_sha256':b.file_digest(b.__file__),
        'source_sha256':{p.name:b.file_digest(p) for p in (ROOT/'NK_Grid/src/aleatoric_nk_grid').glob('*.py')}})
    pools=[ProcessPoolExecutor(max_workers=1) for _ in range(8)];handles=[]
    try:
        futures=[pool.submit(initialize,mask) for pool,mask in zip(pools,hardware['physical_core_masks'][:8])]
        ready=[f.result(timeout=240) for f in futures];b.base.save(OUT/'startup.json',ready)
        handles=[b.base.kernel.OpenProcess(0x410,False,r['pid']) for r in ready]
        handles.append(b.base.kernel.OpenProcess(0x410,False,os.getpid()))
        if not all(handles):raise ctypes.WinError(ctypes.get_last_error())
        results=[];reference={}
        for repeat in range(2):
            for workers,policy in (ARMS if repeat==0 else tuple(reversed(ARMS))):
                if time.monotonic()>deadline:raise TimeoutError('Benchmark bound')
                run=f'{repeat}-{workers}-{policy}';root=OUT/run;root.mkdir()
                for pool in pools:pool.submit(b.configure,policy,run).result(timeout=30)
                tasks=[b.ModelTask(12345,0,n,k,m) for n,k in b.CELLS for m in b.MODELS]
                if policy=='global_single':
                    b.Dispatcher.create(root/'queue',((task,b.WEIGHTS[task.model]*task.N*task.K**.5) for task in tasks),
                        identity={'benchmark':run},lease_seconds=120)
                    q=b.Dispatcher(root/'queue',scratch=root/'scratch')
                else:q=b.CategoryRouter(root/'queues',tasks)
                token='concurrency-benchmark-'+'x'*32
                server=b.make_server(q,token=token);service=threading.Thread(target=server.serve_forever);service.start()
                client=b.Client(f'http://127.0.0.1:{server.server_port}',token,q.queue_id)
                lock=threading.Lock();records=[];busy=[0.]*workers;finished=[0.]*workers
                stop=threading.Event();samples=[];errors=[]
                start=time.perf_counter()
                # Count active numerical workers + parent; the other resident
                # pools are idle and excluded from the active-memory statistic.
                active_handles=handles[:workers]+[handles[-1]]
                def sample():
                    try:
                        while not stop.is_set():
                            working=private=0
                            for handle in active_handles:
                                value=b.base.Counters();value.cb=ctypes.sizeof(value)
                                if not b.base.psapi.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
                                working+=int(value.ws);private+=int(value.private)
                            samples.append({'elapsed':time.perf_counter()-start,'working_set':working,'private_commit':private})
                            stop.wait(.2)
                    except BaseException as exc:errors.append(str(exc))
                sampler=threading.Thread(target=sample);sampler.start()
                def worker(i):
                    def execute(value):
                        at=time.perf_counter()-start
                        packet=pools[i].submit(b.numerical,value).result(timeout=max(1,deadline-time.monotonic()))
                        packet.update(worker=i,category=b.CATEGORY[value['model']],start=at,finish=time.perf_counter()-start)
                        with lock:records.append(packet);busy[i]+=packet['seconds']
                        return packet['row']
                    result=b.execute_worker(client,f'w{i}',execute,spool=root/f'spool-{i}',
                        heartbeat_seconds=20,deadline_seconds=max(1,deadline-time.monotonic()))
                    finished[i]=time.perf_counter()-start
                    return result
                print(json.dumps({'starting':run}),flush=True)
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:reports=list(pool.map(worker,range(workers)))
                    wall=time.perf_counter()-start
                    assert q.stats()['done']==36 and not errors,(q.stats(),errors)
                finally:
                    stop.set();sampler.join();server.shutdown();service.join();server.server_close();q.close()
                seen=set()
                for packet in records:
                    row=packet['row'];key=(row['seed'],row['draw'],row['N'],row['K'],row['model'])
                    assert key not in seen;seen.add(key)
                    if key in reference:assert reference[key]==row['_prediction_sha256'],(key,run)
                    else:reference[key]=row['_prediction_sha256']
                assert len(seen)==36
                result={'run':run,'workers':workers,'policy':policy,'repeat':repeat,'wall_seconds':wall,
                    'worker_busy_seconds':busy,'worker_finished_seconds':finished,'records':records,'reports':reports,
                    'reserved_core_seconds':workers*wall,'model_cpu_seconds':sum(p['cpu_seconds'] for p in records),
                    'peak_active_working_set':max(s['working_set'] for s in samples),
                    'peak_active_private_commit':max(s['private_commit'] for s in samples),'prediction_bitwise_equal':True}
                results.append(result);b.base.save(root/'result.json',result);b.base.save(root/'memory.json',samples)
                b.base.save(OUT/'results.json',results)
                print(json.dumps({k:v for k,v in result.items() if k not in {'records','reports'}}),flush=True)
        b.base.save(OUT/'completion.json',{'complete':True,'model_runs':288,'distinct_model_cells':36,
            'all_bitwise_equal':True,'prior_seven_model_hash_comparisons':224,'wall_seconds':time.monotonic()-start_all})
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)
        for handle in handles:b.base.kernel.CloseHandle(handle)


if __name__=='__main__':main()

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import pytest
from aleatoric_nk_grid.shared_queue import ModelTask, QueueError
from four_class_queue_benchmark import CategoryRouter, MODELS, CATEGORY


def test_class_router_completes_each_independent_task_once(tmp_path):
    tasks=[ModelTask(1,0,n,1,m) for n in (10,20) for m in MODELS]
    q=CategoryRouter(tmp_path/'queues',tasks)
    try:
        def worker(i):
            ids=[]
            while True:
                lease=q.claim(f'w{i}')
                if lease['state'] in ('complete','wait'):return ids
                assert q.claim(f'w{i}')['token']==lease['token']
                result={**lease['task'],'status':'ok','mse':.25}
                assert q.submit(lease['id'],lease['token'],f'w{i}',result)['accepted']
                assert q.submit(lease['id'],lease['token'],f'w{i}',result)['duplicate']
                ids.append(lease['id'])
        with ThreadPoolExecutor(max_workers=4) as pool:ids=sum(pool.map(worker,range(4)),[])
        assert len(ids)==len(set(ids))==18 and q.stats()['done']==18
    finally:q.close()


def test_dedicated_queues_do_not_steal_and_shared_queues_can(tmp_path):
    tasks=[ModelTask(1,0,10,1,m) for m in MODELS]
    q=CategoryRouter(tmp_path/'fixed',tasks,dedicated=True)
    try:
        lease=q.claim('w0');assert lease['task']['model']=='super_learner'
        q.submit(lease['id'],lease['token'],'w0',{**lease['task'],'status':'ok'})
        assert q.claim('w0')['state']=='complete' and q.stats()['pending']==8
        q.dedicated=False
        next_lease=q.claim('w0')
        assert next_lease['state']=='task' and CATEGORY[next_lease['task']['model']]!='sl'
    finally:q.close()

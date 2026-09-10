import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from aleatoric_nk_grid.cell_cache import NodeInputStore, resident_size
from aleatoric_nk_grid.shared_queue import QueueError


def test_shared_readonly_mmap_once_and_namespace_guard(tmp_path):
    calls=[]
    def build():calls.append(1);return {"raw":np.arange(1000,dtype=float)}
    def access(i):
        store=NodeInputStore(tmp_path,namespace="fixed-input",max_bytes=100000)
        value,hit=store.load_or_build("same-cell",build)
        assert not value["raw"].flags.writeable
        return hit
    with ThreadPoolExecutor(max_workers=4) as pool:
        hits=list(pool.map(access,range(4)))
    assert len(calls)==1 and sum(hits)==3
    with pytest.raises(QueueError,match="namespace"):
        NodeInputStore(tmp_path,namespace="different-data",max_bytes=100000).load_or_build("same-cell",build)


def test_disk_admission_is_bounded(tmp_path):
    store=NodeInputStore(tmp_path,namespace="fixed",max_bytes=100)
    value,hit=store.load_or_build("large",lambda:np.ones(1000))
    assert not hit and not list(tmp_path.glob("*.joblib"))


def test_corrupt_shared_input_rejected_before_deserialization(tmp_path):
    store=NodeInputStore(tmp_path,namespace="fixed",max_bytes=100000)
    store.load_or_build("cell",lambda:np.ones(1000))
    path=next(tmp_path.glob("*.joblib"));path.write_bytes(b"not valid")
    with pytest.raises(QueueError,match="checksum"):
        NodeInputStore(tmp_path,namespace="fixed",max_bytes=100000).load_or_build("cell",lambda:np.zeros(1000))


def test_recover_publication_crash_keeps_live_mappings(tmp_path, monkeypatch):
    import aleatoric_nk_grid.cell_cache as module
    store=NodeInputStore(tmp_path,namespace="fixed",max_bytes=20000)
    live,_=store.load_or_build("first",lambda:np.ones(1000))
    actual=module.atomic_json
    def crash(*args,**kw):raise OSError("crash before index publication")
    monkeypatch.setattr(module,"atomic_json",crash)
    with pytest.raises(OSError):store.load_or_build("orphan",lambda:np.ones(1000)*2)
    assert len(list(tmp_path.glob("*.joblib")))==2
    (tmp_path/("0"*64+"."+"1"*32+".tmp")).write_bytes(b"partial")
    monkeypatch.setattr(module,"atomic_json",actual)
    recovered=NodeInputStore(tmp_path,namespace="fixed",max_bytes=20000)
    recovered.load_or_build("third",lambda:np.ones(1000)*3)
    assert len(list(tmp_path.glob("*.joblib")))==2
    assert not list(tmp_path.glob("*.tmp"))
    assert sum(p.stat().st_size for p in tmp_path.glob("*.joblib"))<=20000
    np.testing.assert_array_equal(live,np.ones(1000))


def test_private_lru_eviction_and_oversized_cell_not_retained():
    from types import SimpleNamespace
    from aleatoric_nk_grid.cell_cache import CachedSession
    from aleatoric_nk_grid.shared_queue import ModelTask
    session=SimpleNamespace(_closed=False,config=SimpleNamespace(models=("ols",)),
        repeat_pairs={(1,0)},n_grid=(1,2,3,4),k_grid=(1,),
        _run_model=lambda **args:args["payload"][0])
    one_size=resident_size({"payload":np.ones(1000)})
    cache=CachedSession(session,max_bytes=2*one_size)
    cache._build=lambda task:{"payload":np.ones(10000 if task.N==4 else 1000)*task.N}
    tasks=[ModelTask(1,0,n,1,"ols") for n in range(1,5)]
    for task in (tasks[0],tasks[1],tasks[0],tasks[2]):
        assert cache.run(task)==task.N
        assert cache.bytes<=cache.max_bytes
    assert tasks[0].cell in cache.entries and tasks[1].cell not in cache.entries
    assert cache.stats["memory_hits"]==1 and cache.stats["evictions"]==1
    assert cache.run(tasks[3])==4
    assert tasks[3].cell not in cache.entries and cache.bytes<=cache.max_bytes

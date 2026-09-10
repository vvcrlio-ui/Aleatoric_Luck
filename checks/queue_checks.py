import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import pytest

from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError
from aleatoric_nk_grid.queue_service import Client, execute_worker, make_server


def task(n=10, model="ols"):
    return ModelTask(12345, 0, n, 1, model)


def result(lease, **extra):
    return {**lease["task"], "status": "ok", "mse": .25, **extra}


def create(tmp_path, tasks=None, **kwargs):
    root = tmp_path / "durable"
    Dispatcher.create(root, [(t, cost) for t, cost in (tasks or [(task(), 1)])], identity={"science": "fixed"}, **kwargs)
    return root


def test_http_lost_claim_submit_replies_and_transient_heartbeat(tmp_path, monkeypatch):
    root=create(tmp_path,lease_seconds=2)
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        calls={"claim":0,"submit":0,"heartbeat":0}; executions=[]
        for operation in calls:
            original=getattr(q,operation)
            def flaky(*a,_op=operation,_original=original,**kw):
                calls[_op]+=1
                response=_original(*a,**kw)
                if calls[_op]==1:raise OSError("lost reply after durable commit")
                return response
            monkeypatch.setattr(q,operation,flaky)
        server=make_server(q,token="x"*32)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            client=Client(f"http://127.0.0.1:{server.server_port}","x"*32,q.queue_id)
            def execute(value):
                executions.append(value);time.sleep(.15)
                return {**value,"status":"ok","mse":.2}
            report=execute_worker(client,"w",execute,spool=tmp_path/"spool",
                heartbeat_seconds=.02,idle_seconds=.01,deadline_seconds=5)
            assert report=={"state":"complete","accepted":1}
            assert len(executions)==1 and all(n>=2 for n in calls.values())
            assert not list((tmp_path/"spool").glob("*.json"))
        finally:
            server.shutdown();server.server_close();thread.join()


def test_rejected_stale_submission_keeps_result_spool(tmp_path):
    now=[100.]
    root=create(tmp_path,lease_seconds=2)
    with Dispatcher(root,scratch=tmp_path/"scratch",clock=lambda:now[0]) as q:
        class LocalClient:
            queue_id=q.queue_id
            def call(self,operation,**args):return getattr(q,operation)(**args)
        def execute(value):
            now[0]+=3;q.claim("replacement")
            return {**value,"status":"ok","mse":.2}
        report=execute_worker(LocalClient(),"original",execute,spool=tmp_path/"spool")
        assert report["state"]=="submission_rejected"
        assert Path(report["receipt"]).exists()
        assert q.stats().get("done",0)==0 and q.stats()["leased"]==1


def test_local_drain_file_controls_claims_and_allows_current_result(tmp_path):
    from aleatoric_nk_grid.queue_service import serve_with_drain
    root=create(tmp_path,[(task(10),1),(task(20),2)])
    drain=tmp_path/"drain";stop=threading.Event()
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        lease=q.claim("current")
        drain.touch()
        server=make_server(q,token="x"*32)
        thread=threading.Thread(target=serve_with_drain,
            kwargs={"server":server,"drain_file":drain,"stop":stop.is_set});thread.start()
        client=Client(f"http://127.0.0.1:{server.server_port}","x"*32,q.queue_id)
        try:
            assert client.call("claim",worker="idle")["state"]=="paused"
            client.call("submit",worker="current",task_id=lease["id"],token=lease["token"],result=result(lease))
            drain.unlink()
            until=time.monotonic()+3
            while time.monotonic()<until:
                resumed=client.call("claim",worker="idle")
                if resumed["state"]=="task":break
                time.sleep(.02)
            assert resumed["state"]=="task" and q.stats()["done"]==1
        finally:
            stop.set();thread.join(timeout=3);server.server_close()


def test_single_model_dynamic_idempotence(tmp_path):
    root = create(tmp_path, [(task(10, "ols"), 1), (task(20, "super_learner"), 10)])
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        a = q.claim("worker1"); assert a["task"]["model"] == "super_learner"
        assert q.claim("worker1")["token"] == a["token"]
        b = q.claim("worker2"); assert b["task"]["model"] == "ols"
        assert q.claim("worker3")["state"] == "wait"
        q.submit(a["id"], a["token"], "worker1", result(a))
        assert q.submit(a["id"], a["token"], "worker1", result(a))["duplicate"]
        with pytest.raises(QueueError):
            q.submit(a["id"], a["token"], "worker1", result(a, mse=.5))
        q.submit(b["id"], b["token"], "worker2", result(b))
        assert q.claim("worker1")["state"] == "complete"


def test_lease_heartbeat_expiry_and_bounded_retry(tmp_path):
    now = [100.]
    root = create(tmp_path, lease_seconds=10, max_attempts=2)
    with Dispatcher(root, scratch=tmp_path / "scratch", clock=lambda: now[0]) as q:
        a = q.claim("w1"); now[0] = 105.
        q.heartbeat(a["id"], a["token"], "w1")
        now[0] = 111.; assert q.claim("w2")["state"] == "wait"
        now[0] = 116.; b = q.claim("w2"); assert b["attempt"] == 2
        with pytest.raises(QueueError):
            q.submit(a["id"], a["token"], "w1", result(a))
        now[0] = 127.; assert q.claim("w3")["state"] == "blocked"
        assert q.stats()["exhausted"] == 1


def test_crash_after_fsync_before_index_apply_and_restart_fencing(tmp_path):
    root = create(tmp_path, [(task(10), 1), (task(20), 1)])
    q = Dispatcher(root, scratch=tmp_path / "scratch")
    a = q.claim("w1"); b = q.claim("w2")
    original = q._apply
    def crash(event):
        if event["kind"] == "result":
            raise OSError("Injected index crash after durable result")
        original(event)
    q._apply = crash
    with pytest.raises(OSError):
        q.submit(a["id"], a["token"], "w1", result(a))
    q.close()
    with Dispatcher(root, scratch=tmp_path / "scratch") as recovered:
        assert recovered.stats()["done"] == 1
        assert recovered.submit(a["id"], a["token"], "w1", result(a))["duplicate"]
        with pytest.raises(QueueError):
            recovered.submit(b["id"], b["token"], "w2", result(b))
        c = recovered.claim("w3"); assert c["id"] == b["id"]


def test_drain_survives_restart(tmp_path):
    root = create(tmp_path, [(task(10), 1), (task(20), 1)])
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        lease = q.claim("w"); q.pause()
        assert q.claim("other")["state"] == "paused"
        q.submit(lease["id"], lease["token"], "w", result(lease))
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        assert q.claim("w")["state"] == "paused"
        q.pause(False); assert q.claim("w")["state"] == "task"


def test_torn_tail_repair_and_committed_tamper_rejection(tmp_path):
    root = create(tmp_path)
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        a = q.claim("w"); q.submit(a["id"], a["token"], "w", result(a))
    with (root / "events.jsonl").open("ab") as f:
        f.write(b'{"incomplete')
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        assert q.stats()["done"] == 1
    data = (root / "events.jsonl").read_bytes().replace(b'"mse":0.25', b'"mse":0.99')
    (root / "events.jsonl").write_bytes(data)
    with pytest.raises(QueueError, match="integrity"):
        Dispatcher(root, scratch=tmp_path / "scratch")


def test_double_owner_and_foreign_key(tmp_path):
    root = create(tmp_path)
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        with pytest.raises(QueueError, match="Owner"):
            Dispatcher(root, scratch=tmp_path / "other")
        a = q.claim("w")
        with pytest.raises(QueueError):
            q.submit(a["id"], a["token"], "w", result(a, model="ridge"))
        assert q.stats()["leased"] == 1


def test_duplicate_design_rejected(tmp_path):
    root = create(tmp_path, [(task(), 1), (task(), 2)])
    with pytest.raises(QueueError, match="Duplicate"):
        Dispatcher(root, scratch=tmp_path / "scratch")


def test_concurrent_workers_and_service_auth(tmp_path):
    root = create(tmp_path, [(task(i+10), i+1) for i in range(80)])
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        token = "private-token-" + "x"*32
        server = make_server(q, token=token)
        t = threading.Thread(target=server.serve_forever); t.start()
        client = Client(f"http://127.0.0.1:{server.server_port}", token, q.queue_id)
        bad = Client(client.url, "wrong", q.queue_id)
        try:
            with pytest.raises(QueueError, match="Unauthorized"):
                bad.call("claim", worker="bad")
            with pytest.raises(QueueError):
                client.call("pause")
            seen = []; mutex = threading.Lock()
            def execute(value):
                with mutex:
                    seen.append(ModelTask(**value).id)
                time.sleep(.001)
                return {**value, "status": "ok", "mse": .1}
            with ThreadPoolExecutor(max_workers=8) as pool:
                reports = list(pool.map(lambda i: execute_worker(client, f"w{i}", execute,
                    spool=tmp_path / f"spool{i}", heartbeat_seconds=.1), range(8)))
            assert len(seen) == len(set(seen)) == 80
            assert q.stats()["done"] == 80
            assert sum(x["accepted"] for x in reports) == 80
        finally:
            server.shutdown(); t.join(); server.server_close()


def test_numerical_failure_is_separate_and_blocks_completion(tmp_path):
    root = create(tmp_path)
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        a = q.claim("w")
        q.submit(a["id"], a["token"], "w", result(a, status="failed", error="SVD failed"))
        assert q.claim("w")["state"] == "blocked"
        assert q.stats()["failed"] == 1


def test_affinity_does_not_hide_much_longer_tasks(tmp_path):
    root = create(tmp_path, [(task(10), 1), (task(20, "super_learner"), 20)])
    with Dispatcher(root, scratch=tmp_path / "scratch") as q:
        assert q.claim("w", cached_cells=[task(10).cell])["task"]["N"] == 20

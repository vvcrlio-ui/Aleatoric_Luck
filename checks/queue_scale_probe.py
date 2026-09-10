"""Bounded 100k-plan/2k-result local probe; not an 18M-task capacity claim."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import threading
import time
import tracemalloc
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, atomic_json, file_digest
from aleatoric_nk_grid.queue_service import Client, make_server


def main():
    output=Path(__file__).resolve().parents[2]/"queue-scale-100k"
    if output.exists():raise RuntimeError("Fresh output required")
    output.mkdir()
    root=output/"queue"; scratch=output/"scratch"
    report={"tasks":100000,"completed_models":2000,"workers":16,
            "scope":"local Windows disk and loopback HTTP; minimal synthetic results"}
    t=time.perf_counter()
    Dispatcher.create(root,((ModelTask(1,0,n,1,"ols"),float(n)) for n in range(1,100001)),identity={"probe":"100k-v1"})
    report["manifest_seconds"]=time.perf_counter()-t
    print(json.dumps(report),flush=True)
    t=time.perf_counter()
    with Dispatcher(root,scratch=scratch) as q:
        report["index_build_seconds"]=time.perf_counter()-t
        report["initial_index_bytes"]=q.db_path.stat().st_size
        server=make_server(q,token="scale-probe-"+"x"*32)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        client=Client(f"http://127.0.0.1:{server.server_port}","scale-probe-"+"x"*32,q.queue_id)
        try:
            def worker(i):
                for _ in range(125):
                    lease=client.call("claim",worker=f"worker-{i}")
                    assert lease["state"]=="task"
                    client.call("submit",worker=f"worker-{i}",task_id=lease["id"],token=lease["token"],
                        result={**lease["task"],"status":"ok","mse":.25,"rmse":.5,"mae":.4})
            t=time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(worker,range(16)))
            report["http_claim_and_durable_result_seconds"]=time.perf_counter()-t
            assert q.stats()["done"]==2000 and q.stats()["pending"]==98000
            report["events_bytes"]=(root/"events.jsonl").stat().st_size
            report["index_bytes_with_results"]=q.db_path.stat().st_size
        finally:
            server.shutdown();server.server_close();thread.join()
    print(json.dumps(report),flush=True)
    t=time.perf_counter()
    with Dispatcher(root,scratch=scratch) as q:
        report["restart_rebuild_replay_seconds"]=time.perf_counter()-t
        assert q.stats()["done"]==2000 and q.stats()["pending"]==98000
        report["restart_verified"]=True
    report["accepted_models_per_second"]=2000/report["http_claim_and_durable_result_seconds"]
    report["tasks_manifest_bytes"]=(root/"tasks.jsonl").stat().st_size
    report["source_sha256"]={p.name:file_digest(p) for p in
        (Path(__file__).resolve().parents[1]/"NK_Grid/src/aleatoric_nk_grid").glob("*queue*.py")}
    atomic_json(output/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()

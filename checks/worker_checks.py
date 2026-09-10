"""Portable planner/serialization checks; native execution is a Linux gate."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask
from aleatoric_nk_grid.single_model_worker import json_result


def test_real_planner_cli_streams_one_model_tasks(tmp_path):
    identity={"cell_spec":{"resolved_n_grid":[10,20],"resolved_k_grid":[1,2],
        "resolved_repeat_plan":[[1,0],[2,0]],"models":["ols","super_learner"]}}
    path=tmp_path/"identity.json";path.write_text(json.dumps(identity))
    root=tmp_path/"queue"
    completed=subprocess.run([sys.executable,"-m","aleatoric_nk_grid.single_model_worker",
        "plan",str(root),"--identity",str(path)],capture_output=True,text=True,check=True)
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        assert json.loads(completed.stdout)["queue_id"]==q.queue_id
        assert q.stats()["pending"]==16
        lease=q.claim("w")
        assert lease["task"]["model"]=="super_learner"
        assert lease["task"]["N"]==20 and lease["task"]["K"]==2
        tasks=[ModelTask(**json.loads(line)["task"]) for line in (root/"tasks.jsonl").read_bytes().splitlines()]
        assert len({task.id for task in tasks})==16


def test_numpy_result_serialization_keeps_missing_diagnostics_explicit():
    value=json_result({"mse":np.float64(.25),"N":np.int64(10),"diagnostics":[np.nan,np.inf]})
    assert value=={"mse":.25,"N":10,"diagnostics":[None,None]}
    assert json.loads(json.dumps(value,allow_nan=False))==value

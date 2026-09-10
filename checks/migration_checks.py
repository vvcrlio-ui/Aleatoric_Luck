from dataclasses import asdict
import json
from pathlib import Path
import pytest
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, atomic_json, canonical, digest, file_digest
from aleatoric_nk_grid.result_migration import (OLD_ALGORITHM, NEW_ALGORITHM, assert_compatible_specs,
    import_bundle, source_compatibility)


def setup(tmp_path, *, corrupt_metric=False, conflict=False):
    old = {"git_commit": "old", "algorithm_version": OLD_ALGORITHM, "resolved_model_params": {"ridge": {"n_alphas":63}}, "execution_groups": ["old7"]}
    new = {**old, "git_commit":"new", "algorithm_version":NEW_ALGORITHM, "execution_groups":["single"]}
    certificate = {"policy":"exact-scheduler-plus-conditional-gesvd-v1", "files":{"fold_local.py":{"old":"a","new":"b"}}}
    tasks = [ModelTask(12345,0,10,1,m) for m in ("ols","ridge","super_learner")]
    root = tmp_path/"queue"
    Dispatcher.create(root, ((task,1) for task in tasks), identity={"cell_spec":new,"compatibility_certificate_sha256":digest(certificate)})
    bundle = tmp_path/"bundle"; bundle.mkdir()
    envelopes=[]
    for i, task in enumerate(tasks):
        row = {**asdict(task), "status":"failed" if i==2 else "ok", "error":"SVD" if i==2 else "",
               "mse":"nan" if corrupt_metric and i==1 else ".25", "rmse":".5", "mae":".4"}
        envelopes.append({"result":row, "origin":{"analysis_id":"old-analysis","payload_sha256":digest(row),"row_id":"old-seven-row","sequence":1}})
    if conflict:
        duplicate=json.loads(json.dumps(envelopes[0])); duplicate["result"]["mse"]=".3"
        duplicate["origin"]["payload_sha256"]=digest(duplicate["result"]);envelopes.append(duplicate)
    path=bundle/"results.jsonl";path.write_bytes(b"".join(canonical(e)+b"\n" for e in envelopes))
    atomic_json(bundle/"manifest.json", {"format":"sealed-legacy-export-v1","sealed":True,"source_analysis_id":"old-analysis",
        "cell_spec":old,"rows":len(envelopes),"results_sha256":file_digest(path)})
    return root,bundle,new,certificate


def test_successes_import_failed_model_only_pending_and_idempotent(tmp_path):
    root,bundle,new,certificate=setup(tmp_path)
    args=dict(new_spec=new,certificate=certificate,expected_manifest_sha256=file_digest(bundle/"manifest.json"))
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        receipt=import_bundle(q,bundle,**args)
        assert receipt["imported"]==2 and receipt["failed_records_left_pending"]==1
        assert import_bundle(q,bundle,**args)["imported"]==0
        assert q.stats()["done"]==2 and q.stats()["pending"]==1
        lease=q.claim("new-worker"); assert lease["task"]["model"]=="super_learner"
        q.submit(lease["id"],lease["token"],"new-worker",{**lease["task"],"status":"ok","mse":.2,"rmse":.45,"mae":.35})
        assert q.claim("new-worker")["state"]=="complete"
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        assert q.stats()["done"]==3


@pytest.mark.parametrize("variant",["corrupt_metric","conflict"])
def test_invalid_bundle_import_is_atomic_preflight(tmp_path,variant):
    root,bundle,new,certificate=setup(tmp_path,**{variant:True})
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        with pytest.raises(QueueError):
            import_bundle(q,bundle,new_spec=new,certificate=certificate,expected_manifest_sha256=file_digest(bundle/"manifest.json"))
        assert q.stats()["pending"]==3 and not q.stats().get("done",0)


def test_scientific_change_rejected():
    old={"algorithm_version":OLD_ALGORITHM,"seed":12345,"params":{"alpha":1}}
    assert_compatible_specs(old,{**old,"algorithm_version":NEW_ALGORITHM,"git_commit":"new"})
    for changed in ({**old,"seed":1},{**old,"params":{"alpha":2}},{**old,"algorithm_version":"arbitrary"}):
        with pytest.raises(QueueError):assert_compatible_specs(old,changed)


def test_manifest_and_live_export_rejected(tmp_path):
    root,bundle,new,certificate=setup(tmp_path)
    with Dispatcher(root,scratch=tmp_path/"scratch") as q:
        with pytest.raises(QueueError,match="manifest"):
            import_bundle(q,bundle,new_spec=new,certificate=certificate,expected_manifest_sha256="wrong")
        path=bundle/"manifest.json"; manifest=json.loads(path.read_bytes());manifest["sealed"]=False;atomic_json(path,manifest)
        with pytest.raises(QueueError,match="sealed"):
            import_bundle(q,bundle,new_spec=new,certificate=certificate,expected_manifest_sha256=file_digest(path))


def test_actual_source_diff_is_only_approved_fallback():
    import os
    new=Path(__file__).resolve().parents[1]
    old=Path(os.environ.get('SCHEDULER_OLD_REPO', str(new.parents[2])))
    cert=source_compatibility(old,new)
    assert cert["files"]["nk_grid.py"]["old_sha256"]==cert["files"]["nk_grid.py"]["new_sha256"]
    pair=cert["parameter_files"]["FFCWS/model_params.yaml"]
    old_spec={"algorithm_version":OLD_ALGORITHM,"model_params_sha256":pair["old_sha256"]}
    new_spec={"algorithm_version":NEW_ALGORITHM,"model_params_sha256":pair["new_sha256"]}
    assert_compatible_specs(old_spec,new_spec,certificate=cert)
    with pytest.raises(QueueError,match="Uncertified"):
        assert_compatible_specs(old_spec,new_spec)
    with pytest.raises(QueueError,match="Uncertified"):
        assert_compatible_specs(old_spec,{**new_spec,"model_params_sha256":"unrelated"},certificate=cert)


def test_portable_export_adapter_exact_target_and_legacy_codec(tmp_path,monkeypatch):
    # Interface test only. Genuine POSIX seals/WAL locking remain a Linux gate.
    import hashlib
    import sys
    from types import SimpleNamespace
    import aleatoric_nk_grid
    from aleatoric_nk_grid.result_migration import export_sealed
    frontier=[{"closed_path":"实验/generation.closed.json"}]
    codec=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    receipt=tmp_path/"receipt.json";receipt.write_text("verified fixture")
    snapshot=tmp_path/"snapshot.json";snapshot.write_text("snapshot fixture")
    task=ModelTask(1,0,10,1,"ridge")
    row={**asdict(task),"status":"ok","mse":".25","rmse":".5","mae":".4"}
    analysis=SimpleNamespace(analysis_id="a",payload={"cell_execution_spec":{"algorithm_version":OLD_ALGORITHM}})
    def exact(payload,*,round_index,submission_generation,prep_token,
              expected_previous_generation,expected_pointer_version,prep_job_id):
        assert (round_index,submission_generation,prep_token)==(0,"gen","token")
        assert expected_previous_generation is expected_pointer_version is prep_job_id is None
        return analysis,None,"target"
    legacy=SimpleNamespace(
        verify_rounds=lambda *a,**kw:{"aborted_tasks":[],"sealed_history_digest_sha256":hashlib.sha256(codec(frontier)).hexdigest(),"verification_receipt":str(receipt)},
        _load_snapshot=lambda p:{"output_dir":str(tmp_path)},_exact_target=exact,
        canonical_json_bytes=codec,TASK_RESULT=3,
        _iter_sealed_wal_scans=lambda *a,**kw:iter([(None,Path("original.wal"),SimpleNamespace(records=[
            SimpleNamespace(event_type=3,payload=b"public row",sequence=1,row_id="group")]))]),
        decode_public_rows=lambda raw:(tuple(row),[row]))
    control=SimpleNamespace(GenerationValidationCache=lambda:None,
        classify_exact_afterany_target_read_only=lambda *a,**kw:SimpleNamespace(kind="sealed-generation"),
        frozen_sealed_history=lambda *a:frontier)
    monkeypatch.setitem(sys.modules,"aleatoric_nk_grid.flat_task_table",legacy)
    monkeypatch.setattr(aleatoric_nk_grid,"flat_task_table",legacy,raising=False)
    monkeypatch.setitem(sys.modules,"aleatoric_nk_grid.generation_control",control)
    manifest=export_sealed(snapshot,target_arguments={"round_index":0,"submission_generation":"gen","expected_prep_token":"token"},
        output=tmp_path/"export",tmp_dir=tmp_path/"scratch")
    assert manifest["rows"]==1 and manifest["sealed"]
    envelope=json.loads((tmp_path/"export/results.jsonl").read_bytes())
    assert envelope["origin"]["payload_sha256"]==digest(row)

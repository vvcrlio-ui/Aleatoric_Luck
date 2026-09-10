"""Bounded native Linux GPA worker CLI comparison. Run on a compute node."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

from aleatoric_nk_grid.config import NKGridConfig
from aleatoric_nk_grid import nk_grid as nk
from aleatoric_nk_grid.execution_contract import CellExecutionSpec
from aleatoric_nk_grid.shared_queue import Dispatcher, atomic_json, canonical, file_digest
from aleatoric_nk_grid.queue_service import make_server
from aleatoric_nk_grid.single_model_worker import iter_model_tasks, json_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--schema',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if sys.platform!='linux':raise RuntimeError('Native Linux validation required')
    models=('ols','ridge','lasso','random_forest','shallow_neural_network','extra_trees','super_learner','xgboost','lightgbm')
    config=NKGridConfig(schema=args.schema,out=args.output/'unused.csv',outcome='gpa',models=models,
        seed=12345,test_size=.2,n_seeds=1,n_draws=1,n_sizes_n=1,n_sizes_k=1,max_n=122,max_k=47,
        batch_size=1,n_jobs=1,n_grid=(122,),k_grid=(47,),model_params=args.repo/'FFCWS/model_params.yaml')
    started=time.monotonic()
    with nk.NKGridExecutionSession.open_from_config(config) as session:
        spec=CellExecutionSpec.from_config(config,repo_root=args.repo,resolved_n_grid=session.n_grid,
            resolved_k_grid=session.k_grid,resolved_repeat_plan=session.repeat_pairs,model_n_jobs=1,
            git_commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
            algorithm_version=session.algorithm_version,resolved_model_params=nk.resolved_model_params(session.selected_model_params),
            environment_overrides=nk.model_run_settings(models),execution_groups=[{'k_features':47,'groups':[
                {'group':g,'models':list(ms)} for g,ms in nk.execution_groups_for_models(models)]}],
            input_provenance=nk._frozen_input_provenance_for_schema(session.schema),require_clean_worktree=True)
        baseline=[]
        for _, group in nk.execution_groups_for_models(models):
            baseline.extend(session.run_cell_group(seed=12345,draw=0,n_samples=122,k_features=47,models=group))
    header=nk.public_result_columns('regression')
    expected={row['model']:json_result(nk.project_public_result(row,header=header)) for row in baseline}
    assert all(row['status']=='ok' for row in baseline),[(r['model'],r['status'],r['error']) for r in baseline]
    atomic_json(args.output/'baseline.json',expected)
    root=args.output/'queue';token=args.output/'token'
    token.write_text(secrets.token_hex(32));token.chmod(0o600)
    Dispatcher.create(root,iter_model_tasks(spec),identity={'cell_spec':dict(spec.payload)})
    atomic_json(args.output/'spec.json',dict(spec.payload))
    with Dispatcher(root,scratch=args.scratch) as queue:
        server=make_server(queue,token=token.read_text())
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            command=[sys.executable,'-m','aleatoric_nk_grid.single_model_worker','run',str(root),
                '--url',f'http://127.0.0.1:{server.server_port}','--repo-root',str(args.repo),
                '--token-file',str(token),'--spool',str(args.output/'spool'),'--max-seconds','900']
            with (args.output/'worker.stdout').open('w') as stdout,(args.output/'worker.stderr').open('w') as stderr:
                subprocess.run(command,check=True,timeout=1200,stdout=stdout,stderr=stderr)
            assert queue.stats().get('done')==9 and not queue.stats().get('failed'),queue.stats()
            queue.export_results(args.output/'new-results.jsonl')
        finally:
            server.shutdown();server.server_close();thread.join()
    for line in (args.output/'new-results.jsonl').read_bytes().splitlines():
        row=json.loads(line)['result'];actual=nk.project_public_result(row,header=header)
        assert canonical(actual)==canonical(expected[row['model']]),row['model']
    with Dispatcher(root,scratch=args.scratch) as queue:
        assert queue.stats().get('done')==9
    report=dict(native_linux=True,model_results=9,full_public_rows_equal=True,
        actual_worker_cli=True,restart_verified=True,seconds=time.monotonic()-started,
        source_commit=spec.payload['git_commit'],results_sha256=file_digest(args.output/'new-results.jsonl'),
        limitations='One GPA cell, nine models; no production cutover or 698-worker capacity claim.')
    atomic_json(args.output/'report.json',report);print(json.dumps(report))


if __name__=='__main__':main()

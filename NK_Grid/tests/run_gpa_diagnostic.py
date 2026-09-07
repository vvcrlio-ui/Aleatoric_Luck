"""Bounded real-GPA training-pool paired diagnostic. Never select on test outcomes.

Windows bypasses eager POSIX package initialization only. Sampling functions
are compiled unchanged from the actual engine source; this is NOT full-engine
or checkpoint/recovery acceptance. Writes exclusively to a fresh JSONL file.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType('aleatoric_nk_grid')
package.__path__ = [str(ROOT/'NK_Grid/src/aleatoric_nk_grid')]
sys.modules['aleatoric_nk_grid'] = package
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.preprocessing import source_groups, sampling_units, FoldPreprocessor, preprocess_cell
from aleatoric_nk_grid.model_registry import make_model, load_model_params
from aleatoric_nk_grid.execution_contract import runtime_environment
from aleatoric_nk_grid.execution_contract import sha256_file

source = ROOT/'NK_Grid/src/aleatoric_nk_grid/nk_grid.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef, ast.ClassDef)) and n.name in {'DrawOrders','draw_orders','_model_seed'}]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'))

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schema',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--params',type=Path,default=ROOT/'FFCWS/model_params.yaml')
parser.add_argument('--n',nargs='+',type=int,default=[180,199,200,201,220,250,251,252,300])
parser.add_argument('--k',nargs='+',type=int,default=[10,100])
parser.add_argument('--seeds',nargs='+',type=int,default=[12345,23456])
parser.add_argument('--draws',nargs='+',type=int,default=[0])
parser.add_argument('--method',choices=['legacy','fold-local'],default='fold-local')
parser.add_argument('--model', choices=['super_learner', 'shallow_neural_network'], default='super_learner')
parser.add_argument('--batches', nargs='+', default=['auto', 'full'])
args=parser.parse_args()
policies = [int(p) if p.isdecimal() else p for p in args.batches]
if len(set(policies)) != len(policies) or any(p not in ('auto', 'full') and (type(p) is not int or p <= 0) for p in policies):
    parser.error('--batches must be distinct auto, full, or positive integers')
params=load_model_params(args.params,task='regression',models=[args.model])[args.model]
loaded=load_input(args.schema,'gpa')
groups=source_groups(loaded.predictors,loaded.manifest,loaded.schema.continuous_priors)
units=sampling_units(groups)
pool=loaded.train.dropna(subset=['gpa'])
# One preregistered independent validation split of the original training pool.
fit_ids,valid_ids=train_test_split(pool.index,test_size=.2,random_state=731)
unit_map={u.name:u for u in units}
if max(args.n)>len(fit_ids) or max(args.k)>len(units):
    raise ValueError('requested diagnostic N/K exceeds training-pool capacity')
identity={'kind':'training-pool GPA paired diagnostic','method':args.method,
          'model':args.model, 'batches':policies,
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'runtime':runtime_environment(),'schema_sha256':hashlib.sha256(args.schema.read_bytes()).hexdigest(),
          'train_file_sha256':sha256_file(loaded.schema.table),
          'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in (ROOT/'NK_Grid/src/aleatoric_nk_grid').glob('*.py')},
          'driver_sha256':sha256_file(Path(__file__)),
          'worktree_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--','NK_Grid','FFCWS/model_params.yaml'],cwd=ROOT)).hexdigest(),
          'params_sha256':hashlib.sha256(args.params.read_bytes()).hexdigest(),
          'holdout_seed':731,'holdout_fraction':.2,'pool_rows':len(pool),'fit_rows':len(fit_ids),
          'n':args.n,'k':args.k,'seeds':args.seeds,'draws':args.draws,
          'note':'external test values not used; diagnostic replay adds fits; epochs differ in optimizer steps'}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x',encoding='utf-8') as output, threadpool_limits(limits=1):
    output.write(json.dumps({'identity':identity})+'\n');output.flush()
    for seed in args.seeds:
        for draw in args.draws:
            order=draw_orders(fit_ids,[u.name for u in units],seed=seed,draw=draw)
            for k in args.k:
                selected=[unit_map[str(name)] for name in order.feature_names[:k]]
                cols=[c for u in selected for c in u.features]; selected_groups=tuple(g for u in selected for g in u.groups)
                for n in args.n:
                    rows=order.row_index[:n]; X=pool.loc[rows,cols]; y=pool.loc[rows,'gpa']; V=pool.loc[valid_ids,cols]
                    for policy in policies:
                        record={'seed':seed,'draw':draw,'N':n,'K':k,'K_expanded':len(cols),'batch':policy,
                                'model_seed':_model_seed(seed,draw,n,k),'sources':[u.name for u in selected],
                                'row_order_sha256':hashlib.sha256(np.asarray(rows).tobytes()).hexdigest()}
                        started=time.perf_counter()
                        try:
                            preprocessor=FoldPreprocessor(selected_groups,loaded.schema.imputation,args.model)
                            if args.method=='legacy':
                                prepared=preprocess_cell(X,V,selected_groups,loaded.schema.imputation,model_name=args.model)
                                train_X,valid_X=prepared.X_train,prepared.X_test; preprocessor=None
                            else:
                                train_X,valid_X=X,V
                            run_params = {**params, 'mlp_batch_size':policy}
                            if args.model == 'super_learner':
                                run_params['diagnostics'] = True
                            model=make_model(args.model,seed=record['model_seed'],n_jobs=1,
                                params=run_params,preprocessor=preprocessor).fit(train_X,y)
                            predictions=model.predict(valid_X)
                            if not np.isfinite(predictions).all():raise ValueError('nonfinite predictions')
                            if args.model == 'super_learner':
                                diagnostics = model.diagnostics_
                            else:
                                adaptive = model if args.method == 'fold-local' else model[-1].regressor_
                                fitted = adaptive.model_[-1].regressor_ if args.method == 'fold-local' else adaptive.model_
                                diagnostics = {'alpha':adaptive.alpha_, 'cv_mse':list(adaptive.cv_mse_),
                                    'final_fit':{'N':fitted.fit_n_, 'batch':fitted.effective_batch_size_,
                                        'iterations':fitted.n_iter_, 'max_iter':fitted.max_iter,
                                        'convergence_warnings':fitted.convergence_warnings_}}
                            record.update(status='ok',validation_mse=float(np.mean((predictions-pool.loc[valid_ids,'gpa'])**2)),diagnostics=diagnostics)
                        except Exception as exc:
                            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                        record['seconds']=time.perf_counter()-started
                        output.write(json.dumps(record,allow_nan=False)+'\n');output.flush()
                        print(json.dumps({key:record[key] for key in ('seed','draw','N','K','batch','status','seconds')}),flush=True)

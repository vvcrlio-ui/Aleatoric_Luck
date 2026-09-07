"""Freeze one real FFC legacy-preprocessed boosting cell into a fresh NPZ."""
import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types
from typing import Sequence
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
package=types.ModuleType('aleatoric_nk_grid');package.__path__=[str(ROOT/'NK_Grid/src/aleatoric_nk_grid')]
sys.modules['aleatoric_nk_grid']=package
from aleatoric_nk_grid.ingest import load_input
from aleatoric_nk_grid.preprocessing import source_groups,sampling_units,preprocess_cell
from aleatoric_nk_grid.execution_contract import sha256_file
source=ROOT/'NK_Grid/src/aleatoric_nk_grid/nk_grid.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'DrawOrders','draw_orders','_model_seed'}]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schema',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--n',type=int,default=201);parser.add_argument('--k',type=int,default=3400)
parser.add_argument('--seed',type=int,default=12345);parser.add_argument('--draw',type=int,default=0)
args=parser.parse_args()
loaded=load_input(args.schema,'gpa');pool=loaded.train.dropna(subset=['gpa'])
groups=source_groups(loaded.predictors,loaded.manifest,loaded.schema.continuous_priors);units=sampling_units(groups)
if not 1<=args.n<=len(pool) or not 1<=args.k<=len(units):raise ValueError('N/K capacity violation')
order=draw_orders(pool.index,[u.name for u in units],seed=args.seed,draw=args.draw)
mapping={u.name:u for u in units};selected=[mapping[str(name)] for name in order.feature_names[:args.k]]
columns=[f for u in selected for f in u.features];groups=tuple(g for u in selected for g in u.groups)
X=pool.loc[order.row_index[:args.n],columns];y=pool.loc[order.row_index[:args.n],'gpa']
if loaded.schema.imputation['model_overrides'].get('xgboost')!=loaded.schema.imputation['model_overrides'].get('lightgbm'):
    raise ValueError('models require different legacy preprocessing; prepare separately')
prepared=preprocess_cell(X,X.iloc[:0],groups,loaded.schema.imputation,model_name='lightgbm').X_train
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('xb') as handle:np.savez(handle,X=prepared.to_numpy(),y=y.to_numpy())
metadata={'schema_sha256':sha256_file(args.schema),'train_sha256':sha256_file(loaded.schema.table),
          'npz_sha256':sha256_file(args.output),'N':args.n,'K':args.k,'K_expanded':prepared.shape[1],
          'seed':args.seed,'draw':args.draw,'model_seed':_model_seed(args.seed,args.draw,args.n,args.k),
          'method':'legacy outer-N typed preprocessing, common boosting overrides'}
with args.output.with_suffix('.json').open('x') as handle:json.dump(metadata,handle,indent=2)
print(json.dumps(metadata))

"""Portable calculation test of the actual planner function, not planner acceptance."""
import ast
import json
from pathlib import Path
from typing import Mapping
import numpy as np
from aleatoric_nk_grid.grid_contract import validate_size_grid

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'NK_Grid/src/aleatoric_nk_grid/chunk_planning.py'
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='expanded_columns_for_k')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'))


def test_b6_full_missing_indicator_width():
    assert expanded_columns_for_k(ROOT/'FFCWS/schema/ffc_median_missing_indicator.feature_universe.json',3400)==16085


def test_b6_random_subsets_covered():
    path=ROOT/'FFCWS/schema/ffc_median_missing_indicator.feature_universe.json'
    doc=json.loads(path.read_text()); widths={}
    for group in doc['sources']:
        widths.setdefault(group.get('sampling_source',group['source']),set()).update(f['feature'] for f in group['features'])
    bound=expanded_columns_for_k(path,25)
    names=list(widths)
    for seed in range(40):
        selected=np.random.default_rng(seed).choice(names,25,replace=False)
        assert sum(len(widths[name]) for name in selected)<=bound
    # Independent enumeration of all parent widths, including adversarial top widths.
    assert bound==sum(sorted(map(len,widths.values()))[-25:])

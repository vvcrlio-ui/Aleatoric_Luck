"""Independent boundary oracles; portable checks do not certify POSIX recovery."""
import json
from pathlib import Path

import numpy as np
import pytest

from aleatoric_nk_grid.grid_contract import validate_size_grid
from aleatoric_nk_grid.preprocessing import SourceGroup, sampling_units
from aleatoric_nk_grid.audit_history import audit

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("values", [[True], [0], [-1], [1.5], [float('nan')], [1, 1], [2, 1], []])
def test_a4_reject_without_normalizing(values):
    with pytest.raises(ValueError):
        validate_size_grid(values, "N", 10)


def test_a3_capacity():
    assert validate_size_grid([1, 3], "N", 3) == (1, 3)
    with pytest.raises(ValueError, match="capacity"):
        validate_size_grid([4], "N", 3)


def test_a1_versioned_universes():
    selections = []
    for representation in ('tree_ordinal', 'median_mode', 'median_missing_indicator'):
        doc = json.loads((ROOT / f'FFCWS/schema/ffc_{representation}.feature_universe.json').read_text())
        groups = [SourceGroup(name=s['source'], features=tuple(f['feature'] for f in s['features']),
                              source_order=s['source_order'], unit_type=s['unit_type'],
                              sampling_source=s.get('sampling_source')) for s in doc['sources']]
        units = sampling_units(groups)
        assert len(units) == 3400
        selections.append(np.random.RandomState(37).permutation([u.name for u in units])[:25].tolist())
    assert selections[0] == selections[1] == selections[2]


def test_a2_derived_reordering():
    groups = [SourceGroup('a', ('a',), 0, 'continuous'),
              SourceGroup('b', ('b',), 1, 'continuous'),
              SourceGroup('a_missing', ('am1', 'am2'), 2, 'continuous', sampling_source='a')]
    for order in (groups, groups[::-1]):
        units = sampling_units(order)
        assert [u.name for u in units] == ['a', 'b']
        assert units[0].features == ('a', 'am1', 'am2')
    # Move a derived group ahead of another parent's primary in source_order.
    # Parent sampling order must still follow the original primaries.
    groups = [SourceGroup('a', ('a',), 5, 'continuous'),
              SourceGroup('b', ('b',), 3, 'continuous'),
              SourceGroup('a_missing', ('am',), 1, 'continuous', sampling_source='a')]
    assert [u.name for u in sampling_units(groups)] == ['b', 'a']


def test_history_does_not_rewrite(tmp_path):
    path = tmp_path / 'old.csv'
    data = 'experiment_id,model,seed,draw,N,K,n_train_total,n_features_total\nx,ols,1,1,4,5,3,2\nx,ols,1,1,4,5,3,2\n'
    path.write_text(data)
    before = path.read_bytes()
    result = audit(path)
    assert len(result['over_capacity']) == 4
    assert len(result['duplicate_keys']) == 1
    assert result['exclude_from_analysis']
    assert path.read_bytes() == before


@pytest.mark.skipif(__import__('os').name == 'nt', reason='actual engine imports POSIX resource/fcntl')
def test_a6_actual_session_rejects_before_slice():
    from types import SimpleNamespace
    from aleatoric_nk_grid.nk_grid import NKGridExecutionSession
    session = object.__new__(NKGridExecutionSession)
    session._closed = False
    session.feature_units = ('a', 'b')
    session.config = SimpleNamespace(models=('ols',))
    session.repeat_pairs = ((1, 1),)
    session.n_grid = (3, 4)
    session.k_grid = (2, 5)
    session.split_manager = SimpleNamespace(for_seed=lambda seed: SimpleNamespace(train_index=range(3)))
    session._orders = lambda *args: pytest.fail('oversized task reached sampling')
    for n, k in ((4, 2), (3, 5)):
        with pytest.raises(ValueError, match='capacity'):
            session.run_cell_group(seed=1, draw=1, n_samples=n, k_features=k, models=('ols',))


@pytest.mark.skipif(__import__('os').name == 'nt', reason='actual planner requires POSIX resource/fcntl')
def test_a3_planner_capacity_failure_precedes_task_publication(tmp_path, monkeypatch):
    import pandas as pd
    from conftest import write_repo_schema_bundle
    from aleatoric_nk_grid.nk_grid import NKGridConfig
    from aleatoric_nk_grid import chunk_planning as planning
    schema=write_repo_schema_bundle(tmp_path,pd.DataFrame({'x':range(10),'y':range(10)}))
    config=NKGridConfig(schema=schema,out=tmp_path/'result.csv',outcome='y',models=('ols',),
                       seed=1,test_size=.3,n_seeds=1,n_draws=1,n_sizes_n=1,n_sizes_k=1,
                       max_n=10,max_k=1,batch_size=1,n_jobs=1,min_n=1)
    monkeypatch.setattr(planning,'write_task_table_streaming',lambda *a,**kw:pytest.fail('invalid grid reached writer'))
    table=tmp_path/'tasks.parquet'
    with pytest.raises(ValueError,match='capacity'):
        planning.build_dynamic_plan(config,n_grid=[8],k_grid=[1],
            cluster=planning.ClusterPolicy(1,1,'test','00:10:00','test','none'),
            table_path=table,snapshot_path=tmp_path/'snapshot.json',output_dir=tmp_path/'out',panel='test')
    assert not table.exists()


@pytest.mark.skipif(__import__('os').name == 'nt', reason='actual session requires POSIX resource/fcntl')
def test_a5_missing_outcomes_reduce_training_capacity(tmp_path):
    import pandas as pd
    from conftest import write_schema_bundle
    from aleatoric_nk_grid.nk_grid import NKGridConfig,NKGridExecutionSession
    train=pd.DataFrame({'id':range(10),'x':range(10),'y':[0.,1.,2.,3.,4.,5.,6.,7.,np.nan,np.nan]})
    test=pd.DataFrame({'id':[20,21],'x':[1,2],'y':[1.,2.]})
    schema=write_schema_bundle(tmp_path,train,split_mode='external_test',test=test,id_column='id')
    config=NKGridConfig(schema=schema,out=tmp_path/'result.csv',outcome='y',models=('ols',),
                        seed=1,test_size=.3,n_seeds=1,n_draws=1,n_sizes_n=1,n_sizes_k=1,
                        max_n=10,max_k=1,batch_size=1,n_jobs=1,min_n=1,n_grid=(9,),k_grid=(1,))
    with pytest.raises(ValueError,match='capacity'):
        NKGridExecutionSession._open_config(config)

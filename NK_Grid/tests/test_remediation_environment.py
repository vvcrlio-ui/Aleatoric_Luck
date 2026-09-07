import os
import pytest
from aleatoric_nk_grid.execution_contract import runtime_environment, CellExecutionSpec, AnalysisContract, ContractError, sha256_file, CELL_SPEC_FORMAT_VERSION
from aleatoric_nk_grid.phase_timing import timed_phase


def test_f2_content_hash_ignores_mtime(tmp_path):
    from aleatoric_nk_grid.execution_contract import resolve_repo_locator
    path = tmp_path/'data.csv'; path.write_bytes(b'1,2\n')
    before = path.stat(); digest = sha256_file(path)
    path.write_bytes(b'1,3\n'); os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert sha256_file(path) != digest
    with pytest.raises(ContractError, match='checksum mismatch'):
        resolve_repo_locator('data.csv', digest, repo_root=tmp_path)


def test_f6_sql_key_reducer_catches_duplicate_padding():
    # Test the actual SQLite reduction without importing/faking POSIX locks.
    # This does not certify the surrounding WAL or publication entry points.
    import ast
    from pathlib import Path
    import sqlite3
    import aleatoric_nk_grid
    path = Path(aleatoric_nk_grid.__path__[0]) / 'flat_task_table.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name in ('_queue_key_join', '_queue_missing_model_keys')]
    scope = {'sqlite3':sqlite3, '_QUEUE_KEY_COLUMNS':('model','seed','draw','N','K')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    with sqlite3.connect(':memory:') as connection:
        for table in ('expected','completed'):
            connection.execute(f'CREATE TABLE {table} (model TEXT, seed INTEGER, draw INTEGER, N INTEGER, K INTEGER)')
        expected = [('ols',1,0,10,1), ('ols',1,0,11,1)]
        for actual in (expected, [expected[0], expected[0]]):
            connection.execute('DELETE FROM expected'); connection.execute('DELETE FROM completed')
            connection.executemany('INSERT INTO expected VALUES (?,?,?,?,?)', expected)
            connection.executemany('INSERT INTO completed VALUES (?,?,?,?,?)', actual)
            assert len(actual) == len(expected)
            oracle = len(set(expected) - set(actual))
            assert scope['_queue_missing_model_keys'](connection) == oracle


def test_f3_actual_environment_drift():
    declared = runtime_environment(); declared['numpy'] = 'controlled-incompatible-version'
    with pytest.raises(ContractError, match='runtime environment mismatch'):
        CellExecutionSpec.from_payload({'cell_spec_format_version': CELL_SPEC_FORMAT_VERSION, 'runtime_environment': declared})


def test_timing_failure_does_not_swallow_exception(capsys):
    @timed_phase('injected')
    def fail():
        raise RuntimeError('controlled')
    with pytest.raises(RuntimeError, match='controlled'):
        fail()
    assert '"status": "failed"' in capsys.readouterr().err


def test_e6_reject_old_serializer():
    with pytest.raises(ContractError, match='serializer'):
        AnalysisContract.from_payload({'analysis_contract_format_version': 1, 'serializer_version': 1})


def test_slurm_roles_share_declared_numerical_threads():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'slurm'
    for name in ('plan_production', 'prep_dynamic_queue', 'run_flat_task_table',
                 'verify_dynamic_queue', 'close_dynamic_queue', 'finalize_dynamic_queue'):
        text = (root / (name + '.sbatch')).read_text()
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
            assert f'{variable}=1' in text, (name, variable)
        assert text.index('OMP_NUM_THREADS=1') > text.index('module load'), name


def test_e6_new_analysis_contract_roundtrip_and_nested_serializer_drift():
    import json
    import copy
    spec = CellExecutionSpec.from_payload({
        'cell_spec_format_version':CELL_SPEC_FORMAT_VERSION, 'runtime_environment':runtime_environment(),
        'models':['ols'], 'model_n_jobs':1, 'resolved_n_grid':[10], 'resolved_k_grid':[1],
        'resolved_repeat_plan':[[1,0]], 'git_commit':'d'*40, 'algorithm_version':'test-v1',
        'resolved_model_params':{'ols':{}}, 'environment_overrides':{},
        'execution_groups':[{'k_features':1,'groups':[]}],
        'input_provenance':{'train':{'path':'train.csv','sha256':'a'*64}}, 'require_clean_worktree':False,
    })
    created = AnalysisContract.create(cell_execution_spec=spec, task_design_digest='b'*64,
        expected_task_rows=1, expected_model_rows=1,
        public_result_schema=['mse','null_mse_train_mean','test_target_variance','skill_train_mean','r2_test_mean','r2_test'])
    payload = json.loads(json.dumps(created.to_payload()))
    restored = AnalysisContract.from_payload(payload)
    assert restored.to_payload() == payload
    for version in (1, 2, 999, None):
        bad = copy.deepcopy(payload)
        bad['public_result_schema']['serializer_version'] = version
        # No outer hash is supplied, so rejection must come from version semantics.
        bad.pop('analysis_id'); bad.pop('analysis_contract_sha256')
        with pytest.raises(ContractError, match='serializer'):
            AnalysisContract.from_payload(bad)

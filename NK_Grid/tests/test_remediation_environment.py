import os
import pytest
from aleatoric_nk_grid.execution_contract import runtime_environment, CellExecutionSpec, AnalysisContract, ContractError, sha256_file, CELL_SPEC_FORMAT_VERSION
from aleatoric_nk_grid.phase_timing import timed_phase


def test_f2_content_hash_ignores_mtime(tmp_path):
    path = tmp_path/'data.csv'; path.write_bytes(b'1,2\n')
    before = path.stat(); digest = sha256_file(path)
    path.write_bytes(b'1,3\n'); os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert sha256_file(path) != digest


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

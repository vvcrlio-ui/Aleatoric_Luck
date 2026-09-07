"""Actual production recovery entries; never substitute Windows lock shims."""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='requires real POSIX engine, schedule lease and WAL')


def test_f1_same_snapshot_second_round_never_rebuilds_table(tmp_path, monkeypatch):
    from test_worker_event_wal import _fault_plan
    import aleatoric_nk_grid.chunk_planning as planning
    import aleatoric_nk_grid.flat_task_table as ft
    snapshot, plan = _fault_plan(tmp_path, rows=4, workers=2, rounds=2)
    table = tmp_path / 'tasks.parquet'
    original = table.read_bytes()
    def unexpected(*args, **kwargs):
        pytest.fail('resuming the same snapshot rebuilt the production task table')
    monkeypatch.setattr(planning, 'build_dynamic_plan', unexpected)
    monkeypatch.setattr(planning, 'write_task_table_streaming', unexpected)
    ft.prepare_round(snapshot,round_index=1,prep_token='job-1',prep_job_id='job-1',
                     submission_generation='g1',expected_pointer_version=0)
    for worker in range(2):
        ft.run_slice(snapshot,round_index=1,worker_index=worker,expected_prep_token='job-1',
                     prep_job_id='job-1',submission_generation='g1',expected_pointer_version=0)
    ft.close_generation(snapshot,round_index=1,submission_generation='g1',
                        expected_prep_token='job-1',prep_job_id='job-1',expected_pointer_version=0)
    result = ft.prepare_round(snapshot,round_index=2,prep_token='job-2',prep_job_id='job-2',
        submission_generation='g2',expected_previous_generation='g1',expected_pointer_version=1)
    assert result['no_generation'] is True
    assert table.read_bytes() == original


def test_f7_worker_count_tamper_fails_before_activation(tmp_path):
    from test_worker_event_wal import _fault_plan
    import aleatoric_nk_grid.flat_task_table as ft
    snapshot, plan = _fault_plan(tmp_path, rows=4, workers=2)
    payload = json.loads(snapshot.read_text())
    payload['workers'] = 3
    # Only this fresh disposable fixture is altered, never a user snapshot.
    snapshot.chmod(0o644)
    snapshot.write_text(json.dumps(payload))
    before = {str(p.relative_to(tmp_path / 'out')) for p in (tmp_path / 'out').rglob('*')}
    with pytest.raises(ValueError, match='worker count'):
        ft.prepare_round(snapshot,round_index=1,prep_token='job-1',prep_job_id='job-1',
                         submission_generation='g1',expected_pointer_version=0)
    assert {str(p.relative_to(tmp_path / 'out')) for p in (tmp_path / 'out').rglob('*')} == before

"""Bounded Linux test of real legacy WAL sealing, export and pending recovery.

Run old phase with the frozen old source on PYTHONPATH, then new phase with
the candidate source. Failure/interruption injections affect this fixture only.
No production snapshot or WAL is opened for writing.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def key(row):
    return tuple(str(row[k]) for k in ('seed', 'draw', 'N', 'K', 'model'))


def old_phase(args):
    from aleatoric_nk_grid.config import NKGridConfig
    from aleatoric_nk_grid.chunk_planning import ClusterPolicy, build_dynamic_plan
    from aleatoric_nk_grid import flat_task_table as ft
    from aleatoric_nk_grid.nk_grid import NKGridExecutionSession, public_result_columns, project_public_result
    from aleatoric_nk_grid.worker_event_wal import encode_public_rows, decode_public_rows
    assert Path(ft.__file__).resolve().is_relative_to(args.old_repo.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    config = NKGridConfig(schema=args.schema, out=args.output/'unused.csv', outcome='gpa',
        models=('ols', 'ridge'), seed=12345, test_size=.2, n_seeds=1, n_draws=1,
        n_sizes_n=2, n_sizes_k=1, max_n=123, max_k=47, n_grid=(122, 123), k_grid=(47,),
        batch_size=1, n_jobs=1, model_params=args.old_repo/'FFCWS/model_params.yaml')
    cluster = ClusterPolicy(workers=2, rounds=1, partition='cn', time_limit='00:10:00',
        account='ehpc-dev-2026d08-299', constraint='none', memory_override='4G',
        preparation_tmp_dir=str(args.scratch), verification_tmp_dir=str(args.scratch))
    snapshot = args.output/'snapshot.json'
    plan = build_dynamic_plan(config, n_grid=(122,123), k_grid=(47,), cluster=cluster,
        table_path=args.output/'tasks.parquet', snapshot_path=snapshot,
        output_dir=args.output/'legacy', panel='ffc_median_mode_gpa')
    assert plan['model_row_count'] == 4
    target = dict(round_index=1, submission_generation='linux-resume-fixture', expected_prep_token='fixture-prep')
    prep = ft.prepare_round(snapshot, round_index=1, submission_generation=target['submission_generation'],
        prep_token=target['expected_prep_token'], tmp_dir=args.scratch)
    header = public_result_columns('regression')
    baseline = []
    original = NKGridExecutionSession.run_cell_group

    class InjectedInterruption(RuntimeError):
        pass

    def injected(session, **kwargs):
        rows = original(session, **kwargs)
        assert all(row['status'] == 'ok' for row in rows)
        public = [project_public_result(row, header=header) for row in rows]
        baseline.extend(decode_public_rows(encode_public_rows(public, header=header))[1])
        if kwargs['n_samples'] == 123:
            raise InjectedInterruption('Fixture interruption after STARTED, before RESULT')
        for row in rows:
            if row['model'] == 'ridge':
                row.update(status='failed', error='Injected fixture numerical failure', mse=float('nan'), rmse=float('nan'), mae=float('nan'))
        return rows

    NKGridExecutionSession.run_cell_group = injected
    interrupted = 0
    try:
        for worker in (0, 1):
            try:
                ft.run_slice(snapshot, worker_index=worker, **target)
            except InjectedInterruption:
                interrupted += 1
    finally:
        NKGridExecutionSession.run_cell_group = original
    assert interrupted == 1 and len(baseline) == 4
    save(args.output/'baseline.json', baseline)
    save(args.output/'target.json', target)
    save(args.output/'old-phase.json', dict(plan=plan, prep=prep, injected_interruption=interrupted,
        injected_failures=1, real_model_fits=4, sealed=False, source_commit=subprocess.check_output(
            ['git','-C',str(args.old_repo),'rev-parse','HEAD'], text=True).strip()))


def new_phase(args):
    from aleatoric_nk_grid import flat_task_table as ft
    from aleatoric_nk_grid import nk_grid as nk
    from aleatoric_nk_grid.config import config_from_json
    from aleatoric_nk_grid.generation_control import ControlBusyError, ControlProtocolError, ControlSupersededError
    from aleatoric_nk_grid.result_migration import export_sealed, source_compatibility, NEW_ALGORITHM
    from aleatoric_nk_grid.pending_resume import prepare, merge
    from aleatoric_nk_grid.shared_queue import Dispatcher, QueueError, canonical, digest, file_digest
    from aleatoric_nk_grid.single_model_worker import json_result
    from aleatoric_nk_grid.worker_event_wal import encode_public_rows, decode_public_rows
    assert Path(ft.__file__).resolve().is_relative_to(args.new_repo.resolve())
    started = time.monotonic()
    snapshot = args.output/'snapshot.json'
    target = json.loads((args.output/'target.json').read_bytes())
    rejected = []

    def reject(label, operation):
        try:
            operation()
        except (QueueError, ControlBusyError, ControlProtocolError, ControlSupersededError, ValueError):
            rejected.append(label)
        else:
            raise AssertionError('Did not reject ' + label)

    reject('active_generation_export', lambda: export_sealed(snapshot, target_arguments=target,
        output=args.output/'active-export', tmp_dir=args.scratch))
    reject('unknown_generation_export', lambda: export_sealed(snapshot,
        target_arguments={**target,'submission_generation':'unknown'}, output=args.output/'unknown-export', tmp_dir=args.scratch))
    assert not (args.output/'active-export').exists() and not (args.output/'unknown-export').exists()
    closed = ft.close_generation(snapshot, **target)
    verified = ft.verify_rounds(snapshot, tmp_dir=args.scratch, **target)
    assert not verified['complete'] and verified['missing_model_keys'] == 3
    assert verified['interrupted_row_ids'] and verified['failed_attempt_row_ids']
    bundle = args.output/'export'
    manifest = export_sealed(snapshot, target_arguments=target, output=bundle, tmp_dir=args.scratch)
    assert manifest['rows'] == 2 and manifest['sealed']
    source_wal_hashes = {str(path):file_digest(path) for path in (args.output/'legacy').rglob('*.wal')}
    certificate = source_compatibility(args.old_repo, args.new_repo)
    new_spec = {**manifest['cell_spec'], 'algorithm_version':NEW_ALGORITHM,
        'git_commit':subprocess.check_output(['git','-C',str(args.new_repo),'rev-parse','HEAD'], text=True).strip(),
        'model_params_sha256':file_digest(args.new_repo/'FFCWS/model_params.yaml')}
    common = dict(new_spec=new_spec, certificate=certificate, scratch=args.scratch)
    resumed = args.output/'resume'
    ready = prepare(bundle, resumed, expected_manifest_sha256=file_digest(bundle/'manifest.json'), **common)
    assert (ready['old_valid_unique'], ready['old_failed_records'], ready['pending']) == (1,1,3)
    reject('incomplete_merge', lambda: merge(args.output/'incomplete.csv', resumed=resumed, scratch=args.scratch))
    assert not (args.output/'incomplete.csv').exists()

    # Adversarial copies only; preserve the original sealed export and WALs.
    for variant in ('duplicate', 'conflict'):
        altered = args.output/variant
        shutil.copytree(bundle, altered)
        rows = [json.loads(line) for line in (altered/'results.jsonl').read_bytes().splitlines()]
        extra = json.loads(json.dumps(next(row for row in rows if row['result']['status'] == 'ok')))
        if variant == 'conflict':
            extra['result']['mse'] = str(float(extra['result']['mse']) + .1)
            extra['origin']['payload_sha256'] = digest(extra['result'])
        rows.append(extra)
        (altered/'results.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
        changed = {**manifest, 'rows':len(rows), 'results_sha256':file_digest(altered/'results.jsonl')}
        save(altered/'manifest.json', changed)
        kwargs = dict(expected_manifest_sha256=file_digest(altered/'manifest.json'), **common)
        if variant == 'conflict':
            reject('conflicting_old_results', lambda: prepare(altered, args.output/'conflict-resume', **kwargs))
            assert not (args.output/'conflict-resume/ready.json').exists()
        else:
            duplicate = prepare(altered, args.output/'duplicate-resume', **kwargs)
            assert duplicate['identical_old_duplicates'] == 1 and duplicate['pending'] == 3

    payload = json.loads(snapshot.read_bytes())
    config = replace(config_from_json(payload['config']), model_params=args.new_repo/'FFCWS/model_params.yaml')
    header = manifest['public_columns']
    baseline = {key(row):row for row in json.loads((args.output/'baseline.json').read_bytes())}
    completed = []
    with Dispatcher(resumed/'queue', scratch=args.scratch) as queue:
        with nk.NKGridExecutionSession.open_from_config(config) as session:
            while True:
                lease = queue.claim('native-resume')
                if lease['state'] == 'complete':
                    break
                task = lease['task']
                assert key(task) != ('12345','0','122','47','ols')
                rows = session.run_cell_group(seed=task['seed'], draw=task['draw'], n_samples=task['N'],
                    k_features=task['K'], models=(task['model'],))
                assert len(rows) == 1 and rows[0]['status'] == 'ok'
                row = json_result(nk.project_public_result(rows[0], header=header))
                actual = decode_public_rows(encode_public_rows([row], header=header))[1][0]
                expected = baseline[key(actual)]
                assert {k:v for k,v in actual.items() if k != 'algorithm_version'} == {
                    k:v for k,v in expected.items() if k != 'algorithm_version'}, task
                assert actual['algorithm_version'] == NEW_ALGORITHM
                queue.submit(lease['id'], lease['token'], 'native-resume', row)
                completed.append(task)
        assert queue.stats().get('done') == 3
        queue.export_results(args.output/'new-results.jsonl')
    with Dispatcher(resumed/'queue', scratch=args.scratch) as queue:
        assert queue.stats().get('done') == 3 and queue.claim('restart')['state'] == 'complete'
    final = args.output/'final.csv'
    report = merge(final, resumed=resumed, new_results=args.output/'new-results.jsonl', scratch=args.scratch)
    with final.open(newline='') as handle:
        reader = csv.DictReader(handle); rows = list(reader)
        assert reader.fieldnames == header and len(rows) == len({key(row) for row in rows}) == 4
    assert report['validated_complete'] and report['new_valid_unique'] == 3 and not report['added_source_columns']
    assert source_wal_hashes == {str(path):file_digest(path) for path in (args.output/'legacy').rglob('*.wal')}
    report.update(native_linux=True, real_legacy_protocol=True, old_verification=verified,
        closed_path=str(closed), rejected_cases=rejected, identical_duplicate_deduplicated=True,
        public_rows_equal_except_algorithm_version=True, resumed_tasks=completed, restart_verified=True,
        original_fixture_wals_unchanged=True, seconds=time.monotonic()-started,
        probe_sha256=file_digest(Path(__file__)), candidate_commit=new_spec['git_commit'],
        limitations='Four GPA model keys; one injected failure and one interrupted group. No production cutover, cross-node or scale claim.')
    save(args.output/'report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('old','new'))
    for name in ('old-repo','new-repo','schema','output','scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Linux native validation required')
    args.scratch.mkdir(parents=True, exist_ok=True)
    (old_phase if args.phase == 'old' else new_phase)(args)

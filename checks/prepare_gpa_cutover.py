"""Seal the stopped GPA run and prepare only its missing model keys.

Compute-node operation. Does not submit training or delete source results.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from aleatoric_nk_grid.shared_queue import atomic_json, file_digest
from aleatoric_nk_grid.result_migration import export_sealed, source_compatibility, NEW_ALGORITHM
from aleatoric_nk_grid.pending_resume import prepare
from aleatoric_nk_grid import flat_task_table as ft


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    sys.path.insert(0, str(a.old / 'launch'))
    from discoverer_continuation import controller_lock, target
    run = a.old / 'runs/discoverer-ffc-gpa-production-20260909'
    with controller_lock(run / '.continuation.lock'):
        state = json.loads((run / 'continuation.json').read_bytes())
        assert state['run_id'] == '17820c4142ba439b82911477a3b4b958'
        jobs = subprocess.check_output(['squeue', '-r', '-u', 'xwan', '-h', '-o', '%i|%j|%T'], text=True)
        if 'al-17820c4142ba439b-' in jobs:
            raise RuntimeError('Old GPA jobs still present; refuse sealing')
        a.output.mkdir(parents=True, exist_ok=False)
        atomic_json(a.output / 'stopped-jobs.json', {'queue': jobs, 'state': state})
        scratch = Path('/dev/shm') / ('gpa-cutover-' + os.environ['SLURM_JOB_ID'])
        scratch.mkdir(exist_ok=False)
        round_state = state['rounds'][-1]
        snapshot = Path(round_state['snapshot'])
        kwargs = target(round_state)
        atomic_json(a.output / 'phase.json', {'phase': 'sealing'})
        closed = ft.close_generation(snapshot, **kwargs)
        atomic_json(a.output / 'closed.json', closed)
        atomic_json(a.output / 'phase.json', {'phase': 'exporting'})
        manifest = export_sealed(snapshot, target_arguments=kwargs, output=a.output / 'base', tmp_dir=scratch)
        certificate = source_compatibility(a.old, a.new)
        spec = {**manifest['cell_spec'], 'algorithm_version': NEW_ALGORITHM,
                'git_commit': subprocess.check_output(['git', '-C', str(a.new), 'rev-parse', 'HEAD'], text=True).strip(),
                'model_params_sha256': file_digest(a.new / 'FFCWS/model_params.yaml')}
        # Preserve the exact schema bytes and repository-relative input locator.
        source = a.old / 'runs/discoverer-gpa-timing-balanced-20260909/prepared'
        dest = a.new / 'runs/discoverer-gpa-timing-balanced-20260909/prepared'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, dest)
        from aleatoric_nk_grid.execution_contract import CellExecutionSpec
        CellExecutionSpec.from_payload(spec).resolve_inputs(repo_root=a.new)
        atomic_json(a.output / 'phase.json', {'phase': 'planning_missing'})
        receipt = prepare(a.output / 'base', a.output / 'resume', new_spec=spec,
            certificate=certificate, expected_manifest_sha256=file_digest(a.output / 'base/manifest.json'),
            scratch=scratch, cost_profile=json.loads((a.new / 'NK_Grid/scheduler_profiles/ffc_gpa_discoverer_20260911.json').read_bytes()))
        if receipt['old_valid_unique'] != 14994428 or receipt['pending'] != 3005572:
            raise RuntimeError('Stopped-run coverage disagrees with full audit')
        atomic_json(a.output / 'phase.json', {'phase': 'prepared', 'receipt': receipt})
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

"""Create and verify a nine-model fixture for the real Slurm round runner."""
import argparse
import json
from pathlib import Path
import subprocess
from aleatoric_nk_grid.shared_queue import Dispatcher, atomic_json
from aleatoric_nk_grid.single_model_worker import iter_model_tasks, json_result
from aleatoric_nk_grid.nk_grid import public_result_columns, project_public_result

p = argparse.ArgumentParser()
p.add_argument('command', choices=['prepare', 'verify'])
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.command == 'prepare':
    a.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((a.baseline / 'spec.json').read_bytes())
    spec['git_commit'] = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip()
    Dispatcher.create(a.output / 'queue', iter_model_tasks(spec), identity={'cell_spec': spec})
else:
    receipt = json.loads((a.output / 'control/round-result.json').read_bytes())
    assert receipt['complete'], receipt
    rows = [json.loads(line)['result'] for line in Path(receipt['results']).read_bytes().splitlines()]
    baseline = json.loads((a.baseline / 'baseline.json').read_bytes())
    header = public_result_columns('regression')
    assert len(rows) == 9
    for row in rows:
        assert json_result(project_public_result(row, header=header)) == baseline[row['model']], row['model']
    atomic_json(a.output / 'verified.json', {'passed': True, 'native_models': 9,
        'full_public_rows_equal': True, 'actual_slurm_round': True})

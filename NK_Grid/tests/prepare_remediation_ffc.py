"""Build real FFC acceptance inputs in a NEW directory; never overwrite inputs.

Uses the unchanged adapter configuration and pipeline. On Windows, bypasses
only the package's eager POSIX engine import, not validation or any OS locks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import types
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
destination = args.output.resolve()
destination.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(args.archive) as archive:
    for name in ('background.dta', 'train.csv', 'test.csv'):
        matches = [entry for entry in archive.infolist() if Path(entry.filename).name == name]
        if len(matches) != 1:
            raise ValueError(f'expected exactly one {name}, found {len(matches)}')
        with archive.open(matches[0]) as source, (destination / name).open('xb') as target:
            shutil.copyfileobj(source, target)
config = yaml.safe_load((ROOT / 'FFCWS/adapter/config/ffc.yaml').read_text())
config['paths'] = {key: str(destination / name) for key, name in {
    'background':'background.dta', 'train':'train.csv', 'test':'test.csv',
    'output_root':'adapter_work', 'ard_root':'ard', 'schema_root':'schema'}.items()}
configuration = destination / 'config.yaml'
configuration.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
if sys.platform == 'win32':
    package = types.ModuleType('aleatoric_nk_grid')
    package.__path__ = [str(ROOT / 'NK_Grid/src/aleatoric_nk_grid')]
    sys.modules['aleatoric_nk_grid'] = package
sys.argv = ['adapter.py', '--config', str(configuration)]
runpy.run_path(str(ROOT / 'FFCWS/adapter/adapter.py'), run_name='__main__')
comparisons = {}
for generated in (destination / 'schema').glob('*.feature_universe.json'):
    reference = ROOT / 'FFCWS/schema' / generated.name
    comparisons[generated.name] = json.loads(generated.read_text()) == json.loads(reference.read_text())
print(json.dumps({'versioned_universe_exact_equality': comparisons}, indent=2))
if len(comparisons) != 3 or not all(comparisons.values()):
    raise RuntimeError('real-input feature universes differ from versioned contract')

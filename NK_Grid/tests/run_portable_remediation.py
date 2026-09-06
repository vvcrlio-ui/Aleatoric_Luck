"""Windows numerical checks only: bypass eager POSIX package __init__, no lock shim.

Never use this runner to certify engine, recovery, RSS or Slurm behavior.
"""
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[2]
package = types.ModuleType('aleatoric_nk_grid')
package.__path__ = [str(root / 'NK_Grid/src/aleatoric_nk_grid')]
sys.modules['aleatoric_nk_grid'] = package
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))

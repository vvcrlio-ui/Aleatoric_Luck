#!/bin/bash
#SBATCH --job-name=al-nk-finalize
#SBATCH --open-mode=append

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?snapshot required}"
FINALIZER_MAP="${2:?finalizer map required}"
: "${SLURM_ARRAY_TASK_ID:?array index required}"

[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
if ! "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>&1; then
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then
    export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "aleatoric_nk_grid is not installed in $VENV" >&2
    exit 1
  fi
fi

"$PYTHON" -m aleatoric_nk_grid.seed_shards finalize \
  --snapshot "$SNAPSHOT" --map "$FINALIZER_MAP" --index "$SLURM_ARRAY_TASK_ID"

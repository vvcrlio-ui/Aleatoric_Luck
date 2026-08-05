#!/bin/bash
#SBATCH --job-name=al-nk-flat
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --open-mode=append

# The submitter supplies partition, CPU, memory and wall-time from the frozen
# chunk resource class.  There are intentionally no header defaults here.
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass a frozen flat-task snapshot}"
: "${SLURM_ARRAY_TASK_ID:?Submit this worker as a Slurm array job}"

[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
if ! "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>&1; then
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then
    export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "aleatoric_nk_grid is not installed in $VENV" >&2
    exit 1
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" -m aleatoric_nk_grid.flat_task_table run --snapshot "$SNAPSHOT" --chunk-id "$SLURM_ARRAY_TASK_ID"

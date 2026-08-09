#!/bin/bash
#SBATCH --job-name=al-nk-dynamic
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass a frozen dynamic-work-queue snapshot}"
ROUND="${2:?Pass the dynamic-work-queue round}"
EXPECTED_PREP_TOKEN="${3:?Pass the current prep Slurm job ID}"
: "${SLURM_ARRAY_TASK_ID:?Submit this worker as a Slurm array job}"

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
IMPORT_STDERR="$(mktemp "${TMPDIR:-/tmp}/aleatoric-nk-grid-import.XXXXXX")"
if "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>"$IMPORT_STDERR"; then
  rm -f "$IMPORT_STDERR"
else
  IMPORT_STATUS=$?
  if [ "$IMPORT_STATUS" = "132" ] || [ "$IMPORT_STATUS" = "134" ] || [ "$IMPORT_STATUS" = "139" ]; then
    echo "venv is incompatible with this node CPU architecture" >&2
    echo "BMRC_GCC_ARCH_NATIVE=${BMRC_GCC_ARCH_NATIVE:-?} MODULE_CPU_TYPE=${MODULE_CPU_TYPE:-?} VENV=$VENV" >&2
    echo "Use --constraint to select a compatible architecture or create a venv for this architecture." >&2
    cat "$IMPORT_STDERR" >&2
    rm -f "$IMPORT_STDERR"
    exit 1
  fi
  rm -f "$IMPORT_STDERR"
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then
    export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "aleatoric_nk_grid is not installed in $VENV" >&2
    exit 1
  fi
fi
# One CPU is required both for deterministic estimator execution and to avoid
# wasting cores while each dynamic worker runs exactly one cell group at a time.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "host=$(hostname -s) arch=${BMRC_GCC_ARCH_NATIVE:-?} cpu_type=${MODULE_CPU_TYPE:-?} python=$PYTHON venv=$VENV round=$ROUND worker=$SLURM_ARRAY_TASK_ID"
"$PYTHON" -m aleatoric_nk_grid.flat_task_table run --snapshot "$SNAPSHOT" --round "$ROUND" --worker-index "$SLURM_ARRAY_TASK_ID" --expected-prep-token "$EXPECTED_PREP_TOKEN"

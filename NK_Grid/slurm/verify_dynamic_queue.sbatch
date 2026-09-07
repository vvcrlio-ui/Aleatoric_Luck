#!/bin/bash
#SBATCH --job-name=al-nk-verify
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass a frozen dynamic-work-queue snapshot}"
ROUND="${2:?Pass frozen final round}"
GENERATION="${3:?Pass frozen final submission generation}"
PREP_JOB="${4:?Pass frozen final prep job ID}"
EXPECTED_PREVIOUS="${5:-}"
EXPECTED_POINTER_VERSION="${6:?Pass frozen final pointer version}"
if [ "$#" -eq 7 ]; then
  EXPECTED_PREVIOUS_PLAN=""
  EXPECTED_PREVIOUS_ROUND=""
  VERIFICATION_TMP_DIR="${7:-}"
else
  EXPECTED_PREVIOUS_PLAN="${7:-}"
  EXPECTED_PREVIOUS_ROUND="${8:-}"
  VERIFICATION_TMP_DIR="${9:-}"
fi

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi
# Match the numerical worker environment frozen into CellExecutionSpec.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1

[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
IMPORT_STDERR="$(mktemp "${TMPDIR:-/tmp}/aleatoric-nk-grid-import.XXXXXX")"
if "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>"$IMPORT_STDERR"; then
  rm -f "$IMPORT_STDERR"
else
  IMPORT_STATUS=$?
  if [ "$IMPORT_STATUS" = "132" ] || [ "$IMPORT_STATUS" = "134" ] || [ "$IMPORT_STATUS" = "139" ]; then
    echo "venv is incompatible with this node CPU architecture" >&2
    echo "BMRC_GCC_NATIVE_ARCH=${BMRC_GCC_NATIVE_ARCH:-?} MODULE_CPU_TYPE=${MODULE_CPU_TYPE:-?} VENV=$VENV" >&2
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
echo "host=$(hostname -s) arch=${BMRC_GCC_NATIVE_ARCH:-?} cpu_type=${MODULE_CPU_TYPE:-?} python=$PYTHON venv=$VENV round=$ROUND generation=$GENERATION phase=verify"
VERIFY_ARGS=(verify --snapshot "$SNAPSHOT" --round "$ROUND" --generation "$GENERATION" --expected-prep-token "$PREP_JOB" --prep-job-id "$PREP_JOB" --expected-pointer-version "$EXPECTED_POINTER_VERSION")
if [ -n "$EXPECTED_PREVIOUS" ]; then VERIFY_ARGS+=(--expected-previous-generation "$EXPECTED_PREVIOUS"); fi
if [ -n "$EXPECTED_PREVIOUS_PLAN" ] && [ -n "$EXPECTED_PREVIOUS_ROUND" ]; then VERIFY_ARGS+=(--expected-previous-execution-plan-id "$EXPECTED_PREVIOUS_PLAN" --expected-previous-round "$EXPECTED_PREVIOUS_ROUND"); fi
if [ -n "$VERIFICATION_TMP_DIR" ]; then
  VERIFY_ARGS+=(--tmp-dir "$VERIFICATION_TMP_DIR")
fi
"$PYTHON" -m aleatoric_nk_grid.flat_task_table "${VERIFY_ARGS[@]}"

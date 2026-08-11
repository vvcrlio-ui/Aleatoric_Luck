#!/bin/bash
#SBATCH --job-name=al-nk-close
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass frozen snapshot}"
ROUND="${2:?Pass round}"
GENERATION="${3:?Pass frozen submission generation}"
PREP_JOB="${4:?Pass frozen prep job ID}"
EXPECTED_PREVIOUS="${5:-}"
EXPECTED_POINTER_VERSION="${6:?Pass frozen pointer version}"
EXPECTED_PREVIOUS_PLAN="${7:-}"
EXPECTED_PREVIOUS_ROUND="${8:-}"

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || { echo "PYTHON_MODULE is set but the module command is unavailable" >&2; exit 1; }
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
    echo "BMRC_GCC_NATIVE_ARCH=${BMRC_GCC_NATIVE_ARCH:-${BMRC_GCC_ARCH_NATIVE:-?}} MODULE_CPU_TYPE=${MODULE_CPU_TYPE:-?} VENV=$VENV" >&2
    cat "$IMPORT_STDERR" >&2; rm -f "$IMPORT_STDERR"; exit 1
  fi
  rm -f "$IMPORT_STDERR"
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"; else echo "aleatoric_nk_grid is not installed in $VENV" >&2; exit 1; fi
fi
CLOSE_ARGS=(close --snapshot "$SNAPSHOT" --round "$ROUND" --generation "$GENERATION" --expected-prep-token "$PREP_JOB" --prep-job-id "$PREP_JOB" --expected-pointer-version "$EXPECTED_POINTER_VERSION")
if [ -n "$EXPECTED_PREVIOUS" ]; then CLOSE_ARGS+=(--expected-previous-generation "$EXPECTED_PREVIOUS"); fi
if [ -n "$EXPECTED_PREVIOUS_PLAN" ] && [ -n "$EXPECTED_PREVIOUS_ROUND" ]; then CLOSE_ARGS+=(--expected-previous-execution-plan-id "$EXPECTED_PREVIOUS_PLAN" --expected-previous-round "$EXPECTED_PREVIOUS_ROUND"); fi
echo "host=$(hostname -s) python=$PYTHON round=$ROUND generation=$GENERATION phase=close"
"$PYTHON" -m aleatoric_nk_grid.flat_task_table "${CLOSE_ARGS[@]}"

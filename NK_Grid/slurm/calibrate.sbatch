#!/bin/bash
# Submit with --job-name="$CALIBRATION_JOB_NAME" and explicitly export
# SBATCH_ACCOUNT plus SBATCH_CONSTRAINT before calling sbatch.
# These retained resource requests predate the reduced calibration scope and
# must be re-measured when that scope changes.
#SBATCH --job-name=al-calibrate
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"

: "${CALIBRATION_JOB_NAME:?Set CALIBRATION_JOB_NAME and pass it to sbatch with --job-name}"
: "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT before submitting this calibration job}"
: "${SBATCH_CONSTRAINT:?Set SBATCH_CONSTRAINT before submitting this calibration job}"
: "${CALIBRATION_SHAPE_SCHEMA:?Set CALIBRATION_SHAPE_SCHEMA to a feature-universe schema}"
: "${CALIBRATION_N_TRAIN:?Set CALIBRATION_N_TRAIN to the training-row count}"
: "${CALIBRATION_TASK_MEMORY_CELLS:?Set CALIBRATION_TASK_MEMORY_CELLS}"
: "${CALIBRATION_OUT_DIR:?Set CALIBRATION_OUT_DIR}"

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }

if [ -n "${CALIBRATION_DTYPE_PROFILE:-}" ]; then
  DTYPE_ARGS=(--feature-dtype-profile "$CALIBRATION_DTYPE_PROFILE")
elif [ -n "${CALIBRATION_ASSUME_FEATURE_DTYPE:-}" ]; then
  DTYPE_ARGS=(--assume-feature-dtype "$CALIBRATION_ASSUME_FEATURE_DTYPE")
else
  echo "Set CALIBRATION_DTYPE_PROFILE or CALIBRATION_ASSUME_FEATURE_DTYPE" >&2
  exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "host=$(hostname -s) arch=${BMRC_GCC_ARCH_NATIVE:-?} cpu_type=${MODULE_CPU_TYPE:-?} python=$PYTHON venv=$VENV calibration_job=$CALIBRATION_JOB_NAME"
"$PYTHON" -m aleatoric_nk_grid.calibrate_cost \
  --shape-schema "$CALIBRATION_SHAPE_SCHEMA" \
  "${DTYPE_ARGS[@]}" \
  --n-train "$CALIBRATION_N_TRAIN" \
  --task-memory-cells "$CALIBRATION_TASK_MEMORY_CELLS" \
  --out-dir "$CALIBRATION_OUT_DIR"

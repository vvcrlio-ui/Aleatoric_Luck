#!/bin/bash
#SBATCH --job-name=al-nk-grid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass the frozen Slurm job snapshot created by submit_nk_grid.sh}"
ALLOW_LARGE_RUN="${2:-0}"

: "${SLURM_ARRAY_TASK_ID:?This worker must be submitted as a Slurm array job.}"
echo \
  "[nk_grid] slurm worker snapshot=$SNAPSHOT array_job_id=${SLURM_ARRAY_JOB_ID:-unknown} task_index=$SLURM_ARRAY_TASK_ID" \
  >&2

if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON -- run ./setup_env.sh or export VENV=/path/to/your/venv" >&2
  exit 1
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi

cd "$PROJECT_DIR"
mkdir -p logs outputs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "[nk_grid] native thread limits OMP=1 MKL=1 OPENBLAS=1 NUMEXPR=1" >&2

ARGS=(
  run
  --snapshot "$SNAPSHOT"
  --index "$SLURM_ARRAY_TASK_ID"
)
if [ "$ALLOW_LARGE_RUN" = "1" ]; then
  ARGS+=(--allow-large-run)
fi

"$PYTHON" src/slurm_jobs.py "${ARGS[@]}"

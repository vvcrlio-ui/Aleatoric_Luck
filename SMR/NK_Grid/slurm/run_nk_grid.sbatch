#!/bin/bash
#SBATCH --job-name=al-nk-grid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-9

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "VENV not found at $VENV/bin/activate -- see NK_Grid/README.md Environment section, or export VENV=/path/to/your/venv" >&2
  exit 1
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs outputs

MODELS=(ols ridge lasso elastic_net random_forest xgboost lightgbm shallow_neural_network extra_trees super_learner)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
python src/nk_grid.py \
  --data data/asample2_withlag.csv \
  --dataset asample2_withlag \
  --outcome Cm_lhourlywage \
  --models "$MODEL" \
  --seed 12345 \
  --test-size 0.3 \
  --n-seeds 100 \
  --n-draws 50 \
  --n-sizes-n "${N_SIZES_N:-20}" \
  --n-sizes-k "${N_SIZES_K:-20}" \
  --max-n 0 \
  --max-k 0 \
  --batch-size "${BATCH_SIZE:-50}" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --out "outputs/nk_grid_${MODEL}.csv"

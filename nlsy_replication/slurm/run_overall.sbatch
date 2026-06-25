#!/bin/bash
#SBATCH --job-name=al-overall
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-7

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "VENV not found at $VENV/bin/activate -- see nlsy_replication/README.md Environment section, or export VENV=/path/to/your/venv" >&2
  exit 1
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs outputs

MODELS=(xgboost lightgbm ols ridge lasso elastic_net random_forest bart)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
python src/overall_prediction.py \
  --models "$MODEL" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --out "outputs/overall_${MODEL}.csv"

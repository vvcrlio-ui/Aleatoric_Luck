#!/bin/bash
#SBATCH --job-name=al-samples
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="${VENV:-$HOME/venvs/aleatoric-luck}"
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs outputs
MODELS=(xgboost lightgbm ols ridge lasso elastic_net random_forest bart)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
python src/sample_size.py \
  --models "$MODEL" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --out "outputs/sample_size_${MODEL}.csv" \
  --power-law-out "outputs/sample_size_power_law_${MODEL}.csv"

#!/bin/bash
#SBATCH --job-name=al-samples-light
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6

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

# BART runs separately via run_sample_size_bart.sbatch: it needs more memory
# and far less concurrency than these models, so it is never in this array.
MODELS=(xgboost lightgbm ols ridge lasso elastic_net random_forest)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
python src/sample_size.py \
  --data data/asample2_withlag.csv \
  --outcome Cm_lhourlywage \
  --models "$MODEL" \
  --seed 12345 \
  --test-size 0.3 \
  --n-sizes 10 \
  --n-draws 20 \
  --min-fraction 0.1 \
  --max-fraction 1.0 \
  --bootstrap-iterations 500 \
  --batch-size 20 \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --out "outputs/sample_size_${MODEL}.csv" \
  --power-law-out "outputs/sample_size_power_law_${MODEL}.csv"

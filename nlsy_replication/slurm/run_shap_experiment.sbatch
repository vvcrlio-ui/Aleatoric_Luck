#!/bin/bash
#SBATCH --job-name=al-shap-order
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
python src/SHAP_experiment.py --n-jobs "$SLURM_CPUS_PER_TASK"

#!/bin/bash
#SBATCH --job-name=al-shap-order
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
python src/SHAP_experiment.py --n-jobs "$SLURM_CPUS_PER_TASK"

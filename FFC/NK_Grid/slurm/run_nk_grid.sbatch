#!/bin/bash
#SBATCH --job-name=al-nk-grid-ffc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-5

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "VENV not found at $VENV/bin/activate -- see README.md Environment section, or export VENV=/path/to/your/venv" >&2
  exit 1
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs outputs

# One array task per FFC outcome/panel. Task type (regression/classification),
# predictor prefixes (X_/C_/M_), and the fixed official test split are all
# already declared per panel in panels.yaml, so this script does not repeat
# them. Set preset: production in panels.yaml before submitting a full run
# (defaults to whatever each panel currently declares).
PANELS=(ffc_gpa ffc_grit ffc_materialHardship ffc_eviction ffc_layoff ffc_jobTraining)
PANEL="${PANELS[$SLURM_ARRAY_TASK_ID]}"

# SLURM_CPUS_PER_TASK is read automatically by run_panels.py's DEFAULTS.
python src/run_panels.py --manifest panels.yaml --only "$PANEL"

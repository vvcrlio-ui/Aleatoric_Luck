#!/bin/bash
#SBATCH --job-name=al-nk-prep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass a frozen dynamic-work-queue snapshot}"
ROUND="${2:?Pass the dynamic-work-queue round}"
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
"$PYTHON" -m aleatoric_nk_grid.flat_task_table prep --snapshot "$SNAPSHOT" --round "$ROUND"

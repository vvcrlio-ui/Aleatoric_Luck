#!/bin/bash
# Build the production dynamic plan as a batch job.
#
# Why not on the login node: the task table for the production grid is ~17.1M
# rows.  write_task_table_streaming keeps memory bounded but needs a local
# SQLite scratch database and a long single-core run, both of which login-node
# limits tend to kill part-way through.
#
# Usage (from the repo root, with VENV / PILOT_* exported):
#   sbatch --account="$PILOT_ACCOUNT" --partition=long --constraint="$PILOT_CONSTRAINT" \
#          NK_Grid/slurm/plan_production.sbatch REQUEST_JSON PLAN_OUT
#
#SBATCH --job-name=al-nk-plan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
set -euo pipefail

REQUEST="${1:?Pass the request JSON built by make_production_request.py}"
PLAN_OUT="${2:?Pass the plan.json output path}"

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:?Export VENV before submitting}"
PYTHON="${PYTHON:-$VENV/bin/python}"

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi
# Match the numerical worker environment frozen into CellExecutionSpec.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1

[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }

# Keep the streaming writer's SQLite scratch on node-local disk, not on GPFS.
export TMPDIR="${TMPDIR:-/tmp/slurm-${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"

echo "host=$(hostname) python=$PYTHON request=$REQUEST plan_out=$PLAN_OUT tmpdir=$TMPDIR"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" -m aleatoric_nk_grid.chunk_planning --request "$REQUEST" --plan-out "$PLAN_OUT"

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "plan bytes: $(wc -c < "$PLAN_OUT")"

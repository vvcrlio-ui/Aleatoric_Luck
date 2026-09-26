#!/usr/bin/env bash
# Site-neutral controller/worker entry; the Python path is frozen at preparation.
set -euo pipefail
PYTHON="${1:?frozen Python required}"
ENTRY="${2:?scheduler entry required}"
MODE="${3:?mode required}"
PLAN="${4:?plan required}"
export PYTHON_MODULE="${5-}"
if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || { echo 'module unavailable' >&2; exit 2; }
  module purge
  module load "$PYTHON_MODULE"
fi
# Bind imports to this entry's frozen checkout, never the submitting shell's
# PYTHONPATH (which may still point at a different completed experiment).
REPO_ROOT="$(cd -- "$(dirname -- "$ENTRY")/.." && pwd -P)"
test -d "$REPO_ROOT/NK_Grid/src/aleatoric_nk_grid" || { echo 'frozen package directory missing' >&2; exit 2; }
export PYTHONPATH="$REPO_ROOT/NK_Grid/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
SERVICE_SLOTS="${6:-1}"
SERVICE_MEMORY_MB="${7:-0}"
if [ "$MODE" = work ] && [ "$SERVICE_SLOTS" -gt 1 ]; then
  SERVICE_CPU_PER_CORE="${8:?service CPU accounting factor required}"
  SERVICE_STEP_CPUS="$((SERVICE_SLOTS * SERVICE_CPU_PER_CORE))"
  export NKGRID_JOB_NTASKS="$SLURM_NTASKS"
  export NKGRID_JOB_TASK_CPU_WIDTH="${SLURM_CPUS_PER_TASK:-1}"
  export NKGRID_JOB_NODELIST="$SLURM_JOB_NODELIST"
  export NKGRID_JOB_CPUS_PER_NODE="$SLURM_JOB_CPUS_PER_NODE"
  export NKGRID_JOB_TASKS_PER_NODE="$SLURM_TASKS_PER_NODE"
  export NKGRID_SERVICE_STEP=1
  HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  unset SLURM_NTASKS_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE
  # Slurm 20 supports exclusive steps, not the newer --exact option.
  exec srun --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task="$SERVICE_STEP_CPUS" --threads-per-core=1 --cpu-bind=cores \
    --mem="${SERVICE_MEMORY_MB}M" --nodelist="$HEAD_NODE" \
    "$PYTHON" "$ENTRY" "$MODE" "$PLAN"
fi
exec "$PYTHON" "$ENTRY" "$MODE" "$PLAN"

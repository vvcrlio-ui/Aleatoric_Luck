#!/usr/bin/env bash
# One worker allocation: order the unfinished cells, compute them, publish once all are done.
# Resource arguments, including account and QoS, are explicit in the launcher.
set -euo pipefail
PLAN="${1:?plan required}"
PYTHON="${2:?shared-environment Python required}"
ENGINE_SRC="${3:?engine source directory required}"
PYTHON_MODULE="${4-}"
RUN="${PLAN%/*}"
cd -- "$RUN"
if [ -n "$PYTHON_MODULE" ]; then
  command -v module >/dev/null 2>&1 || { echo 'module unavailable in batch shell' >&2; exit 2; }
  module purge
  module load "$PYTHON_MODULE"
fi
export PYTHONPATH="$ENGINE_SRC" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
ATTEMPT="$RUN/attempt-$SLURM_JOB_ID"
"$PYTHON" -m aleatoric_nk_grid.cell_pipeline attempt --plan "$PLAN" --attempt "$ATTEMPT"
# A worker that fails leaves its cell unfinished; the others go on, and a resumed run recomputes it.
srun --kill-on-bad-exit=0 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$PYTHON" -m aleatoric_nk_grid.cell_pipeline work --plan "$PLAN" --attempt "$ATTEMPT"
"$PYTHON" -m aleatoric_nk_grid.cell_pipeline finalize --plan "$PLAN"

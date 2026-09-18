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
exec "$PYTHON" "$ENTRY" "$MODE" "$PLAN"

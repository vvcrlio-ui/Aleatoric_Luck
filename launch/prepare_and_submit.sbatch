#!/usr/bin/env bash
# All expensive data validation and task-table construction happen on this node.
set -euo pipefail
REQUEST="${1:?request JSON required}"
: "${ENGINE_DIR:?}" "${VENV:?}" "${PYTHON:?}"
if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || { echo 'module unavailable in batch environment' >&2; exit 2; }
  module purge
  module load "$PYTHON_MODULE"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
"$PYTHON" "$ENGINE_DIR/../launch/experiment.py" execute --request "$REQUEST"

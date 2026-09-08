#!/bin/bash
set -euo pipefail
STATE="${1:?Pass continuation.json}"
# Re-enter project storage before module/Python initialization, matching the
# bootstrap's recovery from Discoverer's observed Slurm startup cwd fallback.
case "$STATE" in
  /*) cd -- "${STATE%/*}" ;;
  *) echo "continuation state path must be absolute" >&2; exit 2 ;;
esac
if [ -n "${PYTHON_MODULE:-}" ]; then
  module purge
  module load "$PYTHON_MODULE"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
: "${PYTHON:?Continuation requires the bootstrap-verified Python environment}"
export TMPDIR="$(dirname "$STATE")/tmp/control-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR"
exec "$PYTHON" "$ENGINE_DIR/../launch/discoverer_continuation.py" control "$STATE"

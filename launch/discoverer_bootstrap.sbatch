#!/usr/bin/env bash
# Resource arguments, including account and QoS, are explicit in the launcher.
set -euo pipefail
REQUEST="${1:?request required}"
PYTHON_MODULE="${2:?module required}"
LAUNCHER="${3:?launcher required}"
# Re-enter the shared directory explicitly if Slurm started before its mount was available.
case "$REQUEST" in /*) cd -- "${REQUEST%/*}" ;; *) echo "request path must be absolute" >&2; exit 2 ;; esac
command -v module >/dev/null 2>&1 || { echo 'module unavailable in batch shell' >&2; exit 2; }
module purge
module load "$PYTHON_MODULE"
export PYTHON_MODULE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# bootstrap configures project-local temporary storage before installing packages.
python3 "$LAUNCHER" bootstrap --request "$REQUEST"

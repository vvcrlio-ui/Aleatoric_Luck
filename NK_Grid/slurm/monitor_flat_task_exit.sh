#!/usr/bin/env bash

# Structured operator/runbook classifier for a completed dynamic queue job.
# Slurm may display exit 8 as FAILED; this adapter records it as terminal
# superseded and never asks an automation to retry it.  Only exit 7 retries.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 PHASE EXIT_CODE" >&2
  exit 2
fi

ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"

exec "$PYTHON" -m aleatoric_nk_grid.dynamic_exit_monitor --phase "$1" "$2"

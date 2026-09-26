#!/usr/bin/env bash
# Route by the saved format; never reinterpret historical snapshots.
set -euo pipefail
ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
ORIGINAL_ARGS=("$@")
MODE=preview
if [ "${1:-}" = --submit ]; then MODE=start; shift; fi
if [ "$#" != 1 ] || [ "$1" = --help ] || [ "$1" = -h ]; then
  echo "Usage: $0 [--submit] SINGLE_MODEL_PLAN.json" >&2
  exit 2
fi
FORMAT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("format", "legacy"))' "$1")
if [ "$FORMAT" != single-model-slurm-v1 ]; then
  exec bash "$ENGINE_DIR/slurm/legacy_submit_flat_task_table.sh" "${ORIGINAL_ARGS[@]}"
fi
exec "$PYTHON" "$ENGINE_DIR/../launch/cluster_scheduler.py" "$MODE" "$1"

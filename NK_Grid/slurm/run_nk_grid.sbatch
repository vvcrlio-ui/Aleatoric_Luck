#!/bin/bash
#SBATCH --job-name=al-nk-grid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Resource values are supplied by the submitter from the immutable chunk
# resource class.  Do not add defaults here: partition, time and memory are
# Stage-B calibration/cluster inputs rather than worker-script policy.
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SNAPSHOT="${1:?Pass a frozen Slurm job snapshot}"
ALLOW_LARGE_RUN="${2:-0}"
RERUN_COMPLETED="${3:-0}"
CHUNK_MAP="${4:-}"
MAX_SLURM_RESTARTS="${MAX_SLURM_RESTARTS:-5}"
REQUEUE_WATCHDOG_SECONDS="${REQUEUE_WATCHDOG_SECONDS:-240}"
: "${SLURM_ARRAY_TASK_ID:?Submit this worker as a Slurm array job}"

case "$ALLOW_LARGE_RUN:$RERUN_COMPLETED" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "Worker boolean arguments must be 0 or 1" >&2; exit 2 ;;
esac
case "$MAX_SLURM_RESTARTS" in
  ''|*[!0-9]*) echo "MAX_SLURM_RESTARTS must be a non-negative integer" >&2; exit 2 ;;
esac
case "$REQUEUE_WATCHDOG_SECONDS" in
  ''|*[!0-9]*) echo "REQUEUE_WATCHDOG_SECONDS must be an integer in [0, 240]" >&2; exit 2 ;;
esac
if [ "$REQUEUE_WATCHDOG_SECONDS" -gt 240 ]; then
  echo "REQUEUE_WATCHDOG_SECONDS must be an integer in [0, 240]" >&2
  exit 2
fi
SLURM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
case "$SLURM_RESTART_COUNT" in
  ''|*[!0-9]*) echo "SLURM_RESTART_COUNT must be a non-negative integer" >&2; exit 2 ;;
esac
if [ "$SLURM_RESTART_COUNT" -gt "$MAX_SLURM_RESTARTS" ]; then
  echo "Slurm restart count $SLURM_RESTART_COUNT exceeds configured maximum $MAX_SLURM_RESTARTS" >&2
  exit 75
fi

if [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || {
    echo "PYTHON_MODULE is set but the module command is unavailable" >&2
    exit 1
  }
  module purge
  module load "$PYTHON_MODULE"
fi
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
if ! "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>&1; then
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then
    export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "aleatoric_nk_grid is not installed in $VENV" >&2
    echo "Install the shared package or set ALEATORIC_NK_GRID_SOURCE_FALLBACK=1" >&2
    exit 1
  fi
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STOP_REQUEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aleatoric-nk-grid-stop.XXXXXX")"
STOP_REQUEST_FILE="$STOP_REQUEST_DIR/requested"
WATCHDOG_CANCEL_FILE="$STOP_REQUEST_DIR/watchdog-cancelled"
REQUEUE_LOCK_DIR="$STOP_REQUEST_DIR/requeue-lock"
export NK_GRID_STOP_REQUEST_FILE="$STOP_REQUEST_FILE"
REQUEUE_REQUESTED=0
PYTHON_PID=""
WATCHDOG_PID=""

cleanup_stop_request() {
  if [ -n "$WATCHDOG_PID" ]; then
    : > "$WATCHDOG_CANCEL_FILE"
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  rm -f "$STOP_REQUEST_FILE"
  rm -f "$WATCHDOG_CANCEL_FILE"
  rmdir "$REQUEUE_LOCK_DIR" 2>/dev/null || true
  rmdir "$STOP_REQUEST_DIR" 2>/dev/null || true
}

requeue_current_task() {
  REQUEUE_REASON="${1:?requeue reason is required}"
  if [ "$SLURM_RESTART_COUNT" -ge "$MAX_SLURM_RESTARTS" ]; then
    echo "Maximum Slurm restart count reached ($MAX_SLURM_RESTARTS)" >&2
    return 75
  fi
  command -v scontrol >/dev/null 2>&1 || {
    echo "scontrol is unavailable; cannot requeue the checkpointed task" >&2
    return 75
  }
  if ! mkdir "$REQUEUE_LOCK_DIR" 2>/dev/null; then
    echo "Requeue has already been requested for this task" >&2
    return 0
  fi
  REQUEUE_TARGET="${SLURM_JOB_ID:?SLURM_JOB_ID is required for requeue}"
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  echo "Requeueing $REQUEUE_TARGET ($REQUEUE_REASON)" >&2
  if ! scontrol requeue "$REQUEUE_TARGET"; then
    rmdir "$REQUEUE_LOCK_DIR" 2>/dev/null || true
    return 75
  fi
}

start_requeue_watchdog() {
  (
    WATCHDOG_ELAPSED=0
    while [ "$WATCHDOG_ELAPSED" -lt "$REQUEUE_WATCHDOG_SECONDS" ]; do
      sleep 1
      [ ! -f "$WATCHDOG_CANCEL_FILE" ] || exit 0
      WATCHDOG_ELAPSED=$((WATCHDOG_ELAPSED + 1))
    done
    [ ! -f "$WATCHDOG_CANCEL_FILE" ] || exit 0
    if [ -f "$STOP_REQUEST_FILE" ] \
      && [ -n "$PYTHON_PID" ] \
      && kill -0 "$PYTHON_PID" 2>/dev/null; then
      echo "Checkpoint-boundary stop exceeded ${REQUEUE_WATCHDOG_SECONDS}s; forcing requeue from the last complete checkpoint" >&2
      requeue_current_task "watchdog fallback before wall timeout"
    fi
  ) &
  WATCHDOG_PID=$!
}

cancel_requeue_watchdog() {
  [ -n "$WATCHDOG_PID" ] || return 0
  : > "$WATCHDOG_CANCEL_FILE"
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=""
}

request_requeue() {
  trap '' USR1
  if [ "$REQUEUE_REQUESTED" = "1" ]; then
    return
  fi
  REQUEUE_REQUESTED=1
  : > "$STOP_REQUEST_FILE"
  echo "USR1 received; requesting a checkpoint-boundary stop" >&2
  start_requeue_watchdog
}

trap cleanup_stop_request EXIT
trap request_requeue USR1

ARGS=(run --snapshot "$SNAPSHOT" --index "$SLURM_ARRAY_TASK_ID")
[ -z "$CHUNK_MAP" ] || ARGS+=(--chunk-map "$CHUNK_MAP")
[ "$ALLOW_LARGE_RUN" = "0" ] || ARGS+=(--allow-large-run)
[ "$RERUN_COMPLETED" = "0" ] || ARGS+=(--rerun-completed)
"$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${ARGS[@]}" &
PYTHON_PID=$!

set +e
wait "$PYTHON_PID"
PYTHON_STATUS=$?
if [ "$REQUEUE_REQUESTED" = "1" ] && [ "$PYTHON_STATUS" -gt 128 ]; then
  wait "$PYTHON_PID"
  RETRY_STATUS=$?
  if [ "$RETRY_STATUS" -ne 127 ]; then
    PYTHON_STATUS="$RETRY_STATUS"
  fi
fi
set -e

if [ "$REQUEUE_REQUESTED" = "1" ]; then
  cancel_requeue_watchdog
  if [ -d "$REQUEUE_LOCK_DIR" ]; then
    echo "Watchdog already requested requeue; exiting current invocation" >&2
    exit 0
  fi
  if [ "$PYTHON_STATUS" -ne 0 ]; then
    echo "Checkpoint-boundary stop failed with status $PYTHON_STATUS; not requeueing" >&2
    exit "$PYTHON_STATUS"
  fi
  if [ ! -f "$STOP_REQUEST_FILE" ]; then
    echo "Run completed before requeue; exiting normally" >&2
    exit 0
  fi
  requeue_current_task "checkpoint-boundary stop"
  exit $?
fi

exit "$PYTHON_STATUS"

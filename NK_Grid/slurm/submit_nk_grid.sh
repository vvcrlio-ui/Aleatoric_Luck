#!/usr/bin/env bash

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MANIFEST=""
SNAPSHOT_INPUT=""
ALLOW_LARGE_RUN=0
DRY_RUN=0
RERUN_COMPLETED=0
MAX_CONCURRENT_PER_CLASS=""
CPUS_PER_TASK=8
SERIAL_CPUS_PER_TASK=1
BART_CPUS_PER_TASK=""
MEMORY="48G"
BART_MEMORY=""
TIME_LIMIT="4-00:00:00"
BART_TIME_LIMIT=""
MAX_RESTARTS=5
REQUEUE_WATCHDOG_SECONDS=240
ONLY_RESOURCE_CLASS=""

usage() {
  echo "Usage: $0 (--manifest ARTICLE/panels.yaml | --snapshot JOBS.json) [options]"
  echo "  --snapshot PATH            reuse a frozen snapshot (requires --resource-class)"
  echo "  --allow-large-run          authorize production-sized tasks"
  echo "  --rerun-completed          intentionally recompute completed tasks"
  echo "  --resource-class CLASS     submit only parallel, serial or bart (recovery)"
  echo "  --max-concurrent-per-class N  cap running tasks in each resource-class array"
  echo "  --cpus-per-task N          CPUs for parallel tasks (default: 8)"
  echo "  --serial-cpus-per-task N   CPUs for serial tasks (default: 1)"
  echo "  --bart-cpus-per-task N     CPUs for BART tasks (default: parallel value)"
  echo "  --mem VALUE                memory for parallel/serial tasks (default: 48G)"
  echo "  --bart-mem VALUE           memory for BART tasks (default: --mem)"
  echo "  --time VALUE               time for parallel/serial tasks (default: 4-00:00:00)"
  echo "  --bart-time VALUE          time for BART tasks (default: --time)"
  echo "  --max-restarts N           checkpoint requeues allowed (default: 5)"
  echo "  --requeue-watchdog-seconds N  force requeue after waiting N seconds (default: 240)"
  echo "  --dry-run                  print resolved jobs without submitting"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="${2:?--manifest requires a path}"; shift 2 ;;
    --snapshot) SNAPSHOT_INPUT="${2:?--snapshot requires a path}"; shift 2 ;;
    --allow-large-run) ALLOW_LARGE_RUN=1; shift ;;
    --rerun-completed) RERUN_COMPLETED=1; shift ;;
    --resource-class) ONLY_RESOURCE_CLASS="${2:?--resource-class requires a value}"; shift 2 ;;
    --max-concurrent-per-class) MAX_CONCURRENT_PER_CLASS="${2:?--max-concurrent-per-class requires a value}"; shift 2 ;;
    --max-concurrent)
      echo "--max-concurrent is ambiguous after resource-class splitting; use --max-concurrent-per-class" >&2
      exit 2
      ;;
    --cpus-per-task) CPUS_PER_TASK="${2:?--cpus-per-task requires a value}"; shift 2 ;;
    --serial-cpus-per-task) SERIAL_CPUS_PER_TASK="${2:?--serial-cpus-per-task requires a value}"; shift 2 ;;
    --bart-cpus-per-task) BART_CPUS_PER_TASK="${2:?--bart-cpus-per-task requires a value}"; shift 2 ;;
    --mem) MEMORY="${2:?--mem requires a value}"; shift 2 ;;
    --bart-mem) BART_MEMORY="${2:?--bart-mem requires a value}"; shift 2 ;;
    --time) TIME_LIMIT="${2:?--time requires a value}"; shift 2 ;;
    --bart-time) BART_TIME_LIMIT="${2:?--bart-time requires a value}"; shift 2 ;;
    --max-restarts) MAX_RESTARTS="${2:?--max-restarts requires a value}"; shift 2 ;;
    --requeue-watchdog-seconds) REQUEUE_WATCHDOG_SECONDS="${2:?--requeue-watchdog-seconds requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -n "$MANIFEST" ] && [ -n "$SNAPSHOT_INPUT" ]; then
  echo "--manifest and --snapshot are mutually exclusive" >&2
  exit 2
fi
if [ -z "$MANIFEST" ] && [ -z "$SNAPSHOT_INPUT" ]; then
  usage >&2
  exit 2
fi
case "$ONLY_RESOURCE_CLASS" in
  ""|parallel|serial|bart) ;;
  *) echo "--resource-class must be parallel, serial or bart" >&2; exit 2 ;;
esac
if [ -n "$SNAPSHOT_INPUT" ] && [ -z "$ONLY_RESOURCE_CLASS" ]; then
  echo "--snapshot recovery requires --resource-class to avoid duplicate arrays" >&2
  exit 2
fi
if [ -n "$MANIFEST" ] && [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$(cd "$ENGINE_DIR/.." && pwd)/$MANIFEST"
fi
if [ -n "$SNAPSHOT_INPUT" ] && [[ "$SNAPSHOT_INPUT" != /* ]]; then
  SNAPSHOT_INPUT="$(cd "$(dirname "$SNAPSHOT_INPUT")" && pwd)/$(basename "$SNAPSHOT_INPUT")"
fi
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
if [ -n "$MANIFEST" ]; then
  [ -f "$MANIFEST" ] || { echo "Panel manifest not found: $MANIFEST" >&2; exit 1; }
else
  [ -f "$SNAPSHOT_INPUT" ] || { echo "Job snapshot not found: $SNAPSHOT_INPUT" >&2; exit 1; }
fi
case "$CPUS_PER_TASK" in
  ''|*[!0-9]*|0) echo "--cpus-per-task must be a positive integer" >&2; exit 2 ;;
esac
case "$SERIAL_CPUS_PER_TASK" in
  ''|*[!0-9]*|0) echo "--serial-cpus-per-task must be a positive integer" >&2; exit 2 ;;
esac
BART_CPUS_PER_TASK="${BART_CPUS_PER_TASK:-$CPUS_PER_TASK}"
BART_MEMORY="${BART_MEMORY:-$MEMORY}"
BART_TIME_LIMIT="${BART_TIME_LIMIT:-$TIME_LIMIT}"
case "$BART_CPUS_PER_TASK" in
  ''|*[!0-9]*|0) echo "--bart-cpus-per-task must be a positive integer" >&2; exit 2 ;;
esac
[ -n "$MEMORY" ] || { echo "--mem must not be empty" >&2; exit 2; }
[ -n "$BART_MEMORY" ] || { echo "--bart-mem must not be empty" >&2; exit 2; }
[ -n "$TIME_LIMIT" ] || { echo "--time must not be empty" >&2; exit 2; }
[ -n "$BART_TIME_LIMIT" ] || { echo "--bart-time must not be empty" >&2; exit 2; }
case "$MAX_RESTARTS" in
  ''|*[!0-9]*) echo "--max-restarts must be a non-negative integer" >&2; exit 2 ;;
esac
case "$REQUEUE_WATCHDOG_SECONDS" in
  ''|*[!0-9]*) echo "--requeue-watchdog-seconds must be an integer in [0, 240]" >&2; exit 2 ;;
esac
if [ "$REQUEUE_WATCHDOG_SECONDS" -gt 240 ]; then
  echo "--requeue-watchdog-seconds must be an integer in [0, 240]" >&2
  exit 2
fi
if [ -n "$MAX_CONCURRENT_PER_CLASS" ]; then
  case "$MAX_CONCURRENT_PER_CLASS" in
    *[!0-9]*|0) echo "--max-concurrent-per-class must be a positive integer" >&2; exit 2 ;;
  esac
fi
if ! "$PYTHON" -c "import aleatoric_nk_grid" >/dev/null 2>&1; then
  if [ "${ALEATORIC_NK_GRID_SOURCE_FALLBACK:-0}" = "1" ]; then
    export PYTHONPATH="$ENGINE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "aleatoric_nk_grid is not installed in $VENV" >&2
    echo "Install the shared package or set ALEATORIC_NK_GRID_SOURCE_FALLBACK=1" >&2
    exit 1
  fi
fi

if [ "$DRY_RUN" = "1" ]; then
  if [ -n "$SNAPSHOT_INPUT" ]; then
    DRY_ARGS=(list --snapshot "$SNAPSHOT_INPUT")
  else
    DRY_ARGS=(list --manifest "$MANIFEST")
  fi
  [ "$RERUN_COMPLETED" = "0" ] || DRY_ARGS+=(--rerun-completed)
  "$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${DRY_ARGS[@]}"
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is not available; submit from a Slurm login node." >&2
  exit 1
}

mkdir -p "$ENGINE_DIR/logs" "$ENGINE_DIR/logs/slurm-specs"
if [ -n "$SNAPSHOT_INPUT" ]; then
  SNAPSHOT="$SNAPSHOT_INPUT"
  ARGS=(count --snapshot "$SNAPSHOT")
  [ "$ALLOW_LARGE_RUN" = "0" ] || ARGS+=(--allow-large-run)
  SNAPSHOT_JOB_COUNT=$("$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${ARGS[@]}")
else
  SNAPSHOT="$ENGINE_DIR/logs/slurm-specs/jobs-$(date +%Y%m%d-%H%M%S)-$$.json"
  ARGS=(snapshot --manifest "$MANIFEST" --snapshot "$SNAPSHOT")
  [ "$ALLOW_LARGE_RUN" = "0" ] || ARGS+=(--allow-large-run)
  [ "$RERUN_COMPLETED" = "0" ] || ARGS+=(--rerun-completed)
  SNAPSHOT_JOB_COUNT=$("$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${ARGS[@]}")
fi
case "$SNAPSHOT_JOB_COUNT" in
  ''|*[!0-9]*|0) echo "Snapshot returned an invalid job count: $SNAPSHOT_JOB_COUNT" >&2; exit 1 ;;
esac

RESOURCE_CLASSES=(parallel serial bart)
CLASS_INDICES_BY_POSITION=()
CLASS_JOB_COUNTS_BY_POSITION=()
PLANNED_JOB_COUNT=0
for CLASS_POSITION in "${!RESOURCE_CLASSES[@]}"; do
  RESOURCE_CLASS="${RESOURCE_CLASSES[$CLASS_POSITION]}"
  CLASS_INDICES=$(
    "$PYTHON" -m aleatoric_nk_grid.slurm_jobs indices \
      --snapshot "$SNAPSHOT" \
      --resource-class "$RESOURCE_CLASS"
  )
  CLASS_INDICES_BY_POSITION[$CLASS_POSITION]="$CLASS_INDICES"
  CLASS_JOB_COUNT=0
  if [ -n "$CLASS_INDICES" ]; then
    IFS=',' read -r -a CLASS_INDEX_ARRAY <<< "$CLASS_INDICES"
    CLASS_JOB_COUNT="${#CLASS_INDEX_ARRAY[@]}"
  fi
  CLASS_JOB_COUNTS_BY_POSITION[$CLASS_POSITION]="$CLASS_JOB_COUNT"
  PLANNED_JOB_COUNT=$((PLANNED_JOB_COUNT + CLASS_JOB_COUNT))
done
if [ "$PLANNED_JOB_COUNT" -ne "$SNAPSHOT_JOB_COUNT" ]; then
  echo "Resource-class plan covers $PLANNED_JOB_COUNT of $SNAPSHOT_JOB_COUNT snapshot jobs; nothing was submitted." >&2
  exit 1
fi

MAX_SLURM_RESTARTS="$MAX_RESTARTS"
export ENGINE_DIR VENV PYTHON MAX_SLURM_RESTARTS REQUEUE_WATCHDOG_SECONDS
EXPORT_SPEC="ALL"
SUBMITTED_ARRAYS=0
SUBMITTED_JOBS=()
for CLASS_POSITION in "${!RESOURCE_CLASSES[@]}"; do
  RESOURCE_CLASS="${RESOURCE_CLASSES[$CLASS_POSITION]}"
  if [ -n "$ONLY_RESOURCE_CLASS" ] && [ "$RESOURCE_CLASS" != "$ONLY_RESOURCE_CLASS" ]; then
    continue
  fi
  CLASS_INDICES="${CLASS_INDICES_BY_POSITION[$CLASS_POSITION]}"
  [ -n "$CLASS_INDICES" ] || continue
  CLASS_JOB_COUNT="${CLASS_JOB_COUNTS_BY_POSITION[$CLASS_POSITION]}"

  CLASS_CPUS="$CPUS_PER_TASK"
  CLASS_MEMORY="$MEMORY"
  CLASS_TIME_LIMIT="$TIME_LIMIT"
  case "$RESOURCE_CLASS" in
    serial)
      CLASS_CPUS="$SERIAL_CPUS_PER_TASK"
      ;;
    bart)
      CLASS_CPUS="$BART_CPUS_PER_TASK"
      CLASS_MEMORY="$BART_MEMORY"
      CLASS_TIME_LIMIT="$BART_TIME_LIMIT"
      ;;
  esac

  ARRAY_SPEC="$CLASS_INDICES"
  [ -z "$MAX_CONCURRENT_PER_CLASS" ] || ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_PER_CLASS}"
  if ! SBATCH_OUTPUT=$(sbatch --parsable \
      --requeue \
      --signal=B:USR1@300 \
      --open-mode=append \
      --job-name="al-nk-grid-$RESOURCE_CLASS" \
      --output="$ENGINE_DIR/logs/%x-%A_%a.out" \
      --error="$ENGINE_DIR/logs/%x-%A_%a.err" \
      --chdir="$ENGINE_DIR" \
      --export="$EXPORT_SPEC" \
      --cpus-per-task="$CLASS_CPUS" \
      --mem="$CLASS_MEMORY" \
      --time="$CLASS_TIME_LIMIT" \
      --array="$ARRAY_SPEC" \
      "$ENGINE_DIR/slurm/run_nk_grid.sbatch" \
      "$SNAPSHOT" "$ALLOW_LARGE_RUN" "$RERUN_COMPLETED"); then
    echo "Failed to submit Slurm $RESOURCE_CLASS array." >&2
    echo "Snapshot retained at $SNAPSHOT" >&2
    if [ "${#SUBMITTED_JOBS[@]}" -gt 0 ]; then
      echo "Already submitted: ${SUBMITTED_JOBS[*]}" >&2
      echo "Do not rerun the whole submission command." >&2
    fi
    echo "After checking squeue, recover only this class with --snapshot '$SNAPSHOT' --resource-class '$RESOURCE_CLASS' and the same policy/resource flags." >&2
    exit 1
  fi
  JOB_ID="${SBATCH_OUTPUT%%;*}"
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "Could not parse Slurm job ID: $SBATCH_OUTPUT. A job may already exist; inspect squeue before resubmitting." >&2
    echo "Snapshot retained at $SNAPSHOT" >&2
    if [ "${#SUBMITTED_JOBS[@]}" -gt 0 ]; then
      echo "Previously submitted: ${SUBMITTED_JOBS[*]}" >&2
    fi
    exit 1
  }
  SUBMITTED_ARRAYS=$((SUBMITTED_ARRAYS + 1))
  SUBMITTED_JOBS+=("$RESOURCE_CLASS=$JOB_ID")
  echo "Submitted Slurm $RESOURCE_CLASS array $JOB_ID with $CLASS_JOB_COUNT tasks (array=$ARRAY_SPEC)"

  RECEIPT_ARGS=(
    receipt
    --snapshot "$SNAPSHOT"
    --job-id "$JOB_ID"
    --array-spec "$ARRAY_SPEC"
    --worker-script "$ENGINE_DIR/slurm/run_nk_grid.sbatch"
    --resource-class "$RESOURCE_CLASS"
    --cpus-per-task "$CLASS_CPUS"
    --memory "$CLASS_MEMORY"
    --time-limit "$CLASS_TIME_LIMIT"
    --max-restarts "$MAX_RESTARTS"
    --requeue-watchdog-seconds "$REQUEUE_WATCHDOG_SECONDS"
  )
  [ "$ALLOW_LARGE_RUN" = "0" ] || RECEIPT_ARGS+=(--allow-large-run)
  [ "$RERUN_COMPLETED" = "0" ] || RECEIPT_ARGS+=(--rerun-completed)
  if ! RECEIPT_OUTPUT=$(
    "$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${RECEIPT_ARGS[@]}" 2>&1
  ); then
    echo "WARNING: job $JOB_ID was submitted, but its receipt could not be written." >&2
    echo "$RECEIPT_OUTPUT" >&2
    echo "Snapshot retained at $SNAPSHOT" >&2
    continue
  fi
  echo "Receipt: $RECEIPT_OUTPUT"
done

if [ "$SUBMITTED_ARRAYS" -eq 0 ]; then
  if [ -n "$ONLY_RESOURCE_CLASS" ]; then
    echo "Snapshot contains no $ONLY_RESOURCE_CLASS tasks; nothing was submitted." >&2
  else
    echo "Snapshot contains $SNAPSHOT_JOB_COUNT jobs but no recognized resource-class tasks." >&2
  fi
  exit 1
fi

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
SUPER_LEARNER_CPUS_PER_TASK=1
MAX_ARRAY_SIZE=""
MEMORY="48G"
TIME_LIMIT="4-00:00:00"
FINALIZER_MEMORY="16G"
FINALIZER_TIME_LIMIT="1-00:00:00"
PUBLISH_MEMORY="32G"
PUBLISH_TIME_LIMIT="2-00:00:00"
MAX_RESTARTS=5
REQUEUE_WATCHDOG_SECONDS=240
ONLY_RESOURCE_CLASS=""
RECOVERY_MASTER_INDICES=""
CLEANUP_NEVER_SATISFIED=0

usage() {
  echo "Usage: $0 (--manifest ARTICLE/panels.yaml | --snapshot JOBS.json) [options]"
  echo "  --snapshot PATH            reuse a frozen snapshot (requires --resource-class)"
  echo "  --allow-large-run          authorize production-sized tasks"
  echo "  --rerun-completed          intentionally recompute completed tasks"
  echo "  --resource-class CLASS     submit only parallel, serial, or super_learner (recovery)"
  echo "  --master-indices CSV       explicit master indices for snapshot recovery"
  echo "  --cleanup-never-satisfied  cancel only DependencyNeverSatisfied jobs verified from this snapshot's receipts"
  echo "  --max-concurrent-per-class N  cap running tasks in each resource-class array"
  echo "  --cpus-per-task N          CPUs for parallel tasks (default: 8)"
  echo "  --serial-cpus-per-task N   CPUs for serial tasks (default: 1)"
  echo "  --super-learner-cpus-per-task N  CPUs for SuperLearner tasks (default: 1)"
  echo "  --max-array-size N         Slurm MaxArraySize override (otherwise auto-detect)"
  echo "  --mem VALUE                memory for parallel/serial tasks (default: 48G)"
  echo "  --time VALUE               time for parallel/serial tasks (default: 4-00:00:00)"
  echo "  --finalizer-mem VALUE      finalizer memory (default: 16G)"
  echo "  --finalizer-time VALUE     finalizer time (default: 1-00:00:00)"
  echo "  --publish-mem VALUE        panel publish memory (default: 32G)"
  echo "  --publish-time VALUE       panel publish time (default: 2-00:00:00)"
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
    --master-indices) RECOVERY_MASTER_INDICES="${2:?--master-indices requires a value}"; shift 2 ;;
    --cleanup-never-satisfied) CLEANUP_NEVER_SATISFIED=1; shift ;;
    --max-concurrent-per-class) MAX_CONCURRENT_PER_CLASS="${2:?--max-concurrent-per-class requires a value}"; shift 2 ;;
    --max-concurrent)
      echo "--max-concurrent is ambiguous after resource-class splitting; use --max-concurrent-per-class" >&2
      exit 2
      ;;
    --cpus-per-task) CPUS_PER_TASK="${2:?--cpus-per-task requires a value}"; shift 2 ;;
    --serial-cpus-per-task) SERIAL_CPUS_PER_TASK="${2:?--serial-cpus-per-task requires a value}"; shift 2 ;;
    --super-learner-cpus-per-task) SUPER_LEARNER_CPUS_PER_TASK="${2:?--super-learner-cpus-per-task requires a value}"; shift 2 ;;
    --max-array-size) MAX_ARRAY_SIZE="${2:?--max-array-size requires a value}"; shift 2 ;;
    --mem) MEMORY="${2:?--mem requires a value}"; shift 2 ;;
    --time) TIME_LIMIT="${2:?--time requires a value}"; shift 2 ;;
    --finalizer-mem) FINALIZER_MEMORY="${2:?--finalizer-mem requires a value}"; shift 2 ;;
    --finalizer-time) FINALIZER_TIME_LIMIT="${2:?--finalizer-time requires a value}"; shift 2 ;;
    --publish-mem) PUBLISH_MEMORY="${2:?--publish-mem requires a value}"; shift 2 ;;
    --publish-time) PUBLISH_TIME_LIMIT="${2:?--publish-time requires a value}"; shift 2 ;;
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
  ""|parallel|serial|super_learner) ;;
  *) echo "--resource-class must be parallel, serial, or super_learner" >&2; exit 2 ;;
esac
if [ -n "$SNAPSHOT_INPUT" ] && [ -z "$ONLY_RESOURCE_CLASS" ]; then
  echo "--snapshot recovery requires --resource-class to avoid duplicate arrays" >&2
  exit 2
fi
if [ -n "$RECOVERY_MASTER_INDICES" ] && [ -z "$SNAPSHOT_INPUT" ]; then
  echo "--master-indices is only valid with --snapshot recovery" >&2
  exit 2
fi
if [ "$CLEANUP_NEVER_SATISFIED" = "1" ] && [ -z "$SNAPSHOT_INPUT" ]; then
  echo "--cleanup-never-satisfied is only valid with --snapshot recovery" >&2
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
case "$SUPER_LEARNER_CPUS_PER_TASK" in
  ''|*[!0-9]*|0) echo "--super-learner-cpus-per-task must be a positive integer" >&2; exit 2 ;;
esac
if [ -n "$MAX_ARRAY_SIZE" ]; then
  case "$MAX_ARRAY_SIZE" in
    ''|*[!0-9]*|0) echo "--max-array-size must be a positive integer" >&2; exit 2 ;;
  esac
fi
[ -n "$MEMORY" ] || { echo "--mem must not be empty" >&2; exit 2; }
[ -n "$TIME_LIMIT" ] || { echo "--time must not be empty" >&2; exit 2; }
[ -n "$FINALIZER_MEMORY" ] || { echo "--finalizer-mem must not be empty" >&2; exit 2; }
[ -n "$FINALIZER_TIME_LIMIT" ] || { echo "--finalizer-time must not be empty" >&2; exit 2; }
[ -n "$PUBLISH_MEMORY" ] || { echo "--publish-mem must not be empty" >&2; exit 2; }
[ -n "$PUBLISH_TIME_LIMIT" ] || { echo "--publish-time must not be empty" >&2; exit 2; }
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

if [ -z "$MAX_ARRAY_SIZE" ]; then
  MAX_ARRAY_CONFIG=$(scontrol show config 2>/dev/null || true)
  MAX_ARRAY_SIZE=$(printf '%s\n' "$MAX_ARRAY_CONFIG" | sed -n 's/^[[:space:]]*MaxArraySize[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n 1)
fi
case "$MAX_ARRAY_SIZE" in
  ''|*[!0-9]*|0) echo "Unable to determine Slurm MaxArraySize; pass --max-array-size explicitly." >&2; exit 1 ;;
esac

RESOURCE_CLASSES=(parallel serial super_learner)
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
if [ -n "$SNAPSHOT_INPUT" ]; then
  if FINALIZATION_STATUS=$(
    "$PYTHON" -m aleatoric_nk_grid.slurm_jobs \
      finalization-status --snapshot "$SNAPSHOT" 2>&1
  ); then
    echo "Finalization status: $FINALIZATION_STATUS"
  else
    echo "Recovery refused because this snapshot has an active finalizer/publish job: $FINALIZATION_STATUS" >&2
    exit 1
  fi
  DIAGNOSTIC_ARGS=(
    dependency-diagnostics --snapshot "$SNAPSHOT"
  )
  [ "$CLEANUP_NEVER_SATISFIED" = "0" ] || \
    DIAGNOSTIC_ARGS+=(--cleanup-never-satisfied)
  echo "Dependency diagnostics: $(
    "$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${DIAGNOSTIC_ARGS[@]}"
  )"
  RECOVERY_ARGS=(
    recovery-indices
    --snapshot "$SNAPSHOT"
    --resource-class "$ONLY_RESOURCE_CLASS"
  )
  [ -z "$RECOVERY_MASTER_INDICES" ] || \
    RECOVERY_ARGS+=(--master-indices "$RECOVERY_MASTER_INDICES")
  RECOVERY_MASTER_INDICES=$(
    "$PYTHON" -m aleatoric_nk_grid.seed_shards "${RECOVERY_ARGS[@]}"
  )
  for CLASS_POSITION in "${!RESOURCE_CLASSES[@]}"; do
    [ "${RESOURCE_CLASSES[$CLASS_POSITION]}" = "$ONLY_RESOURCE_CLASS" ] || continue
    CLASS_INDICES_BY_POSITION[$CLASS_POSITION]="$RECOVERY_MASTER_INDICES"
    CLASS_JOB_COUNT=0
    if [ -n "$RECOVERY_MASTER_INDICES" ]; then
      IFS=',' read -r -a RECOVERY_INDEX_ARRAY <<< "$RECOVERY_MASTER_INDICES"
      CLASS_JOB_COUNT="${#RECOVERY_INDEX_ARRAY[@]}"
    fi
    CLASS_JOB_COUNTS_BY_POSITION[$CLASS_POSITION]="$CLASS_JOB_COUNT"
  done
fi

MAX_SLURM_RESTARTS="$MAX_RESTARTS"
export ENGINE_DIR VENV PYTHON MAX_SLURM_RESTARTS REQUEUE_WATCHDOG_SECONDS
EXPORT_SPEC="ALL"
SUBMITTED_ARRAYS=0
SUBMITTED_JOBS=()
LAST_JOB_IDS=()
for CLASS_POSITION in "${!RESOURCE_CLASSES[@]}"; do
  RESOURCE_CLASS="${RESOURCE_CLASSES[$CLASS_POSITION]}"
  [ -z "$ONLY_RESOURCE_CLASS" ] || [ "$RESOURCE_CLASS" = "$ONLY_RESOURCE_CLASS" ] || continue
  CLASS_JOB_COUNT="${CLASS_JOB_COUNTS_BY_POSITION[$CLASS_POSITION]}"
  [ "$CLASS_JOB_COUNT" -gt 0 ] || continue
  case "$RESOURCE_CLASS" in
    parallel) CLASS_CPUS="$CPUS_PER_TASK" ;;
    serial) CLASS_CPUS="$SERIAL_CPUS_PER_TASK" ;;
    super_learner) CLASS_CPUS="$SUPER_LEARNER_CPUS_PER_TASK" ;;
  esac
  CLASS_LAST_JOB=""
  CHUNK_ORDINAL=0
  CHUNK_START=0
  while [ "$CHUNK_START" -lt "$CLASS_JOB_COUNT" ]; do
    CHUNK_SIZE="$MAX_ARRAY_SIZE"
    [ $((CLASS_JOB_COUNT - CHUNK_START)) -lt "$CHUNK_SIZE" ] && CHUNK_SIZE=$((CLASS_JOB_COUNT - CHUNK_START))
    CHUNK_ARGS=(chunk-map --snapshot "$SNAPSHOT" --resource-class "$RESOURCE_CLASS" --max-array-size "$MAX_ARRAY_SIZE" --chunk-ordinal "$CHUNK_ORDINAL")
    [ -z "$SNAPSHOT_INPUT" ] || CHUNK_ARGS+=(--master-indices "$RECOVERY_MASTER_INDICES")
    CHUNK_MAP=$("$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${CHUNK_ARGS[@]}")
    ARRAY_SPEC="0-$((CHUNK_SIZE - 1))"
    [ "$CHUNK_SIZE" -eq 1 ] && ARRAY_SPEC="0"
    [ -z "$MAX_CONCURRENT_PER_CLASS" ] || ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_PER_CLASS}"
    DEPENDENCY_ARGS=()
    [ -z "$CLASS_LAST_JOB" ] || DEPENDENCY_ARGS=("--dependency=afterany:$CLASS_LAST_JOB")
    if ! SBATCH_OUTPUT=$(set +u; sbatch --parsable --requeue --signal=B:USR1@300 --open-mode=append \
      --job-name="al-nk-grid-$RESOURCE_CLASS" --output="$ENGINE_DIR/logs/%x-%A_%a.out" --error="$ENGINE_DIR/logs/%x-%A_%a.err" \
      --chdir="$ENGINE_DIR" --export="$EXPORT_SPEC" --cpus-per-task="$CLASS_CPUS" --mem="$MEMORY" --time="$TIME_LIMIT" \
      --array="$ARRAY_SPEC" "${DEPENDENCY_ARGS[@]}" "$ENGINE_DIR/slurm/run_nk_grid.sbatch" \
      "$SNAPSHOT" "$ALLOW_LARGE_RUN" "$RERUN_COMPLETED" "$CHUNK_MAP"); then
      echo "Failed to submit Slurm $RESOURCE_CLASS array." >&2
      echo "Snapshot retained at $SNAPSHOT; already submitted: ${SUBMITTED_JOBS[*]:-}" >&2
      exit 1
    fi
    JOB_ID="${SBATCH_OUTPUT%%;*}"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Could not parse Slurm job ID: $SBATCH_OUTPUT" >&2; exit 1; }
    SUBMITTED_ARRAYS=$((SUBMITTED_ARRAYS + 1)); SUBMITTED_JOBS+=("$RESOURCE_CLASS=$JOB_ID")
    echo "Submitted Slurm $RESOURCE_CLASS array $JOB_ID with $CHUNK_SIZE tasks (array=$ARRAY_SPEC, chunk=$CHUNK_ORDINAL)"
    RECEIPT_ARGS=(receipt --snapshot "$SNAPSHOT" --job-id "$JOB_ID" --array-spec "$ARRAY_SPEC" --worker-script "$ENGINE_DIR/slurm/run_nk_grid.sbatch" --resource-class "$RESOURCE_CLASS" --cpus-per-task "$CLASS_CPUS" --memory "$MEMORY" --time-limit "$TIME_LIMIT" --max-restarts "$MAX_RESTARTS" --requeue-watchdog-seconds "$REQUEUE_WATCHDOG_SECONDS" --chunk-map-path "$CHUNK_MAP")
    [ -z "$CLASS_LAST_JOB" ] || RECEIPT_ARGS+=(--dependency-job-id "$CLASS_LAST_JOB")
    [ "$ALLOW_LARGE_RUN" = "0" ] || RECEIPT_ARGS+=(--allow-large-run)
    [ "$RERUN_COMPLETED" = "0" ] || RECEIPT_ARGS+=(--rerun-completed)
    if RECEIPT_OUTPUT=$("$PYTHON" -m aleatoric_nk_grid.slurm_jobs "${RECEIPT_ARGS[@]}" 2>&1); then echo "Receipt: $RECEIPT_OUTPUT"; else echo "WARNING: job $JOB_ID was submitted, but its receipt could not be written: $RECEIPT_OUTPUT" >&2; fi
    CLASS_LAST_JOB="$JOB_ID"; CHUNK_START=$((CHUNK_START + CHUNK_SIZE)); CHUNK_ORDINAL=$((CHUNK_ORDINAL + 1))
  done
  LAST_JOB_IDS+=("$CLASS_LAST_JOB")
done

if [ "$SUBMITTED_ARRAYS" -eq 0 ]; then
  echo "No missing/incomplete matching resource-class tasks were submitted." >&2
  exit 1
fi

FINALIZER_MAP=$(
  "$PYTHON" -m aleatoric_nk_grid.seed_shards build-map \
    --snapshot "$SNAPSHOT" --kind finalizer
)
PUBLISH_MAP=$(
  "$PYTHON" -m aleatoric_nk_grid.seed_shards build-map \
    --snapshot "$SNAPSHOT" --kind publish
)
FINALIZER_COUNT=$(
  "$PYTHON" -m aleatoric_nk_grid.seed_shards map-count --map "$FINALIZER_MAP"
)
PUBLISH_COUNT=$(
  "$PYTHON" -m aleatoric_nk_grid.seed_shards map-count --map "$PUBLISH_MAP"
)
case "$FINALIZER_COUNT:$PUBLISH_COUNT" in
  *[!0-9:]*|0:*|*:0) echo "Finalizer/publish maps must be non-empty." >&2; exit 1 ;;
esac
FINALIZER_ARRAY="0-$((FINALIZER_COUNT - 1))"
[ "$FINALIZER_COUNT" -eq 1 ] && FINALIZER_ARRAY="0"
FINALIZER_DEPENDENCY=$(IFS=:; echo "${LAST_JOB_IDS[*]}")
if ! FINALIZER_OUTPUT=$(sbatch --parsable \
  --job-name="al-nk-finalize" \
  --output="$ENGINE_DIR/logs/%x-%A_%a.out" \
  --error="$ENGINE_DIR/logs/%x-%A_%a.err" \
  --chdir="$ENGINE_DIR" --export="$EXPORT_SPEC" \
  --cpus-per-task=1 --mem="$FINALIZER_MEMORY" --time="$FINALIZER_TIME_LIMIT" \
  --array="$FINALIZER_ARRAY" \
  --dependency="afterany:$FINALIZER_DEPENDENCY" \
  "$ENGINE_DIR/slurm/finalize_seed_shards.sbatch" \
  "$SNAPSHOT" "$FINALIZER_MAP" finalize); then
  echo "Seed arrays submitted but finalizer submission failed." >&2
  exit 1
fi
FINALIZER_JOB_ID="${FINALIZER_OUTPUT%%;*}"
[[ "$FINALIZER_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "Could not parse finalizer job ID: $FINALIZER_OUTPUT" >&2
  exit 1
}
FINALIZER_RECEIPT_ARGS=(
  finalization-receipt
  --snapshot "$SNAPSHOT"
  --stage finalizer
  --job-id "$FINALIZER_JOB_ID"
  --array-spec "$FINALIZER_ARRAY"
  --finalization-map "$FINALIZER_MAP"
  --worker-script "$ENGINE_DIR/slurm/finalize_seed_shards.sbatch"
  --dependency-job-ids "${FINALIZER_DEPENDENCY//:/,}"
  --cpus-per-task 1
  --memory "$FINALIZER_MEMORY"
  --time-limit "$FINALIZER_TIME_LIMIT"
)
if FINALIZER_RECEIPT_OUTPUT=$(
  "$PYTHON" -m aleatoric_nk_grid.slurm_jobs \
    "${FINALIZER_RECEIPT_ARGS[@]}" 2>&1
); then
  echo "Finalizer receipt: $FINALIZER_RECEIPT_OUTPUT"
else
  scancel "$FINALIZER_JOB_ID" >/dev/null 2>&1 || true
  echo "Finalizer receipt failed; cancelled newly submitted job $FINALIZER_JOB_ID: $FINALIZER_RECEIPT_OUTPUT" >&2
  exit 1
fi
echo "Submitted seed finalizer array $FINALIZER_JOB_ID with $FINALIZER_COUNT tasks"

PUBLISH_ARRAY="0-$((PUBLISH_COUNT - 1))"
[ "$PUBLISH_COUNT" -eq 1 ] && PUBLISH_ARRAY="0"
if ! PUBLISH_OUTPUT=$(sbatch --parsable \
  --job-name="al-nk-publish" \
  --output="$ENGINE_DIR/logs/%x-%A_%a.out" \
  --error="$ENGINE_DIR/logs/%x-%A_%a.err" \
  --chdir="$ENGINE_DIR" --export="$EXPORT_SPEC" \
  --cpus-per-task=1 --mem="$PUBLISH_MEMORY" --time="$PUBLISH_TIME_LIMIT" \
  --array="$PUBLISH_ARRAY" \
  --dependency="afterany:$FINALIZER_JOB_ID" \
  "$ENGINE_DIR/slurm/finalize_seed_shards.sbatch" \
  "$SNAPSHOT" "$PUBLISH_MAP" publish); then
  echo "Finalizer submitted but panel publish submission failed." >&2
  exit 1
fi
PUBLISH_JOB_ID="${PUBLISH_OUTPUT%%;*}"
[[ "$PUBLISH_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "Could not parse panel publish job ID: $PUBLISH_OUTPUT" >&2
  exit 1
}
PUBLISH_RECEIPT_ARGS=(
  finalization-receipt
  --snapshot "$SNAPSHOT"
  --stage publish
  --job-id "$PUBLISH_JOB_ID"
  --array-spec "$PUBLISH_ARRAY"
  --finalization-map "$PUBLISH_MAP"
  --worker-script "$ENGINE_DIR/slurm/finalize_seed_shards.sbatch"
  --dependency-job-ids "$FINALIZER_JOB_ID"
  --cpus-per-task 1
  --memory "$PUBLISH_MEMORY"
  --time-limit "$PUBLISH_TIME_LIMIT"
)
if PUBLISH_RECEIPT_OUTPUT=$(
  "$PYTHON" -m aleatoric_nk_grid.slurm_jobs \
    "${PUBLISH_RECEIPT_ARGS[@]}" 2>&1
); then
  echo "Publish receipt: $PUBLISH_RECEIPT_OUTPUT"
else
  scancel "$PUBLISH_JOB_ID" >/dev/null 2>&1 || true
  echo "Publish receipt failed; cancelled newly submitted job $PUBLISH_JOB_ID: $PUBLISH_RECEIPT_OUTPUT" >&2
  exit 1
fi
echo "Submitted panel publish array $PUBLISH_JOB_ID with $PUBLISH_COUNT tasks"

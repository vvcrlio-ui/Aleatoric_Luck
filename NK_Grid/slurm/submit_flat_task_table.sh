#!/usr/bin/env bash

# Submit every dynamic-work-queue round at once.  prep/work/verify use
# afterany because Slurm time limits are normal round termination.  Finalize
# alone uses afterok so incomplete verification can never publish a result.
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SUBMIT=0
PLAN=""

usage() {
  echo "Usage: $0 [--submit] PLAN.json" >&2
  echo "  Without --submit, print the dynamic Slurm chain without submitting." >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *) [ -z "$PLAN" ] || { echo "Only one plan JSON path may be supplied" >&2; exit 2; }; PLAN="$1"; shift ;;
  esac
done
[ -n "$PLAN" ] || { usage; exit 2; }
[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
[ -f "$PLAN" ] || { echo "Plan JSON not found: $PLAN" >&2; exit 1; }

FIELDS=$("$PYTHON" -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    snapshot = p["snapshot"]; workers = int(p["workers"]); rounds = int(p["rounds"])
    submission = p["submission"]; array = submission["array"]; args = submission["sbatch_args"]
    account = submission["account"]; constraint = submission["constraint"]
    preparation = p["preparation"]; prep_args = preparation["sbatch_args"]
    verification = p["verification"]; verify_args = verification["sbatch_args"]
    finalization = p["finalization"]; final_args = finalization["sbatch_args"]
    prep_tmp_dir = preparation.get("tmp_dir")
    verify_tmp_dir = verification.get("tmp_dir")
    final_tmp_dir = finalization.get("tmp_dir")
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"invalid dynamic plan JSON: {exc}")
if not isinstance(snapshot, str) or not snapshot or workers < 1 or rounds < 1:
    raise SystemExit("invalid dynamic plan JSON: snapshot, workers, and rounds are required")
if not isinstance(array, str) or not array or not isinstance(args, list) or not args or not all(isinstance(a, str) and a for a in args):
    raise SystemExit("invalid dynamic plan JSON: submission array and sbatch_args are required")
if not isinstance(account, str) or not account or not isinstance(constraint, str) or not constraint:
    raise SystemExit("invalid dynamic plan JSON: submission account and constraint are required")
if "--cpus-per-task=1" not in args:
    raise SystemExit("invalid dynamic plan JSON: dynamic workers must request one CPU")
for phase, phase_args, phase_tmp_dir in (
    ("preparation", prep_args, prep_tmp_dir),
    ("verification", verify_args, verify_tmp_dir),
    ("finalization", final_args, final_tmp_dir),
):
    if not isinstance(phase_args, list) or not phase_args or not all(isinstance(a, str) and a for a in phase_args):
        raise SystemExit(f"invalid dynamic plan JSON: {phase}.sbatch_args are required")
    if "--cpus-per-task=1" not in phase_args:
        raise SystemExit(f"invalid dynamic plan JSON: {phase} must request one CPU")
    if f"--account={account}" not in phase_args:
        raise SystemExit(f"invalid dynamic plan JSON: {phase} account must match submission account")
    if constraint != "none" and f"--constraint={constraint}" not in phase_args:
        raise SystemExit(f"invalid dynamic plan JSON: {phase} constraint must match submission constraint")
    if phase_tmp_dir is not None and (not isinstance(phase_tmp_dir, str) or not phase_tmp_dir):
        raise SystemExit(f"invalid dynamic plan JSON: {phase}.tmp_dir must be null or a non-empty string")
values = [snapshot, str(workers), str(rounds), array, account, constraint, *args, *prep_args, *verify_args, *final_args]
for directory in (prep_tmp_dir, verify_tmp_dir, final_tmp_dir):
    if directory is not None:
        values.append(directory)
if any("\n" in value or "\r" in value for value in values):
    raise SystemExit("invalid dynamic plan JSON: submission fields must be single-line strings")
print(snapshot); print(workers); print(rounds); print(array); print(account); print(constraint)
print(len(args))
for arg in args: print(arg)
print(len(prep_args))
for arg in prep_args: print(arg)
print(len(verify_args))
for arg in verify_args: print(arg)
print(len(final_args))
for arg in final_args: print(arg)
print("__NK_GRID_NONE__" if prep_tmp_dir is None else prep_tmp_dir)
print("__NK_GRID_NONE__" if verify_tmp_dir is None else verify_tmp_dir)
print("__NK_GRID_NONE__" if final_tmp_dir is None else final_tmp_dir)
' "$PLAN")
FIELD_LINES=()
while IFS= read -r line || [ -n "$line" ]; do
  FIELD_LINES+=("$line")
done <<< "$FIELDS"
SNAPSHOT="${FIELD_LINES[0]}"; WORKERS="${FIELD_LINES[1]}"; ROUNDS="${FIELD_LINES[2]}"; ARRAY_SPEC="${FIELD_LINES[3]}"
SBATCH_ACCOUNT="${FIELD_LINES[4]}"; SBATCH_CONSTRAINT="${FIELD_LINES[5]}"
FIELD_INDEX=6
SBATCH_COUNT="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
SBATCH_ARGS=("${FIELD_LINES[@]:$FIELD_INDEX:$SBATCH_COUNT}"); FIELD_INDEX=$((FIELD_INDEX + SBATCH_COUNT))
PREPARATION_COUNT="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
PREPARATION_SBATCH_ARGS=("${FIELD_LINES[@]:$FIELD_INDEX:$PREPARATION_COUNT}"); FIELD_INDEX=$((FIELD_INDEX + PREPARATION_COUNT))
VERIFICATION_COUNT="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
VERIFICATION_SBATCH_ARGS=("${FIELD_LINES[@]:$FIELD_INDEX:$VERIFICATION_COUNT}"); FIELD_INDEX=$((FIELD_INDEX + VERIFICATION_COUNT))
FINALIZATION_COUNT="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
FINALIZATION_SBATCH_ARGS=("${FIELD_LINES[@]:$FIELD_INDEX:$FINALIZATION_COUNT}"); FIELD_INDEX=$((FIELD_INDEX + FINALIZATION_COUNT))
PREPARATION_TMP_DIR="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
VERIFICATION_TMP_DIR="${FIELD_LINES[$FIELD_INDEX]}"; FIELD_INDEX=$((FIELD_INDEX + 1))
FINALIZATION_TMP_DIR="${FIELD_LINES[$FIELD_INDEX]}"

PREP="$ENGINE_DIR/slurm/prep_dynamic_queue.sbatch"
WORKER="$ENGINE_DIR/slurm/run_flat_task_table.sbatch"
CLOSER="$ENGINE_DIR/slurm/close_dynamic_queue.sbatch"
VERIFY="$ENGINE_DIR/slurm/verify_dynamic_queue.sbatch"
FINALIZER="$ENGINE_DIR/slurm/finalize_dynamic_queue.sbatch"
for script in "$PREP" "$WORKER" "$CLOSER" "$VERIFY" "$FINALIZER"; do [ -f "$script" ] || { echo "Dynamic queue script not found: $script" >&2; exit 1; }; done
# Slurm opens these files before executing any sbatch script.
mkdir -p logs

submit_or_print() {
  local label="$1"; shift
  if [ "$SUBMIT" = "0" ]; then
    printf 'DRY RUN (%s):' "$label"; printf ' %q' "$@"; printf '\n'
    JOB_ID="dry-$label"
  else
    JOB_ID=$(sbatch --parsable "${@:2}")
  fi
}

GENERATIONS=($("$PYTHON" -c 'import sys, uuid; [print(uuid.uuid4()) for _ in range(int(sys.argv[1]))]' "$ROUNDS"))
PREVIOUS_CLOSE=""
RECEIPT=""
for ROUND in $(seq 1 "$ROUNDS"); do
  GENERATION="${GENERATIONS[$((ROUND - 1))]}"
  POINTER_VERSION=$((ROUND - 1))
  PREVIOUS_GENERATION=""
  if [ "$ROUND" -gt 1 ]; then PREVIOUS_GENERATION="${GENERATIONS[$((ROUND - 2))]}"; fi
  PREP_COMMAND=("${PREPARATION_SBATCH_ARGS[@]}")
  PREP_TMP_ARG=""
  if [ "$PREPARATION_TMP_DIR" != "__NK_GRID_NONE__" ]; then PREP_TMP_ARG="$PREPARATION_TMP_DIR"; fi
  PREP_SCRIPT_ARGS=("$SNAPSHOT" "$ROUND" "$PREP_TMP_ARG")
  PREP_SCRIPT_ARGS+=("$GENERATION" "$PREVIOUS_GENERATION" "$POINTER_VERSION")
  if [ -z "$PREVIOUS_CLOSE" ]; then
    submit_or_print "prep-$ROUND" sbatch "${PREP_COMMAND[@]}" "$PREP" "${PREP_SCRIPT_ARGS[@]}"
  else
    submit_or_print "prep-$ROUND" sbatch "${PREP_COMMAND[@]}" "--dependency=afterany:$PREVIOUS_CLOSE" "$PREP" "${PREP_SCRIPT_ARGS[@]}"
  fi
  PREP_JOB="$JOB_ID"
  if [ "$SUBMIT" != "0" ]; then
    if [ -z "$PREVIOUS_CLOSE" ]; then RECEIPT+="prep-$ROUND"$'\t'"$PREP_JOB"$'\t'"none"$'\n';
    else RECEIPT+="prep-$ROUND"$'\t'"$PREP_JOB"$'\t'"afterany:$PREVIOUS_CLOSE"$'\n'; fi
  fi
  submit_or_print "work-$ROUND" sbatch "${SBATCH_ARGS[@]}" "--dependency=afterany:$PREP_JOB" "--array=$ARRAY_SPEC" "$WORKER" "$SNAPSHOT" "$ROUND" "$PREP_JOB" "$GENERATION" "$PREVIOUS_GENERATION" "$POINTER_VERSION"
  WORK_JOB="$JOB_ID"
  [ "$SUBMIT" = "0" ] || RECEIPT+="work-$ROUND"$'\t'"$WORK_JOB"$'\t'"afterany:$PREP_JOB"$'\n'
  submit_or_print "close-$ROUND" sbatch "${PREPARATION_SBATCH_ARGS[@]}" "--dependency=afterany:$WORK_JOB" "$CLOSER" "$SNAPSHOT" "$ROUND" "$GENERATION" "$PREP_JOB" "$PREVIOUS_GENERATION" "$POINTER_VERSION"
  CLOSE_JOB="$JOB_ID"
  [ "$SUBMIT" = "0" ] || RECEIPT+="close-$ROUND"$'\t'"$CLOSE_JOB"$'\t'"afterany:$WORK_JOB"$'\n'
  PREVIOUS_CLOSE="$CLOSE_JOB"
done
LAST_GENERATION="${GENERATIONS[$((ROUNDS - 1))]}"
LAST_PREVIOUS=""
if [ "$ROUNDS" -gt 1 ]; then LAST_PREVIOUS="${GENERATIONS[$((ROUNDS - 2))]}"; fi
LAST_POINTER_VERSION=$((ROUNDS - 1))
VERIFY_COMMAND=("${VERIFICATION_SBATCH_ARGS[@]}" "--dependency=afterany:$PREVIOUS_CLOSE" "$VERIFY" "$SNAPSHOT" "$ROUNDS" "$LAST_GENERATION" "$PREP_JOB" "$LAST_PREVIOUS" "$LAST_POINTER_VERSION")
if [ "$VERIFICATION_TMP_DIR" != "__NK_GRID_NONE__" ]; then
  VERIFY_COMMAND+=("$VERIFICATION_TMP_DIR")
fi
submit_or_print "verify" sbatch "${VERIFY_COMMAND[@]}"
VERIFY_JOB="$JOB_ID"
[ "$SUBMIT" = "0" ] || RECEIPT+="verify"$'\t'"$VERIFY_JOB"$'\t'"afterany:$PREVIOUS_CLOSE"$'\n'
FINALIZER_COMMAND=("${FINALIZATION_SBATCH_ARGS[@]}" "--dependency=afterok:$VERIFY_JOB" "$FINALIZER" "$SNAPSHOT" "$ROUNDS" "$LAST_GENERATION" "$PREP_JOB" "$LAST_PREVIOUS" "$LAST_POINTER_VERSION")
if [ "$FINALIZATION_TMP_DIR" != "__NK_GRID_NONE__" ]; then
  FINALIZER_COMMAND+=("$FINALIZATION_TMP_DIR")
fi
submit_or_print "finalize" sbatch "${FINALIZER_COMMAND[@]}"
FINALIZE_JOB="$JOB_ID"
[ "$SUBMIT" = "0" ] || RECEIPT+="finalize"$'\t'"$FINALIZE_JOB"$'\t'"afterok:$VERIFY_JOB"$'\n'

if [ "$SUBMIT" = "1" ]; then
  RECEIPT_PATH=$("$PYTHON" -c '
import datetime, json, sys
from pathlib import Path
plan = Path(sys.argv[1]); rows = [line.rstrip("\n").split("\t") for line in sys.stdin if line.strip()]
p = json.load(plan.open(encoding="utf-8")); submission = p["submission"]
resources = {
    phase: {"sbatch_args": p[phase]["sbatch_args"], "tmp_dir": p[phase].get("tmp_dir")}
    for phase in ("preparation", "verification", "finalization")
}
out = plan.with_name(plan.stem + ".submission-receipt-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
out.write_text(json.dumps({"plan": str(plan.resolve()), "snapshot": p["snapshot"], "sbatch_account": submission["account"], "sbatch_constraint": submission["constraint"], "resources": resources, "jobs": [{"label": label, "slurm_job_id": job_id, "dependency": dependency} for label, job_id, dependency in rows]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
' "$PLAN" <<< "$RECEIPT")
  echo "Receipt: $RECEIPT_PATH"
fi

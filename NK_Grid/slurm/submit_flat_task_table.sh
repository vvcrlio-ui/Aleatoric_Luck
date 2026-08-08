#!/usr/bin/env bash

# Submit every dynamic-work-queue round at once.  afterany is deliberate:
# Slurm time limits are normal round termination, not a reason to stop the
# recovery chain.
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
print(snapshot); print(workers); print(rounds); print(array); print(account); print(constraint)
for arg in args: print(arg)
' "$PLAN")
FIELD_LINES=()
while IFS= read -r line || [ -n "$line" ]; do
  FIELD_LINES+=("$line")
done <<< "$FIELDS"
SNAPSHOT="${FIELD_LINES[0]}"; WORKERS="${FIELD_LINES[1]}"; ROUNDS="${FIELD_LINES[2]}"; ARRAY_SPEC="${FIELD_LINES[3]}"
SBATCH_ACCOUNT="${FIELD_LINES[4]}"; SBATCH_CONSTRAINT="${FIELD_LINES[5]}"
SBATCH_ARGS=("${FIELD_LINES[@]:6}")

PREP="$ENGINE_DIR/slurm/prep_dynamic_queue.sbatch"
WORKER="$ENGINE_DIR/slurm/run_flat_task_table.sbatch"
VERIFY="$ENGINE_DIR/slurm/verify_dynamic_queue.sbatch"
for script in "$PREP" "$WORKER" "$VERIFY"; do [ -f "$script" ] || { echo "Dynamic queue script not found: $script" >&2; exit 1; }; done
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

PREVIOUS=""
RECEIPT=""
for ROUND in $(seq 1 "$ROUNDS"); do
  if [ -z "$PREVIOUS" ]; then
    submit_or_print "prep-$ROUND" sbatch "${SBATCH_ARGS[@]}" "$PREP" "$SNAPSHOT" "$ROUND"
  else
    submit_or_print "prep-$ROUND" sbatch "${SBATCH_ARGS[@]}" "--dependency=afterany:$PREVIOUS" "$PREP" "$SNAPSHOT" "$ROUND"
  fi
  PREP_JOB="$JOB_ID"
  [ "$SUBMIT" = "0" ] || RECEIPT+="prep-$ROUND $PREP_JOB"$'\n'
  submit_or_print "work-$ROUND" sbatch "${SBATCH_ARGS[@]}" "--dependency=afterany:$PREP_JOB" "--array=$ARRAY_SPEC" "$WORKER" "$SNAPSHOT" "$ROUND"
  WORK_JOB="$JOB_ID"
  [ "$SUBMIT" = "0" ] || RECEIPT+="work-$ROUND $WORK_JOB"$'\n'
  PREVIOUS="$WORK_JOB"
done
submit_or_print "verify" sbatch "${SBATCH_ARGS[@]}" "--dependency=afterany:$PREVIOUS" "$VERIFY" "$SNAPSHOT"
VERIFY_JOB="$JOB_ID"
[ "$SUBMIT" = "0" ] || RECEIPT+="verify $VERIFY_JOB"$'\n'

if [ "$SUBMIT" = "1" ]; then
  RECEIPT_PATH=$("$PYTHON" -c '
import datetime, json, sys
from pathlib import Path
plan = Path(sys.argv[1]); rows = [line.split(" ", 1) for line in sys.stdin if line.strip()]
p = json.load(plan.open(encoding="utf-8")); submission = p["submission"]
out = plan.with_name(plan.stem + ".submission-receipt-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
out.write_text(json.dumps({"plan": str(plan.resolve()), "snapshot": p["snapshot"], "sbatch_account": submission["account"], "sbatch_constraint": submission["constraint"], "jobs": [{"label": a, "slurm_job_id": b} for a, b in rows]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
' "$PLAN" <<< "$RECEIPT")
  echo "Receipt: $RECEIPT_PATH"
fi

#!/usr/bin/env bash

# Submit an already-calibrated flat-task plan.  This script deliberately only
# consumes the frozen plan; chunk_planning owns all resource derivation.
set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$(cd "$ENGINE_DIR/.." && pwd)/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
SUBMIT=0
MAX_ARRAY_SIZE=""
PLAN=""

usage() {
  echo "Usage: $0 [--submit --max-array-size N] PLAN.json" >&2
  echo "  Without --submit, print the sbatch commands without submitting." >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --max-array-size) MAX_ARRAY_SIZE="${2:?--max-array-size requires a positive integer}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *)
      if [ -n "$PLAN" ]; then
        echo "Only one plan JSON path may be supplied" >&2
        usage
        exit 2
      fi
      PLAN="$1"
      shift
      ;;
  esac
done
if [ -z "$PLAN" ]; then
  usage
  exit 2
fi
if [ -n "$MAX_ARRAY_SIZE" ]; then
  case "$MAX_ARRAY_SIZE" in
    ''|*[!0-9]*|0) echo "--max-array-size must be a positive integer" >&2; exit 2 ;;
  esac
fi
if [ "$SUBMIT" = "1" ] && [ -z "$MAX_ARRAY_SIZE" ]; then
  echo "--submit requires --max-array-size from the target cluster" >&2
  exit 2
fi

[ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; exit 1; }
[ -f "$PLAN" ] || { echo "Plan JSON not found: $PLAN" >&2; exit 1; }

# Validate all required fields before printing or submitting a single job.  The
# compact JSON records preserve sbatch argument boundaries without inventing
# defaults for any resource value.
SUBMISSIONS=$(
  "$PYTHON" -c '
import json
import sys

path = sys.argv[1]
max_array_size = int(sys.argv[2]) if sys.argv[2] else None
try:
    payload = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid plan JSON {path}: {exc}")
if not isinstance(payload, dict):
    raise SystemExit("invalid plan JSON: expected an object")
snapshot = payload.get("snapshot")
if not isinstance(snapshot, str) or not snapshot:
    raise SystemExit("invalid plan JSON: missing non-empty snapshot")
submissions = payload.get("submissions")
if not isinstance(submissions, list) or not submissions:
    raise SystemExit("invalid plan JSON: missing non-empty submissions")
for index, submission in enumerate(submissions):
    if not isinstance(submission, dict):
        raise SystemExit(f"invalid plan JSON: submissions[{index}] must be an object")
    resource_class = submission.get("resource_class")
    array = submission.get("array")
    sbatch_args = submission.get("sbatch_args")
    if not isinstance(resource_class, str) or not resource_class:
        raise SystemExit(f"invalid plan JSON: submissions[{index}] missing non-empty resource_class")
    if not isinstance(array, str) or not array:
        raise SystemExit(f"invalid plan JSON: submissions[{index}] missing non-empty array")
    array_base = array.split("%", 1)[0]
    parts = array_base.split("-", 1)
    if not all(part.isdigit() for part in parts) or len(parts) > 2:
        raise SystemExit(f"invalid plan JSON: submissions[{index}] has unsupported array {array!r}")
    first = int(parts[0])
    last = int(parts[-1])
    if last < first:
        raise SystemExit(f"invalid plan JSON: submissions[{index}] has descending array {array!r}")
    if max_array_size is not None and last - first + 1 > max_array_size:
        raise SystemExit(
            f"invalid plan JSON: submissions[{index}] array {array!r} has "
            f"{last - first + 1} tasks, exceeding MaxArraySize {max_array_size}"
        )
    if not isinstance(sbatch_args, list) or not sbatch_args or not all(isinstance(arg, str) and arg for arg in sbatch_args):
        raise SystemExit(f"invalid plan JSON: submissions[{index}] missing non-empty sbatch_args")
    print(json.dumps({"resource_class": resource_class, "array": array, "sbatch_args": sbatch_args, "snapshot": snapshot}, separators=(",", ":")))
' "$PLAN" "$MAX_ARRAY_SIZE"
)

WORKER="$ENGINE_DIR/slurm/run_flat_task_table.sbatch"
[ -f "$WORKER" ] || { echo "Flat-task worker not found: $WORKER" >&2; exit 1; }

RECEIPT_LINES=""
while IFS= read -r SUBMISSION; do
  FIELDS=$("$PYTHON" -c '
import json
import sys
entry = json.loads(sys.argv[1])
print(entry["resource_class"])
print(entry["array"])
print(entry["snapshot"])
for arg in entry["sbatch_args"]:
    print(arg)
' "$SUBMISSION")
  IFS= read -r RESOURCE_CLASS <<< "$FIELDS"
  FIELDS="${FIELDS#*$'\n'}"
  IFS= read -r ARRAY_SPEC <<< "$FIELDS"
  FIELDS="${FIELDS#*$'\n'}"
  IFS= read -r SNAPSHOT <<< "$FIELDS"
  FIELDS="${FIELDS#*$'\n'}"
  SBATCH_ARGS=()
  while IFS= read -r ARGUMENT || [ -n "$ARGUMENT" ]; do
    SBATCH_ARGS+=("$ARGUMENT")
  done <<< "$FIELDS"
  COMMAND=(sbatch "${SBATCH_ARGS[@]}" "--array=$ARRAY_SPEC" "$WORKER" "$SNAPSHOT")

  if [ "$SUBMIT" = "0" ]; then
    printf 'DRY RUN:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    continue
  fi

  OUTPUT=$("${COMMAND[@]}")
  JOB_ID="${OUTPUT##* }"
  [ -n "$JOB_ID" ] || { echo "sbatch returned no job ID for $RESOURCE_CLASS" >&2; exit 1; }
  echo "Submitted flat-task $RESOURCE_CLASS array $JOB_ID (array=$ARRAY_SPEC)"
  RECEIPT_LINES+=$("$PYTHON" -c '
import json
import sys
entry = json.loads(sys.argv[1])
entry["slurm_job_id"] = sys.argv[2]
print(json.dumps(entry, separators=(",", ":")))
' "$SUBMISSION" "$JOB_ID")$'\n'
done <<< "$SUBMISSIONS"

if [ "$SUBMIT" = "1" ]; then
  RECEIPT_PATH=$("$PYTHON" -c '
import datetime
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
entries = [json.loads(line) for line in sys.stdin if line.strip()]
payload = json.load(plan_path.open(encoding="utf-8"))
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
receipt = plan_path.with_name(plan_path.stem + ".submission-receipt-" + stamp + ".json")
receipt.write_text(json.dumps({"plan": str(plan_path.resolve()), "snapshot": payload["snapshot"], "submitted_at": timestamp, "jobs": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(receipt)
' "$PLAN" <<< "$RECEIPT_LINES")
  echo "Receipt: $RECEIPT_PATH"
fi

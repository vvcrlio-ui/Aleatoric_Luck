#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
MANIFEST="$PROJECT_DIR/panels.yaml"
ALLOW_LARGE_RUN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./slurm/submit_nk_grid.sh [options]

Options:
  --manifest PATH     Panel manifest (default: panels.yaml)
  --allow-large-run   Authorize array tasks above the safety threshold
  --dry-run           Print the expanded panel/model job list without submitting
  -h, --help          Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest)
      [ "$#" -ge 2 ] || { echo "--manifest requires a path" >&2; exit 2; }
      MANIFEST="$2"
      shift 2
      ;;
    --allow-large-run)
      ALLOW_LARGE_RUN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$PROJECT_DIR/$MANIFEST"
fi
if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON" >&2
  echo "Create the environment with: ./setup_env.sh" >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Panel manifest not found: $MANIFEST" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  "$PYTHON" "$PROJECT_DIR/src/slurm_jobs.py" list --manifest "$MANIFEST"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this command on a Slurm login node." >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs" "$PROJECT_DIR/logs/slurm-specs"
SNAPSHOT="$PROJECT_DIR/logs/slurm-specs/jobs-$(date +%Y%m%d-%H%M%S)-$$.json"
SNAPSHOT_ARGS=(snapshot --manifest "$MANIFEST" --snapshot "$SNAPSHOT")
if [ "$ALLOW_LARGE_RUN" = "1" ]; then
  SNAPSHOT_ARGS+=(--allow-large-run)
fi
JOB_COUNT=$("$PYTHON" "$PROJECT_DIR/src/slurm_jobs.py" "${SNAPSHOT_ARGS[@]}")
ARRAY_MAX=$((JOB_COUNT - 1))

export PROJECT_DIR VENV PYTHON
cd "$PROJECT_DIR"
echo "Submitting $JOB_COUNT panel/model jobs from snapshot $SNAPSHOT"
sbatch \
  --array="0-$ARRAY_MAX" \
  "$PROJECT_DIR/slurm/run_nk_grid.sbatch" \
  "$SNAPSHOT" \
  "$ALLOW_LARGE_RUN"

#!/usr/bin/env bash
# One entry point for Linux/WSL and Slurm. No shell-evaluated user arguments.
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ] || SCRIPT_DIR=.
ROOT="$(cd "$SCRIPT_DIR" && pwd)"
PROFILE="local"; UPDATE=0; PREVIEW=0; ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) [ "$#" -ge 2 ] || { echo '--profile needs a name' >&2; exit 2; }; PROFILE="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --update) UPDATE=1; shift ;;
    --dry-run|-h|--help) PREVIEW=1; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$PROFILE" in
  local) ;;
  bmrc) source "$ROOT/launch/profiles/bmrc.sh" ;;
  discoverer) source "$ROOT/launch/profiles/discoverer.sh" ;;
  *) echo "Unknown profile: $PROFILE (choose local, bmrc or discoverer)" >&2; exit 2 ;;
esac
cd "$ROOT"
if [ "$UPDATE" = 1 ] && [ "$PREVIEW" = 0 ]; then
  [ "$(git branch --show-current)" = 'SMR&FFC' ] || { echo 'Update requires branch SMR&FFC; switch explicitly first.' >&2; exit 2; }
  [ -z "$(git status --porcelain)" ] || { echo 'Update refused: worktree has changes.' >&2; exit 2; }
  git pull --ff-only origin 'SMR&FFC'
  exec bash "$ROOT/run.sh" "${ARGS[@]}"
fi
if [ "$PREVIEW" = 0 ] && [ -n "${PYTHON_MODULE:-}" ]; then
  command -v module >/dev/null 2>&1 || { echo "module command unavailable; use a cluster login shell for $PYTHON_MODULE" >&2; exit 2; }
  module purge
  module load "$PYTHON_MODULE"
fi
if [ "$PROFILE" = bmrc ] && [ -z "${VENV:-}" ]; then
  # Resolve AFTER module load: its CPU ABI can differ across cluster nodes.
  if [ "$PREVIEW" = 0 ]; then
    : "${MODULE_CPU_TYPE:?Python module did not set MODULE_CPU_TYPE; set VENV explicitly}"
  fi
  export VENV="$ROOT/venv/aleatoric-${MODULE_CPU_TYPE:-module-cpu}"
fi
export ENGINE_DIR="$ROOT/NK_Grid"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
BOOTSTRAP="${NKGRID_BOOTSTRAP_PYTHON:-python3}"
command -v "$BOOTSTRAP" >/dev/null 2>&1 || BOOTSTRAP=python
bootstrap_compatible() {
  "$BOOTSTRAP" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)' >/dev/null 2>&1
}
# A preview skips installation/submission, but still needs a supported parser.
# Login nodes may default to an older system Python even with a site profile.
if ! bootstrap_compatible; then
  if [ "$PREVIEW" = 1 ] && [ -n "${PYTHON_MODULE:-}" ] && [ -z "${NKGRID_BOOTSTRAP_PYTHON:-}" ] && command -v module >/dev/null 2>&1; then
    module purge
    module load "$PYTHON_MODULE"
    hash -r
    BOOTSTRAP=python3
    command -v "$BOOTSTRAP" >/dev/null 2>&1 || BOOTSTRAP=python
  fi
  bootstrap_compatible || { echo 'Python 3.11–3.14 required; load the site Python module or set NKGRID_BOOTSTRAP_PYTHON to a supported interpreter.' >&2; exit 2; }
fi
exec "$BOOTSTRAP" "$ROOT/launch/experiment.py" "${ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=gpa-scheduler-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
# Explicit paths supplied at submission; no production controller is touched.
repo=$(realpath "$1")
old_repo=$(realpath "$2")
validation=$(realpath "$3")
python_bin="$old_repo/runs/discoverer-gpa-timing-balanced-20260909/venv/bin/python"
module purge
module load python/3/3.12/3.12.4
cd "$repo"
export SCHEDULER_OLD_REPO="$old_repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}/gpa-scheduler-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$validation/matplotlib"
export MPLCONFIGDIR="$validation/matplotlib"
# Test dependencies are isolated; the old production environment stays frozen.
"$python_bin" -m pip install --disable-pip-version-check --target "$validation/test-deps" 'pytest>=8,<10'
export PYTHONPATH="$repo/NK_Grid/src:$validation/test-deps"
"$python_bin" -m pytest -q checks/queue_checks.py checks/svd_checks.py checks/migration_checks.py checks/cache_checks.py checks/worker_checks.py checks/resume_checks.py checks/cutover_checks.py checks/tls_checks.py | tee "$validation/pytest.txt"
mkdir -p "$validation/input"
cp -a "$old_repo/runs/discoverer-gpa-timing-balanced-20260909/prepared" "$validation/input/prepared"
"$python_bin" checks/linux_native_worker_probe.py --repo "$repo" --schema "$validation/input/prepared/schema/ffc_median_mode_gpa.json" --output "$validation/native" --scratch "$TMPDIR/dispatcher" | tee "$validation/native.txt"

#!/usr/bin/env bash
#SBATCH --job-name=controller-lockf-acceptance
#SBATCH --account=mills.prj
#SBATCH --partition=short
#SBATCH --constraint=skl-compat
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:03:00
#SBATCH --output=controller-lockf-acceptance-%j.out
#SBATCH --error=controller-lockf-acceptance-%j.err

set -euo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
PYTHON=$(command -v python3)
export SLURM_EXPORT_ENV=ALL
repo_root=${1:?Pass the checkout containing the repaired controller}
scratch_parent=${2:?Pass an existing compute-visible scratch directory}
test -d "$scratch_parent"
run_root=$(mktemp -d "$scratch_parent/controller-lockf-acceptance-${SLURM_JOB_ID}.XXXXXX")
printf 'Acceptance directory: %s\n' "$run_root"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 "$PYTHON" -u \
  "$repo_root/runs/controller-lockf-repair-20261001/acceptance.py" "$run_root"

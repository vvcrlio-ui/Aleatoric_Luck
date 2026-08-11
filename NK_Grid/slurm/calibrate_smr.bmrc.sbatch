#!/bin/bash
# Cost calibration for the SMR panel shape.
#SBATCH --job-name=al-calibrate-smr
#SBATCH -A mills.prj
#SBATCH --partition=short
#SBATCH --constraint=skl-compat
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

# calibrate_cost spawns its own workers; pin BLAS/OpenMP to avoid oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "host=$(hostname -s) arch=${BMRC_GCC_ARCH_NATIVE:-?} python=$(which python)"
echo "started $(date)"

CELLS="ols:1000:497,ridge:1000:497,lasso:1000:497,random_forest:1000:497,extra_trees:1000:497,shallow_neural_network:1000:497,lightgbm:1000:497,xgboost:1000:497"

python -m aleatoric_nk_grid.calibrate_cost \
  --shape-schema SMR/schema/asample2_withlag.feature_universe.json \
  --feature-dtype-profile SMR/dtype_profile.json \
  --n-train 5970 \
  --task-memory-cells "$CELLS" \
  --parallel-efficiency-cells "$CELLS" \
  --out-dir NK_Grid/calibration

echo "finished $(date)"

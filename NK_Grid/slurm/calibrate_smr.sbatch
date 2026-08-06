#!/bin/bash
# Cost calibration for the SMR panel shape.
# Reads only the feature-universe schema (structure) and an externally measured
# dtype profile; the panel it measures on is synthetic and generated in-process.
#SBATCH --job-name=al-calibrate-smr
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CELLS="ols:1000:497,ridge:1000:497,lasso:1000:497,random_forest:1000:497,extra_trees:1000:497,shallow_neural_network:1000:497,lightgbm:1000:497,xgboost:1000:497"

.venv/bin/python -m aleatoric_nk_grid.calibrate_cost \
  --shape-schema SMR/schema/asample2_withlag.feature_universe.json \
  --feature-dtype-profile SMR/dtype_profile.json \
  --n-train 5970 \
  --task-memory-cells "$CELLS" \
  --parallel-efficiency-cells "$CELLS" \
  --out-dir NK_Grid/calibration

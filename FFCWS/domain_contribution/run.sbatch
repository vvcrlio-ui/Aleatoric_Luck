#!/bin/bash
#SBATCH --job-name=ffc-domains
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One array task per outcome (0 gpa, 1 grit, 2 materialHardship, 3 eviction, 4 layoff,
# 5 jobTraining); each fits all 513 domain models for every listed seed.
# Submit from the repository root with the Python environment's modules loaded:
#   sbatch -A ACCOUNT -p PARTITION --cpus-per-task=128 --time=02:00:00 --array=0-5 \
#     --output=OUT_DIR/%x-%A_%a.out FFCWS/domain_contribution/run.sbatch PYTHON DATA_DIR OUT_DIR SEED...
set -euo pipefail

outcomes=(gpa grit materialHardship eviction layoff jobTraining)
python=$1 data=$2 out=$3
shift 3

"$python" "$SLURM_SUBMIT_DIR/FFCWS/domain_contribution/fit.py" --data "$data" --out "$out" \
  --outcomes "${outcomes[$SLURM_ARRAY_TASK_ID]}" --seeds "$@" --workers "$SLURM_CPUS_PER_TASK"

#!/bin/bash
#SBATCH --job-name=al-samples-bart
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
# BMRC: none of the scripts in this repo set --partition/--qos. If the
# default partition rejects a 96G request, add the right one here, e.g.:
#   #SBATCH --partition=<your-high-mem-partition>

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "VENV not found at $VENV/bin/activate -- see nlsy_replication/README.md Environment section, or export VENV=/path/to/your/venv" >&2
  exit 1
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-Python/3.11}"
fi
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs outputs

# BART alone, deliberately at low concurrency: experiment.py's
# parallel_preference() forces process-based parallelism for BART because
# BartPy's RNG is process-global, and each process holds its own full copy
# of the training data plus tree/MCMC sample state. --n-jobs is capped at 2
# (matching --cpus-per-task) rather than scaled up, to bound peak memory.
# If this still OOMs or you want more throughput, --n-jobs/--cpus-per-task
# and --mem are the two knobs to revisit together.
#
# The script checkpoints after every batch and skips already-completed
# (model, n_samples, seed) rows on restart, so if this hits the time limit
# it is safe to resubmit and it will resume rather than redo finished work.
python src/sample_size.py \
  --data data/asample2_withlag.csv \
  --outcome Cm_lhourlywage \
  --models bart \
  --seed 12345 \
  --test-size 0.3 \
  --n-sizes 10 \
  --n-draws 20 \
  --min-fraction 0.1 \
  --max-fraction 1.0 \
  --bootstrap-iterations 500 \
  --batch-size 20 \
  --n-jobs 2 \
  --out outputs/sample_size_bart.csv \
  --power-law-out outputs/sample_size_power_law_bart.csv

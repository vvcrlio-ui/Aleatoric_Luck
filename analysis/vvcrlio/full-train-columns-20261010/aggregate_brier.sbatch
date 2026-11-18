#!/bin/bash
#SBATCH --job-name=ffc-brier-aggregate
#SBATCH --account=ehpc-dev-2026d08-299
#SBATCH --qos=ehpc-dev-2026d08-299
#SBATCH --partition=cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/valhalla/projects/ehpc-dev-2026d08-299/full-train-columns-20261010/aggregate-brier-%j.out
# Usage: sbatch aggregate_brier.sbatch RUN_DIR
module purge
module load python/3/3.12/3.12.4
W=/valhalla/projects/ehpc-dev-2026d08-299/full-train-columns-20261010
mkdir -p $W/aggregates-ffc
/valhalla/projects/ehpc-dev-2026d08-299/nkgrid-envs/deps-31716bded8124e01/bin/python $W/aggregate_brier.py $W/aggregates-ffc $W/targets-$(basename $1).npz $1

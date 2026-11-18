#!/bin/bash
#SBATCH --job-name=nk-matrix-export
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
#SBATCH --output=/valhalla/projects/ehpc-dev-2026d08-299/nk-analysis-20261011/export-%j.out
# Usage: sbatch export_matrix.sbatch RUN_DIR...
module purge
module load python/3/3.12/3.12.4
W=/valhalla/projects/ehpc-dev-2026d08-299/nk-analysis-20261011
mkdir -p $W/matrices
cd $W
/valhalla/projects/ehpc-dev-2026d08-299/nkgrid-envs/deps-31716bded8124e01/bin/python $W/export_matrix.py $W/matrices "$@"

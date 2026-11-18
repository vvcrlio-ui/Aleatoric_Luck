#!/bin/bash
#SBATCH --job-name=ftc-aggregate
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
#SBATCH --output=/valhalla/projects/ehpc-dev-2026d08-299/full-train-columns-20261010/aggregate-%j.out
# Arguments: TABLE (final.csv or final.pending.csv) then published run directories.
module purge
module load python/3/3.12/3.12.4
W=/valhalla/projects/ehpc-dev-2026d08-299/full-train-columns-20261010
mkdir -p $W/aggregates
python3 $W/aggregate_panels.py $W/aggregates "$@"

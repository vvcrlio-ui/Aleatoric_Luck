#!/usr/bin/env bash
#SBATCH --job-name=mh-range-bars
#SBATCH --account=ehpc-dev-2026d08-299
#SBATCH --qos=ehpc-dev-2026d08-299
#SBATCH --partition=cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/valhalla/projects/ehpc-dev-2026d08-299/mh-range-bars-20261010
#SBATCH --output=/valhalla/projects/ehpc-dev-2026d08-299/mh-range-bars-20261010/aggregate-%j.log
set -euo pipefail
module purge
module load python/3/3.12/3.12.4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd /valhalla/projects/ehpc-dev-2026d08-299/mh-range-bars-20261010
python3 aggregate_remote.py aggregate.json

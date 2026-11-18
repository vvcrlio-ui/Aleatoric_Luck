#!/bin/bash
#SBATCH --job-name=seed-draw-design
#SBATCH --account=ehpc-dev-2026d08-299
#SBATCH --qos=ehpc-dev-2026d08-299
#SBATCH --partition=cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-core=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/valhalla/projects/ehpc-dev-2026d08-299/seed-draw-design-20261010/analyze-%j.out
module load python/3/3.12/3.12.4
P=/valhalla/projects/ehpc-dev-2026d08-299
D=$P/seed-draw-design-20261010
PY=$P/nkgrid-envs/deps-31716bded8124e01/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" $D/analyze.py $P/aleatoric_luck-mh-8ba797e/runs/prod-mh-6h3g-b/final.csv mh $D/mh.json > $D/mh.log 2>&1 &
"$PY" $D/analyze.py $P/aleatoric-tree-gpa-sl7-d63b29f/runs/tree-ordinal-gpa-sl7-production/final.csv gpa $D/gpa.json > $D/gpa.log 2>&1 &
"$PY" $D/analyze.py $P/aleatoric-smr-hourlywage-sl7-a2d0bad/runs/smr-hourlywage-sl7-production/final.csv smr_hourlywage $D/smr.json > $D/smr.log 2>&1 &
wait
tail -n 30 $D/*.log

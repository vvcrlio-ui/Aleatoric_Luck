# BMRC CPU partition; supply the project account on the command line.
export PYTHON_MODULE=Python/3.11.3-GCCcore-12.3.0
export NKGRID_PARTITION=long
export NKGRID_CONSTRAINT=skl-compat
export NKGRID_MAX_TIME=10-00:00:00
# BMRC asks automated tools to leave at least 100 s between squeue/sacct calls.
export NKGRID_SLURM_QUERY_INTERVAL=100
# CPU compute nodes cannot reach PyPI; wheels are downloaded on the login node.
export NKGRID_OFFLINE_COMPUTE=1

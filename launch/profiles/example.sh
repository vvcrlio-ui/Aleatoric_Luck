# Copy to NAME.sh and select it with --profile NAME.
# Set the partition and its maximum worker wall time.
export NKGRID_PARTITION=compute
export NKGRID_MAX_TIME=24:00:00
# Include these only when the site needs them:
# export PYTHON_MODULE=Python/3.12
# export NKGRID_CONSTRAINT=cpu_feature
# export NKGRID_QOS=normal
# NKGRID_QOS=account means the account supplied with --account.
# With no QoS here or on the CLI, Slurm's account default applies.

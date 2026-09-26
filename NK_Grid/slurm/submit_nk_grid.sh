#!/usr/bin/env bash
# The pre-table submission protocol is retired in this checkout.
set -euo pipefail
echo "Use ./run.sh slurm --profile SITE --account ACCOUNT [panel/options]. All new cluster submissions use the shared single-model scheduler." >&2
echo "For a stopped legacy experiment, use its original frozen checkout or explicitly migrate sealed results." >&2
exit 2

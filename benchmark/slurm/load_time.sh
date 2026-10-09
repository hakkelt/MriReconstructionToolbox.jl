#!/bin/bash
#SBATCH --job-name=ristretto-load-time
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Load time and time to first solve (benchmark/load_time.jl) at 1 and 8 threads, for this
# checkout or for each `name:path` checkout given with `--refs`. Load time is dominated by
# compilation and file reads, not by memory bandwidth, so the job books 16 cores rather than a
# whole node (which the test partition's CPU-minute limit would not admit for an hour).
#
#   benchmark/slurm/submit.sh load_time.sh [--refs=master:/path,head:/path] [reps]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

REFS="here:$REPO_ROOT"
if [ $# -gt 0 ] && [ "${1#--refs=}" != "$1" ]; then
    REFS="${1#--refs=}"
    shift
fi

for t in 1 8; do
    for ref in ${REFS//,/ }; do
        name="${ref%%:*}"
        path="${ref#*:}"
        echo "### $name ($path), $t threads"
        "$JULIA_BIN" --project="$path" --threads=$t "$REPO_ROOT/benchmark/load_time.jl" "$@"
    done
done

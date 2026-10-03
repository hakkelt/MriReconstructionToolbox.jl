#!/bin/bash
#SBATCH --job-name=mrt-wavelet-threads
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#
# Threaded against serial wavelet transforms (benchmark/wavelet_threads.jl) at 2, 4, 8 and 16
# threads, each run pinned to the first cores of one NUMA domain.
#
#   benchmark/slurm/submit.sh wavelet_threads.sh [reps]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

for t in 2 4 8 16; do
    echo "### $t threads"
    taskset -c 0-$((t - 1)) "$JULIA_BIN" --project=test --threads=$t benchmark/wavelet_threads.jl "$@"
done

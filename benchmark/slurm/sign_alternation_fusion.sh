#!/bin/bash
#SBATCH --job-name=ristretto-sign-alternation-fusion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Sign alternation fused into a pointwise run against a pass of its own
# (benchmark/sign_alternation_fusion.jl) at 1, 4 and 8 threads.
#
#   benchmark/slurm/submit.sh sign_alternation_fusion.sh [reps]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

for t in 1 4 8; do
    echo "### $t threads"
    "$JULIA_BIN" --project=benchmark --threads=$t benchmark/sign_alternation_fusion.jl "$@"
done

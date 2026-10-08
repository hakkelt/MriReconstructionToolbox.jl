#!/bin/bash
#SBATCH --job-name=ristretto-gpu-batched-svt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#
# Batched singular value thresholding variants (benchmark/gpu_batched_svt.jl) on one GPU.
#
#   benchmark/slurm/submit.sh gpu_batched_svt.sh [reps]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$JULIA_BIN" --project=test --threads=8 benchmark/gpu_batched_svt.jl "$@"

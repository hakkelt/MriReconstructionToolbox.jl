#!/bin/bash
#SBATCH --job-name=ristretto-gpu-task-splitting
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Split against unsplit device reconstructions (benchmark/gpu_task_splitting.jl) on one GPU.
#
#   benchmark/slurm/submit.sh gpu_task_splitting.sh [reps]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$JULIA_BIN" --project=test --threads=8 benchmark/gpu_task_splitting.jl "$@"

#!/bin/bash
#SBATCH --job-name=ristretto-comparison-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# The cross-toolkit comparison on one GPU (benchmark/comparison/scripts/run_all.jl --device=cuda):
# Ristretto, BART, SigPy, MRIReco and MRpro, each on the device, from host data to a host image.
#
#   benchmark/slurm/submit.sh comparison_gpu.sh [run_all.jl args...]
#   benchmark/slurm/submit.sh comparison_gpu.sh --sections=cgsense,sparsity --cases=shepp_logan_2d
#   benchmark/slurm/submit.sh --production --time=08:00:00 comparison_gpu.sh --data=all
#
# One host thread and OpenBLAS as the host BLAS, always (see `DEVICE` in _setup.jl for why); every
# argument goes to run_all.jl. BART needs `RISTRETTO_BENCH_BART_CUDA`, and SigPy and MRpro an interpreter
# with CuPy and a CUDA build of PyTorch (`RISTRETTO_BENCH_GPU_PYTHON`), both in
# `benchmark/slurm/site.env`; a toolkit whose GPU build is missing is skipped with a warning.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$JULIA_BIN" --project=benchmark/comparison -t 1 benchmark/comparison/scripts/run_all.jl \
    --threads=1 --device=cuda "$@"

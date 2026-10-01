#!/bin/bash
#SBATCH --job-name=mrt-comparison-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# The cross-toolkit comparison on one GPU (benchmark/comparison/scripts/run_all.jl --device=cuda):
# MRT, BART, SigPy, MRIReco and MRpro, each on the device, from host data to a host image.
#
#   benchmark/slurm/submit.sh comparison_gpu.sh [--threads=N] [run_all.jl args...]
#   benchmark/slurm/submit.sh comparison_gpu.sh --sections=cgsense,sparsity --cases=shepp_logan_2d
#   benchmark/slurm/submit.sh --production --time=08:00:00 comparison_gpu.sh --data=all
#
# `--threads=N` (default 1) is the number of host threads, which serve whatever a toolkit keeps on
# the host. One, because the device does the work, and because MRIReco's GPU path races with more
# than one Julia thread (`MRIRECO_GPU_SAFE` in _toolkits.jl), so its rows are skipped above one.
# Every other argument goes to run_all.jl. BART needs `MRT_BENCH_BART_CUDA`, and SigPy
# and MRpro an interpreter with CuPy and a CUDA build of PyTorch (`MRT_BENCH_GPU_PYTHON`), both in
# `benchmark/slurm/site.env`; a toolkit whose GPU build is missing is skipped with a warning.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

NT=1
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --threads=*) NT="${a#*=}" ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$JULIA_BIN" --project=benchmark/comparison -t "$NT" benchmark/comparison/scripts/run_all.jl \
    --threads="$NT" --device=cuda ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

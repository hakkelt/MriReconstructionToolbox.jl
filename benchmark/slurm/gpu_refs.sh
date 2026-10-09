#!/bin/bash
#SBATCH --job-name=ristretto-gpu-refs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# The same GPU benchmark on several Ristretto checkouts in one job, on one device, alternating the
# checkouts round by round so drift on the node hits each of them alike.
#
#   benchmark/slurm/submit.sh gpu_refs.sh --refs=master:/wt/master,dev:/wt/dev [--rounds=2] <bench> [args...]
#
# <bench> is one of
#   comparison [run_all.jl args...]   Ristretto's rows of the comparison suite on CUDA (the suite runs
#                                     with --frameworks=Ristretto, one host thread, OpenBLAS)
#   task_splitting [reps]             benchmark/gpu_task_splitting.jl
#
# Round r runs the refs in the given order when r is odd and in reverse when it is even. Each
# checkout runs its own copy of the benchmark scripts. The comparison run files a ref writes are
# copied to `benchmark/results/slurm/gpu_refs_<job>/<ref>/runs/`, which `benchmark/compare_gpu.jl` reads;
# the other benchmarks print their tables to the job log, one block per ref and round.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

REFS=()
ROUNDS=2
while [ $# -gt 0 ]; do
    case "$1" in
        --refs=*) IFS=, read -ra REFS <<<"${1#*=}"; shift ;;
        --rounds=*) ROUNDS="${1#*=}"; shift ;;
        *) break ;;
    esac
done
[ ${#REFS[@]} -ge 1 ] && [ $# -ge 1 ] || {
    echo "usage: gpu_refs.sh --refs=name:path,... [--rounds=N] comparison|task_splitting [args...]" >&2
    exit 2
}
BENCH="$1"; shift
case "$BENCH" in
    comparison | task_splitting) ;;
    *) echo "### unknown benchmark $BENCH" >&2; exit 2 ;;
esac

OUT_DIR="$BENCH_RESULTS/slurm/gpu_refs_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

run_ref() {
    local name="$1" path="$2" marker
    shift 2
    marker="$(mktemp)"
    echo "### ref $name ($path), round $round: $BENCH $*"
    (
        cd "$path"
        case "$BENCH" in
            comparison)
                "$JULIA_BIN" --project=benchmark/comparison -t 1 benchmark/comparison/scripts/run_all.jl \
                    --threads=1 --device=cuda --frameworks=Ristretto "$@" ;;
            task_splitting)
                "$JULIA_BIN" --project=test --threads=8 benchmark/gpu_task_splitting.jl "$@" ;;
        esac
    ) || echo "### ref $name, round $round failed"
    if [ "$BENCH" = comparison ]; then
        mkdir -p "$OUT_DIR/$name/runs"
        find "$path/benchmark/comparison/results/runs" -name "*_job${SLURM_JOB_ID:-}_*.json" -newer "$marker" \
            -exec cp {} "$OUT_DIR/$name/runs/" \;
    fi
    rm -f "$marker"
}

for ((round = 1; round <= ROUNDS; round++)); do
    order=("${REFS[@]}")
    if ((round % 2 == 0)); then
        order=()
        for ((i = ${#REFS[@]} - 1; i >= 0; i--)); do order+=("${REFS[$i]}"); done
    fi
    for spec in "${order[@]}"; do
        run_ref "${spec%%:*}" "${spec#*:}" "$@"
    done
done
echo "### results in $OUT_DIR"

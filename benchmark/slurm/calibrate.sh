#!/bin/bash
# Per-case λ calibration (benchmark/comparison/scripts/calibrate_lambda.jl) as a SLURM array job:
# one array task per case, each writing its own results/lambda/<case id>.json, so the tasks never
# share a file and a failed case can be resubmitted alone.
#
#   benchmark/slurm/submit.sh --array=0-6 calibrate.sh
#   benchmark/slurm/submit.sh calibrate.sh --cases=shepp_logan_2d_8ch_cartesian
#
# `--cases=a,b,...` lists the cases (default: every synthetic catalog case, in catalog order); array
# task i calibrates the i-th of them, and without an array the job calibrates all of them in turn.
# Every other argument (--frameworks=, --use-mkl, ...) is passed to calibrate_lambda.jl.
#
#SBATCH --job-name=calibrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/benchmark/slurm/common.sh"

CASES=(
    shepp_logan_2d_1ch_cartesian shepp_logan_2d_8ch_cartesian shepp_logan_2d_8ch_radial
    shepp_logan_multislice_8ch_cartesian shepp_logan_3d_8ch_cartesian
    torso_cine_8ch_cartesian torso_cine_8ch_radial
)
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --cases=*) IFS=, read -ra CASES <<<"${a#*=}" ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CASES[@]}" ]; then
        echo "### array task $SLURM_ARRAY_TASK_ID has no case (${#CASES[@]} listed)" >&2
        exit 2
    fi
    CASES=("${CASES[$SLURM_ARRAY_TASK_ID]}")
fi

NT="${SLURM_CPUS_PER_TASK:-16}"
for c in "${CASES[@]}"; do
    echo "### calibrating $c on $NT threads"
    "$JULIA_BIN" --project=benchmark/comparison -t "$NT" benchmark/comparison/scripts/calibrate_lambda.jl \
        --threads="$NT" --cases="$c" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
done

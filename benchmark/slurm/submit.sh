#!/bin/bash
# Submit one of the benchmark SLURM scripts from the repository root, on the test partition unless
# `--production` is given. The partition names come from `benchmark/slurm/site.env`.
#
#   benchmark/slurm/submit.sh [--production] [--time=HH:MM:SS] [--array=SPEC] <script> [script args...]
#
#   benchmark/slurm/submit.sh matrix.sh --suite=harness --matrix-threads=1,8 --cases=shepp_logan_2d
#   benchmark/slurm/submit.sh --production matrix.sh --suite=comparison
#   benchmark/slurm/submit.sh calibrate.sh --cases=shepp_logan_2d_8ch_cartesian
#   benchmark/slurm/submit.sh --array=0-6 calibrate.sh
#
# The test partition has a one-hour limit, so the script's own `--time` is capped at that there.
# Production runs occupy a whole node for hours: submit them only when that has been agreed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/benchmark/slurm/common.sh"

PARTITION="${SLURM_TEST_PARTITION:-test}"
TIME="01:00:00"
ARRAY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --production) PARTITION="${SLURM_PRODUCTION_PARTITION:-cpu}"; TIME=""; shift ;;
        --time=*) TIME="${1#*=}"; shift ;;
        --array=*) ARRAY="${1#*=}"; shift ;;
        *) break ;;
    esac
done
[ $# -ge 1 ] || { echo "usage: $0 [--production] [--time=HH:MM:SS] <script> [args...]" >&2; exit 2; }
SCRIPT="$1"; shift
[ -f "$SCRIPT" ] || SCRIPT="benchmark/slurm/$SCRIPT"
[ -f "$SCRIPT" ] || { echo "### no such script: $SCRIPT" >&2; exit 2; }

opts=(--partition="$PARTITION" --output="$BENCH_RESULTS/slurm/%x_%j.txt")
[ -n "$TIME" ] && opts+=(--time="$TIME")
[ -n "$ARRAY" ] && opts+=(--array="$ARRAY" --output="$BENCH_RESULTS/slurm/%x_%A_%a.txt")
[ -n "${SLURM_ACCOUNT:-}" ] && opts+=(--account="$SLURM_ACCOUNT")
echo "### sbatch ${opts[*]} $SCRIPT $*"
sbatch "${opts[@]}" "$SCRIPT" "$@"

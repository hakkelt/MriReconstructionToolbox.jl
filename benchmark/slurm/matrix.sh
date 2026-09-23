#!/bin/bash
# NUMA-isolated concurrent benchmark matrix. Books one node --exclusive, then runs several
# configurations of the matrix AT THE SAME TIME inside that one job, each pinned via `numactl` to
# its own disjoint NUMA domain (no two tasks ever share cores, L3, or memory channels). This is the
# fix for the NUMA-fragmentation anomaly: a task scattered across domains by a busy shared-node
# scheduler saw BART/OpenBLAS/FFTW timings inflate 10-700x.
#
# The node topology is discovered with ThreadPinning.jl, never hardcoded. Each domain must have at
# least 16 cores (the largest thread count in the default matrix).
#
# Suites:
#   --suite=comparison   benchmark/comparison/scripts/run_all.jl (MRT vs BART/SigPy/MRIReco/MIRT)
#   --suite=harness      benchmark/run.jl (MRT only, every catalog case and method)
#
# Matrix dimensions (each a comma-separated list):
#   --matrix-threads=1,2,4,8,16          thread counts (default: 1,2,4,8,16)
#   --matrix-backends=openblas,mkl       BLAS backends (default: both)
#   --matrix-refs=master:/path,perf:/path   harness only: MRT checkouts to measure, each passed to
#                                        run.jl as --mrt=<path> --ref-name=<name>. Default: this
#                                        checkout.
#   --matrix-env=KMP_BLOCKTIME=0,KMP_BLOCKTIME=200   environment variants; `+` joins several
#                                        assignments into one variant (A=1+B=2). Default: none.
# Everything else (--sections=, --cases=, --methods=, --frameworks=, --remeasure, ...) is passed
# verbatim to every launched Julia process.
#
# usage (through submit.sh, which picks the partition):
#   benchmark/slurm/submit.sh matrix.sh --suite=comparison --sections=base
#   benchmark/slurm/submit.sh matrix.sh --suite=harness --matrix-refs=master:/wt/master,perf:/wt/perf
#
#SBATCH --job-name=matrix
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/benchmark/slurm/common.sh"

### ---- 1. Parse flags: matrix-level vs pass-through --------------------------------------------

SUITE=comparison
MATRIX_THREADS=(1 2 4 8 16)
MATRIX_BACKENDS=(openblas mkl)
MATRIX_REFS=("")
MATRIX_ENV=("")
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --suite=*) SUITE="${a#*=}" ;;
        --matrix-threads=*) IFS=, read -ra MATRIX_THREADS <<<"${a#*=}" ;;
        --matrix-backends=*) IFS=, read -ra MATRIX_BACKENDS <<<"${a#*=}" ;;
        --matrix-refs=*) IFS=, read -ra MATRIX_REFS <<<"${a#*=}" ;;
        --matrix-env=*) IFS=, read -ra MATRIX_ENV <<<"${a#*=}" ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done

case "$SUITE" in
    comparison)
        PROJECT_DIR="benchmark/comparison"
        SCRIPT_PATH="benchmark/comparison/scripts/run_all.jl"
        RESULTS_DIR="benchmark/comparison/results/runs"
        if [ "${MATRIX_REFS[*]}" != "" ]; then
            echo "### --matrix-refs is only supported by --suite=harness" >&2
            exit 2
        fi
        ;;
    harness)
        PROJECT_DIR="benchmark"
        SCRIPT_PATH="benchmark/run.jl"
        RESULTS_DIR="benchmark/results/runs"
        ;;
    *) echo "### unknown --suite=$SUITE (comparison or harness)" >&2; exit 2 ;;
esac
LOG_DIR="$BENCH_RESULTS/slurm/matrix_${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
START_MARKER="$LOG_DIR/.start"
touch "$START_MARKER"

### ---- 2. NUMA topology discovery (ThreadPinning.jl, never hardcoded) ---------------------------

mapfile -t NUMA_CORES < <("$JULIA_BIN" --project="$PROJECT_DIR" -e '
    using ThreadPinning
    for i in 1:ThreadPinning.nnuma()
        println(join(ThreadPinning.numa(i), " "))
    end')
N_DOMAINS=${#NUMA_CORES[@]}
if [ "$N_DOMAINS" -lt 1 ]; then
    echo "### NUMA topology discovery failed (no domains reported)" >&2
    exit 1
fi
MAX_THREADS=$(printf '%s\n' "${MATRIX_THREADS[@]}" | sort -n | tail -1)
for d in "${!NUMA_CORES[@]}"; do
    n=$(wc -w <<<"${NUMA_CORES[$d]}")
    if [ "$n" -lt "$MAX_THREADS" ]; then
        echo "### domain $d only has $n cores (< $MAX_THREADS threads requested), aborting" >&2
        exit 1
    fi
done
echo "### discovered $N_DOMAINS NUMA domains, $(wc -w <<<"${NUMA_CORES[0]}") cores each"

QUEUE=()
for r in "${!MATRIX_REFS[@]}"; do
    for e in "${!MATRIX_ENV[@]}"; do
        for t in "${MATRIX_THREADS[@]}"; do
            for b in "${MATRIX_BACKENDS[@]}"; do
                QUEUE+=("$t:$b:$r:$e")
            done
        done
    done
done
echo "### suite $SUITE, queue: ${QUEUE[*]}"
echo "### refs: ${MATRIX_REFS[*]:-<this checkout>}  env variants: ${MATRIX_ENV[*]:-<none>}"
echo "### pass-through args: ${PASS_ARGS[*]:-<none>}"

### ---- 3. Scheduler: one task per NUMA domain, enforced via numactl ----------------------------

declare -a DOMAIN_BUSY
for ((d = 0; d < N_DOMAINS; d++)); do DOMAIN_BUSY[d]=0; done
declare -A PID_DOMAIN   # background pid -> domain id

launch_task() {  # threads backend ref_index env_index domain
    local threads=$1 backend=$2 r=$3 e=$4 domain=$5
    local pin
    pin="$(tr ' ' '\n' <<<"${NUMA_CORES[$domain]}" | head -n "$threads" | paste -sd,)"
    local args=(--threads="$threads")
    [ "$backend" = mkl ] && args+=(--use-mkl)
    local tag="${backend}_${threads}t"
    local ref="${MATRIX_REFS[$r]}"
    if [ -n "$ref" ]; then
        args+=(--ref-name="${ref%%:*}" --mrt="${ref#*:}")
        tag="${ref%%:*}_$tag"
    fi
    local envs=()
    if [ -n "${MATRIX_ENV[$e]}" ]; then
        IFS=+ read -ra envs <<<"${MATRIX_ENV[$e]}"
        tag="${tag}_env$e"
        args+=(--env-variant="${MATRIX_ENV[$e]}")
    fi
    local log="$LOG_DIR/${tag}_domain${domain}.log"

    # A plain background process, not an `srun` step: this SLURM build's step allocator picks the
    # step's cpu set itself (always starting from the lowest free id) before looking at --cpu-bind,
    # and rejects a mask outside what it picked, so concurrent --overlap steps can never receive
    # disjoint, caller-chosen cpu ranges. The job owns the whole node (--exclusive), so a background
    # process inherits the full cpuset and numactl enforces the placement. Without steps there is
    # no per-task `sacct` row; `/usr/bin/time -v` gives the per-task efficiency instead (see
    # report_efficiency.sh).
    numactl --physcpubind="$pin" --membind="$domain" \
        /usr/bin/time -v \
        env JULIA_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
            MKL_NUM_THREADS="$threads" "${envs[@]}" \
        "$JULIA_BIN" --project="$PROJECT_DIR" -t "$threads" "$SCRIPT_PATH" \
            "${args[@]}" "${PASS_ARGS[@]}" \
        >"$log" 2>&1 &
    PID_DOMAIN[$!]=$domain
    DOMAIN_BUSY[$domain]=1
    echo "### launched $tag on domain $domain (cores $pin), pid $! -> $log"
}

next_free_domain() {
    for ((d = 0; d < N_DOMAINS; d++)); do
        [ "${DOMAIN_BUSY[$d]}" = 0 ] && { echo "$d"; return; }
    done
    echo -1
}

for cfg in "${QUEUE[@]}"; do
    IFS=: read -r threads backend r e <<<"$cfg"
    while [ "$(next_free_domain)" = -1 ]; do
        wait -n
        for pid in "${!PID_DOMAIN[@]}"; do
            kill -0 "$pid" 2>/dev/null || { DOMAIN_BUSY[${PID_DOMAIN[$pid]}]=0; unset "PID_DOMAIN[$pid]"; }
        done
    done
    launch_task "$threads" "$backend" "$r" "$e" "$(next_free_domain)"
done
wait

### ---- 4. Post-run sanity check: no run this job wrote spans more than one domain --------------

echo "### post-run NUMA sanity check"
find "$RESULTS_DIR" -name '*.json' -newer "$START_MARKER" | "$JULIA_BIN" --project="$PROJECT_DIR" -e '
    using JSON
    for f in eachline(stdin)
        d = try JSON.parsefile(f) catch; continue end
        pinned = get(d, "pinned_cpus", "")
        isempty(pinned) && continue
        ids = sort(parse.(Int, split(pinned, ",")))
        span = ids[end] - ids[1]
        println(basename(f), ": ", span < 16 ? "PASS" : "FAIL (span=$span, cpus=$pinned)")
    end
'

echo "### matrix.sh done $(date '+%H:%M:%S'); per-task logs in $LOG_DIR"

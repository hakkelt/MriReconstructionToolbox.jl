# Shared environment for every benchmark SLURM script. Sourced, not executed:
#
#   source "$REPO_ROOT/benchmark/slurm/common.sh"
#
# Sets REPO_ROOT, sources the untracked site file `benchmark/slurm/site.env` (copy
# `site.env.example`), resolves JULIA_BIN, and exports the process-wide settings every benchmark
# process must start with. Fails loudly instead of benchmarking under a half-configured site.

# sbatch runs a copy of the script from the spool directory, so the repository is found through the
# submit directory (`submit.sh` always submits from the repository root) rather than through
# BASH_SOURCE.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [ ! -f "$REPO_ROOT/benchmark/slurm/common.sh" ]; then
    echo "### $REPO_ROOT is not the MriReconstructionToolbox repository root" >&2
    exit 1
fi
cd "$REPO_ROOT" || exit 1

SITE_ENV="$REPO_ROOT/benchmark/slurm/site.env"
if [ ! -f "$SITE_ENV" ]; then
    echo "### $SITE_ENV is missing: copy benchmark/slurm/site.env.example and fill it in" >&2
    exit 1
fi
set -a
# shellcheck source=/dev/null
source "$SITE_ENV"
set +a

# The login-node `julia` wrapper applies its own `taskset` over a random core window, which fails
# inside a SLURM cgroup, so the depot binary is called directly.
if [ -z "${JULIA_BIN:-}" ] && [ -n "${JULIAUP_DEPOT_PATH:-}" ]; then
    JULIA_BIN="$(ls -d "${JULIAUP_DEPOT_PATH}"/juliaup/julia-*/bin/julia 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "${JULIA_BIN:-}" ] || [ ! -x "$JULIA_BIN" ]; then
    echo "### no Julia binary: set JULIA_BIN or JULIAUP_DEPOT_PATH in $SITE_ENV" >&2
    exit 1
fi
export JULIA_BIN

# libiomp5 (MKL) reads this once at load: without it MKL's idle workers spin and crowd out the
# threads being measured. Kept until the BLAS-policy work shows it is no longer needed.
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
# BartIO writes every BART input/output .cfl through `tempdir()`. Keep that on tmpfs whatever the
# node's /tmp happens to be: on Lustre an 8 MiB `bart copy` measured 2230 ms against 130 ms on
# tmpfs (test node, 2026-09-23). Pages land on the writer's NUMA domain under --membind.
export TMPDIR=/dev/shm

BENCH_RESULTS="$REPO_ROOT/benchmark/results"
mkdir -p "$BENCH_RESULTS/slurm"

#!/bin/bash
# One-shot precompile job on a compute node, ahead of the per-notebook export array
# (run_export_slurm.sh). Instantiates the root, docs/notebooks and test environments under the
# depot's current default Julia (juliaup default, picked up via sort -V so a juliaup upgrade
# needs no edit here) and (re)builds the IJulia kernelspec, so no export-array task pays first-use
# precompilation or a missing-kernel failure.
#
#   sbatch docs/notebooks/run_precompile_slurm.sh
#
#SBATCH --job-name=mrt-precompile
#SBATCH --partition=test
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=docs/notebooks/build/slurm_precompile_%j.txt
set -eu

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p docs/notebooks/build

# Resolve the depot binary directly: both the shell `julia` function and ~/.local/bin/julia wrap
# it in a `taskset` over a random 16-87 core window, which fails inside a SLURM cgroup.
JULIAUP_DEPOT_PATH="${JULIAUP_DEPOT_PATH:-/scratch/c_mrrecon/juliaup_depot}"
JULIA_BIN="${JULIA_BIN:-$(ls -d "${JULIAUP_DEPOT_PATH}"/juliaup/julia-*/bin/julia 2>/dev/null | sort -V | tail -1)}"
echo "### julia binary: $JULIA_BIN"
"$JULIA_BIN" --version

# Julia 1.13.0's stdlib JLLs (LibGit2_jll, LibCURL_jll, PCRE2_jll, Zstd_jll) fail their very first
# pkgimage precompile on this cluster's shared (Lustre) depot with "Precompiled image ... not
# available with flags", reproducibly, when the precompiling process itself runs multi-threaded
# (-t > 1) -- consistent with concurrent native-codegen writes from Julia's own worker threads
# racing each other on the network filesystem, not a real flag mismatch (neither pkgimages=no nor
# capping Pkg's precompile task pool alone fixed it). Do this one-time instantiate/build step
# single-threaded so the depot's compiled/vX.Y cache is built cleanly once; the actual
# notebook/test runs afterward reuse that cache and can run with the full thread count.
echo "### instantiating root project"
JULIA_NUM_THREADS=1 "$JULIA_BIN" --project=. -e \
    'using Pkg; Pkg.resolve(); Pkg.instantiate(); using MriReconstructionToolbox'

echo "### instantiating test environment"
JULIA_NUM_THREADS=1 "$JULIA_BIN" --project=test -e \
    'using Pkg; Pkg.resolve(); Pkg.instantiate(); using MriReconstructionToolbox'

echo "### instantiating docs/notebooks environment and building IJulia kernel"
JULIA_NUM_THREADS=1 "$JULIA_BIN" --project=docs/notebooks -e '
    using Pkg
    Pkg.resolve()
    Pkg.instantiate()
    using MriReconstructionToolbox
    Pkg.build("IJulia")
'

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "### registered Jupyter kernels"
python3 -m jupyter kernelspec list

echo "### done rc=$?"

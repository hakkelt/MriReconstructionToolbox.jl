#!/bin/bash
# Export all twelve notebooks on a compute node, one SLURM array task per notebook, on the `cpu`
# partition (7-day limit; `test` caps at 1h and notebook 10 alone needs ~39 min). Run
# run_precompile_slurm.sh first so no array task pays first-use precompilation, and regenerate the
# .ipynb files once before submitting (all 12 array tasks share them; regenerating per-task would
# race):
#
#   python3 -m jupytext --to ipynb docs/notebooks/src/*.jl --output-dir docs/notebooks
#   sbatch docs/notebooks/run_export_slurm.sh
#
#SBATCH --job-name=mrt-export
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --output=docs/notebooks/build/slurm_export_%a.txt
set -eu

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p docs/notebooks/build

# Resolve the depot binary directly: both the shell `julia` function and ~/.local/bin/julia wrap
# it in a `taskset` over a random 16-87 core window, which fails inside a SLURM cgroup.
JULIAUP_DEPOT_PATH="${JULIAUP_DEPOT_PATH:-/scratch/c_mrrecon/juliaup_depot}"
JULIA_BIN="${JULIA_BIN:-$(ls -d "${JULIAUP_DEPOT_PATH}"/juliaup/julia-*/bin/julia 2>/dev/null | sort -V | tail -1)}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# All 12 array tasks share the same .ipynb files; export.jl's own regenerate_ipynb! would have
# every task regenerate every notebook concurrently, racing on the same output files. Skip that
# here -- the .ipynb files must already be regenerated (once, sequentially) before this job is
# submitted; see the submission instructions above.
export MRT_SKIP_REGEN=1

NB=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-1}")
echo "### julia binary: $JULIA_BIN"
echo "### exporting notebook $NB"

# The registered IJulia kernel's argv uses --project=@. (cwd-relative, so it works whether a
# person launches Jupyter from docs/notebooks or from the repo root with -dir). nbconvert spawns
# the kernel with whatever cwd this script has when it runs export.jl, so cd into docs/notebooks
# first -- from the repo root the kernel would activate the wrong project (root Project.toml, no
# IJulia) and fail to start.
cd docs/notebooks
"$JULIA_BIN" --project=. export.jl "$NB" --timeout=3000

echo "### done rc=$?"

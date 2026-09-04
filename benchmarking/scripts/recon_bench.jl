# MRT-only reconstruction benchmark, run per group so pieces can go in parallel.
#
#   julia --project=benchmarking -t 8 benchmarking/scripts/recon_bench.jl \
#         --threads=8 [--use-mkl] [--groups=base,sparsity,...] [--num-runs=3]
#
# --groups: comma list of ReconBench.GROUPS, or "synthetic" / "real" / "all"
#           (default: "synthetic"; "real"/"real3d"/"realdyn" download data on first use).
#           Back-compat: MRT_BENCH_REAL_DATA=1 with no --groups → "synthetic,real".
#
# Each invocation writes benchmarking/results/mrt_<backend>_<n>threads__<groupsig>.json.
# `merge_results.jl` stitches them into mrt_<backend>_<n>threads.json for the comparison suite.

using Printf, JSON

use_mkl = "--use-mkl" in ARGS
_arg(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=")[2])
num_threads = parse(Int, _arg("threads", string(Threads.nthreads())))
num_runs = parse(Int, _arg("num-runs", "3"))

default_groups = get(ENV, "MRT_BENCH_REAL_DATA", "0") == "1" ? "synthetic,real" : "synthetic"
groups_arg = _arg("groups", default_groups)

use_mkl && @eval using MKL

using ThreadPinning
_aff = getaffinity()
_allowed = findall(==(1), _aff) .- 1
isempty(_allowed) && (_allowed = collect(0:(Threads.nthreads() - 1)))
pinthreads(_allowed[1:min(length(_allowed), Threads.nthreads())])
# MKL_DYNAMIC defaults on and lets MKL resize/repin its pool mid-run, undoing pinthreads.
use_mkl && try
    ThreadPinning.MKL.mkl_set_dynamic(0)
catch e
    @warn "mkl_set_dynamic failed" e
end

using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "ReconBench.jl"))
using .ReconBench

backend = use_mkl ? "mkl" : "openblas"
groups = ReconBench._expand_groups(groups_arg)
@info "recon_bench" backend num_threads julia_threads = Threads.nthreads() blas = BLAS.get_num_threads() groups

results = run_cases(build_cases(groups); num_runs)

groupsig = join(groups, "-")
outdir = joinpath(@__DIR__, "..", "results")
mkpath(outdir)
outfile = joinpath(outdir, "mrt_$(backend)_$(num_threads)threads__$(groupsig).json")
open(outfile, "w") do io
    JSON.print(
        io,
        Dict(
            "backend" => backend,
            "num_threads" => num_threads,
            "julia_version" => string(VERSION),
            "host" => gethostname(),
            "groups" => groups,
            "benchmarks" => [
                Dict(
                        "category" => r.category, "method" => r.method,
                        "framework" => "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))",
                        "threads" => num_threads, "time_ms" => r.time_ms,
                        "nrmse_gt" => r.nrmse_gt, "nrmse_mrt" => 0.0,
                    ) for r in results
            ],
        ),
        2,
    )
end
@info "wrote" outfile

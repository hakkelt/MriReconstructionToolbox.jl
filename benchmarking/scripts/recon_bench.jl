# MRT-only reconstruction benchmark. Writes benchmarking/results/mrt_<backend>_<n>threads.json,
# which comparison/scripts/run_benchmarks.jl picks up as the MRT baseline column.
#
# usage: julia --project=benchmarking -t 8 benchmarking/scripts/recon_bench.jl --threads=8 [--use-mkl]

using Printf, JSON

use_mkl = "--use-mkl" in ARGS
ti = findfirst(a -> startswith(a, "--threads="), ARGS)
num_threads = ti === nothing ? Threads.nthreads() : parse(Int, split(ARGS[ti], "=")[2])

use_mkl && @eval using MKL

using ThreadPinning
mask = getaffinity()
allowed = findall(==(1), mask) .- 1
isempty(allowed) && (allowed = collect(0:(Threads.nthreads() - 1)))
pinthreads(allowed[1:min(length(allowed), Threads.nthreads())])
# MKL_DYNAMIC defaults on and lets MKL resize/repin its pool mid-run, undoing pinthreads.
use_mkl && try; ThreadPinning.MKL.mkl_set_dynamic(0); catch e; @warn "mkl_set_dynamic failed" e; end

using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "ReconBench.jl"))
using .ReconBench

backend = use_mkl ? "mkl" : "openblas"
@info "recon_bench" backend num_threads julia_threads = Threads.nthreads() blas = BLAS.get_num_threads()

results = run_cases(build_cases())

outdir = joinpath(@__DIR__, "..", "results")
mkpath(outdir)
outfile = joinpath(outdir, "mrt_$(backend)_$(num_threads)threads.json")
open(outfile, "w") do io
    JSON.print(
        io,
        Dict(
            "backend" => backend,
            "num_threads" => num_threads,
            "julia_version" => string(VERSION),
            "host" => gethostname(),
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

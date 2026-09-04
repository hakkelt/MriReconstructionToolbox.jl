# Shared prelude for the decomposed comparison suite. Every `run_<section>.jl` does
# `include(joinpath(@__DIR__, "_setup.jl"))` first: it parses the CLI, pins threads, configures
# the BART / OpenMP / MKL environment, loads MRT + BART + SigPy + MRIReco, defines the timing
# helpers and `BenchResult`, and builds the shared multi-coil brain phantom.
#
# Each section script then appends to `results::Vector{BenchResult}` and calls
# `write_section("<name>")`, which writes
#   comparison/results/benchmark_<backend>_<n>threads__<name>.json
# `merge_benchmarks.jl` stitches those into benchmark_<backend>_<n>threads.json.

using Printf
using JSON

const USE_MKL = "--use-mkl" in ARGS
let i = findfirst(a -> startswith(a, "--threads="), ARGS)
    global const NUM_THREADS = i === nothing ? Threads.nthreads() : parse(Int, split(ARGS[i], "=")[2])
end

if USE_MKL
    @info "Enabling Intel MKL backend via MKL.jl"
    using MKL
    const BART_BINARY = "/project/c_mrrecon/bart_mkl"
else
    const BART_BINARY = "/project/c_mrrecon/bart_openblas"
end
const FW = "MRT ($(USE_MKL ? "MKL" : "OpenBLAS"))"
const BART_FW = "BART ($(USE_MKL ? "MKL" : "OpenBLAS"))"

using ThreadPinning
let mask = getaffinity()
    allowed = findall(==(1), mask) .- 1
    isempty(allowed) && (allowed = collect(0:(Threads.nthreads() - 1)))
    global const PINNED_CPUS = allowed[1:min(length(allowed), Threads.nthreads())]
end
pinthreads(PINNED_CPUS)
USE_MKL && try
    ThreadPinning.MKL.mkl_set_dynamic(0)
catch e
    @warn "mkl_set_dynamic failed" e
end
const CPU_STR = join(PINNED_CPUS, ",")
@info "Julia threads pinned" CPU_STR

ENV["TOOLBOX_PATH"] = BART_BINARY
# BART_USE_FFTW_WISDOM=1 was measured to *hurt* (6x): it forces FFTW_MEASURE on every fresh
# `bart` process and the wisdom file is never persisted, so MEASURE planning is never amortised.
ENV["BART_USE_FFTW_WISDOM"] = "0"
ENV["OMP_NUM_THREADS"] = string(NUM_THREADS)
ENV["OPENBLAS_NUM_THREADS"] = string(NUM_THREADS)
ENV["MKL_NUM_THREADS"] = string(NUM_THREADS)
ENV["GOMP_CPU_AFFINITY"] = CPU_STR
ENV["KMP_AFFINITY"] = "granularity=fine,proclist=[$CPU_STR],explicit"
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "{$CPU_STR}"

using MriReconstructionToolbox
using GeometricMedicalPhantoms
using LinearAlgebra
using Statistics
using Random
using FFTW
using BartIO
using PyCall
using MRIReco

# MRIReco's own `__init__` pins BLAS only when `Threads.nthreads() > 1`
# (MRIReco.jl:20-26 — the other branch is Windows-only), so a `-t 1` run on Linux leaves OpenBLAS
# at its `jl_effective_threads`-derived default, i.e. most of the node. RegularizedLeastSquares
# then makes ~30 BLAS-1 calls per ADMM outer iteration (`norm`/`dot`/`rmul!` in `cg.jl` and the
# residual block), each spawning and joining a full thread team over a ~9k-element vector. That
# alone measured 346 s for a 20-iteration TV solve that takes 1.08 s with BLAS pinned — a 320x
# artifact that has nothing to do with MRIReco's reconstruction math. Pin it explicitly, for every
# toolkit, so the thread count under test is the one we asked for.
BLAS.set_num_threads(NUM_THREADS)
FFTW.set_num_threads(NUM_THREADS)
@info "BLAS/FFTW pinned" blas_threads = BLAS.get_num_threads() fftw_threads = FFTW.get_num_threads()

include(joinpath(@__DIR__, "..", "src", "ComparisonHarness.jl"))
using .ComparisonHarness: check_nrmse, nrmse, run_bart, generate_multicoil_brain,
    generate_dynamic_multicoil_brain, load_real_case, load_real_case_3d, load_real_dynamic

const sigpy = pyimport("sigpy")
const sp_mri = pyimport("sigpy.mri")
const sp_app = pyimport("sigpy.mri.app")

@info "comparison setup" host = gethostname() julia = VERSION threads = Threads.nthreads() blas = BLAS.get_config().loaded_libs[1].libname mkl = USE_MKL bart = BART_BINARY

# Bare process spawn cost (`bart version` does no file I/O) — the floor for an input-less call.
const BART_SPAWN = let times = Float64[]
    for _ in 1:10
        t0 = time_ns()
        read(pipeline(ignorestatus(`$BART_BINARY version`)), String)
        push!(times, (time_ns() - t0) / 1e9)
    end
    minimum(times)
end
@info @sprintf("BART spawn cost: %.1f ms", BART_SPAWN * 1000)

"""
    bart_overhead(inputs...; reps = 5) -> seconds

What a real `pics` call pays outside its solver: one process spawn, reading each input
`.cfl/.hdr` from disk, and writing one output. `bart copy in out` spawns once and does one read
**and** one write of the array, so `(copy_time − spawn)` ≈ read + write ≈ 2·read for that array.
A `pics` call only *reads* each input (no write-back) and writes a single (image-sized) output,
so the estimate is `spawn + Σᵢ (copyᵢ − spawn)/2 + (mean inputs)/2` — half the measured per-array
transfer per input, plus one more half for the output. Subtracted from BART recon timings so the
reported figure is solver time, comparable to the in-process toolkits.
"""
function bart_overhead(inputs...; reps = 5)
    isempty(inputs) && return BART_SPAWN
    io = Float64[]
    for inp in inputs
        run_bart(1, "copy", inp)                       # warm the path
        ts = Float64[]
        for _ in 1:reps
            t0 = time_ns()
            run_bart(1, "copy", inp)
            push!(ts, (time_ns() - t0) / 1e9)
        end
        push!(io, max(0.0, minimum(ts) - BART_SPAWN) / 2)   # one-way transfer for this array
    end
    return BART_SPAWN + sum(io) + sum(io) / length(io)      # + one output write (≈ mean input)
end

"""
    time_bart(cmd, inputs...; nout = 1, num_runs = 3, heavy_threshold = 5.0)
        -> (t_min_s, t_med_s, result)

Run BART `cmd` on `inputs`, timed `num_runs` times after a warm-up, with `bart_overhead(inputs...)`
subtracted. `BART_USE_FFTW_WISDOM` is enabled for this recon only if the warm-up solver time
exceeds `heavy_threshold` seconds — i.e. only when the one-off `FFTW_MEASURE` planning is small
against the recon's own compute (per the user's ">5 s" rule).
"""
function time_bart(cmd::AbstractString, inputs...; nout::Int = 1, num_runs::Int = 3, heavy_threshold::Real = 5.0)
    ovh = bart_overhead(inputs...)
    t0 = time_ns()
    res = run_bart(nout, cmd, inputs...)
    warm = (time_ns() - t0) / 1e9 - ovh
    wis = warm > heavy_threshold
    wis && run_bart(nout, cmd, inputs...; wisdom = true)   # build the measured plan once
    times = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        res = run_bart(nout, cmd, inputs...; wisdom = wis)
        push!(times, (time_ns() - t0) / 1e9)
    end
    @info @sprintf("BART '%s': overhead %.1f ms, wisdom %s", first(split(cmd)), ovh * 1000, wis)
    return max(1.0e-5, minimum(times) - ovh), max(1.0e-5, median(times) - ovh), res
end

"""
    time_reconstruction(f; num_runs = 3) -> (t_min_s, t_med_s, result)

Warm up `f` once, then time `num_runs` more (in-process toolkits: MRT, SigPy, MRIReco).
BART goes through [`time_bart`](@ref) instead.
"""
function time_reconstruction(f; num_runs = 3, is_bart = false)
    is_bart && error("route BART through time_bart(cmd, inputs...) so I/O overhead and the FFTW-wisdom rule are handled")
    res = f()
    times = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        res = f()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return minimum(times), median(times), res
end

# MRT baseline times owned by benchmarking/recon_bench.jl. If the merged JSON is present we take
# its `time_ms` for the MRT rows instead of re-timing (the recon still runs once for NRMSE).
const _MRT_BASELINE = let
    f = normpath(joinpath(
        @__DIR__, "..", "..", "benchmarking", "results",
        "mrt_$(USE_MKL ? "mkl" : "openblas")_$(NUM_THREADS)threads.json",
    ))
    d = Dict{Tuple{String, String}, Float64}()
    if isfile(f)
        for b in JSON.parsefile(f)["benchmarks"]
            d[(b["category"], b["method"])] = b["time_ms"]
        end
        @info "using MRT baseline from benchmarking/" file = f n = length(d)
    else
        @info "no MRT baseline; timing MRT rows inline" expected = f
    end
    d
end

function time_mrt(category, method, f)
    k = (category, method)
    haskey(_MRT_BASELINE, k) && return _MRT_BASELINE[k] / 1000, _MRT_BASELINE[k] / 1000, f()
    return time_reconstruction(f)
end

# magnitude, scale-aligned — real recons carry a receive phase the magnitude reference lacks.
mag_nrmse(est, ref) = (a = abs.(est); r = abs.(ref); nrmse(a .* (norm(r) / norm(a)), r))

struct BenchResult
    category::String
    method::String
    framework::String
    threads::Int
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_mrt::Float64
end

results = BenchResult[]

"""Replace NaN / Inf with -1.0 so a single bad toolkit row does not sink the section's JSON."""
_json_num(x::Real) = isfinite(x) ? Float64(x) : -1.0

function write_section(name::AbstractString)
    dir = normpath(joinpath(@__DIR__, "..", "results"))
    mkpath(dir)
    path = joinpath(dir, "benchmark_$(USE_MKL ? "mkl" : "openblas")_$(NUM_THREADS)threads__$(name).json")
    open(path, "w") do io
        JSON.print(
            io,
            Dict(
                "hostname" => gethostname(), "julia_version" => string(VERSION),
                "julia_threads" => Threads.nthreads(),
                "blas_vendor" => BLAS.get_config().loaded_libs[1].libname,
                "use_mkl" => USE_MKL, "bart_binary" => BART_BINARY,
                "pinned_cpus" => CPU_STR, "section" => name,
                "bart_spawn_ms" => BART_SPAWN * 1000,
                "benchmarks" => [
                    Dict(
                        "category" => r.category, "method" => r.method, "framework" => r.framework,
                        "threads" => r.threads, "time_ms" => _json_num(r.time_ms),
                        "nrmse_gt" => _json_num(r.nrmse_gt), "nrmse_mrt" => _json_num(r.nrmse_mrt),
                    ) for r in results
                ],
            ),
            4,
        )
    end
    for r in results
        @printf("%-14s | %-26s | %-22s | %7d | %10.2f ms | %10.2e | %10.2e\n",
            r.category, r.method, r.framework, r.threads, r.time_ms, r.nrmse_gt, r.nrmse_mrt)
    end
    @info "wrote section" path n = length(results)
    return path
end

# Shared 2D multi-coil brain phantom (most sections).
const N = 128
const Nc = 8
const IMG_MC, KSPACE_MC, CMAP = generate_multicoil_brain(N = N, num_coils = Nc)

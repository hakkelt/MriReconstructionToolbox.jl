# Re-fit SERIAL_BLAS_THRESHOLD_BYTES on *real* solves (IMPLEMENTATION_PLAN.md C6.3).
#
# The constant is one number fitted on one node against the synthetic CG-shaped reproducer in
# threading_sweep.jl. This script asks the same question of the reconstruction path itself:
# for a solve whose image variable is B bytes, is the whole solve faster threaded or serial?
#
# No constant has to be redefined to ask that. The threshold has exactly two effects, and both
# are all-or-nothing per solve:
#
#   * below it, `maybe_disable_undecomposed_threading` (and, since C6.1, `slice_threading`)
#     forces `threaded = false` for the whole solve;
#   * above it, `solve_core.jl` takes the `with_serial_blas` branch, which then also declines to
#     narrow anything because the item is over the same threshold — i.e. fully threaded.
#
# So `threaded = true` vs `threaded = false` on the same problem *is* the A/B for the gate, and
# the crossover between them is the number being fitted. Runs are interleaved (A/B/A/B/...)
# because this cluster's run-to-run spread on a whole solve is ±30-60%, which a naive
# before/after loop happily mistakes for signal.
#
# usage: julia --project=benchmarking -t 8 benchmarking/scripts/serial_blas_threshold_sweep.jl \
#            <openblas|mkl> [--rounds=3] [--reps=3] [--maxit=10]

using Printf, JSON

const BACKEND = length(ARGS) >= 1 ? ARGS[1] : "openblas"
BACKEND == "mkl" && (@eval using MKL)

argi(name, default) = begin
    i = findfirst(a -> startswith(a, "--$name="), ARGS)
    i === nothing ? default : parse(Int, split(ARGS[i], "=")[2])
end

const ROUNDS = argi("rounds", 3)
const REPS = argi("reps", 3)
const MAXIT = argi("maxit", 10)

using ThreadPinning
let mask = getaffinity()
    allowed = findall(==(1), mask) .- 1
    isempty(allowed) && (allowed = collect(0:(Threads.nthreads() - 1)))
    pinthreads(allowed[1:min(length(allowed), Threads.nthreads())])
end
BACKEND == "mkl" && try
    ThreadPinning.MKL.mkl_set_dynamic(0)
catch e
    @warn "mkl_set_dynamic failed" e
end

using LinearAlgebra
using NamedDims
using Random: MersenneTwister
using MriReconstructionToolbox
const MRT = MriReconstructionToolbox
include(joinpath(@__DIR__, "..", "src", "Phantoms.jl"))
using .Phantoms

@info "serial_blas_threshold_sweep" BACKEND julia_threads = Threads.nthreads() blas = BLAS.get_num_threads() threshold = MRT.SERIAL_BLAS_THRESHOLD_BYTES

# One case = one undersampled TV solve at a given image size, so the work item the gate sees is
# `N^2 * Nt * sizeof(ComplexF32)`. Static cases sweep N; dynamic cases sweep the time extent
# with a regularizer that couples time, so the whole volume stays a single work item (a
# separable regularizer would decompose it into per-frame slices, which is a different question).
struct Case
    label::String
    bytes::Int
    run::Function
end

function static_case(N; Nc = 8)
    _, kspace, cmap = generate_multicoil_brain(N = N, num_coils = Nc)
    mask = rand(MersenneTwister(42), Bool, N, N)
    mask[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true
    kdata = NamedDimsArray(kspace[mask, :], (:kxy, :coil))
    acq = CartesianAcquisitionInfo(
        kdata; is3D = false, image_size = (N, N),
        sensitivity_maps = NamedDimsArray(cmap, (:x, :y, :coil)),
        shifted_image_dims = (:x, :y), subsampling = mask,
    )
    bytes = N * N * sizeof(ComplexF32)
    run = (threaded) -> reconstruct(
        acq, IterativeReconstruction(regularization = TotalVariation2D(0.01));
        maxit = MAXIT, tol = 1.0e-5, verbose = false, threaded = threaded,
    )
    return Case(@sprintf("TV %d²", N), bytes, run)
end

function dynamic_case(N, Nt; Nc = 4)
    _, kspace, cmap = generate_dynamic_multicoil_brain(N = N, num_coils = Nc, num_frames = Nt)
    mask_pe = rand(MersenneTwister(42), Bool, N)
    mask_pe[(N ÷ 2 - 4):(N ÷ 2 + 4)] .= true
    kdata = NamedDimsArray(
        permutedims(kspace[:, mask_pe, :, :], (1, 2, 4, 3)), (:kx, :ky, :coil, :time)
    )
    acq = CartesianAcquisitionInfo(
        kdata; is3D = false, image_size = (N, N),
        sensitivity_maps = NamedDimsArray(cmap, (:x, :y, :coil)),
        subsampling = (:, mask_pe), shifted_image_dims = (:x, :y),
    )
    bytes = N * N * Nt * sizeof(ComplexF32)
    run = (threaded) -> reconstruct(
        acq, IterativeReconstruction(regularization = TemporalTotalVariation(0.01; time_dim = :time));
        maxit = MAXIT, tol = 1.0e-4, verbose = false, threaded = threaded,
    )
    return Case(@sprintf("tTV %d²×%d", N, Nt), bytes, run)
end

mib(b) = b / 2^20

function measure(case)
    case.run(true)
    case.run(false)   # warm up both paths before timing either
    serial = Float64[]
    threaded = Float64[]
    for _ in 1:ROUNDS
        push!(serial, minimum(@elapsed(case.run(false)) for _ in 1:REPS))
        push!(threaded, minimum(@elapsed(case.run(true)) for _ in 1:REPS))
    end
    return (serial = minimum(serial) * 1000, threaded = minimum(threaded) * 1000)
end

# `--smoke` builds only the two cheapest cases, for checking the script runs at all before
# spending a SLURM allocation on it.
const SMOKE = "--smoke" in ARGS

cases = SMOKE ? Case[static_case(128), dynamic_case(128, 8)] : Case[
        static_case(128),
        static_case(256),
        static_case(512),
        dynamic_case(128, 16),
        dynamic_case(128, 32),
        dynamic_case(256, 16),
        dynamic_case(256, 32),
        static_case(1024),
        dynamic_case(512, 32),
    ]

results = NamedTuple[]
@printf("%-14s %10s %12s %12s %10s\n", "case", "item MiB", "serial ms", "threaded ms", "winner")
for c in cases
    t = measure(c)
    winner = t.serial <= t.threaded ? "serial" : "threaded"
    ratio = t.serial / t.threaded
    @printf(
        "%-14s %10.2f %12.1f %12.1f %10s (%.2fx)\n",
        c.label, mib(c.bytes), t.serial, t.threaded, winner, ratio
    )
    flush(stdout)
    push!(
        results,
        (
            label = c.label, bytes = c.bytes, serial_ms = t.serial,
            threaded_ms = t.threaded, ratio = ratio,
        ),
    )
end

# The crossover bracket. `ratio = serial / threaded`, so > 1 means threading won. Differences
# inside ±3% are called a tie rather than a winner: on this node a whole-solve A/B at that
# margin is not separable from run-to-run noise, and treating one as a data point is how a
# threshold gets fitted to nothing. The bracket is therefore the largest item where serial
# *clearly* won and the smallest larger one where threading clearly won.
const TIE = 0.03

sorted = sort(results; by = r -> r.bytes)
serial_wins = [r for r in sorted if r.ratio < 1 - TIE]
threaded_wins = [r for r in sorted if r.ratio > 1 + TIE]
println()
for r in sorted
    verdict = r.ratio < 1 - TIE ? "serial" : r.ratio > 1 + TIE ? "threaded" : "tie"
    @printf("  %-14s %8.2f MiB  %.2fx  %s\n", r.label, mib(r.bytes), r.ratio, verdict)
end
println()
if isempty(threaded_wins)
    println(
        "no size measured favours threading -- the threshold should go above ",
        @sprintf("%.1f MiB", mib(sorted[end].bytes))
    )
elseif isempty(serial_wins)
    println("no size measured favours serial -- the threshold should go to 0")
else
    lo = maximum(r -> r.bytes, serial_wins)
    above = filter(r -> r.bytes > lo, threaded_wins)
    hi = isempty(above) ? minimum(r -> r.bytes, threaded_wins) : minimum(r -> r.bytes, above)
    @printf(
        "crossover bracket: serial still wins at %.1f MiB, threading wins from %.1f MiB\n",
        mib(lo), mib(hi)
    )
    @printf("current threshold: %.1f MiB\n", mib(MRT.SERIAL_BLAS_THRESHOLD_BYTES))
end

outdir = joinpath(@__DIR__, "..", "results")
mkpath(outdir)
outfile = joinpath(outdir, "serial_blas_threshold_$(BACKEND)_$(Threads.nthreads())threads.json")
open(outfile, "w") do io
    JSON.print(
        io,
        Dict(
            "backend" => BACKEND,
            "julia_threads" => Threads.nthreads(),
            "host" => gethostname(),
            "maxit" => MAXIT,
            "current_threshold_bytes" => MRT.SERIAL_BLAS_THRESHOLD_BYTES,
            "cases" => [Dict(pairs(r)) for r in results],
        ),
        2,
    )
end
println("wrote ", outfile)

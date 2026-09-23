# Shared prelude for the decomposed comparison suite. Every `run_<section>.jl` does
# `include(joinpath(@__DIR__, "_setup.jl"))` first: it parses the CLI, pins threads, configures
# the BART / OpenMP / MKL environment, loads MRT + BART + SigPy + MRIReco and the case catalog
# (benchmark/utils/), and defines the timing helpers and `BenchResult`. Sections take their data
# from the catalog only.
#
# Each section script then appends to `results::Vector{BenchResult}` and calls
# `write_section("<name>")`, which records one immutable run file under
# `results/runs/` via `ResultsStore.jl`. `query_results.jl` reads that directory back for
# analysis -- there is no merge step (see `ResultsStore.jl`'s module docstring for why).

using Printf
using JSON

const USE_MKL = "--use-mkl" in ARGS
let i = findfirst(a -> startswith(a, "--threads="), ARGS)
    global const NUM_THREADS = i === nothing ? Threads.nthreads() : parse(Int, split(ARGS[i], "=")[2])
end

# Machine paths (BART builds, SigPy's interpreter, the data cache) come from the environment, filled
# from the untracked `benchmark/slurm/site.env` for anything not already set.
include(joinpath(@__DIR__, "..", "..", "utils", "config.jl"))
load_site_env!()

# Which BART build to time against. Two builds because the comparison is per BLAS backend. With no
# build configured for this backend, BART is left out of every section (see `should_run_framework`).
if USE_MKL
    @info "Enabling Intel MKL backend via MKL.jl"
    using MKL
end
const BART_BINARY = get(ENV, USE_MKL ? "MRT_BENCH_BART_MKL" : "MRT_BENCH_BART_OPENBLAS", "")
const BART_AVAILABLE = !isempty(BART_BINARY) && isfile(BART_BINARY)
BART_AVAILABLE || @warn "No BART build configured for this backend, so BART rows are skipped" key = USE_MKL ? "MRT_BENCH_BART_MKL" : "MRT_BENCH_BART_OPENBLAS" value = BART_BINARY
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

BART_AVAILABLE && (ENV["TOOLBOX_PATH"] = BART_BINARY)
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
using MriReconstructionToolbox: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
using GeometricMedicalPhantoms
using LinearAlgebra
using Statistics
using Random
using FFTW
using BartIO
using PyCall
using MRIReco

"""
    MRIRECO_BLAS_THREADS

The BLAS thread count MRIReco chose for itself, captured before this file overrides it.

`MRIReco.__init__` sets `BLAS.set_num_threads(1)` when `Threads.nthreads() > 1`
(MRIReco.jl:20-26 — the other branch is Windows-only). That is a deliberate choice by the
package under test, and the harness must not silently undo it: setting the count here, *after*
`using MRIReco`, left every timed MRIReco row at `NUM_THREADS` while MRT pinned BLAS inside its
own solve, so the two toolkits were compared under different BLAS policies for no reason other
than the order of two lines in this file.

At `-t 1` on Linux MRIReco makes no choice at all, and OpenBLAS stays at its
`jl_effective_threads`-derived default — most of the node. That is not a policy to respect but a
known artifact: RegularizedLeastSquares makes ~30 BLAS-1 calls per ADMM outer iteration
(`norm`/`dot`/`rmul!` in `cg.jl` and the residual block), each spawning a full thread team over a
~9k-element vector, measured at 346 s for a TV solve that takes 1.08 s with BLAS pinned. So the
single-threaded case still gets `NUM_THREADS` (which is 1 there anyway).
"""
const MRIRECO_BLAS_THREADS = Threads.nthreads() > 1 ? BLAS.get_num_threads() : NUM_THREADS

# Everything else runs at the thread count under test. `with_mrireco_blas` puts MRIReco's own
# choice back for the duration of an MRIReco call, and restores this afterwards.
BLAS.set_num_threads(NUM_THREADS)
FFTW.set_num_threads(NUM_THREADS)
@info "BLAS/FFTW pinned" blas_threads = BLAS.get_num_threads() fftw_threads = FFTW.get_num_threads() mrireco_blas_threads = MRIRECO_BLAS_THREADS

"""
    check_environment()

Fail loudly instead of silently benchmarking under an environment that does not match what was
requested. It is an assertion, run automatically by every section, because a mismatch silently
invalidates the numbers rather than crashing anything -- exactly the kind of thing this suite
exists to catch in a toolkit, not commit on its own.
"""
function check_environment()
    cpus_allowed = let line = ""
        for l in eachline("/proc/self/status")
            startswith(l, "Cpus_allowed_list:") && (line = strip(split(l, ":")[2]))
        end
        line
    end
    n_allowed = sum(
        r -> (p = split(r, "-"); length(p) == 1 ? 1 : parse(Int, p[2]) - parse(Int, p[1]) + 1),
        split(cpus_allowed, ","),
    )
    n_allowed < NUM_THREADS && error(
        "requested $NUM_THREADS threads but only $n_allowed CPUs are allowed " *
            "(Cpus_allowed_list=$cpus_allowed) -- the SLURM allocation does not cover what was " *
            "asked for; rerun with a matching --cpus-per-task",
    )
    Threads.nthreads() != NUM_THREADS && error(
        "requested $NUM_THREADS threads but Julia started with $(Threads.nthreads()) -- pass -t $NUM_THREADS",
    )
    BLAS.get_num_threads() != NUM_THREADS && error(
        "BLAS is pinned to $(BLAS.get_num_threads()) threads, not the requested $NUM_THREADS",
    )
    USE_MKL && get(ENV, "KMP_BLOCKTIME", "") != "0" && error(
        "MKL is enabled but KMP_BLOCKTIME is $(get(ENV, "KMP_BLOCKTIME", "unset")), not \"0\" -- " *
            "export it before starting Julia (see docs/src/high-level/performance.md); MKL's " *
            "worker threads will otherwise spin and crowd out the ones being measured",
    )
    return nothing
end
check_environment()

"""
    with_mrireco_blas(f)

Run `f()` with BLAS at [`MRIRECO_BLAS_THREADS`](@ref) — what MRIReco set for itself — and restore
`NUM_THREADS` afterwards, including on exception.

Every timed MRIReco call goes through this, so MRIReco is measured under its own threading policy
and MRT under its own, rather than both under whichever one happened to be set last.
"""
function with_mrireco_blas(f)
    MRIRECO_BLAS_THREADS == NUM_THREADS && return f()
    BLAS.set_num_threads(MRIRECO_BLAS_THREADS)
    try
        return f()
    finally
        BLAS.set_num_threads(NUM_THREADS)
    end
end

include(joinpath(@__DIR__, "..", "src", "ComparisonHarness.jl"))
using .ComparisonHarness: check_nrmse, run_bart

# The case catalog, MRT's reconstruction of each method, `time_run` and the result store, shared
# with the MRT harness (benchmark/run.jl). No section prepares data of its own.
include(joinpath(@__DIR__, "..", "..", "utils", "bench_utils.jl"))
using .BenchUtils

const sigpy = pyimport("sigpy")
const sp_mri = pyimport("sigpy.mri")
const sp_app = pyimport("sigpy.mri.app")

@info "comparison setup" host = gethostname() julia = VERSION threads = Threads.nthreads() blas = BLAS.get_config().loaded_libs[1].libname mkl = USE_MKL bart = BART_BINARY

# Bare process spawn cost (`bart version` does no file I/O) — the floor for an input-less call.
const BART_SPAWN = BART_AVAILABLE ? let times = Float64[]
        for _ in 1:10
            t0 = time_ns()
            read(pipeline(ignorestatus(`$BART_BINARY version`)), String)
            push!(times, (time_ns() - t0) / 1.0e9)
    end
        minimum(times)
end : NaN
BART_AVAILABLE && @info @sprintf("BART spawn cost: %.1f ms", BART_SPAWN * 1000)

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
            push!(ts, (time_ns() - t0) / 1.0e9)
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
function time_bart(cmd::AbstractString, inputs...; nout::Int = 1, num_runs::Int = RUNS[], heavy_threshold::Real = 5.0)
    if WARMUP[] == 0         # images only (calibration): one untimed-for-the-record run
        t0 = time_ns()
        res = run_bart(nout, cmd, inputs...)
        t = (time_ns() - t0) / 1.0e9
        return t, t, res
    end
    ovh = bart_overhead(inputs...)
    t0 = time_ns()
    res = run_bart(nout, cmd, inputs...)
    warm = (time_ns() - t0) / 1.0e9 - ovh
    wis = warm > heavy_threshold
    wis && run_bart(nout, cmd, inputs...; wisdom = true)   # build the measured plan once
    times = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        res = run_bart(nout, cmd, inputs...; wisdom = wis)
        push!(times, (time_ns() - t0) / 1.0e9)
    end
    @info @sprintf("BART '%s': overhead %.1f ms, wisdom %s", first(split(cmd)), ovh * 1000, wis)
    return max(1.0e-5, minimum(times) - ovh), max(1.0e-5, median(times) - ovh), res
end

"""
    RUNS

Timed runs of the current case: 3, or 1 for a heavy case (`timed_runs`). `toolkit_run` sets it, and
[`time_reconstruction`](@ref) and [`time_bart`](@ref) default to it.
"""
const RUNS = Ref(3)

"""
    WARMUP

Untimed warm-up runs before the timed ones: 1, or 0 in `calibrate_lambda.jl`, which only needs
the images.
"""
const WARMUP = Ref(1)

"""
    time_reconstruction(f; num_runs = RUNS[]) -> (t_min_s, t_med_s, result)

`time_run` (`WARMUP[]` warm-ups, then the minimum and median of `num_runs`) for the in-process
toolkits, MRT's timing function in the harness too. BART goes through [`time_bart`](@ref) instead.
"""
time_reconstruction(f; num_runs::Int = RUNS[]) = time_run(f; warmup = WARMUP[], runs = num_runs)

"""
    BenchResult

One row: `category` is the section, `method` the method label, `case_id` the catalog case and
`data_source` where its data came from (`"synthetic"` or the real dataset).
"""
struct BenchResult
    category::String
    method::String
    framework::String
    threads::Int
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_mrt::Float64
    case_id::String
    data_source::String
end

results = BenchResult[]

"""
    CASE_FILTER

Parsed from `--cases=pat1,pat2,...`, or `nothing` when not passed (run everything). Each `pat` is a
case-insensitive substring matched against a catalog case id (`--cases=shepp_logan_2d` runs the
three 2D Shepp-Logan cases, `--cases=cine` both cine cases), a section, or a method label
(`--cases=low-rank` runs every low-rank row). Every section checks [`should_run_case`](@ref) and
[`should_run`](@ref) before paying for a solve, so a rerun of one suspect case does not have to pay
for the whole section.
"""
const CASE_FILTER = let i = findfirst(a -> startswith(a, "--cases="), ARGS)
    i === nothing ? nothing : [lowercase(s) for s in split(ARGS[i][(length("--cases=") + 1):end], ",")]
end

"""
    DATA

`--data=synthetic` (default), `real` or `all`: which catalog cases the sections iterate. Real data
are the real-data analogues of the synthetic cases (benchmark/utils/real_data.jl).
"""
const DATA = let i = findfirst(a -> startswith(a, "--data="), ARGS)
    d = i === nothing ? "synthetic" : ARGS[i][(length("--data=") + 1):end]
    d in ("synthetic", "real", "all") || error("--data=$d: expected synthetic, real or all")
    d
end

"""
    should_run(label, method) -> Bool

True unless [`CASE_FILTER`](@ref) is set and no pattern in it is a substring of `label` (a case id
or a section) or of `method` (case-insensitive).
"""
should_run(category, method) = CASE_FILTER === nothing ||
    any(p -> occursin(p, lowercase(category)) || occursin(p, lowercase(method)), CASE_FILTER)

"""
    should_run_case(id) -> Bool

Whether any row of case `id` can pass [`CASE_FILTER`](@ref): true when no filter is set, or when a
pattern does not name a case at all (then it filters by section or method instead).
"""
function should_run_case(id)
    CASE_FILTER === nothing && return true
    all_ids = lowercase.(case_ids(; real = true))
    return any(p -> occursin(p, lowercase(id)) || !any(i -> occursin(p, i), all_ids), CASE_FILTER)
end

"""
    FRAMEWORK_FILTER

Parsed from `--frameworks=pat1,pat2,...`, or `nothing`. Each `pat` is a case-insensitive substring
matched against a framework label (`"BART"` matches `"BART (MKL)"` and `"BART (OpenBLAS)"` alike).
Gates only the *competitor* toolkits (SigPy/BART/MRIReco/MIRT) in each case, never MRT itself: MRT's
own solve is the reference every other framework's `nrmse_mrt` is computed against, so it always
runs regardless of this filter, and stays cheap next to whichever toolkit is under suspicion.
"""
const FRAMEWORK_FILTER = let i = findfirst(a -> startswith(a, "--frameworks="), ARGS)
    i === nothing ? nothing : [lowercase(s) for s in split(ARGS[i][(length("--frameworks=") + 1):end], ",")]
end

"""
    should_run_framework(framework) -> Bool

True unless [`FRAMEWORK_FILTER`](@ref) is set and no pattern in it is a substring of `framework`
(case-insensitive), or `framework` is BART and no BART build is configured for this backend. See
[`FRAMEWORK_FILTER`](@ref) -- never call this for MRT's own row.
"""
function should_run_framework(framework)
    occursin("bart", lowercase(framework)) && !BART_AVAILABLE && return false
    return FRAMEWORK_FILTER === nothing || any(p -> occursin(p, lowercase(framework)), FRAMEWORK_FILTER)
end

"""Replace NaN / Inf with -1.0 so a single bad toolkit row does not sink the section's JSON."""
_json_num(x::Real) = isfinite(x) ? Float64(x) : -1.0

using .BenchUtils.ResultsStore: record_run

"""
    flush_results!(name) -> path or nothing

Write and clear whatever is currently in `results`, as its own immutable run file, right now.
Every section calls this after each case (see `should_run`'s guarded blocks in the `run_<section>.jl`
scripts) rather than only once at the end, so a crash partway through a long section -- a slow
real-data solve, a hung toolkit subprocess -- does not lose the cases that already finished. Returns
`nothing` when there is nothing to flush (already flushed, or the case was skipped).
"""
function flush_results!(name::AbstractString)
    isempty(results) && return nothing
    path = record_run(
        name, USE_MKL ? "mkl" : "openblas", NUM_THREADS, results;
        hostname = gethostname(), julia_version = string(VERSION),
        julia_threads = Threads.nthreads(), blas_vendor = BLAS.get_config().loaded_libs[1].libname,
        use_mkl = USE_MKL, bart_binary = BART_BINARY, pinned_cpus = CPU_STR,
        bart_spawn_ms = BART_SPAWN * 1000,
        cases_filter = CASE_FILTER, frameworks_filter = FRAMEWORK_FILTER, data = DATA,
        small = small_mode(), cine_frames = cine_frames(),
    )
    for r in results
        @printf(
            "%-38s | %-26s | %-22s | %3d | %10.2f ms | %10.2e | %10.2e\n",
            r.case_id, r.method, r.framework, r.threads, r.time_ms, r.nrmse_gt, r.nrmse_mrt
        )
    end
    @info "flushed run" path n = length(results) source = ResultsStore.source_tag()
    empty!(results)
    return path
end

"""Final catch-all flush at the end of a section script -- a no-op if every case already flushed
itself via [`flush_results!`](@ref)."""
write_section(name::AbstractString) = flush_results!(name)

"""
    CMP_CTYPE / CMP_RTYPE

The complex (and matching real) element type every toolkit reconstructs in. **`ComplexF32`**, which
is what BART is: its `complex float` is a pair of `float32`, with no double-precision build option,
so a double-precision run of the other four compares a toolkit doing twice the memory traffic
against one that is not. Single precision is also what MRI reconstruction is done in — k-space off
the scanner is 16-bit integer or 32-bit float.

Set `CMP_PRECISION=double` to go back to `ComplexF64` for everything except BART, which cannot.

Only the *solve* runs in this type. Each bridge promotes its result to `ComplexF64` on the way out,
outside the timed region, so the NRMSE column is not itself computed at the precision under test.
"""
const CMP_CTYPE = get(ENV, "CMP_PRECISION", "single") == "double" ? ComplexF64 : ComplexF32
const CMP_RTYPE = real(CMP_CTYPE)
@info "comparison precision" ctype = CMP_CTYPE

# Every section's data comes from the case catalog (`get_case`), in `ComplexF32`. The sensitivity
# maps are handed to every toolkit exactly as the catalog produces them: `normalize_sensitivity_maps`
# is deliberately not called, since it would give MRT a known operator norm and so a free step size,
# while MRIReco's `SensitivityOp` and BART's `pics` normalize nothing.

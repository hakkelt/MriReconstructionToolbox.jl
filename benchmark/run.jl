# MRT benchmark harness: every catalog case (benchmark/utils/cases.jl) × every applicable method,
# timed with `time_run` (one warm-up, then the minimum of three runs; one run for heavy cases).
#
#   julia --project=benchmark -t N benchmark/run.jl --threads=N [options]
#
# Options:
#   --use-mkl                 load MKL before MRT (BLAS backend "mkl"; default "openblas")
#   --cases=pat,...           case-insensitive substrings of case ids (default: every case)
#   --methods=tv,llr,...      methods to run (default: every applicable one)
#   --real                    include the real-data analogues (also MRT_BENCH_REAL_DATA=1)
#   --ref-name=NAME           label for the measured checkout (default: its branch name)
#   --mrt=PATH                measure the MRT checkout at PATH instead of this one: re-runs this
#                             script in PATH/benchmark's environment. PATH needs a Manifest.toml;
#                             when it has none, this checkout's is copied in.
#   --dev=Pkg=PATH,...        with --mrt: point the environment's manifest entry of Pkg at PATH
#                             (e.g. --dev=NestedThreading=/path/to/checkout)
#   --remeasure               measure configurations the store already has
#   --runs=K / --warmup=K     override the number of timed / warm-up runs
#   --env-variant=STR         a label for the environment variant (recorded; see matrix.sh)
#
# Results go to benchmark/results/runs/ (one immutable JSON file per case, see results_store.jl),
# tagged with the measured checkout's commit, the tree hashes of its deps/ and of the
# NestedThreading checkout it loads, and the node class. A configuration whose result is already
# stored for the same clean code on the same node class is skipped, so a baseline measured once is
# never measured again. `benchmark/compare.jl` compares two refs.

const _ARGS = copy(ARGS)
_arg(name, default = nothing) = (i = findlast(a -> startswith(a, "--$name="), _ARGS)) === nothing ? default : _ARGS[i][(length(name) + 4):end]
_flag(name) = "--$name" in _ARGS
_list(name) = (v = _arg(name); v === nothing ? nothing : String.(split(v, ",")))

# ---------------------------------------------------------------- --mrt: re-run in that checkout

if _arg("mrt") !== nothing && get(ENV, "MRT_BENCH_REEXEC", "") != "1"
    using FileWatching: mkpidlock
    mrt = abspath(_arg("mrt"))
    proj = joinpath(mrt, "benchmark")
    isfile(joinpath(proj, "Project.toml")) || error("$mrt has no benchmark/Project.toml: it predates the benchmark harness")
    manifest = joinpath(mrt, "Manifest.toml")
    mkpidlock(joinpath(mrt, ".benchmark_env.pid"); stale_age = 3600) do
        if !isfile(manifest)
            cp(joinpath(@__DIR__, "..", "Manifest.toml"), manifest)
            @info "copied this checkout's Manifest.toml into $mrt, so both measure the same dependency versions"
        end
        for spec in something(_list("dev"), String[])
            pkg, path = split(spec, "="; limit = 2)
            text = read(manifest, String)
            block = Regex("(\\[\\[deps\\.$pkg\\]\\][^\\[]*?\\npath = )\"[^\"]*\"")
            occursin(block, text) || error("no path entry for $pkg in $manifest (only a dev'd package can be redirected)")
            write(manifest, replace(text, block => SubstitutionString("\\1\"$(escape_string(abspath(path)))\"")))
        end
        run(`$(Base.julia_cmd()) --project=$proj -e "using Pkg; Pkg.instantiate()"`)
    end
    cmd = `$(Base.julia_cmd()) --project=$proj -t $(Threads.nthreads()) $(@__FILE__) $_ARGS`
    exit(run(addenv(ignorestatus(cmd), "MRT_BENCH_REEXEC" => "1")).exitcode)
end

# ---------------------------------------------------------------- environment

const USE_MKL = _flag("use-mkl")
USE_MKL && @eval using MKL
const NUM_THREADS = parse(Int, something(_arg("threads"), string(Threads.nthreads())))
Threads.nthreads() == NUM_THREADS || error("--threads=$NUM_THREADS but Julia runs $(Threads.nthreads()) threads: pass -t $NUM_THREADS")

using LinearAlgebra, Printf, Dates
using ThreadPinning
using FFTW
using MriReconstructionToolbox

include(joinpath(@__DIR__, "utils", "bench_utils.jl"))
using .BenchUtils
using .BenchUtils: ResultsStore, time_run, timed_runs, git_ref, tree_hash, node_class, recorded_env

load_site_env!()
_flag("real") && (ENV["MRT_BENCH_REAL_DATA"] = "1")

# Threads go only to allowed CPUs that are not SMT siblings: a sibling shares its core with another
# thread of the same run, which halves that core for both.
const PINNED_CPUS = let allowed = findall(!iszero, getaffinity()) .- 1
    isempty(allowed) && (allowed = collect(0:(NUM_THREADS - 1)))
    physical = filter(!ThreadPinning.ishyperthread, allowed)
    length(physical) >= NUM_THREADS || error("only $(length(physical)) physical cores among the allowed CPUs $allowed, $NUM_THREADS threads requested")
    physical[1:NUM_THREADS]
end
pinthreads(PINNED_CPUS)
BLAS.set_num_threads(NUM_THREADS)
FFTW.set_num_threads(NUM_THREADS)
const BACKEND = USE_MKL ? "mkl" : "openblas"

# ---------------------------------------------------------------- provenance

const MRT_DIR = pkgdir(MriReconstructionToolbox)
const MRT_REF = git_ref(MRT_DIR)
const NT_DIR = let id = Base.PkgId(Base.UUID("e10243f7-6390-482b-b557-80f41d32665c"), "NestedThreading")
    haskey(Base.loaded_modules, id) ? pkgdir(Base.loaded_modules[id]) : ""
end
const NT_REF = isempty(NT_DIR) ? (commit = "", branch = "", dirty = false) : git_ref(NT_DIR)
const DEPS_HASH = tree_hash(MRT_DIR, "deps")
const SRC_HASH = tree_hash(MRT_DIR, "src")
const EXT_HASH = tree_hash(MRT_DIR, "ext")
const PROJECT_HASH = tree_hash(MRT_DIR, "Project.toml")
const NT_HASH = isempty(NT_DIR) ? "" : tree_hash(NT_DIR)
const REF_NAME = something(_arg("ref-name"), MRT_REF.branch, "unknown")
# Two runs with equal code keys ran the same MRT package code (src/, ext/, Project.toml), the same
# vendored dependencies (deps/) and the same NestedThreading, whatever else differs between their
# commits -- a commit touching only benchmark/ keeps its baseline. Uncommitted changes to any of
# them leave no key: such a run is always measured, and never reused.
const CODE_PARTS = (SRC_HASH, EXT_HASH, PROJECT_HASH, DEPS_HASH, NT_HASH)
const CLEAN = all(h -> !isempty(h) && !startswith(h, "dirty"), CODE_PARTS[1:4]) && !startswith(NT_HASH, "dirty")
const CODE_KEY = CLEAN ? join(CODE_PARTS, ":") : ""
const NODE = node_class()
const ENV_VARIANT = something(_arg("env-variant"), "")

@info "MRT harness" ref = REF_NAME commit = MRT_REF.commit[1:min(end, 12)] dirty = !CLEAN mrt = MRT_DIR nested_threading = NT_DIR threads = NUM_THREADS backend = BACKEND node = NODE small = small_mode()
CLEAN || @warn "the measured code has uncommitted changes: results are recorded but never reused"

config_key(case, method) = (CODE_KEY, NODE, case, string(method), NUM_THREADS, BACKEND, small_mode(), cine_frames(), ENV_VARIANT)

const STORED = let keys = Set{Tuple}()
    for d in ResultsStore.load_run_files(ResultsStore.HARNESS_RESULTS_DIR)
        k = get(d, "code_key", "")
        isempty(k) && continue
        for r in get(d, "benchmarks", [])
            get(r, "status", "") == "ok" || continue
            push!(keys, (k, get(d, "node_class", ""), r["case"], r["method"], d["threads"], d["backend"], get(d, "small", false), get(d, "cine_frames", 0), get(d, "env_variant", "")))
        end
    end
    keys
end

const RUN_META = (;
    ref_name = REF_NAME, git_commit = MRT_REF.commit, git_branch = MRT_REF.branch, git_dirty = MRT_REF.dirty,
    deps_hash = DEPS_HASH, src_hash = SRC_HASH, ext_hash = EXT_HASH, project_hash = PROJECT_HASH, mrt_dir = MRT_DIR,
    nested_threading_dir = NT_DIR, nested_threading_commit = NT_REF.commit,
    nested_threading_branch = NT_REF.branch, nested_threading_hash = NT_HASH,
    code_key = CODE_KEY, node_class = NODE, hostname = gethostname(), pinned_cpus = join(PINNED_CPUS, ","),
    julia_version = string(VERSION), blas_vendor = string(BLAS.get_config().loaded_libs[1].libname),
    env = recorded_env(), env_variant = ENV_VARIANT, small = small_mode(), cine_frames = cine_frames(),
    harness_commit = git_ref(@__DIR__).commit,
)

# ---------------------------------------------------------------- measurement

const IDS = filter_case_ids(case_ids(), _list("cases"))
const METHOD_FILTER = _list("methods")
const REMEASURE = _flag("remeasure")
const RUNS = _arg("runs") === nothing ? nothing : parse(Int, _arg("runs"))
const WARMUP = parse(Int, something(_arg("warmup"), "1"))

isempty(IDS) && error("no case matches --cases=$(_arg("cases"))")
n_measured = n_skipped = n_failed = 0
for id in IDS
    todo = nothing
    c = nothing
    rows = Dict{String, Any}[]
    try
        c = get_case(id)
    catch err
        @error "case $id could not be prepared, skipping it" exception = (err, catch_backtrace())
        global n_failed += 1
        continue
    end
    methods = applicable_methods(c)
    METHOD_FILTER === nothing || (methods = [m for m in methods if string(m) in METHOD_FILTER])
    for m in methods
        if !REMEASURE && !isempty(CODE_KEY) && config_key(id, m) in STORED
            @info "stored, skipping (--remeasure to rerun)" case = id method = m
            global n_skipped += 1
            continue
        end
        runs = something(RUNS, timed_runs(c))
        row = Dict{String, Any}(
            "case" => id, "method" => string(m), "framework" => "MRT", "runs" => runs, "warmup" => WARMUP,
            "lambda" => get(DEFAULT_LAMBDA, m, 0.0), "source" => c.source, "real" => c.real,
        )
        try
            f = mrt_reconstructor(c, m)
            tmin, tmed, x = time_run(f; warmup = WARMUP, runs)
            e = mag_nrmse(Array(parent(x)), c.reference)
            merge!(row, Dict("status" => "ok", "time_min_ms" => 1.0e3 * tmin, "time_median_ms" => 1.0e3 * tmed, "nrmse" => Float64(e)))
            @info @sprintf("%-40s %-9s %10.1f ms (median %.1f, %d run%s)  NRMSE %.4f", id, m, 1.0e3 * tmin, 1.0e3 * tmed, runs, runs == 1 ? "" : "s", e)
            global n_measured += 1
        catch err
            @error "$id / $m failed" exception = (err, catch_backtrace())
            merge!(row, Dict("status" => "error", "error" => first(sprint(showerror, err), 2000)))
            global n_failed += 1
        end
        push!(rows, row)
    end
    isempty(rows) && continue
    path = ResultsStore.record_run_rows(
        "harness", BACKEND, NUM_THREADS, rows; root = ResultsStore.HARNESS_RESULTS_DIR, source = ResultsStore.source_tag(), RUN_META...,
    )
    @info "recorded" path
    GC.gc()
end
@info "done" measured = n_measured skipped = n_skipped failed = n_failed
exit(n_failed == 0 ? 0 : 1)

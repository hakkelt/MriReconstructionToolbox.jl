#!/usr/bin/env julia
# Benchmark comparison of two revisions for a pull-request comment.
#
#     julia --project=benchmark/ci benchmark/ci/compare.jl \
#         --base-dir <base checkout> --head-dir <head checkout> \
#         --output-dir <dir> [--pr <number>] [--threads 1,2]
#
# The suite of the head checkout (`benchmark/ci/benchmarks.jl`) runs against both revisions, so a
# pull request that adds or changes entries is compared on the same entries. Each revision gets
# its own copy of this environment with that revision `Pkg.develop`ed into it, and each
# (revision, thread count) pair runs in a fresh process pinned with `-t`. The rendered comment
# body is written to `<output-dir>/body.md`.
#
# Ported from the AbstractOperators.jl benchmark driver (MIT).

using ArgParse
using BenchmarkTools
using Pkg
using Printf
using Serialization
using Statistics

function parse_args_local(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--base-dir"
        help = "path to the base revision checkout"
        arg_type = String
        required = true
        "--head-dir"
        help = "path to the head revision checkout"
        arg_type = String
        required = true
        "--output-dir"
        help = "directory for body.md, the environments and the raw results"
        arg_type = String
        default = "benchmark-output"
        "--pr"
        help = "pull-request number"
        arg_type = String
        default = ""
        "--julia-version"
        help = "Julia version for the comment header"
        arg_type = String
        default = string(VERSION)
        "--threads"
        help = "comma-separated thread counts; the suite runs once per count and revision"
        arg_type = String
        default = "1,2"
        "--repeats"
        help = "processes per revision and thread count, alternating base and head; each entry keeps its fastest process"
        arg_type = Int
        default = 1
    end
    return parse_args(args, s)
end

parse_thread_counts(s::AbstractString) = [parse(Int, strip(t)) for t in split(s, ',') if !isempty(strip(t))]

# ---------------------------------------------------------------- running the suite

"""
    prepare_env(head_dir, repo_dir, env_dir)

Copy the head's `benchmark/ci/Project.toml` to `env_dir` and develop the revision at `repo_dir`
into it. Done once per revision, before the thread-count loop.
"""
function prepare_env(head_dir::AbstractString, repo_dir::AbstractString, env_dir::AbstractString)
    mkpath(env_dir)
    cp(joinpath(head_dir, "benchmark", "ci", "Project.toml"), joinpath(env_dir, "Project.toml"); force = true)
    code = "using Pkg; Pkg.develop(; path = $(repr(abspath(repo_dir))), io = devnull); Pkg.instantiate(; io = devnull); Pkg.precompile()"
    @info "Preparing benchmark environment for $repo_dir"
    run(`$(Base.julia_cmd()) --startup-file=no --project=$(env_dir) -e $(code)`)
    return env_dir
end

"""
    run_suite(head_dir, env_dir, result_path, threads)

Run the head's suite in `env_dir` in a fresh process with `threads` threads, and serialize
`(results, nrmse, failed)` to `result_path`.
"""
function run_suite(head_dir::AbstractString, env_dir::AbstractString, result_path::AbstractString, threads::Int)
    script = joinpath(head_dir, "benchmark", "ci", "benchmarks.jl")
    runner = """
    include($(repr(script)))
    results = BenchmarkTools.run(SUITE; verbose = true)
    using Serialization
    Serialization.serialize($(repr(result_path)), (results, NRMSE, FAILED))
    """
    @info "Running benchmarks in $env_dir (threads = $threads)"
    ok = success(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(env_dir) -t $(threads) -e $(runner)`; stdout, stderr))
    ok || @error "The suite process failed in $env_dir (threads = $threads)"
    return result_path
end

# ---------------------------------------------------------------- comparison

struct RevisionResult
    trials::Dict{String, BenchmarkTools.Trial}
    nrmse::Dict{String, Float64}
    failed::Dict{String, String}
end

function flatten_group(group::BenchmarkGroup, prefix = "")
    out = Dict{String, BenchmarkTools.Trial}()
    for (k, v) in group
        key = isempty(prefix) ? string(k) : "$prefix/$k"
        if v isa BenchmarkGroup
            merge!(out, flatten_group(v, key))
        elseif v isa BenchmarkTools.Trial
            out[key] = v
        end
    end
    return out
end

"""
    best_of(results) -> RevisionResult

Per entry, the trial of the process with the lowest median time. Two processes of the same
revision differ by more than the spread within one (FFTW plans, memory placement, a busy
neighbour on the runner), so a single process per side flags changes that are not there.
"""
function best_of(rs::Vector{RevisionResult})
    trials = Dict{String, BenchmarkTools.Trial}()
    nrmse = Dict{String, Float64}()
    for r in rs, (k, t) in r.trials
        if !haskey(trials, k) || median(t.times) < median(trials[k].times)
            trials[k] = t
            haskey(r.nrmse, k) && (nrmse[k] = r.nrmse[k])
        end
    end
    failed = Dict{String, String}()
    for r in rs, (k, e) in r.failed
        haskey(trials, k) || (failed[k] = e)
    end
    return RevisionResult(trials, nrmse, failed)
end

function load_result(path)
    isfile(path) || return RevisionResult(Dict(), Dict(), Dict("(whole suite)" => "the suite process failed"))
    results, nrmse, failed = Serialization.deserialize(path)
    return RevisionResult(flatten_group(results), nrmse, failed)
end

const TIME_UNITS = [(:ns, 1.0e0), (Symbol("μs"), 1.0e3), (:ms, 1.0e6), (:s, 1.0e9)]

function auto_time_unit(t_ns::Float64)
    for (unit, scale) in reverse(TIME_UNITS)
        t_ns / scale >= 1.0 && return (unit, scale)
    end
    return TIME_UNITS[1]
end

function format_time(t::BenchmarkTools.Trial)
    med = median(t.times)
    err = max(0.0, quantile(t.times, 0.75) - quantile(t.times, 0.25)) / 2
    unit, scale = auto_time_unit(med)
    err > 0 && return @sprintf("%.3g ± %.2g %s", med / scale, err / scale, unit)
    return @sprintf("%.3g %s", med / scale, unit)
end

function format_memory(t::BenchmarkTools.Trial)
    allocs, bytes = t.allocs, t.memory
    bytes == 0 && return "0 allocs (0 bytes)"
    bytes < 1024 && return "$allocs allocs ($bytes bytes)"
    bytes < 1024^2 && return @sprintf("%d allocs (%.2f KiB)", allocs, bytes / 1024)
    return @sprintf("%d allocs (%.2f MiB)", allocs, bytes / 1024^2)
end

"""
    compute_ratio(base, head, mode) -> (ratio, err)

`base / head` of the median time (`err` from the interquartile ranges) or of the memory (`err`
is `NaN`). Above 1 the pull request is faster or allocates less.
"""
function compute_ratio(base::BenchmarkTools.Trial, head::BenchmarkTools.Trial, mode::Symbol)
    if mode === :time
        bm, hm = median(base.times), median(head.times)
        be = max(0.0, quantile(base.times, 0.75) - quantile(base.times, 0.25))
        he = max(0.0, quantile(head.times, 0.75) - quantile(head.times, 0.25))
        r = bm / hm
        return r, abs(r) * sqrt((be / bm)^2 + (he / hm)^2)
    end
    head.memory == 0 && return (base.memory == 0 ? 1.0 : Inf), NaN
    return base.memory / head.memory, NaN
end

function ratio_emoji(ratio::Float64, err::Float64, mode::Symbol)
    if mode === :time && isfinite(err)
        ratio + err < 0.8 && return " 🐢"
        ratio - err > 1.2 && return " 🚀"
    else
        ratio < 0.5 && return " 🐢"
        ratio > 1.5 && return " 🚀"
    end
    return ""
end

function format_ratio(ratio::Float64, err::Float64, mode::Symbol)
    emoji = ratio_emoji(ratio, err, mode)
    isfinite(ratio) || return "N/A$emoji"
    isfinite(err) && err > 0 && return @sprintf("%.3g ± %.2g%s", ratio, err, emoji)
    return @sprintf("%.3g%s", ratio, emoji)
end

"""
    NRMSE_TOLERANCE

Relative change of a solve's error against the ground truth above which the NRMSE column is
flagged: a solver change that is faster but converges somewhere else is not a speedup.
"""
const NRMSE_TOLERANCE = 0.05

function format_nrmse(base::RevisionResult, head::RevisionResult, k::String)
    b, h = get(base.nrmse, k, NaN), get(head.nrmse, k, NaN)
    isnan(b) && isnan(h) && return ""
    bs = isnan(b) ? "—" : @sprintf("%.4f", b)
    hs = isnan(h) ? "—" : @sprintf("%.4f", h)
    flag = isfinite(b) && isfinite(h) && abs(h - b) > NRMSE_TOLERANCE * b ? " ⚠️" : ""
    return "$bs → $hs$flag"
end

function markdown_table(header::AbstractVector, rows::AbstractVector)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|:---|", join(fill("---:", length(header) - 1), "|"), "|")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

function build_table(base::RevisionResult, head::RevisionResult, mode::Symbol, base_label, head_label)
    ks = sort!(collect(union(keys(base.trials), keys(head.trials), keys(base.failed), keys(head.failed))))
    with_nrmse = mode === :time && !(isempty(base.nrmse) && isempty(head.nrmse))
    header = ["Benchmark", base_label, head_label, "Ratio (base/head)"]
    with_nrmse && push!(header, "NRMSE (base → head)")
    fmt = mode === :time ? format_time : format_memory
    rows = Vector{String}[]
    for k in ks
        b, h = get(base.trials, k, nothing), get(head.trials, k, nothing)
        row = [
            "`$k`",
            b === nothing ? "—" : fmt(b),
            h === nothing ? "—" : fmt(h),
            b === nothing || h === nothing ? "—" : format_ratio(compute_ratio(b, h, mode)..., mode),
        ]
        with_nrmse && push!(row, format_nrmse(base, head, k))
        push!(rows, row)
    end
    return markdown_table(header, rows)
end

function build_summary(base::RevisionResult, head::RevisionResult)
    common = intersect(keys(base.trials), keys(head.trials))
    counts = Dict(:faster => 0, :slower => 0, :less => 0, :more => 0, :nrmse => 0)
    for k in common
        r, e = compute_ratio(base.trials[k], head.trials[k], :time)
        r - e > 1.2 && (counts[:faster] += 1)
        r + e < 0.8 && (counts[:slower] += 1)
        m, _ = compute_ratio(base.trials[k], head.trials[k], :memory)
        m > 1.5 && (counts[:less] += 1)
        m < 0.5 && (counts[:more] += 1)
        b, h = get(base.nrmse, k, NaN), get(head.nrmse, k, NaN)
        isfinite(b) && isfinite(h) && abs(h - b) > NRMSE_TOLERANCE * b && (counts[:nrmse] += 1)
    end
    plural(n, one, many) = "$n $(n == 1 ? one : many)"
    parts = String[]
    counts[:faster] > 0 && push!(parts, "🚀 " * plural(counts[:faster], "benchmark", "benchmarks") * " faster")
    counts[:slower] > 0 && push!(parts, "🐢 " * plural(counts[:slower], "time regression", "time regressions"))
    counts[:less] > 0 && push!(parts, "🚀 " * plural(counts[:less], "benchmark allocates", "benchmarks allocate") * " less")
    counts[:more] > 0 && push!(parts, "🐢 " * plural(counts[:more], "memory regression", "memory regressions"))
    counts[:nrmse] > 0 && push!(parts, "⚠️ " * plural(counts[:nrmse], "solve", "solves") * " changed NRMSE by more than $(round(Int, 100NRMSE_TOLERANCE)) %")
    summary = isempty(parts) ? "No significant change in time, memory or accuracy." : join(parts, " · ")
    for (label, r) in (("base", base), ("head", head))
        isempty(r.failed) && continue
        summary *= "\n\nFailed on $label: " * join(("`$k`" for k in sort!(collect(keys(r.failed)))), ", ")
    end
    return summary
end

function build_section(base::RevisionResult, head::RevisionResult, base_label, head_label, threads::Int)
    return """
    ### $(threads) thread$(threads == 1 ? "" : "s")

    $(build_summary(base, head))

    <details>
    <summary>Time</summary>

    $(build_table(base, head, :time, base_label, head_label))
    </details>

    <details>
    <summary>Memory</summary>

    $(build_table(base, head, :memory, base_label, head_label))
    </details>
    """
end

function build_body(sections, base_label, head_label, julia_version, repeats)
    processes = repeats == 1 ? "one process" : "the fastest of $repeats alternating processes"
    rendered = [build_section(b, h, base_label, head_label, t) for (t, b, h) in sections]
    return """
    ## Benchmark Results (Julia v$(julia_version))

    Small synthetic cases (`benchmark/ci/benchmarks.jl`), the head's suite run against both
    revisions on the same runner, $(processes) per revision and thread count.

    $(join(rendered, "\n"))

    > **Ratio:** base/head; above 1 the pull request is faster. 🚀 significant speedup (time beyond
    > 1.2 outside the interquartile error, memory beyond 1.5) · 🐢 significant slowdown (below 0.8,
    > memory below 0.5) · ⚠️ NRMSE against the ground truth changed by more than $(round(Int, 100NRMSE_TOLERANCE)) %.
    > Shared runners are noisy: treat single 🐢 rows near the threshold with suspicion.
    """
end

function short_label(dir)
    sha = try
        readchomp(`git -C $dir rev-parse --short HEAD`)
    catch
        ""
    end
    return isempty(sha) ? basename(rstrip(dir, '/')) : sha
end

function main(argv = ARGS)
    opts = parse_args_local(argv)
    base_dir, head_dir = abspath(opts["base-dir"]), abspath(opts["head-dir"])
    output_dir = abspath(opts["output-dir"])
    mkpath(output_dir)
    base_label, head_label = "base (" * short_label(base_dir) * ")", "head (" * short_label(head_dir) * ")"

    base_env = prepare_env(head_dir, base_dir, joinpath(output_dir, "env_base"))
    head_env = prepare_env(head_dir, head_dir, joinpath(output_dir, "env_head"))

    sections = Tuple{Int, RevisionResult, RevisionResult}[]
    for threads in parse_thread_counts(opts["threads"])
        base_runs, head_runs = RevisionResult[], RevisionResult[]
        for i in 1:opts["repeats"]
            push!(base_runs, load_result(run_suite(head_dir, base_env, joinpath(output_dir, "results_base_t$(threads)_$i.jls"), threads)))
            push!(head_runs, load_result(run_suite(head_dir, head_env, joinpath(output_dir, "results_head_t$(threads)_$i.jls"), threads)))
        end
        push!(sections, (threads, best_of(base_runs), best_of(head_runs)))
    end

    body = build_body(sections, base_label, head_label, opts["julia-version"], opts["repeats"])
    write(joinpath(output_dir, "body.md"), body)
    write(joinpath(output_dir, "pr_number.txt"), opts["pr"])
    write(joinpath(output_dir, "julia_version.txt"), opts["julia-version"])
    @info "Benchmark comparison written to $(joinpath(output_dir, "body.md"))"
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

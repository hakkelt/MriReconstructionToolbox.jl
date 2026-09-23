# Concurrency-safe result storage: every recorded run is its own immutable file under
# `results/runs/`, never overwritten. That is the whole design -- a shared mutable file (the old
# `write_section` one-file-per-section-per-config JSON, or a single SQLite database on the same
# NFS-backed /project mount) has exactly the failure mode this replaces: two writers on different
# SLURM nodes (or a login-node smoke test run alongside a cluster job) can race on the same path,
# and the loser's data is gone with no trace it ever ran. A brand-new, uniquely-named file per run
# cannot collide with anything, on any filesystem, with no locking required.
#
# Querying is a read-time concern, not a write-time one: `query_results.jl` globs `results/runs/`
# and filters/sorts the rows in plain Julia -- no database involved at all, so there is nothing to
# keep in sync with the flat files, which stay the only source of truth.
module ResultsStore

using JSON, Dates

const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const RUNS_DIR = joinpath(RESULTS_DIR, "runs")

"""
    source_tag(hostname = gethostname()) -> "slurm" | "other"

SLURM compute nodes on this cluster are named `x<digits>c...`; the login node and anything else
(a laptop, a different cluster) is "other". Queries default to preferring `"slurm"` rows -- a
login-node run is real data (useful for a quick smoke test or an isolated repro) but was not taken
under the pinned/exclusive conditions the committed comparison numbers assume.
"""
source_tag(hostname::AbstractString = gethostname()) = occursin(r"^x\d+c", hostname) ? "slurm" : "other"

"""
    record_run(section, backend, threads, rows; kwargs...) -> path

Write one immutable JSON file recording this process's rows for `section` at (`backend`,
`threads`). `rows` is a `Vector{BenchResult}` (see `_setup.jl`). `kwargs` are extra metadata
(hostname, julia_version, blas_vendor, bart_binary, pinned_cpus, bart_spawn_ms, cases_filter, ...)
stored alongside, exactly as the old per-section JSON did.

The filename encodes everything a query needs to filter without opening the file: timestamp,
backend, threads, section, source, pid. `ispath` is checked and a random suffix appended on the
(astronomically unlikely) chance two runs land in the same millisecond under the same pid.
"""
function record_run(section::AbstractString, backend::AbstractString, threads::Integer, rows; kwargs...)
    benchmarks = [
        Dict(
            "category" => r.category, "method" => r.method, "framework" => r.framework,
            "threads" => r.threads, "time_ms" => _json_num(r.time_ms),
            "nrmse_gt" => _json_num(r.nrmse_gt), "nrmse_mrt" => _json_num(r.nrmse_mrt),
        ) for r in rows
    ]
    return record_run_rows(section, backend, threads, benchmarks; kwargs...)
end

"""
    record_run_rows(section, backend, threads, benchmarks::Vector{<:AbstractDict}; ts, source, kwargs...)

Same as [`record_run`](@ref) but `benchmarks` is already JSON-shaped (one `Dict` per row, keys
`category`/`method`/`framework`/`threads`/`time_ms`/`nrmse_gt`/`nrmse_mrt`). Used directly by
[`record_run`](@ref) and by `migrate_to_store.jl`, which passes rows parsed back out of the old
per-section JSON files and overrides `ts`/`source` to preserve when/where they actually ran instead
of stamping them as recorded now.
"""
function record_run_rows(
        section::AbstractString, backend::AbstractString, threads::Integer, benchmarks::AbstractVector;
        ts::AbstractString = Dates.format(now(), dateformat"yyyymmdd-HHMMSS-sss"),
        source::AbstractString = source_tag(), kwargs...,
    )
    mkpath(RUNS_DIR)
    run_id = "$(ts)_$(backend)_$(threads)threads_$(section)_$(source)_pid$(getpid())"
    path = joinpath(RUNS_DIR, "$run_id.json")
    ispath(path) && (path = joinpath(RUNS_DIR, "$(run_id)_$(rand(UInt32)).json"))
    meta = Dict{String, Any}(string(k) => v for (k, v) in kwargs)
    open(path, "w") do io
        JSON.print(
            io,
            merge(
                meta,
                Dict(
                    "run_id" => run_id, "ts" => ts, "source" => source, "section" => section,
                    "backend" => backend, "threads" => threads, "benchmarks" => benchmarks,
                ),
            ),
            2,
        )
    end
    return path
end

_json_num(x::Real) = isfinite(x) ? Float64(x) : -1.0
_json_num(x) = x  # already JSON-safe (e.g. read back from a migrated file)

"""
    run_files() -> Vector{String}

Every recorded run file, oldest first (filename timestamp order).
"""
run_files() = sort(filter(f -> endswith(f, ".json"), readdir(RUNS_DIR; join = true)))

end

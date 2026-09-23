# Flexible query layer over `results/runs/` (see `ResultsStore.jl`): loads every run file, keeps
# the latest row per (backend, threads, category, method, framework), and prints it. Plain Julia
# over the JSON rows -- no database, no merge step; a run file is never mutated, so this can be
# rerun any time without racing a writer.
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --source=slurm
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --backend=mkl --threads=16
using JSON, Printf

include(joinpath(@__DIR__, "..", "src", "ResultsStore.jl"))
using .ResultsStore: run_files

"""
    Row

One benchmark row, flattened with its run's metadata (backend, threads, source, ts).
"""
struct Row
    backend::String
    threads::Int
    category::String
    method::String
    framework::String
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_mrt::Float64
    source::String
    ts::String
end

function load_rows()
    rows = Row[]
    for f in run_files()
        d = try
            JSON.parsefile(f)
        catch e
            @warn "unreadable run file, skipping" f exception = e
            continue
        end
        backend = get(d, "backend", "")
        threads = get(d, "threads", 0)
        source = get(d, "source", "unknown")
        ts = get(d, "ts", "")
        for b in get(d, "benchmarks", [])
            push!(
                rows, Row(
                    backend, threads, b["category"], b["method"], b["framework"],
                    Float64(b["time_ms"]), Float64(b["nrmse_gt"]), Float64(b["nrmse_mrt"]), source, ts,
                )
            )
        end
    end
    return rows
end

"""
    latest_per_case(rows; prefer_source = nothing) -> Vector{Row}

One row per (backend, threads, category, method, framework): the most recent (`ts`), among rows
matching `prefer_source` if given, else preferring `"slurm"` over anything else and falling back to
whatever exists when no `"slurm"` row is present for that case.
"""
function latest_per_case(rows::Vector{Row}; prefer_source::Union{Nothing, AbstractString} = nothing)
    best = Dict{NTuple{5, Any}, Row}()
    for r in rows
        prefer_source !== nothing && r.source != prefer_source && continue
        key = (r.backend, r.threads, r.category, r.method, r.framework)
        cur = get(best, key, nothing)
        if cur === nothing
            best[key] = r
        else
            better = prefer_source !== nothing ? r.ts > cur.ts :
                (r.source == "slurm", r.ts) > (cur.source == "slurm", cur.ts)
            better && (best[key] = r)
        end
    end
    return collect(values(best))
end

const ARG = Dict(
    m.captures[1] => m.captures[2] for m in (match(r"^--(\w+)=(.*)$", a) for a in ARGS) if m !== nothing
)

rows = load_rows()
isempty(rows) && (@info "no rows in results/runs/ (run migrate_to_store.jl or a benchmark script first)"; exit(0))

haskey(ARG, "backend") && filter!(r -> r.backend == ARG["backend"], rows)
haskey(ARG, "threads") && filter!(r -> r.threads == parse(Int, ARG["threads"]), rows)
haskey(ARG, "category") && filter!(r -> r.category == ARG["category"], rows)
haskey(ARG, "method") && filter!(r -> r.method == ARG["method"], rows)

latest = latest_per_case(rows; prefer_source = get(ARG, "source", nothing))
isempty(latest) && (@info "no rows match the given filters"; exit(0))

fmt_ms(t) = (isnan(t) || t < 0) ? "-" : @sprintf("%.2f ms", t)
for backend in sort(unique(r.backend for r in latest))
    for threads in sort(unique(r.threads for r in latest if r.backend == backend))
        section_rows = filter(r -> r.backend == backend && r.threads == threads, latest)
        isempty(section_rows) && continue
        println("\n=== $backend $(threads)T ($(length(section_rows)) rows) ===")
        @printf("%-16s | %-30s | %-22s | %10s | %10s | %8s | %s\n", "category", "method", "framework", "time", "nrmse_gt", "source", "ts")
        for r in sort(section_rows; by = r -> (r.category, r.method, r.framework))
            @printf(
                "%-16s | %-30s | %-22s | %10s | %10.4f | %8s | %s\n",
                r.category, r.method, r.framework, fmt_ms(r.time_ms), r.nrmse_gt, r.source, r.ts
            )
        end
    end
end

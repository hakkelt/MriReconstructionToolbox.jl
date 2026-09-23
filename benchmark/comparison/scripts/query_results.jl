# Flexible query layer over `results/runs/` (see `ResultsStore.jl`): loads every run file, keeps
# the latest row per (backend, threads, category, method, framework), and prints it. Plain Julia
# over the JSON rows -- no database, no merge step; a run file is never mutated, so this can be
# rerun any time without racing a writer.
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --source=slurm
#   julia --project=benchmark/comparison benchmark/comparison/scripts/query_results.jl --backend=mkl --threads=16
using Printf

include(joinpath(@__DIR__, "..", "src", "ResultsStore.jl"))
using .ResultsStore: load_rows, latest_per_case

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

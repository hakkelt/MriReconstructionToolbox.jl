# Write the committed doc snapshot: `results/benchmark_<backend>_<n>threads.json`, one per
# (backend, threads) pair found in `results/runs/`, each holding the latest `source = "slurm"` row
# per (category, method, framework) -- the documentation's comparison page reads these, not
# `results/runs/` directly. This replaces `merge_benchmarks.jl`'s old role; there is still no merge
# of raw fragments, only a query (`ResultsStore.latest_per_case`) written out to a stable filename.
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl
#   julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl --backend=mkl --threads=16
using JSON

include(joinpath(@__DIR__, "..", "..", "utils", "results_store.jl"))
using .ResultsStore: load_rows, latest_per_case, RESULTS_DIR

const ARG = Dict(
    m.captures[1] => m.captures[2] for m in (match(r"^--(\w+)=(.*)$", a) for a in ARGS) if m !== nothing
)

rows = load_rows()
isempty(rows) && (@info "no rows in results/runs/ -- nothing to export"; exit(0))

haskey(ARG, "backend") && filter!(r -> r.backend == ARG["backend"], rows)
haskey(ARG, "threads") && filter!(r -> r.threads == parse(Int, ARG["threads"]), rows)

latest = latest_per_case(rows; prefer_source = "slurm")
isempty(latest) && (@info "no source=\"slurm\" rows match; nothing exported (rerun on the cluster first)"; exit(0))

for backend in sort(unique(r.backend for r in latest))
    for threads in sort(unique(r.threads for r in latest if r.backend == backend))
        section_rows = filter(r -> r.backend == backend && r.threads == threads, latest)
        isempty(section_rows) && continue
        path = joinpath(RESULTS_DIR, "benchmark_$(backend)_$(threads)threads.json")
        open(path, "w") do io
            JSON.print(
                io,
                Dict(
                    "backend" => backend, "threads" => threads,
                    "benchmarks" => [
                        Dict(
                            "category" => r.category, "method" => r.method, "framework" => r.framework,
                            "threads" => r.threads, "time_ms" => r.time_ms,
                            "nrmse_gt" => r.nrmse_gt, "nrmse_mrt" => r.nrmse_mrt,
                        ) for r in sort(section_rows; by = r -> (r.category, r.method, r.framework))
                    ],
                ),
                4,
            )
        end
        @info "exported" path n = length(section_rows)
    end
end

# One-shot migration: convert the old JSON layout (per-section fragments
# `benchmark_<backend>_<n>threads__<section>.json`, and the merged
# `benchmark_<backend>_<n>threads.json` files that used to be regenerated from them) into the new
# `results/runs/` store (see `ResultsStore.jl`), so existing measurements survive the switch
# without rerunning anything.
#
# Prefers fragments (they carry a `section` field and are the finer-grained, more recent source);
# falls back to a merged file for a (backend, threads) pair that has no fragments at all, tagging
# it `section = "legacy-merged"` since the rows may span several sections with no way to recover
# which. The original file's own `hostname` field (not the current machine) decides `source`, and
# its mtime becomes `ts`, so a login-node-contaminated old file migrates as `source = "other"` and
# keeps its place in time rather than jumping to "now".
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/migrate_to_store.jl [--dry-run]
using JSON, Dates

include(joinpath(@__DIR__, "..", "src", "ResultsStore.jl"))
using .ResultsStore: record_run_rows, source_tag

const DRY_RUN = "--dry-run" in ARGS
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))

function migrate_file(path; section, backend, threads)
    d = JSON.parsefile(path)
    haskey(d, "benchmarks") || (@warn "no benchmarks[] in $path, skipping"; return)
    isempty(d["benchmarks"]) && (@warn "empty benchmarks[] in $path, skipping"; return)
    hostname = get(d, "hostname", "unknown")
    src = source_tag(hostname)
    ts = Dates.format(unix2datetime(mtime(path)), dateformat"yyyymmdd-HHMMSS-sss")
    meta = Dict{Symbol, Any}(
        Symbol(k) => v for (k, v) in d if !(k in ("benchmarks", "hostname", "use_mkl", "section", "backend", "threads"))
    )
    meta[:hostname] = hostname
    meta[:use_mkl] = backend == "mkl"
    meta[:migrated_from] = basename(path)
    if DRY_RUN
        @info "would migrate" path section backend threads ts source = src n = length(d["benchmarks"])
    else
        out = record_run_rows(section, backend, threads, d["benchmarks"]; ts, source = src, meta...)
        @info "migrated" path "->" out
    end
end

# Fragments first: `benchmark_<backend>_<n>threads__<section>.json`
fragment_re = r"^benchmark_(openblas|mkl)_(\d+)threads__(\w+)\.json$"
covered = Set{Tuple{String, String}}()   # (backend, threads) that had >=1 fragment
for f in sort(readdir(RESULTS_DIR))
    m = match(fragment_re, f)
    m === nothing && continue
    backend, threads_str, section = m.captures
    push!(covered, (backend, threads_str))
    migrate_file(joinpath(RESULTS_DIR, f); section, backend, threads = parse(Int, threads_str))
end

# Merged files, only for (backend, threads) pairs with no fragments at all.
merged_re = r"^benchmark_(openblas|mkl)_(\d+)threads\.json$"
for f in sort(readdir(RESULTS_DIR))
    m = match(merged_re, f)
    m === nothing && continue
    backend, threads_str = m.captures
    (backend, threads_str) in covered && continue
    migrate_file(joinpath(RESULTS_DIR, f); section = "legacy-merged", backend, threads = parse(Int, threads_str))
end

@info "migration done" dry_run = DRY_RUN

# Stitch the per-section comparison fragments written by run_<section>.jl
#   comparison/results/benchmark_<backend>_<n>threads__<section>.json
# into the single file the docs / analysis read
#   comparison/results/benchmark_<backend>_<n>threads.json
#
#   julia --project=comparison comparison/scripts/merge_benchmarks.jl [backend] [threads]

using JSON

resdir = normpath(joinpath(@__DIR__, "..", "results"))
frags = filter(f -> occursin(r"^benchmark_(openblas|mkl)_\d+threads__.+\.json$", f), readdir(resdir))
isempty(frags) && (@info "no fragments to merge"; exit(0))

wb = length(ARGS) >= 1 ? ARGS[1] : nothing
wt = length(ARGS) >= 2 ? ARGS[2] : nothing

bykey = Dict{Tuple{String, String}, Vector{String}}()
for f in frags
    m = match(r"^benchmark_(openblas|mkl)_(\d+)threads__", f)
    k = (m.captures[1], m.captures[2])
    (wb === nothing || wb == k[1]) || continue
    (wt === nothing || wt == k[2]) || continue
    push!(get!(bykey, k, String[]), f)
end

const _SECTION_ORDER = ["base", "noncart", "cgsense", "sparsity", "dynamic", "kspace", "real", "accuracy_race"]
secrank(f) = (i = findfirst(s -> occursin("__$s.json", f), _SECTION_ORDER); i === nothing ? 99 : i)

for ((backend, threads), files) in bykey
    meta = nothing
    rows = Any[]
    seen = Dict{Tuple{String, String, String}, Int}()
    for f in sort(files; by = secrank)
        d = JSON.parsefile(joinpath(resdir, f))
        meta === nothing && (meta = d)
        for b in d["benchmarks"]
            key = (b["category"], b["method"], b["framework"])
            if haskey(seen, key)
                rows[seen[key]] = b
            else
                push!(rows, b)
                seen[key] = length(rows)
            end
        end
    end
    out = Dict(
        "hostname" => meta["hostname"], "julia_version" => meta["julia_version"],
        "julia_threads" => meta["julia_threads"], "blas_vendor" => meta["blas_vendor"],
        "use_mkl" => meta["use_mkl"], "bart_binary" => meta["bart_binary"],
        "pinned_cpus" => meta["pinned_cpus"], "bart_spawn_ms" => get(meta, "bart_spawn_ms", get(meta, "bart_startup_time_ms", nothing)),
        "benchmarks" => rows,
    )
    path = joinpath(resdir, "benchmark_$(backend)_$(threads)threads.json")
    open(path, "w") do io
        JSON.print(io, out, 4)
    end
    @info "merged" path nfrag = length(files) nrows = length(rows)
end

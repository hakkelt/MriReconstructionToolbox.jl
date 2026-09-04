# Stitch the per-group MRT benchmark fragments written by recon_bench.jl
#   benchmarking/results/mrt_<backend>_<n>threads__<groupsig>.json
# into the single file the comparison suite reads
#   benchmarking/results/mrt_<backend>_<n>threads.json
#
#   julia --project=benchmarking benchmarking/scripts/merge_results.jl [backend] [threads]
# with no args, merges every (backend, threads) pair found.

using JSON

resdir = joinpath(@__DIR__, "..", "results")
frags = filter(f -> occursin(r"^mrt_(openblas|mkl)_\d+threads__.+\.json$", f), readdir(resdir))
isempty(frags) && (@info "no fragments to merge"; exit(0))

want_backend = length(ARGS) >= 1 ? ARGS[1] : nothing
want_threads = length(ARGS) >= 2 ? ARGS[2] : nothing

groups = Dict{Tuple{String, String}, Vector{String}}()
for f in frags
    m = match(r"^mrt_(openblas|mkl)_(\d+)threads__", f)
    key = (m.captures[1], m.captures[2])
    (want_backend === nothing || want_backend == key[1]) || continue
    (want_threads === nothing || want_threads == key[2]) || continue
    push!(get!(groups, key, String[]), f)
end

for ((backend, threads), files) in groups
    benchmarks = Any[]
    meta = nothing
    for f in sort(files)
        d = JSON.parsefile(joinpath(resdir, f))
        meta === nothing && (meta = d)
        append!(benchmarks, d["benchmarks"])
    end
    # de-dupe on (category, method), last fragment wins
    seen = Dict{Tuple{String, String}, Int}()
    merged = Any[]
    for b in benchmarks
        k = (b["category"], b["method"])
        if haskey(seen, k)
            merged[seen[k]] = b
        else
            push!(merged, b)
            seen[k] = length(merged)
        end
    end
    out = Dict(
        "backend" => backend, "num_threads" => parse(Int, threads),
        "julia_version" => meta["julia_version"], "host" => meta["host"],
        "benchmarks" => merged,
    )
    path = joinpath(resdir, "mrt_$(backend)_$(threads)threads.json")
    open(path, "w") do io
        JSON.print(io, out, 2)
    end
    @info "merged" path nfrag = length(files) nrows = length(merged)
end

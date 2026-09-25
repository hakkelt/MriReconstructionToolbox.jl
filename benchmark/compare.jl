# Compare two refs measured by the MRT harness (benchmark/run.jl).
#
#   julia --project=benchmark benchmark/compare.jl <A> <B> [--threads=1,8] [--backend=openblas]
#                                                          [--cases=pat,...] [--methods=tv,...]
#                                                          [--placement=isolated|shared|any]
#                                                          [--pick=latest|min]
#
# `A` and `B` match a result's `--ref-name` label, or a prefix of its commit. For every
# (case, method, threads, backend, environment variant) measured under both, one result of each
# side is compared: B/A of the minimum time, each side's spread (median / minimum), and
# ΔNRMSE = NRMSE(B) - NRMSE(A). A |ΔNRMSE| above 1e-3 is flagged as a change in what is computed,
# not only in how fast; a ratio outside the larger spread plus 5% is flagged as a real change.
#
# --placement: which runs count. `isolated` (default) leaves out the runs that shared their NUMA
#   domain with other tasks (`matrix.sh --pack`), which are 1.2-4x slower on the memory-bound
#   cases; without this a packed run recorded later than an isolated one would silently be the
#   one compared. `shared` takes only packed runs, `any` both.
# --pick: which of several matching results is used. `latest` (default) takes the most recent;
#   `min` the fastest, for repeated measurements such as `matrix.sh --swap-repeat`, whose repeats
#   ran each ref on the other NUMA domain.

include(joinpath(@__DIR__, "utils", "results_store.jl"))
using .ResultsStore
using Printf, Statistics

length(ARGS) >= 2 && !startswith(ARGS[1], "--") && !startswith(ARGS[2], "--") ||
    error("usage: compare.jl <A> <B> [--threads=..] [--backend=..] [--cases=..] [--methods=..]")
const A, B = ARGS[1], ARGS[2]
_list(name) = (i = findlast(a -> startswith(a, "--$name="), ARGS); i === nothing ? nothing : String.(split(ARGS[i][(length(name) + 4):end], ",")))
const THREADS = let t = _list("threads")
    t === nothing ? nothing : parse.(Int, t)
end
const BACKENDS = _list("backend")
const CASES = _list("cases")
const METHODS = _list("methods")
const PLACEMENT = let p = _list("placement")
    p === nothing ? "isolated" : only(p)
end
PLACEMENT in ("isolated", "shared", "any") || error("--placement=$PLACEMENT: isolated, shared or any")
const PICK = let p = _list("pick")
    p === nothing ? "latest" : only(p)
end
PICK in ("latest", "min") || error("--pick=$PICK: latest or min")

matches(d, ref) = get(d, "ref_name", "") == ref || startswith(get(d, "git_commit", ""), ref)
is_shared(d) = endswith(get(d, "node_class", ""), "/ shared domain")
placement_ok(d) = PLACEMENT == "any" || (PLACEMENT == "shared") == is_shared(d)

# One ok row per key for one side: the latest, or with --pick=min the fastest.
function side(ref)
    best = Dict{Tuple, Tuple{Any, Dict, Dict}}()
    for d in ResultsStore.load_run_files(ResultsStore.HARNESS_RESULTS_DIR)
        matches(d, ref) || continue
        placement_ok(d) || continue
        THREADS === nothing || d["threads"] in THREADS || continue
        BACKENDS === nothing || d["backend"] in BACKENDS || continue
        for r in get(d, "benchmarks", [])
            get(r, "status", "") == "ok" || continue
            CASES === nothing || any(p -> occursin(lowercase(p), lowercase(r["case"])), CASES) || continue
            METHODS === nothing || r["method"] in METHODS || continue
            key = (r["case"], r["method"], d["threads"], d["backend"], get(d, "env_variant", ""), get(d, "small", false), get(d, "cine_frames", 0))
            rank = PICK == "latest" ? get(d, "ts", "") : -r["time_min_ms"]
            (!haskey(best, key) || best[key][1] < rank) && (best[key] = (rank, d, r))
        end
    end
    return best
end

const SA, SB = side(A), side(B)
isempty(SA) && error("no harness results for $A (placement $PLACEMENT)")
isempty(SB) && error("no harness results for $B (placement $PLACEMENT)")
keys_both = sort!(collect(intersect(keys(SA), keys(SB))))
isempty(keys_both) && error("$A and $B have no configuration in common")

nodes = unique([d["node_class"] for (_, d, _) in Iterators.flatten((values(SA), values(SB)))])
length(nodes) > 1 && @warn "the two sides ran on different node classes; ratios mix hardware" nodes
for (label, S) in ((A, SA), (B, SB))
    commits = unique(first(get(d, "git_commit", ""), 12) for (_, d, _) in values(S))
    length(commits) > 1 && @warn "$label matches results from several commits; the $(PICK == "min" ? "fastest" : "latest") per configuration is used" commits
end

@printf("%-38s %-8s %3s %-8s %11s %11s %7s %6s %6s %9s  %s\n", "case", "method", "thr", "backend", "$A ms", "$B ms", "B/A", "sprA", "sprB", "ΔNRMSE", "")
ratios = Float64[]
flagged = 0
for k in keys_both
    (_, _, ra), (_, _, rb) = SA[k], SB[k]
    ta, tb = ra["time_min_ms"], rb["time_min_ms"]
    spa, spb = ra["time_median_ms"] / ta, rb["time_median_ms"] / tb
    ratio = tb / ta
    dn = rb["nrmse"] - ra["nrmse"]
    push!(ratios, ratio)
    tol = max(spa, spb) - 1 + 0.05
    flag = String[]
    abs(dn) > 1.0e-3 && push!(flag, "NRMSE CHANGED")
    ratio > 1 + tol && push!(flag, "slower")
    ratio < 1 - tol && push!(flag, "faster")
    any(==("NRMSE CHANGED"), flag) || any(==("slower"), flag) ? (global flagged += 1) : nothing
    variant = isempty(k[5]) ? "" : " [$(k[5])]"
    @printf(
        "%-38s %-8s %3d %-8s %11.1f %11.1f %7.3f %6.2f %6.2f %+9.1e  %s\n",
        k[1] * variant, k[2], k[3], k[4], ta, tb, ratio, spa, spb, dn, join(flag, ", ")
    )
end
@printf("\n%d configurations, geometric-mean B/A %.3f, %d flagged (slower or NRMSE changed)\n", length(ratios), exp(mean(log.(ratios))), flagged)

# Compare the MRT (CUDA) rows that `benchmark/slurm/gpu_refs.sh comparison` measured for two refs.
#
#   julia --project=benchmark benchmark/compare_gpu.jl <dir> <A> <B> [--slower=1.05]
#
# `dir` is the job's output directory (`benchmark/results/slurm/gpu_refs_<job>/`), holding one
# subdirectory of comparison run files per ref. For every (case, category, method) row measured
# under both refs, the minimum time over the rounds of each side is compared: B/A, and
# ΔNRMSE = NRMSE(B) - NRMSE(A) against the ground truth. A ratio above `--slower` is flagged
# `slower`, below its inverse `faster`, and a |ΔNRMSE| above 1e-3 `NRMSE CHANGED`.

include(joinpath(@__DIR__, "utils", "results_store.jl"))
using .ResultsStore
using Printf, Statistics

length(ARGS) >= 3 || error("usage: compare_gpu.jl <dir> <A> <B> [--slower=1.05]")
const DIR, A, B = ARGS[1], ARGS[2], ARGS[3]
const SLOWER = let i = findlast(a -> startswith(a, "--slower="), ARGS)
    i === nothing ? 1.05 : parse(Float64, ARGS[i][(length("--slower=") + 1):end])
end

# Minimum time and the NRMSE of that run, per row key, over every run file of one ref.
function side(ref)
    best = Dict{Tuple{String, String, String}, Tuple{Float64, Float64}}()
    for d in ResultsStore.load_run_files(joinpath(DIR, ref))
        for r in get(d, "benchmarks", [])
            r["framework"] == "MRT" || continue
            t = Float64(r["time_ms"])
            t > 0 || continue
            k = (r["case_id"], r["category"], r["method"])
            (!haskey(best, k) || t < best[k][1]) && (best[k] = (t, Float64(r["nrmse_gt"])))
        end
    end
    return best
end

const SA, SB = side(A), side(B)
keys_ = sort!(collect(intersect(keys(SA), keys(SB))))
isempty(keys_) && error("no row measured under both $A and $B in $DIR")
@printf("%-38s %-14s %-26s %10s %10s %7s %10s\n", "case", "category", "method", "$A [ms]", "$B [ms]", "B/A", "ΔNRMSE")
ratios = Float64[]
for k in keys_
    (ta, ea), (tb, eb) = SA[k], SB[k]
    q = tb / ta
    push!(ratios, q)
    flag = q > SLOWER ? "  slower" : q < 1 / SLOWER ? "  faster" : ""
    abs(eb - ea) > 1.0e-3 && (flag *= "  NRMSE CHANGED")
    @printf("%-38s %-14s %-26s %10.1f %10.1f %7.3f %+10.1e%s\n", k..., ta, tb, q, eb - ea, flag)
end
@printf("geomean B/A over %d rows: %.3f\n", length(ratios), exp(mean(log.(ratios))))
for (name, s, o) in ((A, SA, SB), (B, SB, SA))
    only_here = setdiff(keys(s), keys(o))
    isempty(only_here) || println("only under $name: ", join(sort!(collect(only_here)), ", "))
end

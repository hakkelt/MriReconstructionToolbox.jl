module ReconBench

# MRT-only reconstruction timings, split into independently runnable groups. The phantoms,
# masks and iteration counts match the MRT rows of `benchmark/comparison/scripts/`, so a group's JSON can
# be consumed there directly as the baseline column.
#
# Groups (`GROUPS`): synthetic — `base`, `sparsity`, `dynamic`, `kspace`;
#                    real scanner data — `real`, `real3d`, `realdyn` (need MRITestData downloads).
# `recon_bench.jl --groups=a,b,c` runs a subset; `merge_results.jl` stitches the per-group JSON
# back into the single `mrt_<backend>_<n>threads.json` the comparison suite reads.

using LinearAlgebra
using Random: MersenneTwister
using NamedDims
using NamedDims: unname, dimnames
using MriReconstructionToolbox

include(joinpath(@__DIR__, "Phantoms.jl"))
using .Phantoms
include(joinpath(@__DIR__, "RealData.jl"))
using .RealData

include(joinpath(@__DIR__, "cases", "synthetic.jl"))
include(joinpath(@__DIR__, "cases", "real.jl"))

export GROUPS, SYNTHETIC_GROUPS, REAL_GROUPS, build_cases, build_group, run_cases

const SYNTHETIC_GROUPS = ("base", "sparsity", "dynamic", "kspace")
const REAL_GROUPS = ("real", "real3d", "realdyn")
const GROUPS = (SYNTHETIC_GROUPS..., REAL_GROUPS...)

const _BUILDERS = Dict{String, Function}(
    "base" => build_base,
    "sparsity" => build_sparsity,
    "dynamic" => build_dynamic,
    "kspace" => build_kspace,
    "real" => build_real,
    "real3d" => build_real3d,
    "realdyn" => build_realdyn,
)

nrmse(x, xref) = norm(vec(x) .- vec(xref)) / norm(vec(xref))

# Compare on magnitude, scale-aligned. Synthetic references are real non-negative and their
# recons come out essentially real (so `abs` is a no-op); real scanner data carries a spatially
# varying receive phase, so its complex recon vs the magnitude reference needs `abs` on both to
# avoid a spurious NRMSE ≈ √2.
function aligned_nrmse(est, ref)
    a = abs.(est)
    r = abs.(ref)
    return nrmse(a .* (norm(r) / norm(a)), r)
end

"""
    build_group(name) -> Vector{NamedTuple}

Build the cases for one group (`in(GROUPS)`). Unknown name throws.
"""
function build_group(name::AbstractString)
    haskey(_BUILDERS, name) || error("unknown group ", repr(name), "; known: ", join(GROUPS, ", "))
    return _BUILDERS[name]()
end

"""
    build_cases(groups = SYNTHETIC_GROUPS) -> Vector{NamedTuple}

Concatenate the cases for several groups. `groups` is an iterable of names or a comma-separated
string; `"all"` expands to [`GROUPS`](@ref), `"synthetic"` / `"real"` to the respective subsets.
"""
function build_cases(groups = SYNTHETIC_GROUPS)
    names = _expand_groups(groups)
    cases = NamedTuple[]
    for g in names
        append!(cases, build_group(g))
    end
    return cases
end

function _expand_groups(groups)
    groups isa AbstractString && (groups = split(groups, ',', keepempty = false))
    out = String[]
    for g in groups
        g = strip(String(g))
        if g == "all"
            append!(out, GROUPS)
        elseif g == "synthetic"
            append!(out, SYNTHETIC_GROUPS)
        elseif g == "real"
            append!(out, REAL_GROUPS)
        else
            push!(out, g)
        end
    end
    return unique(out)
end

"""
    run_cases(cases; num_runs = 3) -> Vector{NamedTuple}

Warm up each case once, then time `num_runs` more; report `time_ms` (minimum) and the aligned
NRMSE against the case reference.
"""
function run_cases(cases; num_runs = 3)
    out = NamedTuple[]
    for c in cases
        res = c.run()
        ts = Float64[]
        for _ in 1:num_runs
            t0 = time_ns()
            res = c.run()
            push!(ts, (time_ns() - t0) / 1.0e9)
        end
        img = res isa DecomposedImage ? total_image(res) : res
        e = aligned_nrmse(img, c.reference)
        push!(
            out,
            (category = c.category, method = c.method, time_ms = minimum(ts) * 1000, nrmse_gt = e),
        )
        println(rpad(string(c.category, " / ", c.method), 42), " ", round(minimum(ts) * 1000; digits = 2), " ms   nrmse=", round(e; digits = 5))
    end
    return out
end

end

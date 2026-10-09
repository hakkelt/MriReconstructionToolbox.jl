# The pull-request benchmark suite: small synthetic cases of the harness catalog
# (`benchmark/utils/`), at `RISTRETTO_BENCH_SMALL` sizes, so a run of both revisions at two thread
# counts fits a shared CI runner.
#
#     SUITE["solve"][case][method]                       reconstruct, fixed iteration count
#     SUITE["operator"][case]["forward"|"adjoint"|"normal"]  mul! with the encoding operator
#     SUITE["setup"][case]["operator"|"opnorm"]          building it, estimating its norm
#
# `NRMSE` holds the magnitude NRMSE against the ground truth of every solve, and `FAILED` the
# entries whose first call threw (the head's suite may use what the base revision lacks); neither
# is registered in `SUITE`. `compare.jl` runs this file once per revision and thread count.

ENV["RISTRETTO_BENCH_SMALL"] = "1"

using BenchmarkTools
using LinearAlgebra: mul!

include(joinpath(@__DIR__, "..", "utils", "bench_utils.jl"))
using .BenchUtils
using Ristretto
using Ristretto: AbstractOperators

const SUITE = BenchmarkGroup()
const NRMSE = Dict{String, Float64}()
const FAILED = Dict{String, String}()

"""
    SOLVES

`case => methods` timed end to end: one case per family, the first-order (FISTA), ADMM, PDHG and
Krylov paths, and the low-rank cine penalties.
"""
const SOLVES = (
    "shepp_logan_2d_1ch_cartesian" => (:wavelet,),
    "shepp_logan_2d_8ch_cartesian" => (:cgsense, :tv, :tv_pd),
    "shepp_logan_2d_8ch_radial" => (:cgsense, :tv),
    "shepp_logan_multislice_8ch_cartesian" => (:tv,),
    "shepp_logan_3d_8ch_cartesian" => (:wavelet,),
    "torso_cine_8ch_cartesian" => (:lowrank, :llr),
)

"""
    OPERATOR_CASES

Cases whose encoding operator is timed on its own.
"""
const OPERATOR_CASES = (
    "shepp_logan_2d_8ch_cartesian",
    "shepp_logan_2d_8ch_radial",
    "shepp_logan_multislice_8ch_cartesian",
    "shepp_logan_3d_8ch_cartesian",
    "torso_cine_8ch_cartesian",
)

# Register `b` under `path` if `probe` (one call of the timed work) runs; record the error
# otherwise. The probe also compiles everything the timed call needs.
function register!(probe, path::Vector{String}, b)
    key = join(path, "/")
    try
        probe()
    catch err
        FAILED[key] = sprint(showerror, err; context = :limit => true)
        @warn "Skipping $key" exception = err
        return nothing
    end
    g = SUITE
    for p in path[1:(end - 1)]
        haskey(g, p) || (g[p] = BenchmarkGroup())
        g = g[p]
    end
    g[path[end]] = b
    return nothing
end

for (id, methods) in SOLVES
    c = get_case(id)
    for m in methods
        key = "solve/$id/$m"
        f = try
            ristretto_reconstructor(c, m)
        catch err
            FAILED[key] = sprint(showerror, err; context = :limit => true)
            continue
        end
        register!(["solve", id, string(m)], @benchmarkable $f()) do
            NRMSE[key] = mag_nrmse(Array(parent(f())), c.reference)
        end
    end
end

for id in OPERATOR_CASES
    c = get_case(id)
    acq = ristretto_acquisition(c)
    𝒜 = try
        Ristretto.get_encoding_operator(acq)
    catch err
        FAILED["operator/$id"] = sprint(showerror, err; context = :limit => true)
        continue
    end
    x = AbstractOperators.allocate_in_domain(𝒜)
    y = AbstractOperators.allocate_in_codomain(𝒜)
    x .= randn.(eltype(x))
    y .= randn.(eltype(y))
    𝒩 = 𝒜' * 𝒜
    x2 = similar(x)
    register!(() -> mul!(y, 𝒜, x), ["operator", id, "forward"], @benchmarkable mul!($y, $𝒜, $x))
    register!(() -> mul!(x, 𝒜', y), ["operator", id, "adjoint"], @benchmarkable mul!($x, $(𝒜'), $y))
    register!(() -> mul!(x2, 𝒩, x), ["operator", id, "normal"], @benchmarkable mul!($x2, $𝒩, $x))
    register!(() -> Ristretto.get_encoding_operator(acq), ["setup", id, "operator"], @benchmarkable Ristretto.get_encoding_operator($acq))
    opnorm = isdefined(Ristretto, :_encoding_opnorm) ? Ristretto._encoding_opnorm : AbstractOperators.estimate_opnorm
    register!(() -> opnorm(𝒜), ["setup", id, "opnorm"], @benchmarkable $opnorm($𝒜))
end

# One second per entry, one evaluation per sample: a solve is milliseconds to a few hundred, and
# the whole suite runs four times (two revisions, two thread counts) inside the CI budget.
for (_, b) in BenchmarkTools.leaves(SUITE)
    b.params.seconds = 1.0
    b.params.samples = 10_000
    b.params.evals = 1
end

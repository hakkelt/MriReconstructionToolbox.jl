module ComparisonHarness

using LinearAlgebra: norm
using Test: @test

include("bart_bridge.jl")
include("sigpy_bridge.jl")
include("matlab_bridge.jl")
include("mirt_bridge.jl")
# Phantom generators live in benchmark/hpc/ (single source of truth: the MRT baseline the
# comparison suite diffs against is measured there on exactly these phantoms).
include(joinpath(@__DIR__, "..", "..", "hpc", "src", "Phantoms.jl"))
# Real scanner k-space via MRITestData.jl — shared with benchmark/hpc/ (single source of truth).
include(joinpath(@__DIR__, "..", "..", "hpc", "src", "RealData.jl"))

using .BARTBridge
using .SigPyBridge
using .MATLABBridge
using .MIRTBridge
using .Phantoms
using .RealData

export run_bart
export sigpy, np, sigpy_mri_app
export setup_matlab_paths
export MIRT
export generate_multicoil_brain, generate_dynamic_multicoil_brain
export load_real_case, real_data_available, real_data_source
export nrmse, check_nrmse

"""
    nrmse(x, xref)

Normalized root-mean-square error `‖x - xref‖ / ‖xref‖` (Frobenius over all entries).
"""
nrmse(x, xref) = norm(vec(x) .- vec(xref)) / norm(vec(xref))

"""
    check_nrmse(est, ref, tol; label)

Magnitude-align `est` to `ref` (by `‖|ref|‖ / ‖|est|‖`, absorbing the arbitrary global scale
between reconstructions), compute the [`nrmse`](@ref), log it under `label`, and assert it is
below `tol`. Returns the NRMSE.
"""
function check_nrmse(est, ref, tol; label)
    e = nrmse(est .* (norm(abs.(ref)) / norm(abs.(est))), ref)
    @info "$label NRMSE: $e"
    @test e < tol
    return e
end

end

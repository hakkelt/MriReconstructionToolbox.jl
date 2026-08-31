module ComparisonHarness

using LinearAlgebra: norm
using Test: @test

include("bart_bridge.jl")
include("sigpy_bridge.jl")
include("matlab_bridge.jl")
include("mirt_bridge.jl")
include("phantoms.jl")

using .BARTBridge
using .SigPyBridge
using .MATLABBridge
using .MIRTBridge
using .Phantoms

export run_bart
export sigpy, np, sigpy_mri_app
export setup_matlab_paths
export MIRT
export generate_multicoil_brain, generate_dynamic_multicoil_brain
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

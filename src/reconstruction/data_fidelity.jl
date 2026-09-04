"""
	DataFidelity

Abstract supertype for data consistency loss terms.
"""
abstract type DataFidelity end

"""
	L2Loss <: DataFidelity

Standard squared Euclidean loss: ‖𝒜x - y‖₂².
"""
struct L2Loss <: DataFidelity end

"""
	HardConsistency(; inner_maxit = 50, inner_tol = 1e-6) <: DataFidelity

Hard data consistency indicator constraint: {x | 𝒜x = y}.
Projections onto the constraint are evaluated via `HardConsistencyProx`. When `is_AAc_diagonal(𝒜)`
is true (single-coil Cartesian, or a `KSpaceToImage` signal model), projection is evaluated directly in closed form;
otherwise an inner Conjugate Gradient solver with maximum iterations `inner_maxit` and relative
tolerance `inner_tol` is used.
"""
Base.@kwdef struct HardConsistency{R <: Real} <: DataFidelity
    inner_maxit::Int = 50
    inner_tol::R = 1.0e-6
end

"""
	NoFidelity <: DataFidelity

No data consistency term (e.g. for pure regularization or custom models).
"""
struct NoFidelity <: DataFidelity end

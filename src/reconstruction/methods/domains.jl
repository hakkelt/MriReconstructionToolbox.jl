"""
	CoilCombination

Abstract supertype for multi-coil combination strategies.
"""
abstract type CoilCombination end

"""
	AdjointSensitivity <: CoilCombination

Combines multi-coil data using the adjoint sensitivity encoding operator (sensitivity-weighted sum).
"""
struct AdjointSensitivity <: CoilCombination end

"""
	RootSumSquares <: CoilCombination

Combines multi-coil data using root sum of squares across coil channels.
"""
struct RootSumSquares <: CoilCombination end

"""
	NoCoilCombination <: CoilCombination

Leaves individual coil channels uncombined.
"""
struct NoCoilCombination <: CoilCombination end

"""
	KSpaceToImage(coil_combination = RootSumSquares())

Signal model for a k-space-domain reconstruction: the optimization variable is the full
multi-channel k-space, the encoding operator during the solve is just the subsampling operator
`𝒫`, and the recovered k-space is mapped to an image afterwards by an inverse Fourier transform
followed by `coil_combination` (see `_kspace_to_image`). Used by `SPIRiT(; iterative = true)`.
"""
struct KSpaceToImage{C <: CoilCombination}
    coil_combination::C
end

KSpaceToImage() = KSpaceToImage(RootSumSquares())

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

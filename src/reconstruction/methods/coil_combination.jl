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

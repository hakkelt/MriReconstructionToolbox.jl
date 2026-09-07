"""
	DirectReconstruction{C<:CoilCombination} <: DirectMethod

Direct non-iterative reconstruction (e.g. adjoint encoding 𝒜'y or gridding).

# Fields
- `coil_combination::C`: Coil combination strategy (default `AdjointSensitivity()`).
"""
struct DirectReconstruction{C <: CoilCombination} <: DirectMethod
    coil_combination::C
end

DirectReconstruction(; coil_combination::CoilCombination = AdjointSensitivity()) =
    DirectReconstruction(coil_combination)

"""
	DirectReconstruction{C} <: DirectMethod

Direct non-iterative reconstruction (e.g. adjoint encoding 𝒜'y or gridding).

# Fields
- `coil_combination::C`: Coil combination strategy. The default, `nothing`, resolves against the
  acquisition (see [`lower`](@ref)): `AdjointSensitivity()` when it carries sensitivity maps, and
  `NoCoilCombination()` when it does not, because without maps the coil axis is a batch dimension
  and there is nothing to combine. Naming `AdjointSensitivity()` or `RootSumSquares()` explicitly
  on a map-less acquisition is an error rather than a silently uncombined result.
"""
struct DirectReconstruction{C <: Union{Nothing, CoilCombination}} <: DirectMethod
    coil_combination::C
end

DirectReconstruction(; coil_combination::Union{Nothing, CoilCombination} = nothing) =
    DirectReconstruction(coil_combination)

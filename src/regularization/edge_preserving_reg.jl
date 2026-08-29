"""
	EdgePreservingRoughness2D(λ; δ=0.01)

Create an edge-preserving roughness penalty for 2D images: `λ ∑_voxels ∑_d ψ_δ(∂_d x)`, where `ψ_δ` is the
Huber potential

	ψ_δ(t) = t²/(2δ)      if |t| ≤ δ
	ψ_δ(t) = |t| − δ/2    otherwise,

applied to every first-order finite difference along the two spatial dimensions.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `δ`: (optional) Width of the quadratic region, i.e. the gradient magnitude above which a difference counts
  as an edge. Should be set relative to the image intensity scale — a good starting point is a small
  percentile of the gradient magnitudes of a preliminary reconstruction. Default `0.01`.

# Notes
- This is the classical edge-preserving roughness penalty of statistical image reconstruction (Fessler,
  *Statistical image reconstruction methods for transmission tomography*, Handbook of Medical Imaging 2000;
  Charbonnier et al., *Deterministic edge-preserving regularization in computed imaging*, IEEE TIP 1997). It
  interpolates between the two terms this package already offers: as `δ → ∞` it approaches
  `λ/(2δ)‖∇x‖²`, a quadratic roughness penalty in the spirit of [`Tikhonov`](@ref), and as `δ → 0` it
  approaches the anisotropic total variation `λ‖∇x‖₁`. Small differences are therefore penalized
  quadratically (which suppresses noise without the amplitude bias of an ℓ₁ term) while large ones are
  penalized only linearly (which preserves edges).
- Unlike [`TotalVariation2D`](@ref), the penalty is differentiable everywhere, so gradient-based algorithms
  can use it directly instead of taking a proximal step. That also means it does not produce the exactly
  piecewise-constant, staircased solutions of TV.
- The Huber potential is applied to each directional difference separately (an anisotropic penalty), matching
  the usual formulation of roughness penalties; [`TotalVariation2D`](@ref) instead couples the directions
  through a per-voxel ℓ₂ norm.
"""
struct EdgePreservingRoughness2D{T, S} <: Regularization
    λ::T
    δ::S
    function EdgePreservingRoughness2D(λ::T; δ::S = 0.01) where {T, S}
        @argcheck λ isa Real "EdgePreservingRoughness2D requires a scalar λ"
        @argcheck λ >= 0 "λ must be non-negative"
        @argcheck δ > 0 "δ must be positive"
        return new{T, S}(λ, δ)
    end
end

"""
	EdgePreservingRoughness3D(λ; δ=0.01)

Create an edge-preserving roughness penalty for 3D images: the Huber potential applied to every first-order
finite difference along the three spatial dimensions. See [`EdgePreservingRoughness2D`](@ref) for details.
"""
struct EdgePreservingRoughness3D{T, S} <: Regularization
    λ::T
    δ::S
    function EdgePreservingRoughness3D(λ::T; δ::S = 0.01) where {T, S}
        @argcheck λ isa Real "EdgePreservingRoughness3D requires a scalar λ"
        @argcheck λ >= 0 "λ must be non-negative"
        @argcheck δ > 0 "δ must be positive"
        return new{T, S}(λ, δ)
    end
end

# The gradient operator is the same one total variation uses; only the potential applied to it differs.
_epr_gradient(reg::EdgePreservingRoughness2D) = TotalVariation2D(reg.λ)
_epr_gradient(reg::EdgePreservingRoughness3D) = TotalVariation3D(reg.λ)

function get_operator(
        reg::Union{EdgePreservingRoughness2D, EdgePreservingRoughness3D}, x::AbstractArray; threaded::Bool = true
    )
    return get_operator(_epr_gradient(reg), x; threaded)
end

get_affected_dims(::EdgePreservingRoughness2D, ::AcquisitionInfo, image_dims) = image_dims[1:2]
get_affected_dims(::EdgePreservingRoughness3D, ::AcquisitionInfo, image_dims) = image_dims[1:3]

# ψ_δ is not homogeneous, because δ is an absolute intensity threshold rather than a weight. It obeys
# `ψ_{cδ}(c t) = c ψ_δ(t)`, so an image scaled by `factor` needs δ scaled by the same factor, and λ then
# scales linearly exactly as it does for the ℓ₁-type terms (see the scale_regularization docstring).
function scale_regularization(reg::EdgePreservingRoughness2D, factor::Real)
    return EdgePreservingRoughness2D(reg.λ * factor; δ = reg.δ * factor)
end
function scale_regularization(reg::EdgePreservingRoughness3D, factor::Real)
    return EdgePreservingRoughness3D(reg.λ * factor; δ = reg.δ * factor)
end

function materialize(
        reg::Union{EdgePreservingRoughness2D, EdgePreservingRoughness3D}, x::Variable{T}; threaded::Bool
    ) where {T}
    ∇ = get_operator(reg, ~x; threaded)
    R = real(T)
    λ, δ = R(reg.λ), R(reg.δ)
    # SeparableHuberLoss(ρ, μ) is `μ/2 t²` below ρ and `ρμ(|t| − ρ/2)` above it, so ρ = δ and μ = λ/δ give
    # exactly `λ ψ_δ`. At λ = 0 that is the zero function, but SeparableHuberLoss rejects μ = 0, so
    # spell it out -- the constructor accepts λ = 0 and every other term in the package treats it as a
    # disabled no-op rather than an error.
    f = iszero(λ) ? ProximalOperators.Zero() : SeparableHuberLoss(δ, λ / δ)
    repr = @sprintf "%g ⋅ ∑ ψ_%g(∇%s)" λ δ get_name(x)
    return StructuredOptimization.Term(1, f, ∇ * x, repr)
end

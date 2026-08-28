"""
	ReferencePrior(λ, reference)

Create a reference-image (prior-image constrained) regularization term with parameter `λ`. The regularization
term is given by `λ‖x - x_ref‖₁`, i.e. sparsity is promoted in the *difference* to a reference image instead
of in the image itself.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `reference`: The reference image `x_ref`, an array of the same size as `x`.

# Notes
- This is the MRI analogue of the prior-image constrained compressed sensing (PICCS) idea of
  Chen, Tang & Leng, *Prior image constrained compressed sensing (PICCS)*, Med Phys 2008. It is useful when
  a high-quality image of the same anatomy is already available: a previous time frame or contrast, a fully
  sampled pre-contrast scan, or a temporal average of a dynamic series.
- The reference must be in the same units and scaling as the reconstructed image. When automatic data
  scaling is used (see [`BartScaling`](@ref) and friends), the reference is rescaled together with the
  regularization parameter.
- Because the reference has the size of the full image, this term blocks
  [problem decomposition](../high-level/decomposition.md) over batch dimensions.
- To combine a reference prior with ordinary sparsity, add both terms
  (e.g. `(ReferencePrior(λ₁, x_ref), L1Wavelet2D(λ₂))`), which reproduces the convex combination used by
  PICCS-style reconstructions.
"""
struct ReferencePrior{T, A <: AbstractArray} <: Regularization
    λ::T
    reference::A
end

get_operator(::ReferencePrior, x::AbstractArray; threaded::Bool = true) = Eye(x)
function get_operator(::ReferencePrior, x::NamedDimsArray; threaded::Bool = true)
    return NamedDimsOp{dimnames(x), dimnames(x)}(Eye(parent(x)))
end

# The penalty itself acts element-wise, but the reference image has the size of the *full* image, so the
# problem must not be split over batch dimensions (a slice subproblem would get a mismatched reference).
function get_affected_dims(::ReferencePrior, acq_info::AcquisitionInfo, image_dims)
    return Tuple(image_dims)
end

get_affected_dims(::ReferencePrior, ::Nothing, image_dims) = Tuple(image_dims)

# λ‖x - x_ref‖₁ is homogeneous of degree 1 in (x, x_ref) jointly, so scaling the variable by `factor`
# requires scaling both λ and the reference (see scale_regularization docstring).
function scale_regularization(reg::ReferencePrior, factor::Real)
    return ReferencePrior(reg.λ .* factor, reg.reference .* factor)
end

function materialize(reg::ReferencePrior, x::Variable{T}; threaded::Bool) where {T}
    @argcheck size(reg.reference) == size(~x) "reference must have the same size as the image variable"
    if reg.λ isa AbstractArray
        @argcheck size(reg.λ) == size(~x) "Incompatible sizes"
    end
    R = real(T)
    λ = R.(reg.λ)
    reference = T.(unname(reg.reference))
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "‖Γ .* ($(get_name(x)) - xᵣₑ)‖₁"
    else
        @sprintf "%g ⋅ ‖%s - xᵣₑ‖₁" λ get_name(x)
    end
    # `Translate(f, b)` evaluates `f(x + b)`, hence the negated reference.
    return StructuredOptimization.Term(1, Translate(NormL1(λ), -reference), op * x, repr)
end

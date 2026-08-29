"""
	SecondOrderTotalVariation2D(λ)

Create a second-order Total Variation regularization term for 2D images with parameter `λ`:
`λ‖∇²x‖_{2,1} = λ ∑_voxels ‖(∂ₓₓx, ∂ₓᵧx, ∂ᵧₓx, ∂ᵧᵧx)‖₂`, where the four second derivatives are obtained by
applying the finite-difference gradient twice.

# Arguments
- `λ`: Regularization parameter, must be a scalar.

# Notes
- First-order TV ([`TotalVariation2D`](@ref)) penalizes any deviation from a *piecewise constant* image and
  therefore produces staircasing on smooth intensity ramps — a well known artifact on, e.g., coil-shading or
  slow tissue transitions. Second-order TV penalizes deviations from a *piecewise linear* image instead, so
  ramps cost nothing and staircasing disappears; the price is that genuine edges are blurred more than by
  first-order TV, because a jump is now penalized through its (large) second derivative.
- In practice second-order TV is rarely used alone. Combining it with first-order TV, either as a plain sum
  or as the infimal convolution of the two (see the *Infimal convolution TV* section of the regularization
  documentation), keeps the edge preservation of first-order TV and the ramp fidelity of the second-order
  term; that combination is also the motivation behind [total generalized variation](@ref
  TotalGeneralizedVariation2D). See Chambolle & Lions, *Image recovery via total variation minimization and
  related problems*, Numer Math 1997, and Bredies, Kunisch & Pock, *Total generalized variation*,
  SIAM J Imaging Sci 2010.
- Dimensions beyond the first two are treated as batch dimensions: each 2D slice is regularized
  independently.
"""
struct SecondOrderTotalVariation2D{T} <: Regularization
    λ::T
end

"""
	SecondOrderTotalVariation3D(λ)

Create a second-order Total Variation regularization term for 3D images with parameter `λ`:
`λ‖∇²x‖_{2,1}`, the sum over voxels of the Euclidean norm of the nine second derivatives along the three
spatial dimensions. See [`SecondOrderTotalVariation2D`](@ref) for when to prefer second-order TV over
first-order TV.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
"""
struct SecondOrderTotalVariation3D{T} <: Regularization
    λ::T
end

_sotv_spatial_dims(::SecondOrderTotalVariation2D) = 2
_sotv_spatial_dims(::SecondOrderTotalVariation3D) = 3

_sotv_first_order(reg::SecondOrderTotalVariation2D) = TotalVariation2D(reg.λ)
_sotv_first_order(reg::SecondOrderTotalVariation3D) = TotalVariation3D(reg.λ)

"""
	get_operator(reg::SecondOrderTotalVariation2D, x; threaded=true)

Return the operator computing all second derivatives of `x`. The gradient operator is applied twice: the
first application appends a direction axis, and the second is batched over that axis (and over any
non-spatial dimension of `x`), so the result carries two direction axes as its two trailing dimensions.
"""
function get_operator(
        reg::Union{SecondOrderTotalVariation2D, SecondOrderTotalVariation3D}, x::AbstractArray; threaded::Bool = true
    )
    n_spatial = _sotv_spatial_dims(reg)
    @argcheck ndims(x) >= n_spatial "$(typeof(reg).name.name) requires at least $n_spatial dimensions in the input variable"
    # First derivative: (spatial..., batch...) -> (spatial..., batch..., direction)
    Δ = get_operator(_sotv_first_order(reg), x; threaded)
    inner = Δ isa NamedDimsOp ? parent(Δ) : Δ
    spatial_size = size(x)[1:n_spatial]
    # Everything the first gradient leaves after the spatial dimensions -- the original batch dimensions and
    # the direction axis it appended -- is a batch dimension for the second gradient.
    batch_size = size(inner, 1)[(n_spatial + 1):end]
    Δ2 = Variation(
        view(unname(x), ntuple(_ -> Colon(), n_spatial)..., (ones(Int, ndims(x) - n_spatial)...));
        threaded = threaded && isempty(batch_size)
    )
    if !isempty(batch_size)
        input_dims = (ntuple(_ -> :_, n_spatial)..., fill(:b, length(batch_size))...)
        image_dims = (:_, fill(:b, length(batch_size))..., :_)
        Δ2 = BatchOp(Δ2, batch_size, input_dims => image_dims; threaded)
    end
    Δ2 = reshape(Δ2, spatial_size..., size(Δ2, 1)[2:end]...)
    composed = Δ2 * inner
    if x isa NamedDimsArray
        output_dimnames = (
            dimnames(x)[1:n_spatial]..., dimnames(x)[(n_spatial + 1):end]...,
            :direction, :direction2,
        )
        composed = NamedDimsOp{dimnames(x), output_dimnames}(composed)
    end
    return composed
end

function get_affected_dims(reg::SecondOrderTotalVariation2D, ::Nothing, image_dims)
    return image_dims[1:2]
end

function get_affected_dims(reg::SecondOrderTotalVariation3D, ::Nothing, image_dims)
    return image_dims[1:3]
end

# λ‖∇²x‖_{2,1} is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::SecondOrderTotalVariation2D, factor::Real) = SecondOrderTotalVariation2D(reg.λ .* factor)
scale_regularization(reg::SecondOrderTotalVariation3D, factor::Real) = SecondOrderTotalVariation3D(reg.λ .* factor)

function materialize(
        reg::Union{SecondOrderTotalVariation2D, SecondOrderTotalVariation3D}, x::Variable{T}; threaded::Bool
    ) where {T}
    n_spatial = _sotv_spatial_dims(reg)
    Δ² = get_operator(reg, ~x; threaded)
    # The two trailing axes are the two direction axes, so collapsing them groups the `n_spatial^2` second
    # derivatives of one voxel into one row of the matrix `NormL21` takes the row-wise ℓ₂ norm of.
    Δ² = _collapse_direction_axes(Δ², ~x, n_spatial^2)
    λ = real(T)(reg.λ)
    repr = @sprintf "%g ⋅ ‖∇²%s‖₂,₁" λ get_name(x)
    return StructuredOptimization.Term(1, NormL21(λ, 2), Δ² * x, repr)
end

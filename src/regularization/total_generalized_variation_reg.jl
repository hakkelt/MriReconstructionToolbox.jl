"""
	TotalGeneralizedVariation2D(λ; ratio=2.0)
	TotalGeneralizedVariation3D(λ; ratio=2.0)

Create a second-order total generalized variation (TGV²) regularization term for 2D (resp. 3D) images:

	TGV(x) = min_w  λ‖∇x − w‖₂,₁ + λ ⋅ ratio ⋅ ‖ℰw‖₂,₁

where `w` is an auxiliary vector field and `ℰw = ½(∇w + ∇wᵀ)` is its symmetrized gradient. The minimization
over `w` is not carried out separately: `w` becomes a variable of the reconstruction problem and is solved
for jointly with the image (see [`materialize_with_auxiliaries`](@ref)).

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `ratio`: (optional) Weight of the second-order term relative to the first, i.e. `α₀/α₁` in the usual
  notation. The value `2.0` recommended by Knoll et al. is the default and works well across a wide range of
  images; the result is not very sensitive to it.

# Notes
- TGV is the standard answer to the staircasing artifact of total variation. Where
  [`TotalVariation2D`](@ref) forces the image towards piecewise constant and
  [`SecondOrderTotalVariation2D`](@ref) forces it towards piecewise linear (blurring genuine edges), TGV
  balances the two automatically: `w` takes over the smooth part of the gradient, so `∇x − w` stays small on
  ramps, while at an edge `w` cannot follow the jump and the first-order term acts exactly as TV does. It
  preserves edges *and* reproduces smooth intensity variation. Introduced by Bredies, Kunisch & Pock,
  *Total generalized variation*, SIAM J Imaging Sci 2010, and brought to MRI by Knoll, Bredies, Pock &
  Stollberger, *Second order total generalized variation (TGV) for MRI*, Magn Reson Med 2011.
- As `ratio → ∞` the auxiliary field is driven to a constant and TGV degenerates to plain total variation, so
  TV is a limiting case rather than a different model.
- The auxiliary field adds one vector per voxel, which is the price of the improved model: its memory cost is
  `N × length(image)` for an `N`-dimensional term. The symmetrized gradient has `N(N+1)/2` independent
  components (3 in 2D, 6 in 3D).
- Only splitting algorithms handle the coupled two-variable term set: use `ADMM`. The first-order term
  couples the two variables through `∇x − w`, which the proximal-gradient algorithms cannot separate.
  Convergence is sensitive to ADMM's penalty parameter on this term; if the adaptive default does not
  settle, pass a fixed `rho` (`ADMM(rho = 1.0)` is what the tests here use).
- Dimensions beyond the spatial ones are batch dimensions, regularized independently.
"""
struct TotalGeneralizedVariation2D{T, S} <: Regularization
    λ::T
    ratio::S
    function TotalGeneralizedVariation2D(λ::T; ratio::S = 2.0) where {T, S}
        @argcheck λ isa Real "TotalGeneralizedVariation2D requires a scalar λ"
        @argcheck λ >= 0 "λ must be non-negative"
        @argcheck ratio > 0 "ratio must be positive"
        return new{T, S}(λ, ratio)
    end
end

"""
	TotalGeneralizedVariation3D(λ; ratio=2.0)

The 3D member of the TGV family: identical to [`TotalGeneralizedVariation2D`](@ref) except that the
gradient, the auxiliary field and the symmetrized gradient run over three spatial dimensions instead of
two, so `w` has three components per voxel and `ℰw` has six. See
[`TotalGeneralizedVariation2D`](@ref) for the model, the arguments and the solver requirements.
"""
struct TotalGeneralizedVariation3D{T, S} <: Regularization
    λ::T
    ratio::S
    function TotalGeneralizedVariation3D(λ::T; ratio::S = 2.0) where {T, S}
        @argcheck λ isa Real "TotalGeneralizedVariation3D requires a scalar λ"
        @argcheck λ >= 0 "λ must be non-negative"
        @argcheck ratio > 0 "ratio must be positive"
        return new{T, S}(λ, ratio)
    end
end

const _TGV = Union{TotalGeneralizedVariation2D, TotalGeneralizedVariation3D}

# The spatial dimensionality is fixed by the type; everything below is written against it so that the 2D and
# 3D members share one implementation.
_tgv_ndims(::TotalGeneralizedVariation2D) = 2
_tgv_ndims(::TotalGeneralizedVariation3D) = 3

_tgv_gradient(reg::TotalGeneralizedVariation2D) = TotalVariation2D(reg.λ)
_tgv_gradient(reg::TotalGeneralizedVariation3D) = TotalVariation3D(reg.λ)

"""
	get_operator(reg::TotalGeneralizedVariation2D, x; threaded=true)

Return the gradient operator of the first-order part of the TGV term, collapsed to the
`(length(x), N)` shape the auxiliary field `w` lives in, `N` being the spatial dimensionality of the term.
The second-order part uses [`SymmetrizedVariation`](@ref), which is built in
[`materialize_with_auxiliaries`](@ref) because it acts on `w` rather than on `x`.
"""
function get_operator(reg::_TGV, x::AbstractArray; threaded::Bool = true)
    N = _tgv_ndims(reg)
    @argcheck ndims(x) >= N "$(typeof(reg).name.name) requires at least $N dimensions in the input variable"
    ∇ = get_operator(_tgv_gradient(reg), x; threaded)
    return _collapse_direction_axes(∇, x, N)
end

function get_affected_dims(reg::_TGV, ::Nothing, image_dims)
    return image_dims[1:_tgv_ndims(reg)]
end

# Both terms are homogeneous of degree 1 in (x, w) jointly, so λ scales linearly as for the ℓ₁-type terms.
function scale_regularization(reg::TotalGeneralizedVariation2D, factor::Real)
    return TotalGeneralizedVariation2D(reg.λ * factor; ratio = reg.ratio)
end

function scale_regularization(reg::TotalGeneralizedVariation3D, factor::Real)
    return TotalGeneralizedVariation3D(reg.λ * factor; ratio = reg.ratio)
end

function materialize(reg::_TGV, x::Variable; threaded::Bool)
    terms, _ = materialize_with_auxiliaries(reg, x; threaded)
    return terms
end

"""
	_tgv_symmetrized_operator(T, spatial_size, batch; threaded)

Return the symmetrized-gradient operator acting on the auxiliary field in its flat `(voxels, N)` layout,
`N = length(spatial_size)`.

The field is stored flat, exactly as `Variation` stores its output, so that `∇x − w` is a plain difference of
two matrices and `NormL21` (which is two-dimensional) can be applied to both terms. `SymmetrizedVariation`
however acts on one spatial grid at a time, so with batch dimensions present the flat layout has to be
unfolded into `(voxels_per_slice, N, batch)`, the operator applied per slice, and the result folded back.
Folding the batch into a spatial extent instead would be wrong: the differences would then run across slice
boundaries. The codomain has `N(N+1)/2` components — the independent entries of the symmetric matrix `ℰw`.
"""
function _tgv_symmetrized_operator(::Type{T}, spatial_size::NTuple{N, Int}, batch::Int; threaded::Bool) where {T, N}
    M = prod(spatial_size)
    C = N * (N + 1) ÷ 2
    ℰ = SymmetrizedVariation(T, spatial_size; threaded = threaded && batch == 1)
    # Without batch dimensions the flat layout is already what the operator wants.
    batch == 1 && return ℰ
    unfold = Reshape(Eye(T, (M * batch, N)), M, batch, N)
    to_slices = PermuteDims(T, (M, batch, N), (1, 3, 2))
    ℰ_batched = BatchOp(ℰ, (batch,), (:_, :_, :b) => (:_, :_, :b); threaded)
    from_slices = PermuteDims(T, (M, C, batch), (1, 3, 2))
    return Reshape(from_slices * ℰ_batched * to_slices * unfold, M * batch, C)
end

function materialize_with_auxiliaries(reg::TotalGeneralizedVariation2D, x::Variable; threaded::Bool)
    return _tgv_materialize(reg, x, Val(2); threaded)
end

function materialize_with_auxiliaries(reg::TotalGeneralizedVariation3D, x::Variable; threaded::Bool)
    return _tgv_materialize(reg, x, Val(3); threaded)
end

function _tgv_materialize(reg::_TGV, x::Variable{T}, ::Val{N}; threaded::Bool) where {T, N}
    x_val = ~x
    ∇ = get_operator(reg, x_val; threaded)
    ∇_inner = ∇ isa NamedDimsOp ? parent(∇) : ∇
    spatial_size = ntuple(d -> size(x_val, d), Val(N))
    # The auxiliary field has one vector per voxel of the *whole* array (batch dimensions included), laid out
    # the way `Variation` lays out its output, so that `∇x - w` is a plain difference of two matrices.
    w = Variable(zeros(T, length(x_val), N))
    batch = length(x_val) ÷ prod(spatial_size)
    ℰ = _tgv_symmetrized_operator(T, spatial_size, batch; threaded)
    R = real(T)
    λ = R(reg.λ)
    λ₀ = R(reg.λ * reg.ratio)
    first_order = StructuredOptimization.Term(
        1, NormL21(λ, 2), ∇_inner * x - w,
        @sprintf("%g ⋅ ‖∇%s − w‖₂,₁", λ, get_name(x))
    )
    second_order = StructuredOptimization.Term(
        1, NormL21(λ₀, 2), ℰ * w,
        @sprintf("%g ⋅ ‖Ɛw‖₂,₁", λ₀)
    )
    return first_order + second_order, (w,)
end

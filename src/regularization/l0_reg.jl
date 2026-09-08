"""
	_sparsifying_operator(::Val{domain}, wavelet, levels, x; threaded)

Return the operator that maps the image to the domain in which sparsity is enforced by the
`L0*` family ([`L0Image`](@ref), [`L0Wavelet2D`](@ref), [`L0Wavelet3D`](@ref)). `domain` is
`:image`, `:wavelet2d` or `:wavelet3d`.
"""
_sparsifying_operator(::Val{:image}, wavelet_type, levels::Int, x::AbstractArray; threaded::Bool) =
    identity_operator(x)
_sparsifying_operator(::Val{:wavelet2d}, wavelet_type, levels::Int, x::AbstractArray; threaded::Bool) =
    get_operator(L1Wavelet2D(1; wavelet = wavelet_type, levels), x; threaded)
_sparsifying_operator(::Val{:wavelet3d}, wavelet_type, levels::Int, x::AbstractArray; threaded::Bool) =
    get_operator(L1Wavelet3D(1; wavelet = wavelet_type, levels), x; threaded)

_sparsifying_affected_dims(::Val{:image}, image_dims) = ()
_sparsifying_affected_dims(::Val{:wavelet2d}, image_dims) = Tuple(image_dims[1:2])
_sparsifying_affected_dims(::Val{:wavelet3d}, image_dims) = Tuple(image_dims[1:3])

_sparsifying_repr(::Val{:image}) = ""
_sparsifying_repr(::Val{:wavelet2d}) = "𝒲"
_sparsifying_repr(::Val{:wavelet3d}) = "𝒲"

# Shared validation for the `L0*` family: `threshold` (penalty form) and `count` (constraint form)
# are mutually exclusive ways to specify the same ℓ₀ term, and exactly one must be given.
function _check_l0_args(threshold, count, name::String)
    @argcheck (threshold === nothing) != (count === nothing) "$name requires exactly one of `threshold` or `count` (they are mutually exclusive), got threshold=$(repr(threshold)), count=$(repr(count))"
    threshold !== nothing && @argcheck threshold >= 0 "threshold must be non-negative"
    count !== nothing && @argcheck count > 0 "count must be positive"
    return nothing
end

"""
	L0Image(; threshold=nothing, count=nothing)

Create an `ℓ₀` regularization term on the image domain directly (no transform). Exactly one of
`threshold` or `count` must be given.

# Arguments
- `threshold`: penalty form. The term is `λ‖x‖₀`, the number of non-zero voxels weighted by `λ = threshold`.
  Its proximal operator is hard thresholding: coefficients with magnitude below `sqrt(2γλ)` are set to
  zero and the surviving ones are left untouched.
- `count`: constraint form. The image is required to have at most `count` non-zero voxels. Its proximal
  operator keeps the `count` largest-magnitude voxels and zeroes the rest.

# Notes
- `threshold` and `count` are mutually exclusive; passing both, or neither, is an `ArgumentError`.
- Penalty form: hard thresholding does not shrink the coefficients it keeps, so unlike the `ℓ₁` terms
  ([`L1Image`](@ref), [`L1Wavelet2D`](@ref)) it introduces no amplitude bias in the retained features — at
  the price of a non-convex objective, for which the solvers only guarantee a stationary point. It is
  BART's `-R H`. The threshold is `sqrt(2γλ)`, not `γλ` as for the `ℓ₁` terms, so a `λ` transplanted from
  an `ℓ₁` term will not give a comparable amount of sparsity.
- Constraint form: the sparsity analogue of [`RankLimit`](@ref) — instead of tuning a penalty weight whose
  relation to the resulting sparsity is indirect, the sparsity itself is prescribed. It is the thresholding
  step of iterative hard thresholding (Blumensath & Davies, *Iterative hard thresholding for compressed
  sensing*, ACHA 2009). The count is over the entire coefficient array, including all batch dimensions;
  because a per-sub-problem budget would not be the same constraint, this term blocks task splitting —
  reconstruct slice by slice explicitly if a per-slice budget is what is wanted.
- Because the term is non-convex (either form), use it with a good starting point (e.g. the zero-filled
  reconstruction or an `ℓ₁` solution) and prefer `ISTA`/`FISTA`; results depend on the initialization.
"""
struct L0Image{T} <: Regularization
    threshold::Union{T, Nothing}
    count::Union{Int, Nothing}
    function L0Image(; threshold::Union{Real, Nothing} = nothing, count::Union{Integer, Nothing} = nothing)
        _check_l0_args(threshold, count, "L0Image")
        T = threshold === nothing ? Float64 : typeof(threshold)
        return new{T}(threshold, count === nothing ? nothing : Int(count))
    end
end

function get_operator(reg::L0Image, x::AbstractArray; threaded::Bool = true)
    return _sparsifying_operator(Val(:image), nothing, 0, x; threaded)
end

function get_affected_dims(reg::L0Image, ::Nothing, image_dims)
    # A count budget couples every voxel of the coefficient array (see the docstring), so it blocks task
    # splitting regardless of domain; a threshold penalty is separable and only affects the domain's dims.
    return reg.count !== nothing ? Tuple(image_dims) : _sparsifying_affected_dims(Val(:image), image_dims)
end

# ‖·‖₀ is homogeneous of degree 0, so with `k=1, p=0` the docstring's rule gives λ_eff = λ ⋅ factor² for the
# threshold form. The count form is scale-invariant (nnz(factor * x) == nnz(x)); no correction needed.
function scale_regularization(reg::L0Image, factor::Real)
    reg.threshold === nothing && return reg
    return L0Image(threshold = reg.threshold * factor^2)
end

function materialize(reg::L0Image, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    if reg.threshold !== nothing
        λ = real(T)(reg.threshold)
        repr = @sprintf "%g ⋅ ‖%s‖₀" λ get_name(x)
        return StructuredOptimization.Term(1, NormL0(λ), op * x, repr)
    else
        repr = @sprintf "nnz(%s) ≤ %d" get_name(x) reg.count
        return StructuredOptimization.Term(1, IndBallL0(reg.count), op * x, repr)
    end
end

"""
	L0Wavelet2D(; threshold=nothing, count=nothing, wavelet=WT.db2, levels=2)

Create an `ℓ₀` regularization term in a 2D wavelet domain. Exactly one of `threshold` or `count` must be
given.

# Arguments
- `threshold`: penalty form. The term is `λ‖𝒲x‖₀`, the number of non-zero wavelet coefficients weighted by
  `λ = threshold`. Its proximal operator is hard thresholding (see [`L0Image`](@ref) for details).
- `count`: constraint form. The image is required to have at most `count` non-zero coefficients in the
  wavelet domain. Its proximal operator keeps the `count` largest-magnitude coefficients and zeroes the rest.
- `wavelet`, `levels`: (optional) Wavelet family and number of decomposition levels. Defaults match
  [`L1Wavelet2D`](@ref).

# Notes
See [`L0Image`](@ref) for the shared penalty/constraint semantics, references and the task-splitting
caveat of the `count` form (which applies regardless of domain, since the budget couples the whole
coefficient array).
"""
struct L0Wavelet2D{T, W} <: Regularization
    threshold::Union{T, Nothing}
    count::Union{Int, Nothing}
    wavelet::W
    levels::Int
    function L0Wavelet2D(;
            threshold::Union{Real, Nothing} = nothing, count::Union{Integer, Nothing} = nothing,
            wavelet::W = WT.db2, levels::Int = 2,
        ) where {W}
        _check_l0_args(threshold, count, "L0Wavelet2D")
        T = threshold === nothing ? Float64 : typeof(threshold)
        return new{T, W}(threshold, count === nothing ? nothing : Int(count), wavelet, levels)
    end
end

function get_operator(reg::L0Wavelet2D, x::AbstractArray; threaded::Bool = true)
    return _sparsifying_operator(Val(:wavelet2d), reg.wavelet, reg.levels, x; threaded)
end

function get_affected_dims(reg::L0Wavelet2D, ::Nothing, image_dims)
    return reg.count !== nothing ? Tuple(image_dims) : _sparsifying_affected_dims(Val(:wavelet2d), image_dims)
end

function scale_regularization(reg::L0Wavelet2D, factor::Real)
    reg.threshold === nothing && return reg
    return L0Wavelet2D(threshold = reg.threshold * factor^2, wavelet = reg.wavelet, levels = reg.levels)
end

function materialize(reg::L0Wavelet2D, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    if reg.threshold !== nothing
        λ = real(T)(reg.threshold)
        repr = @sprintf "%g ⋅ ‖%s%s‖₀" λ _sparsifying_repr(Val(:wavelet2d)) get_name(x)
        return StructuredOptimization.Term(1, NormL0(λ), op * x, repr)
    else
        repr = @sprintf "nnz(%s%s) ≤ %d" _sparsifying_repr(Val(:wavelet2d)) get_name(x) reg.count
        return StructuredOptimization.Term(1, IndBallL0(reg.count), op * x, repr)
    end
end

"""
	L0Wavelet3D(; threshold=nothing, count=nothing, wavelet=WT.db2, levels=2)

Create an `ℓ₀` regularization term in a 3D wavelet domain. Exactly one of `threshold` or `count` must be
given.

# Arguments
- `threshold`: penalty form. The term is `λ‖𝒲x‖₀`, the number of non-zero wavelet coefficients weighted by
  `λ = threshold`. Its proximal operator is hard thresholding (see [`L0Image`](@ref) for details).
- `count`: constraint form. The image is required to have at most `count` non-zero coefficients in the
  wavelet domain. Its proximal operator keeps the `count` largest-magnitude coefficients and zeroes the rest.
- `wavelet`, `levels`: (optional) Wavelet family and number of decomposition levels. Defaults match
  [`L1Wavelet3D`](@ref).

# Notes
See [`L0Image`](@ref) for the shared penalty/constraint semantics, references and the task-splitting
caveat of the `count` form (which applies regardless of domain, since the budget couples the whole
coefficient array).
"""
struct L0Wavelet3D{T, W} <: Regularization
    threshold::Union{T, Nothing}
    count::Union{Int, Nothing}
    wavelet::W
    levels::Int
    function L0Wavelet3D(;
            threshold::Union{Real, Nothing} = nothing, count::Union{Integer, Nothing} = nothing,
            wavelet::W = WT.db2, levels::Int = 2,
        ) where {W}
        _check_l0_args(threshold, count, "L0Wavelet3D")
        T = threshold === nothing ? Float64 : typeof(threshold)
        return new{T, W}(threshold, count === nothing ? nothing : Int(count), wavelet, levels)
    end
end

function get_operator(reg::L0Wavelet3D, x::AbstractArray; threaded::Bool = true)
    return _sparsifying_operator(Val(:wavelet3d), reg.wavelet, reg.levels, x; threaded)
end

function get_affected_dims(reg::L0Wavelet3D, ::Nothing, image_dims)
    return reg.count !== nothing ? Tuple(image_dims) : _sparsifying_affected_dims(Val(:wavelet3d), image_dims)
end

function scale_regularization(reg::L0Wavelet3D, factor::Real)
    reg.threshold === nothing && return reg
    return L0Wavelet3D(threshold = reg.threshold * factor^2, wavelet = reg.wavelet, levels = reg.levels)
end

function materialize(reg::L0Wavelet3D, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    if reg.threshold !== nothing
        λ = real(T)(reg.threshold)
        repr = @sprintf "%g ⋅ ‖%s%s‖₀" λ _sparsifying_repr(Val(:wavelet3d)) get_name(x)
        return StructuredOptimization.Term(1, NormL0(λ), op * x, repr)
    else
        repr = @sprintf "nnz(%s%s) ≤ %d" _sparsifying_repr(Val(:wavelet3d)) get_name(x) reg.count
        return StructuredOptimization.Term(1, IndBallL0(reg.count), op * x, repr)
    end
end

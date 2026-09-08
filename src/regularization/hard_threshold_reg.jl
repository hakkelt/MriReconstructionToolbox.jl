"""
	_sparsifying_operator(::Val{domain}, wavelet, levels, x; threaded)

Return the operator that maps the image to the domain in which sparsity is enforced by
[`HardThreshold`](@ref) and [`SparsityLimit`](@ref). `domain` is `:image`, `:wavelet2d` or `:wavelet3d`.
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

function _check_sparsifying_domain(domain::Symbol)
    return @argcheck domain in (:image, :wavelet2d, :wavelet3d) "domain must be :image, :wavelet2d or :wavelet3d, got :$domain"
end

"""
	HardThreshold(λ; domain=:image, wavelet=WT.db2, levels=2)

Create an `ℓ₀` regularization term with parameter `λ`: `λ‖Ψx‖₀`, the number of non-zero coefficients of the
image in the sparsifying domain `Ψ`, weighted by `λ`. Its proximal operator is hard thresholding, i.e.
coefficients with magnitude below `sqrt(2γλ)` are set to zero and the surviving ones are left untouched.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `domain`: (optional) Sparsifying transform: `:image` (default, identity), `:wavelet2d` or `:wavelet3d`.
- `wavelet`, `levels`: (optional) Wavelet family and number of decomposition levels, used only for the
  wavelet domains. Defaults match [`L1Wavelet2D`](@ref).

# Notes
- Hard thresholding does not shrink the coefficients it keeps, so unlike the `ℓ₁` terms ([`L1Image`](@ref),
  [`L1Wavelet2D`](@ref)) it introduces no amplitude bias in the retained features — at the price of a
  non-convex objective, for which the solvers only guarantee a stationary point. It is BART's `-R H`.
- Because the term is non-convex, use it with a good starting point (e.g. the zero-filled reconstruction or
  an `ℓ₁` solution) and prefer `ISTA`/`FISTA`; results depend on the initialization.
- The threshold is `sqrt(2γλ)`, not `γλ` as for the `ℓ₁` terms, so a `λ` transplanted from an `ℓ₁` term will
  not give a comparable amount of sparsity.
"""
struct HardThreshold{T, W} <: Regularization
    λ::T
    domain::Symbol
    wavelet::W
    levels::Int
    function HardThreshold(λ::T; domain::Symbol = :image, wavelet::W = WT.db2, levels::Int = 2) where {T, W}
        @argcheck λ isa Real "HardThreshold requires a scalar λ"
        @argcheck λ >= 0 "λ must be non-negative"
        _check_sparsifying_domain(domain)
        return new{T, W}(λ, domain, wavelet, levels)
    end
end

function get_operator(reg::HardThreshold, x::AbstractArray; threaded::Bool = true)
    return _sparsifying_operator(Val(reg.domain), reg.wavelet, reg.levels, x; threaded)
end

get_affected_dims(reg::HardThreshold, ::Nothing, image_dims) = _sparsifying_affected_dims(Val(reg.domain), image_dims)

# ‖·‖₀ is homogeneous of degree 0, so with `k=1, p=0` the docstring's rule gives λ_eff = λ ⋅ factor².
function scale_regularization(reg::HardThreshold, factor::Real)
    return HardThreshold(reg.λ * factor^2; domain = reg.domain, wavelet = reg.wavelet, levels = reg.levels)
end

function materialize(reg::HardThreshold, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    λ = real(T)(reg.λ)
    repr = @sprintf "%g ⋅ ‖%s%s‖₀" λ _sparsifying_repr(Val(reg.domain)) get_name(x)
    return StructuredOptimization.Term(1, NormL0(λ), op * x, repr)
end

"""
	SparsityLimit(max_nonzeros; domain=:image, wavelet=WT.db2, levels=2)

Create a hard sparsity constraint: the image is required to have at most `max_nonzeros` non-zero
coefficients in the sparsifying domain `Ψ`. Its proximal operator keeps the `max_nonzeros` largest-magnitude
coefficients and zeroes the rest.

# Arguments
- `max_nonzeros`: Maximum number of non-zero coefficients. Must be a positive integer.
- `domain`: (optional) Sparsifying transform: `:image` (default, identity), `:wavelet2d` or `:wavelet3d`.
- `wavelet`, `levels`: (optional) Wavelet family and number of decomposition levels, used only for the
  wavelet domains.

# Notes
- This is the constrained counterpart of [`HardThreshold`](@ref), and the sparsity analogue of
  [`RankLimit`](@ref): instead of tuning a penalty weight whose relation to the resulting sparsity is
  indirect, the sparsity itself is prescribed. It is the thresholding step of iterative hard thresholding
  (Blumensath & Davies, *Iterative hard thresholding for compressed sensing*, ACHA 2009).
- The constraint set is non-convex, so the solvers only guarantee a stationary point, and the result depends
  on the initialization.
- The count is over the entire coefficient array, including all batch dimensions. Because a per-sub-problem
  budget would not be the same constraint, this term blocks task splitting; reconstruct slice by slice
  explicitly if a per-slice budget is what is wanted.
"""
struct SparsityLimit{W} <: Regularization
    max_nonzeros::Int
    domain::Symbol
    wavelet::W
    levels::Int
    function SparsityLimit(max_nonzeros::Int; domain::Symbol = :image, wavelet::W = WT.db2, levels::Int = 2) where {W}
        @argcheck max_nonzeros > 0 "max_nonzeros must be positive"
        _check_sparsifying_domain(domain)
        return new{W}(max_nonzeros, domain, wavelet, levels)
    end
end

function get_operator(reg::SparsityLimit, x::AbstractArray; threaded::Bool = true)
    return _sparsifying_operator(Val(reg.domain), reg.wavelet, reg.levels, x; threaded)
end

# Unlike the separable penalties, the budget couples every voxel of the coefficient array: splitting the
# problem would give each sub-problem its own budget of `max_nonzeros` and so change the constraint. All
# image dimensions are therefore reported as affected, which blocks task splitting.
get_affected_dims(::SparsityLimit, ::Nothing, image_dims) = Tuple(image_dims)

# The constraint is scale-invariant (nnz(factor * x) == nnz(x)); no correction needed.
scale_regularization(reg::SparsityLimit, ::Real) = reg

function materialize(reg::SparsityLimit, x::Variable; threaded::Bool)
    op = get_operator(reg, ~x; threaded)
    repr = @sprintf "nnz(%s%s) ≤ %d" _sparsifying_repr(Val(reg.domain)) get_name(x) reg.max_nonzeros
    return StructuredOptimization.Term(1, IndBallL0(reg.max_nonzeros), op * x, repr)
end

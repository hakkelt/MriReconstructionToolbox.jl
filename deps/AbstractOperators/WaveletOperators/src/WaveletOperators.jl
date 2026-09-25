module WaveletOperators

export WaveletOp, wavelet, WT

using ..AbstractOperators
using Wavelets
import LinearAlgebra: mul!, opnorm
import Base: size
import ..AbstractOperators:
    domain_type,
    codomain_type,
    domain_array_type,
    codomain_array_type,
    fun_name,
    is_thread_safe,
    supports_threading,
    is_threaded,
    has_fast_opnorm,
    has_optimized_normalop,
    get_normal_op,
    _normalize_array_type,
    _array_wrapper_type,
    _copy_operator_impl,
    _elementwise_threaded,
    @budgeted_threads
import ..OperatorCore:
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank

"""
	WaveletOp(wavelet::DiscreteWavelet, dim_in::Integer)
	WaveletOp(wavelet::DiscreteWavelet, dim_in::Tuple; threaded = true)

Creates a `LinearOperator` which, when multiplied with a vector `x::AbstractVector`, returns the wavelet
transform of `x` using the given `wavelet` and `levels`.

A 2-D or 3-D transform with a filter-bank wavelet splits the lines of each stage across threads
when `threaded` is `true` and the array is large enough; the result is the same as serially.

```jldoctest
julia> using WaveletOperators

julia> W = WaveletOp(wavelet(WT.db4), 4)
𝒲  ℝ^4 -> ℝ^4

julia> W * ones(4)
4-element Vector{Float64}:
  2.0
 -5.551115123125783e-17
 -8.326672684688674e-17
 -8.326672684688674e-17

```
"""
struct WaveletOp{T, N, W <: DiscreteWavelet, S <: AbstractArray{T}, Th} <: LinearOperator
    wavelet::W
    dim_in::NTuple{N, Int}
    levels::Int
end

# Constructors

"""
    _check_transformable(wavelet)

Reject a wavelet `mul!` cannot actually transform with.

`mul!` calls the level-taking `dwt!`/`idwt!`, and Wavelets.jl defines those only for an
`OrthoFilter`: a lifting-scheme `GLS` -- what `wavelet(c, WT.Lifting)` builds, for a
biorthogonal class such as CDF *and* for an orthogonal one such as `db2` -- has only the
single-array in-place form and the allocating `dwt(x, gls, levels)`. Constructing such an
operator therefore produced one that raised a `MethodError` from inside Wavelets on its first
application, and nothing else in the package could tell beforehand. Fail at construction, where
the message can say what is wrong.
"""
function _check_transformable(wavelet::DiscreteWavelet)
    wavelet isa Wavelets.WT.OrthoFilter && return nothing
    throw(
        ArgumentError(
            "WaveletOp supports filter-bank wavelets (`Wavelets.WT.OrthoFilter`, i.e. " *
                "`wavelet(class)` or `wavelet(class, WT.Filter)`); got a $(typeof(wavelet)). " *
                "Wavelets.jl provides no level-taking `dwt!` for a lifting scheme, so such an " *
                "operator could not be applied."
        )
    )
end

function WaveletOp(
        wavelet::DiscreteWavelet, dim_in, levels = nothing;
        array_type::Type{<:AbstractArray} = Array{Float64}, threaded::Bool = true
    )
    if isnothing(levels)
        levels = get_max_transform_levels(dim_in)
    end
    return WaveletOp(Float64, wavelet, dim_in, levels; array_type, threaded)
end

function WaveletOp(
        A::AbstractArray, wavelet::DiscreteWavelet, levels::Int = get_max_transform_levels(size(A));
        threaded::Bool = true
    )
    return WaveletOp(eltype(A), wavelet, size(A), levels; array_type = typeof(A isa SubArray ? parent(A) : A), threaded)
end

function WaveletOp(
        T::Type, wavelet::DiscreteWavelet, dim_in::Integer, levels::Int = get_max_transform_levels(dim_in);
        array_type::Type{<:AbstractArray} = Array{T}, threaded::Bool = true
    )
    _check_transformable(wavelet)
    if isodd(dim_in)
        throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
    end
    if levels > get_max_transform_levels(dim_in)
        throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimension $dim_in: $(get_max_transform_levels(dim_in))."))
    end
    S = _normalize_array_type(array_type, T)
    return WaveletOp{T, 1, typeof(wavelet), S, false}(wavelet, (dim_in,), levels)
end

function WaveletOp(
        T::Type, wavelet::DiscreteWavelet, dim_in::NTuple{N, Int}, levels::Int = get_max_transform_levels(dim_in);
        array_type::Type{<:AbstractArray} = Array{T}, threaded::Bool = true
    ) where {N}
    _check_transformable(wavelet)
    if any(isodd.(dim_in))
        throw(ArgumentError("The input dimension $dim_in is not suitable for wavelet transform: only even dimensions are allowed."))
    end
    if levels > get_max_transform_levels(dim_in)
        throw(ArgumentError("The number of levels $levels exceeds the maximum allowed for dimensions $dim_in: $(get_max_transform_levels(dim_in))."))
    end
    S = _normalize_array_type(array_type, T)
    Th = _wavelet_threaded(threaded, T, wavelet, dim_in, S)
    return WaveletOp{T, N, typeof(wavelet), S, Th}(wavelet, dim_in, levels)
end

# Mappings

function mul!(y::AbstractArray{T}, L::WaveletOp{T}, x::AbstractArray{T}) where {T}
    AbstractOperators.check(y, L, x)
    is_threaded(L) && return _threaded_dwt!(y, x, L.wavelet, L.levels, true)
    return dwt!(y, x, L.wavelet, L.levels)
end

function mul!(
        y::AbstractArray{T}, L::AdjointOperator{<:WaveletOp{T}}, x::AbstractArray{T}
    ) where {T}
    AbstractOperators.check(y, L, x)
    is_threaded(L.A) && return _threaded_dwt!(y, x, L.A.wavelet, L.A.levels, false)
    return idwt!(y, x, L.A.wavelet, L.A.levels)
end

# Properties

fun_name(::WaveletOp) = "𝒲"

size(L::WaveletOp) = (L.dim_in, L.dim_in)

domain_type(::WaveletOp{T}) where {T} = T
codomain_type(::WaveletOp{T}) where {T} = T
domain_array_type(::WaveletOp{T, N, W, S}) where {T, N, W, S} = S
codomain_array_type(::WaveletOp{T, N, W, S}) where {T, N, W, S} = S

# `WᴴW = I` only holds for an orthogonal wavelet family, so every trait below that assumes the
# identity is guarded on this. With `_check_transformable` in place this currently always answers
# `true` -- only an `OrthoFilter` can be constructed at all -- but the guards stay, so relaxing
# that constructor check cannot silently turn the traits into wrong answers.
#
# Orthogonality is a property of the wavelet *class*, not of the transform representation:
# `wavelet(c, WT.Filter)` builds an `OrthoFilter` and `wavelet(c, WT.Lifting)` a `GLS` for *any*
# class, orthogonal ones included (Wavelets.jl `WT/wt_main.jl`). So `wavelet(WT.db4, WT.Lifting)`
# is orthonormal but is a `GLS`, and testing `isa OrthoFilter` alone would reject it. A `GLS`
# carries only its scheme name, so classify on that; an unrecognized name counts as
# non-orthogonal, which is the conservative direction — it costs the generic `L'*L` / power
# iteration instead of asserting a fast path that may not hold.
const _ORTHOGONAL_LIFTING_SCHEMES = ("haar", "db1", "db2")

_is_orthogonal(L::WaveletOp) = _is_orthogonal_wavelet(L.wavelet)
_is_orthogonal_wavelet(::Wavelets.WT.OrthoFilter) = true
_is_orthogonal_wavelet(w::Wavelets.WT.GLS) = Wavelets.WT.name(w) in _ORTHOGONAL_LIFTING_SCHEMES
_is_orthogonal_wavelet(::Any) = false

is_AcA_diagonal(L::WaveletOp) = _is_orthogonal(L)
is_AAc_diagonal(L::WaveletOp) = _is_orthogonal(L)
is_invertible(L::WaveletOp) = true
is_full_row_rank(L::WaveletOp) = true
is_full_column_rank(L::WaveletOp) = true

diag_AcA(L::WaveletOp{T}) where {T} = _is_orthogonal(L) ? real(T(1)) : throw(ArgumentError("diag_AcA is only defined for orthogonal wavelets"))
diag_AAc(L::WaveletOp{T}) where {T} = _is_orthogonal(L) ? real(T(1)) : throw(ArgumentError("diag_AAc is only defined for orthogonal wavelets"))

AbstractOperators.is_thread_safe(::WaveletOp) = true

# The non-orthogonal branches defer to `AbstractOperators`' generic definitions rather than
# throwing: `opnorm(::AbstractOperator)` is a power iteration and `get_normal_op(L) = L' * L`,
# both correct for a biorthogonal wavelet. Throwing here would turn a working call into an
# error at every unguarded call site -- `ProximalAlgorithms`' primal-dual solvers call `opnorm`
# directly, and `get_normal_op(::DCAT)` calls it on every block once *any* block reports an
# optimized normal operator.
has_fast_opnorm(L::WaveletOp) = _is_orthogonal(L)
has_fast_opnorm(L::AdjointOperator{<:WaveletOp}) = _is_orthogonal(L.A)
opnorm(L::WaveletOp{T}) where {T} = _is_orthogonal(L) ? one(T) : AbstractOperators.powerit(L)
opnorm(L::AdjointOperator{<:WaveletOp}) =
    _is_orthogonal(L.A) ? one(eltype(domain_type(L.A))) : AbstractOperators.powerit(L)

has_optimized_normalop(L::WaveletOp) = _is_orthogonal(L)
function get_normal_op(L::WaveletOp)
    _is_orthogonal(L) || return L' * L
    return Eye(domain_type(L), size(L, 2); array_type = domain_array_type(L))
end

# Utils

get_max_transform_levels(dim_in::Integer) = maxtransformlevels(dim_in)
get_max_transform_levels(dim_in::Tuple) = minimum(maxtransformlevels.(dim_in))


# ─── Threading ────────────────────────────────────────────────────────────────
#
# Wavelets.jl runs serially. The threaded path (`threaded_dwt.jl`) exists for the 2-D and 3-D
# filter-bank transforms; 1-D transforms and lifting schemes always run in Wavelets.jl.
#
# The threshold is the package default (`THRESHOLD_MEMORY_BOUND`), not a swept value: each
# element costs a filter's length in multiply-adds per level, so the true crossover is likely
# lower. Measured at 128³ ComplexF32, db2, 3 levels (EPYC 7763, one NUMA domain): forward
# 240 ms serial.
_has_threaded_path(N::Int, wavelet) = N in (2, 3) && wavelet isa Wavelets.WT.OrthoFilter

function _wavelet_threaded(threaded::Bool, ::Type{T}, wavelet, dim_in::NTuple{N, Int}, ::Type{S}) where {T, N, S}
    _has_threaded_path(N, wavelet) || return false
    return _elementwise_threaded(WaveletOp, threaded, T, dim_in, S)
end

AbstractOperators.is_threaded(::WaveletOp{T, N, W, S, Th}) where {T, N, W, S, Th} = Th
AbstractOperators.supports_threading(L::WaveletOp{T, N}) where {T, N} = _has_threaded_path(N, L.wavelet)

# No buffers to deep-copy (only `wavelet`/`dim_in`/`levels`, all immutable), so this method
# rebuilds the storage and threading type parameters.
function _copy_operator_impl(
        op::WaveletOp{T, N, W, S, Th}; storage_type = nothing, threaded = nothing
    ) where {T, N, W, S, Th}
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    new_S = _normalize_array_type(new_at, T)
    new_Th = threaded === nothing && new_S === S ? Th :
        _wavelet_threaded(something(threaded, Th), T, op.wavelet, op.dim_in, new_S)
    return WaveletOp{T, N, W, new_S, new_Th}(op.wavelet, op.dim_in, op.levels)
end

include("threaded_dwt.jl")

end # module

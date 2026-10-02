# Fused pointwise runs of a `Compose` on device arrays (see `src/calculus/pointwise.jl`).
#
# A run is one kernel launch instead of one per operator: an expansion followed by maps writes
# each element of the full array once from the small one, and maps followed by a reduction read
# each copy once and write each element of the small array once. A thread computes one element
# of the array the run writes, adding a reduction's copies in order, so the result is the one
# the operators give one after another.

using .AbstractOperators:
    DiagOp, NoOperatorBroadCast,
    PwMapKind, PwExpandKind, PwReduceKind, PwPlain, PwExpand, PwReduce,
    PwLeftMul, PwConjLeftMul, PwRealConjLeftMul, PwMask, PwTrailingMask,
    _pw_apply_all, _pw_shape_fits
import .AbstractOperators: _pw_kind, _pw_fits, _pw_kernel!, _pw_apply
import GPUArrays.Adapt: Adapt, adapt_structure

const _GPUDiagOp = DiagOp{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGPUArray}
_pw_kind(::Type{<:_GPUDiagOp}) = PwMapKind()
_pw_kind(::Type{<:AdjointOperator{<:_GPUDiagOp}}) = PwMapKind()

_pw_kind(::Type{<:NoOperatorBroadCast{T, N, M, Th, S}}) where {T, N, M, Th, S <: AbstractGPUArray} =
    PwExpandKind()
_pw_kind(::Type{<:AdjointOperator{<:NoOperatorBroadCast{T, N, M, Th, S}}}) where {T, N, M, Th, S <: AbstractGPUArray} =
    PwReduceKind()

_pw_fits(y::AbstractGPUArray, x::AbstractGPUArray, steps, shape, layout) =
    _pw_shape_fits(y, x, steps, shape, layout)

# The CPU kernels resolve a trailing mask once per segment; a device thread reads it per element.
Base.@propagate_inbounds _pw_apply(s::PwTrailingMask{T}, v, j) where {T} =
    ifelse(s.m[div(j - 1, s.inner) + 1], convert(T, v), zero(T))

# Steps travel into the kernel with their arrays converted to the backend's device form.
adapt_structure(to, s::PwLeftMul{T}) where {T} = PwLeftMul{T}(Adapt.adapt(to, s.d))
adapt_structure(to, s::PwConjLeftMul{T}) where {T} = PwConjLeftMul{T}(Adapt.adapt(to, s.d))
adapt_structure(to, s::PwRealConjLeftMul{T}) where {T} = PwRealConjLeftMul{T}(Adapt.adapt(to, s.d))
adapt_structure(to, s::PwMask{T}) where {T} = PwMask{T}(Adapt.adapt(to, s.m))
adapt_structure(to, s::PwTrailingMask{T}) where {T} = PwTrailingMask{T}(Adapt.adapt(to, s.m), s.inner)

@kernel function _pw_plain_kernel!(y, @Const(x), steps)
    j = @index(Global, Linear)
    @inbounds y[j] = _pw_apply_all(steps, x[j], j)
end

# `y` is `(inner, K, outer)` and `x` `(inner, outer)`.
@kernel function _pw_expand_kernel!(y, @Const(x), steps, inner, K)
    j = @index(Global, Linear)
    q, r = divrem(j - 1, inner)
    @inbounds y[j] = _pw_apply_all(steps, x[r + (q ÷ K) * inner + 1], j)
end

# `y` is `(inner, outer)` and `x` `(inner, K, outer)`; the copies are added in order, starting
# from zero, as `sum!` does.
@kernel function _pw_reduce_kernel!(y, @Const(x), steps, inner, K)
    i = @index(Global, Linear)
    o, r = divrem(i - 1, inner)
    j = r + o * inner * K + 1
    @inbounds acc = zero(eltype(y)) + _pw_apply_all(steps, x[j], j)
    for k in 1:(K - 1)
        j += inner
        @inbounds acc += _pw_apply_all(steps, x[j], j)
    end
    @inbounds y[i] = acc
end

function _pw_kernel!(y::AbstractGPUArray, x::AbstractGPUArray, steps, shape, layout, threaded::Bool)
    backend = get_backend(y)
    if shape isa PwPlain
        _pw_plain_kernel!(backend)(y, x, steps; ndrange = length(y))
    else
        inner, K = layout
        kernel = shape isa PwExpand ? _pw_expand_kernel! : _pw_reduce_kernel!
        kernel(backend)(y, x, steps, inner, K; ndrange = length(y))
    end
    return y
end

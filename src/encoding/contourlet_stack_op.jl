import Base: size
import LinearAlgebra: mul!
import AbstractOperators: domain_type, codomain_type, fun_name, is_thread_safe, is_invertible

"""
    StackedNSCTOp(T, op)

Wraps a single-slice `NSCTOp` (`op`, acting on a `NTuple{2,Int}` image) so its multi-component
`ArrayPartition` codomain (coarse band + directional subbands) is exposed as one plain
`Array{T,3}` instead: `(dim_in..., n_bands)`, one band per trailing-axis slice. This works only
because the NSCT is shift-invariant, so every band shares the input's spatial size -- unlike
[`ContourletOp`](@ref), whose (nearly critically sampled) bands are ragged and cannot be stacked
this way.

Stacking into a plain array (rather than keeping the `ArrayPartition`) is what lets
[`L1Contourlet`](@ref) reuse `AbstractOperators.BatchOp` for extra batch/time dimensions and
`StructuredOptimization.Term`/`NormL1`, mirroring `WaveletOp`; neither accepts an operator
whose codomain has more than one component (`ndoms(L, 1) > 1`, which an `ArrayPartition` codomain
is).

`T` is the *reported* domain/codomain element type (the caller's variable type, e.g. `ComplexF32`)
and may differ from `op`'s own element type: Contourlets.jl's default filters are `Float64`, so
`NSCTOp`'s own domain type is `promote_type(Float64, S)` for a requested `S` -- e.g. `ComplexF64`
even when the caller works in `ComplexF32`. `mul!` stages `x`/`y` through internal buffers at
`op`'s own element type and converts on `copyto!`, so `T` can be any type convertible to/from it.

Not thread-safe: `mul!` reuses this wrapper's own staging buffers (and `op`'s own scratch
buffers) sequentially.
"""
struct StackedNSCTOp{T, Op, Td} <: LinearOperator
    op::Op
    n_bands::Int
    dim_in::NTuple{2, Int}
    template_x::Matrix{Td}
    template_y::ArrayPartition
end

function StackedNSCTOp(T::Type, op::Op) where {Op}
    Td = domain_type(op)
    n_bands = length(op.band_sizes)
    dim_in = op.dim_in
    template_x = Matrix{Td}(undef, dim_in)
    template_y = ArrayPartition(ntuple(_ -> Matrix{Td}(undef, dim_in), n_bands)...)
    return StackedNSCTOp{T, Op, Td}(op, n_bands, dim_in, template_x, template_y)
end

size(L::StackedNSCTOp) = ((L.dim_in..., L.n_bands), L.dim_in)
domain_type(::StackedNSCTOp{T}) where {T} = T
codomain_type(::StackedNSCTOp{T}) where {T} = T
is_thread_safe(::StackedNSCTOp) = false
is_invertible(::StackedNSCTOp) = true
fun_name(L::StackedNSCTOp) = fun_name(L.op)

function mul!(y::AbstractArray{T, 3}, L::StackedNSCTOp{T}, x::AbstractMatrix{T}) where {T}
    AbstractOperators.check(y, L, x)
    copyto!(L.template_x, x)
    mul!(L.template_y, L.op, L.template_x)
    for k in 1:L.n_bands
        copyto!(view(y, :, :, k), L.template_y.x[k])
    end
    return y
end

function mul!(
        y::AbstractMatrix{T}, L::AdjointOperator{<:StackedNSCTOp{T}}, x::AbstractArray{T, 3}
    ) where {T}
    AbstractOperators.check(y, L, x)
    B = L.A
    for k in 1:B.n_bands
        copyto!(B.template_y.x[k], view(x, :, :, k))
    end
    mul!(B.template_x, B.op', B.template_y)
    copyto!(y, B.template_x)
    return y
end

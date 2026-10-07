"""
	mul!(y, L::AbstractOperator, x, α::Number, β::Number)

`y = α * (L * x) + β * y`, the 5-argument `mul!` of `LinearAlgebra`. When `β` is zero, `y` is
overwritten and what it held before, `NaN` included, is ignored.

`Eye`, `Zeros`, `DiagOp`, `MatrixOp`, `LMatrixOp`, `FiniteDiff`, `HigherOrderDiff`, `GetIndex`
and `ZeroPad`, and their adjoints, accumulate into `y` in the same pass that computes `L * x`;
`Scale` and `Reshape` pass `α` and `β` on to the operator they wrap.

Every other operator computes `L * x` and then combines it with `y`. With `β` zero that happens
in `y` itself. Otherwise it needs an array of the size of the codomain: inside
[`with_operator_pool`](@ref) one of the pool's arrays, returned to the pool after the call, so
repeated calls allocate nothing; outside a pool a new array, which is released with the call.
"""
function mul!(y::AbstractArray, L::AbstractOperator, x, α::Number, β::Number)
    if iszero(β)
        mul!(y, L, x)
        return isone(α) ? y : _store!(y, y, α, false)
    end
    buf = _codomain_buffer(L)
    mul!(buf, L, x)
    _store!(y, buf, α, β)
    _return_to_pool!(buf)
    return y
end

# The pool holds arrays of any type; the assertion keeps the caller type-stable.
function _codomain_buffer(L::AbstractOperator)
    buf = _pooled_codomain_buffer(L)
    return codomain_array_type(L) <: Array ? buf::Array{codomain_type(L), length(size(L, 1))} : buf
end

function _return_to_pool!(buf)
    pool = _active_pool()
    pool === nothing || !(buf isa Array) || @lock pool.lock push!(pool.buffers, buf)
    return nothing
end

# The operators whose 5-argument `mul!` accumulates into `y`; for them `add_mul!` does not need
# its buffer.
const _AccumulatingMul = Union{
    Eye, Zeros, AdjointOperator{<:Zeros}, DiagOp, AdjointOperator{<:DiagOp},
    MatrixOp, AdjointOperator{<:MatrixOp}, LMatrixOp, AdjointOperator{<:LMatrixOp},
    FiniteDiff, AdjointOperator{<:FiniteDiff}, HigherOrderDiff, AdjointOperator{<:HigherOrderDiff},
    GetIndex, AdjointOperator{<:GetIndex}, ZeroPad, AdjointOperator{<:ZeroPad},
}

function add_mul!(y::AbstractArray, L::_AccumulatingMul, b, ::AbstractArray, α::Number = true, β::Number = true)
    return mul!(y, L, b, α, β)
end

# A `Scale` of an accumulating operator applies its coefficient in the operator's own pass instead
# of scaling the output afterwards.
function mul!(y::AbstractArray, L::Scale{Th, T, <:_AccumulatingMul}, x::AbstractArray) where {Th, T <: Number}
    check(y, L, x)
    return mul!(y, L.A, x, L.coeff, false)
end

function mul!(
        y::AbstractArray, S::AdjointOperator{<:Scale{Th, T, <:_AccumulatingMul}}, x::AbstractArray
    ) where {Th, T <: Number}
    check(y, S, x)
    return mul!(y, S.A.A', x, S.A.coeff_conj, false)
end

# `Scale` and `Reshape` hand the buffer on with the operator they wrap, which may not accumulate.
function add_mul!(y::AbstractArray, L::Scale, b, buf::AbstractArray, α::Number = true, β::Number = true)
    return add_mul!(y, L.A, b, buf, α * L.coeff, β)
end

function add_mul!(
        y::AbstractArray, L::AdjointOperator{<:Scale}, b, buf::AbstractArray, α::Number = true, β::Number = true
    )
    return add_mul!(y, L.A.A', b, buf, α * L.A.coeff_conj, β)
end

function add_mul!(y::AbstractArray, R::Reshape, b, buf::AbstractArray, α::Number = true, β::Number = true)
    sz = size(R.A, 1)
    add_mul!(reshape(y, sz), R.A, b, reshape(buf, sz), α, β)
    return y
end

function add_mul!(
        y::AbstractArray, R::AdjointOperator{<:Reshape}, b, buf::AbstractArray, α::Number = true, β::Number = true
    )
    return add_mul!(y, R.A.A', reshape(b, size(R.A.A, 1)), buf, α, β)
end

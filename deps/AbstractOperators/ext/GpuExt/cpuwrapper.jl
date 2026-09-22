# GPU mul! overrides for OperatorWrapper.
# check(y, A, x) passes because domain_array_type/codomain_array_type reflect
# the constructor-supplied array_type (e.g. CuArray), not the inner CPU op's type.

# Forward: GPU → CPU → op → CPU → GPU
function mul!(y::AbstractGPUArray, A::OperatorWrapper, x::AbstractGPUArray)
    check(y, A, x)
    copyto!(A.dom_buf, x)
    mul!(A.cod_buf, A.op, A.dom_buf)
    copyto!(y, A.cod_buf)
    return y
end

# Adjoint: GPU → CPU → op' → CPU → GPU
function mul!(
        y::AbstractGPUArray,
        Ac::AdjointOperator{<:OperatorWrapper},
        x::AbstractGPUArray,
    )
    check(y, Ac, x)
    A = Ac.A
    copyto!(A.cod_buf, x)
    mul!(A.dom_buf, A.op', A.cod_buf)
    copyto!(y, A.dom_buf)
    return y
end

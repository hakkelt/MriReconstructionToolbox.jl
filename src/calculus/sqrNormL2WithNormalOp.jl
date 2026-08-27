# squared L2 norm (times a constant, or weighted) precomposed with an operator

"""
    SqrNormL2WithNormalOp(λ=1, L::LinearOperator)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2}\\|L * x\\|^2.
```
With a nonnegative array `λ`, return the weighted squared Euclidean norm
```math
f(x) = \\tfrac{1}{2}∑_i λ_i y_i^2 where y = L * x.
```

This is a special case of the more general `Precompose(SqrNormL2(), L, 1, 0)` operator,
where `L` is a linear operator, and only the gradient is needed, not the proximal operator.
The gradient of the precomposed squared norm is
```math
\nabla f(x) = Lᴴ * L * x,
```
and in many cases, there is an optimized implementation of the normal operator `Lᴴ * L`
that makes the compution of the gradient much faster than the naive implementation.

A notable drawback of this method is that gradient! does not return the
squared norm of `L * x`, but rather the squared norm of `Lᴴ * L * x` (i.e. the
squared norm of the gradient). Most algorithms, however, tolerate this
difference, and it is much faster to compute.
"""
struct SqrNormL2WithNormalOp{T,SC,L<:AbstractOperator,L2<:AbstractOperator}
    A::L
    AᴴA::L2
    lambda::T
    function SqrNormL2WithNormalOp(A, lambda)
        @assert A isa AbstractOperator
        @assert is_linear(A)
        if any(lambda .< 0)
            error("coefficients in λ must be nonnegative")
        else
            AᴴA = A' * A
            new{typeof(lambda),all(lambda .> 0),typeof(A),typeof(AᴴA)}(A, AᴴA, lambda)
        end
    end
end

is_convex(::Type{<:SqrNormL2WithNormalOp}) = true
is_smooth(::Type{<:SqrNormL2WithNormalOp}) = true
is_separable(::Type{<:SqrNormL2WithNormalOp}) = true
is_generalized_quadratic(::Type{<:SqrNormL2WithNormalOp}) = true
is_strongly_convex(::Type{SqrNormL2WithNormalOp{T,SC}}) where {T,SC} = SC

SqrNormL2WithNormalOp(A) = SqrNormL2WithNormalOp(A, 1)

function (f::SqrNormL2WithNormalOp{S})(x) where {S <: Real}
    y = f.A * x
    return f.lambda / real(eltype(y))(2) * norm(y)^2
end

function (f::SqrNormL2WithNormalOp{<:AbstractArray})(x)
    y = f.A * x
    R = real(eltype(y))
    sqnorm = R(0)
    for k in eachindex(y)
        sqnorm += f.lambda[k] * abs2(y[k])
    end
    return sqnorm / R(2)
end

function gradient!(y, f::SqrNormL2WithNormalOp{<:Real}, x)
    R = real(eltype(y))
    mul!(y, f.AᴴA, x)
    sqnx = R(0)
    for k in eachindex(y)
        y[k] *= f.lambda
        sqnx += abs2(y[k])
    end
    return f.lambda / R(2) * sqnx
end

function gradient!(y, f::SqrNormL2WithNormalOp{<:AbstractArray}, x)
    R = real(eltype(y))
    mul!(y, f.AᴴA, x)
    sqnx = R(0)
    for k in eachindex(y)
        y[k] *= f.lambda[k]
        sqnx += f.lambda[k] * abs2(y[k])
    end
    return sqnx / R(2)
end

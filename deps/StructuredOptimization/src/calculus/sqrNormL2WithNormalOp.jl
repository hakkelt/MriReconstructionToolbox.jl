# squared L2 norm (times a constant) precomposed with an operator

"""
    SqrNormL2WithNormalOp(L::AbstractOperator, λ = 1)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2}\\|L * x\\|^2.
```

This is a special case of the more general `Precompose(SqrNormL2(), L, 1, 0)` operator,
where `L` is a linear operator, and only the gradient is needed, not the proximal operator.
The gradient of the precomposed squared norm is
```math
\\nabla f(x) = λ \\, Lᴴ * L * x,
```
and in many cases, there is an optimized implementation of the normal operator `Lᴴ * L`
that makes the computation of the gradient much faster than the naive implementation.

`L` may be affine (an `AffineAdd`, as produced by `ls(A*x - b)`): writing `L*x = A*x + d`,
the normal operator carries the displacement `Aᴴd` (see `get_normal_op(::AffineAdd)`), so
`gradient!` still computes `λ(AᴴA x + Aᴴd)` in a single pass.

`gradient!` returns the function value `f(x)`, as `ProximalCore.value_and_gradient!`
requires. It is obtained from the gradient without a second application of `L`: with
`y = λ(AᴴA x + Aᴴd)` already computed,
```math
f(x) = \\tfrac{1}{2}\\mathrm{Re}⟨x, y⟩ + \\tfrac{λ}{2}\\left(\\mathrm{Re}⟨x, Aᴴd⟩ + \\|d\\|^2\\right),
```
where `Aᴴd` and `‖d‖²/2` are computed once, at construction time. When `L` has no
displacement both correction terms vanish and the value is just `½Re⟨x, y⟩`.
"""
struct SqrNormL2WithNormalOp{T <: Real, SC, L <: AbstractOperator, L2 <: AbstractOperator, D, R <: Real}
    A::L
    AᴴA::L2
    lambda::T
    # `Aᴴd`: the normal operator's displacement, `Aᴴ * d` with `L*x = A*x + d`, or
    # `nothing` when `L` is purely linear (the overwhelmingly common case), so that the
    # per-gradient correction is skipped entirely rather than paying a dot with zeros.
    Aᴴd::D
    # `‖d‖²/2`, the constant term of the quadratic.
    half_sqnorm_d::R
    function SqrNormL2WithNormalOp(A, lambda)
        @assert A isa AbstractOperator
        @assert is_linear(A)
        if !(lambda isa Real)
            error("λ must be a real scalar")
        end
        if lambda < 0
            error("coefficients in λ must be nonnegative")
        end
        AᴴA = A' * A
        # `A * 0` is the displacement `d`, and `AᴴA * 0` is `Aᴴd` — taken through the
        # very operators `gradient!` uses, so the constants cannot drift from them.
        z = AbstractOperators.allocate_in_domain(A)
        fill!(z, 0)
        d = A * z
        sqnorm_d = real(dot(d, d))
        Aᴴd = sqnorm_d == 0 ? nothing : AᴴA * z
        half_sqnorm_d = sqnorm_d / 2
        return new{
            typeof(lambda), lambda > 0, typeof(A), typeof(AᴴA),
            typeof(Aᴴd), typeof(half_sqnorm_d),
        }(A, AᴴA, lambda, Aᴴd, half_sqnorm_d)
    end
end

is_convex(::Type{<:SqrNormL2WithNormalOp}) = true
is_smooth(::Type{<:SqrNormL2WithNormalOp}) = true
is_separable(::Type{<:SqrNormL2WithNormalOp}) = true
is_generalized_quadratic(::Type{<:SqrNormL2WithNormalOp}) = true
is_strongly_convex(::Type{SqrNormL2WithNormalOp{T,SC}}) where {T,SC} = SC

SqrNormL2WithNormalOp(A) = SqrNormL2WithNormalOp(A, 1)

function (f::SqrNormL2WithNormalOp)(x)
    y = f.A * x
    return f.lambda * real(dot(y, y)) / 2
end

function gradient!(y, f::SqrNormL2WithNormalOp, x)
    mul!(y, f.AᴴA, x)
    if f.lambda != 1
        y .*= f.lambda
    end
    v = real(dot(x, y)) / 2
    if f.Aᴴd !== nothing
        v += f.lambda * (real(dot(x, f.Aᴴd)) / 2 + f.half_sqnorm_d)
    end
    return v
end

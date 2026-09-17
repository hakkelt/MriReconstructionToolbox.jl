# squared L2 norm (times a constant) precomposed with an operator

"""
    SqrNormL2WithNormalOp(L::AbstractOperator, λ = 1)

With a nonnegative scalar `λ`, return the squared Euclidean norm
```math
f(x) = \\tfrac{λ}{2σ}\\|L * x\\|^2,
```
where `σ` is the adjoint scaling of `L` described below (`σ = 1`, and the factor disappears,
whenever `L'` is the true adjoint of `L`).

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
f(x) = \\tfrac{1}{2}\\mathrm{Re}⟨x, y⟩ + \\tfrac{λ}{2}\\left(\\mathrm{Re}⟨x, Aᴴd⟩ + \\|d\\|^2/σ\\right),
```
where `Aᴴd` and `‖d‖²/(2σ)` are computed once, at construction time. When `L` has no
displacement both correction terms vanish and the value is just `½Re⟨x, y⟩`.

# Adjoint scaling

`A'` is not always the true adjoint of `A`. A `BACKWARD`-normalized `DFT`, for instance, has
`A' = A⁻¹ = Aᴴ/N`: the pair is off by a positive scalar `σ` defined by
```math
\\mathrm{Re}⟨A u, A u⟩ = σ \\, \\mathrm{Re}⟨u, (A'A) u⟩ .
```
The gradient built from that pair is then the gradient of `‖A x + d‖²/(2σ)`, not of
`‖A x + d‖²/2`, so the value must be the potential of the gradient the caller actually gets
— otherwise the two disagree by a constant offset and a factor `σ`, and anything that reads
both (a backtracking line search, a printed objective) is meaningless. `σ` is measured once,
at construction, with a single probe through `A` and `A'A`; with a genuine adjoint it is `1`
and every formula above reduces to the usual one.
"""
struct SqrNormL2WithNormalOp{T <: Real, SC, L <: AbstractOperator, L2 <: AbstractOperator, D, R <: Real}
    A::L
    AᴴA::L2
    lambda::T
    # `Aᴴd`: the normal operator's displacement, `Aᴴ * d` with `L*x = A*x + d`, or
    # `nothing` when `L` is purely linear (the overwhelmingly common case), so that the
    # per-gradient correction is skipped entirely rather than paying a dot with zeros.
    Aᴴd::D
    # `‖d‖²/(2σ)`, the constant term of the quadratic, in the same scaling as the gradient.
    half_sqnorm_d::R
    # `1/σ`, the adjoint scaling of `A` (see the docstring); `1` for a true adjoint pair.
    inv_scaling::R
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
        inv_scaling = _inv_adjoint_scaling(A, AᴴA, z, d, Aᴴd)
        half_sqnorm_d = sqnorm_d * inv_scaling / 2
        return new{
            typeof(lambda), lambda > 0, typeof(A), typeof(AᴴA),
            typeof(Aᴴd), typeof(half_sqnorm_d),
        }(A, AᴴA, lambda, Aᴴd, half_sqnorm_d, oftype(half_sqnorm_d, inv_scaling))
    end
end

# `σ` from the docstring, as `1/σ`: `Re⟨A u, A u⟩ / Re⟨u, (A'A) u⟩` for a probe `u`, with the
# displacement of an affine `A` subtracted so that only the linear parts are compared.
#
# The probe is the constant vector, which is deterministic (no RNG dependency, so the value a
# solver prints does not move between runs) and is annihilated by no operator this is used
# with. Should it nevertheless land in the null space, `Aᴴd` — which is in the domain, and
# nonzero exactly when there is a displacement to correct — is tried next; if that fails too
# the scaling is left at 1, which is the behaviour of a true adjoint pair.
function _inv_adjoint_scaling(A, AᴴA, z, d, Aᴴd)
    R = real(eltype(z))
    u = similar(z)
    for probe in 1:2
        if probe == 1
            fill!(u, one(eltype(z)))
        elseif Aᴴd !== nothing
            copyto!(u, Aᴴd)
        else
            break
        end
        Au = A * u
        w = AᴴA * u
        # strip the affine displacement: `A u = A_lin u + d` and `(A'A) u = (A'A)_lin u + Aᴴd`
        if Aᴴd !== nothing
            Au = Au .- d
            w = w .- Aᴴd
        end
        num = real(dot(Au, Au))
        den = real(dot(u, w))
        isfinite(num) && isfinite(den) && den > 0 && return R(den / num)
    end
    return one(R)
end

is_convex(::Type{<:SqrNormL2WithNormalOp}) = true
is_smooth(::Type{<:SqrNormL2WithNormalOp}) = true
is_separable(::Type{<:SqrNormL2WithNormalOp}) = true
is_generalized_quadratic(::Type{<:SqrNormL2WithNormalOp}) = true
is_strongly_convex(::Type{SqrNormL2WithNormalOp{T,SC}}) where {T,SC} = SC

SqrNormL2WithNormalOp(A) = SqrNormL2WithNormalOp(A, 1)

function (f::SqrNormL2WithNormalOp)(x)
    y = f.A * x
    return f.lambda * real(dot(y, y)) * f.inv_scaling / 2
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

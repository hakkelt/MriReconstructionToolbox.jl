# Norms

import LinearAlgebra: norm
export norm

"""
    norm(x::AbstractExpression, p=2, [q]; [dim=1])

Returns the norm of `x`.

Supported norms:

* `p = 0` ``l_0``-pseudo-norm

* `p = 1` ``l_1``-norm

* `p = 2` ``l_2``-norm

* `p = Inf` ``l_{\\infty}``-norm

* `p = *` nuclear norm

* `p = 2`, `q = 1` ``l_{2,1}`` mixed norm (aka Sum-of-``l_2``-norms)
```math
f(\\mathbf{X}) = \\sum_i \\| \\mathbf{x}_i \\|
```
where ``\\mathbf{x}_i`` is the ``i``-th column if `dim == 1` (or row if  `dim == 2`) of ``\\mathbf{X}``.

"""
function norm(ex::AbstractExpression, p::Real = 2)
    if p == 0
        f = NormL0()
    elseif p == 1
        f = NormL1()
    elseif p == 2
        f = NormL2()
    elseif p == Inf
        f = NormLinf()
    else
        error("function not implemented")
    end
    return Term(f, ex)
end

# Nuclear norm
function norm(ex::AbstractExpression, ::typeof(*))
    return Term(NuclearNorm(), ex)
end

# Mixed Norm
function norm(ex::AbstractExpression, p1::Int, p2::Int; dim::Int = 1)
    if p1 == 2 && p2 == 1
        f = NormL21(1.0, dim)
    else
        error("function not implemented")
    end
    return Term(f, ex)
end

# Least square terms

export ls

"""
    ls(x::AbstractExpression)

Returns the squared norm (least squares) of `x`:
```math
f (\\mathbf{x}) = \\frac{1}{2} \\| \\mathbf{x} \\|^2
```
(shorthand of `1/2*norm(x)^2`).

The term keeps `x`'s operator where the expression put it, separate from the function: this
is a plain `SqrNormL2` composed with whatever `x` is. The faster formulations — evaluating
the gradient through the normal operator `Lᴴ * L` in a single pass for an `x` of the form
`L*v + d` (see `SqrNormL2WithNormalOp`), folding a diagonal `L` into the weight, keeping
the exact prox of an `L` with diagonal `L*Lᴴ` — are all chosen when the problem is parsed,
by `StructuredOptimization.merge_function_with_operator`.

Deferring the choice is what makes it a choice at all. Only at parse time is it known which
of the formulations the selected algorithm can actually use (a prox, a gradient, or the
operator on its own), and only then has the operator been expanded to the problem's full —
possibly multi-variable — domain, where the normal-operator rewrite is both applicable and
cheap to judge. Folding `L` into the function here would hide it from every one of those
decisions.
"""
ls(ex::AbstractExpression) = Term(SqrNormL2(), ex)

import Base: ^

"""
    (t::Term{<:Any,<:NormL2})^2

Square a Euclidean-norm term: `norm(ex, 2)^2` is ``\\|\\mathbf{A}\\mathbf{x}+\\mathbf{d}\\|^2``.

Only the exponent `2` is defined — no other power of a norm is proximable or smooth in a
form this package can use — and anything else raises an error.

Note the weighting convention. `ls(ex)` is ``\\tfrac{1}{2}\\|\\cdot\\|^2`` while
`norm(ex, 2)^2` is ``\\|\\cdot\\|^2`` without the half, so `norm(ex, 2)^2 == 2 * ls(ex)`.
This is the form the CG-family solvers expect for a Tikhonov regularizer, which is why the
`SquaredL2Term` assumption maps it to `λ = t.lambda * f.lambda` with no factor of one half.

```jldoctest
julia> x = Variable(4);

julia> t = norm(x, 2)^2;

julia> t.f
SqrNormL2(2.0)

julia> (2 * ls(x)).f
SqrNormL2(1)
```

See also [`ls`](@ref), [`norm`](@ref).
"""
function (^)(t::Term{T1, T2, T3}, exp::Integer) where {T1, T2 <: NormL2, T3}
    if exp == 2
        # The coefficient 2.0 is due to the fact that SqrNormL2 divides by 2.0
        return t.lambda^2 * Term(SqrNormL2(2.0), t.A)
    else
        error("function not implemented")
    end
end

# HingeLoss

export hingeloss

"""
    hingeloss(x::AbstractExpression, y::Array)

Applies the Hinge loss function
```math
f( \\mathbf{x} ) = \\sum_{i} \\max\\{0, 1 - y_i x_i \\},
```
where `y` is an array containing ``y_i``.
"""
hingeloss(ex::AbstractExpression, b::AbstractVector{R}) where {R <: Real} =
    Term(HingeLoss(b), ex)

# HingeLoss

export sqrhingeloss

"""
    sqrhingeloss(x::AbstractExpression, y::Array)

Applies the squared Hinge loss function
```math
f( \\mathbf{x} ) = \\sum_{i} \\max\\{0, 1 - y_i x_i \\}^2,
```
where `y` is an array containing ``y_i``.
"""
sqrhingeloss(ex::AbstractExpression, b::AbstractVector{R}) where {R <: Real} =
    Term(SqrHingeLoss(b), ex)

# CrossEntropy

export crossentropy

"""
    crossentropy(x::AbstractExpression, y::Array)

Applies the cross entropy loss function:
```math
f(\\mathbf{x}) = -1/N \\sum_{i}^{N} y_i \\log (x_i)+(1-y_i) \\log (1-x_i),
```
where `y` is an array of length ``N`` containing ``y_i`` having ``0 \\leq y_i \\leq 1``.
"""
crossentropy(ex::AbstractExpression, b::AbstractVector{R}) where {R <: Real} =
    Term(CrossEntropy(b), ex)

# LogisticLoss

export logisticloss

"""
    logisticloss(x::AbstractExpression, y::Array)

Applies the logistic loss function:
```math
f(\\mathbf{x}) = \\sum_i \\log(1 + \\exp(-y_i x_i)).
```
"""
logisticloss(ex::AbstractExpression, y::AbstractArray) =
    Term(LogisticLoss(y, 1.0), ex)

# LogBarrier

export logbarrier

"""
    logbarrier(x::AbstractExpression)

Applies the log barrier function:
```math
f(\\mathbf{x}) = -\\sum_i \\log( x_i ).
```
"""
logbarrier(ex::AbstractExpression) =
    Term(LogBarrier(1.0), ex)

# HuberLoss

export huberloss

"""
    huberloss(x::AbstractExpression, ρ=1.0)

Applies the Huber loss function:
```math
f(\\mathbf{x}) = \\begin{cases}
  \\tfrac{1}{2}\\| \\mathbf{x} \\|^2 & \\text{if} \\ \\| \\mathbf{x} \\| \\leq \\rho \\\\
  \\rho (\\| \\mathbf{x} \\| - \\tfrac{\\rho}{2}) & \\text{otherwise}.
\\end{cases}
```
"""
huberloss(ex::AbstractExpression, rho::R = 1.0) where {R <: Real} =
    Term(HuberLoss(rho), ex)

import Base: maximum

"""
    maximum(x::AbstractExpression)

Applies the function:
```math
f(\\mathbf{x}) = \\max \\{x_i : i = 1,\\ldots, n \\}.
```
"""
maximum(ex::AbstractExpression) =
    Term(Maximum(), ex)

export sumpositive

"""
    sumpositive(x::AbstractExpression, ρ=1.0)

Applies the function:
```math
f(\\mathbf{x}) = \\sum_i \\max \\{x_i, 0\\}.
```
"""
sumpositive(ex::AbstractExpression) =
    Term(SumPositive(), ex)

import LinearAlgebra: dot
export dot

"""
    dot(c::AbstractVector, x::AbstractExpression)

Applies the function:
```math
f(\\mathbf{x}) = \\mathbf{c}^{T}\\mathbf{x}.
```
"""
dot(c::AbstractVector, ex::AbstractExpression) =
    Term(Linear(c), ex)


# Inequalities

import Base: <=
"""
Inequalities constrains

## Norm Inequalities constraints

* `norm(x::AbstractExpression, 0) <= n::Integer`

  ``\\mathrm{nnz}(\\mathbf{x}) \\leq n``

* `norm(x::AbstractExpression, 1) <= r::Number`

  ``\\sum_i \\| x_i \\| \\leq r``

* `norm(x::AbstractExpression, 2) <= r::Number`

  ``\\| \\mathbf{x} \\| \\leq r``

* `norm(x::AbstractExpression, Inf) <= r::Number`

  `` \\max \\{ x_1, x_2, \\dots \\}  \\leq r``

## Box inequality constraints

* `x::AbstractExpression <= u::Union{AbstractArray, Real}`

  `` x_i \\leq u_i ``

* `x::AbstractExpression >= l::Union{AbstractArray, Real}`

  `` x_i \\geq l_i ``

  Notice that the notation `x in [l,u]` is also possible.

## Rank inequality constraints

* `rank(X::AbstractExpression) <= n::Integer`

  ``\\mathrm{rank}(\\mathbf{X}) \\leq r``

  Notice that the expression `X` must have a codomain with dimension equal to 2.

"""
(<=)(t::Term{T1, T2, T3}, r::Integer) where {T1, T2 <: NormL0, T3} =
    Term(IndBallL0(round(Int, r / t.lambda)), t.A)
(<=)(t::Term{T1, T2, T3}, r::Real) where {T1, T2 <: NormL1, T3} = Term(IndBallL1(r / t.lambda), t.A)
(<=)(t::Term{T1, T2, T3}, r::Real) where {T1, T2 <: NormL2, T3} = Term(IndBallL2(r / t.lambda), t.A)
(<=)(t::Term{T1, T2, T3}, r::Real) where {T1, T4 <: IndBallL1, T2 <: Conjugate{T4}, T3} = Term(IndBallLinf(r / t.lambda), t.A)

# Box constraints

import Base: <=, >=, in

(<=)(ex::AbstractExpression, ub) = Term(IndBox(-Inf, ub), ex)
(<=)(lb, ex::AbstractExpression) = Term(IndBox(lb, +Inf), ex)
(>=)(ex::AbstractExpression, lb) = Term(IndBox(lb, +Inf), ex)
(>=)(ub, ex::AbstractExpression) = Term(IndBox(-Inf, ub), ex)

function in(ex::AbstractExpression, bnds::AbstractArray)
    if length(bnds) != 2
        error("should provide 2 bounds!")
    end
    return Term(IndBox(bnds[1], bnds[2]), ex)
end

# Rank constraints

import LinearAlgebra: rank
export rank

# Dirty trick: the "rank" function only makes sense in constraints such as
#   rank(X) <= r,
# therefore here the parameter (1) doesn't really have a role.
# We should probably fix this: it allows weird things in expressing problems.
# Maybe we should have Rank (with no prox! nor gradient!
# defined), that gives IndBallRank when combined with <=.
struct Rank end

"""
    rank(ex::AbstractExpression)

A placeholder term that is only meaningful inside a rank constraint,

    rank(X) <= r

which becomes the indicator of ``\\{\\mathbf{X} : \\mathrm{rank}(\\mathbf{X}) \\leq r\\}``
(`IndBallRank`). The prox is a truncated SVD, so `X` must be a matrix `Variable`.

`rank(ex)` on its own is not a usable objective term: `Rank` implements neither `prox!` nor
`gradient!`, so a problem containing one will not parse. This is the extension of
`LinearAlgebra.rank` to expressions, not a computation of an expression's rank.

```julia
julia> X = Variable(10, 10);

julia> c = rank(X) <= 3
```

See also [`norm`](@ref).
"""
rank(ex::AbstractExpression) = Term(Rank(), ex)

import Base: <=

(<=)(t::Term{T1, T2, T3} where {T1, T2 <: Rank, T3}, r::Int) = Term(IndBallRank(round(Int, r / t.lambda)), t.A)

import Base: ==

"""
Equalities constraints

## Affine space constraint

* `ex == b::Union{Real,AbstractArray}`

  Requires expression to be affine.

  ### Example
  ```julia
  julia> A,b  = randn(10,5), randn(10);

  julia> x = Variable(5);

  julia> A*x == b
  ```

## Norm equality constraint

* `norm(x::AbstractExpression) == r::Number`

  ``\\| \\mathbf{x} \\| = r``

## Binary constraint

* `x::AbstractExpression == (l, u)`

  ``\\mathbf{x} = \\mathbf{l}`` or ``\\mathbf{x} = \\mathbf{u}``

"""
(==)(t::Term{T1, T2, T3}, r::Real) where {T1, T2 <: NormL2, T3} = Term(IndSphereL2(r / t.lambda), t.A)
# IndSphereL2

(==)(ex::AbstractExpression, lu::Tuple{Union{Real, AbstractArray}, Union{Real, AbstractArray}}) =
    Term(IndBinary(lu...), ex)
# IndBinary

# IndPoint, rewritten to IndAffine at parse time where that is the better formulation.
#
# The syntax layer builds `λ · f(A·x + d)` triples and nothing else (PLAN.md 2.6): the
# equality `ex == b` is the indicator of the singleton `{b}` composed with whatever affine
# expression `ex` happens to be. Folding `A` into an `IndAffine` here would hide it from
# every later decision — which is what used to make `DiagOp(a)*x == b` and `fft(x) == b`
# errors, although the first is a trivial projection and the second is AAᴴ-diagonal, and
# what used to discard every variable of `ex` after the first.
# `merge_function_with_operator` now picks the formulation, including today's `IndAffine`
# for a general `MatrixOp`.
(==)(ex::AbstractExpression, b::Union{Real, AbstractArray}) = Term(IndPoint(b), ex)

# Transforms
# Convex conjugate
import Base: conj

"""
    conj(t::Term)

Returns the convex conjugate transform of `t`:
```math
f^*(\\mathbf{x}) = \\sup_{\\mathbf{y}} \\{ \\langle \\mathbf{y}, \\mathbf{x} \\rangle - f(\\mathbf{y}) \\}.
```

Conjugation needs the term's operator to be the identity: `f∘A` has no conjugate this
package can build from `f`'s alone, so anything else is an error.

# Example
```jldoctest
julia> x = Variable(4);

julia> t = conj(norm(x, 1));

julia> t.f isa Conjugate
true

julia> conj(norm(randn(3, 4) * x, 1))
ERROR: cannot perform convex conjugation
```

"""
function conj(t::Term)
    if typeof(operator(t)) <: Eye
        return Term(1.0, Conjugate(Postcompose(t.f, t.lambda)), t.A)
    else
        error("cannot perform convex conjugation")
    end

end

# Moreau Envelope
export smooth

"""
    smooth(t::Term, gamma = 1.0)

Smooths the nonsmooth term `t` using Moreau envelope:

```math
f^{\\gamma}(\\mathbf{x}) = \\min_{\\mathbf{z}} \\left\\{ f(\\mathbf{z}) + \\tfrac{1}{2\\gamma}\\|\\mathbf{z}-\\mathbf{x}\\|^2 \\right\\}.
```

A term that is already smooth is returned unchanged.

# Example
```jldoctest
julia> x = Variable(4);

julia> StructuredOptimization.is_smooth(norm(x, 1))
false

julia> StructuredOptimization.is_smooth(smooth(norm(x, 1)))
true

julia> smooth(ls(x)) === ls(x) || StructuredOptimization.is_smooth(smooth(ls(x)))
true
```

"""
function smooth(t::Term, gamma = 1.0)
    if !is_smooth(t)
        return Term(1.0, MoreauEnvelope(Postcompose(t.f, t.lambda), gamma), t.A)
    else
        return t
    end
end

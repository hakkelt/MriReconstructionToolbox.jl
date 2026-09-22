import ProximalOperators: gradient!, gradient, preallocate # this can be removed when moved to Prox

export PrecomposeNonlinear

"""
    PrecomposeNonlinear(g, G::AbstractOperator)

The composition ``f(\\mathbf{x}) = g(G(\\mathbf{x}))`` of a smooth function `g` with a
*non-linear* operator `G`, exposing only a gradient:
```math
\\nabla f(\\mathbf{x}) = [\\mathrm{D}G(\\mathbf{x})]^* \\, \\nabla g(G(\\mathbf{x})),
```
where ``\\mathrm{D}G(\\mathbf{x})`` is the Jacobian of `G` at `x` — `AbstractOperators`
provides it as `jacobian(G, x)`, so no automatic differentiation is involved.

This is the last formulation [`merge_function_with_operator`](@ref) will pick, reached when
`G` is not linear at all (`ls(sin(x) - b)`, say). It has **no** proximal operator: the
composition of a prox-friendly `g` with a non-linear `G` generally has none in closed form,
so a solver that needs one must not be offered this term. Convexity is likewise not
preserved, which is why such problems only parse for algorithms that tolerate a non-convex
smooth term (`ZeroFPR`, `PANOCplus`), not for `FastForwardBackward`.

The domain, codomain and Jacobian-application buffers are allocated once at construction
and `g` is `preallocate`d for the codomain shape, so a solver iteration allocates nothing
here.

See also [`SqrNormL2WithNormalOp`](@ref), which is the corresponding fused formulation for a
*linear* operator.
"""
struct PrecomposeNonlinear{
        P,
        T <: AbstractOperator,
        D <: AbstractArray,
        C <: AbstractArray,
    }
    g::P
    G::T
    bufD::D
    bufC::C
    bufC2::C
end

function PrecomposeNonlinear(g::P, G::T) where {P, T}
    bufD = AbstractOperators.allocate_in_domain(G)
    bufC = AbstractOperators.allocate_in_codomain(G)
    bufC2 = AbstractOperators.allocate_in_codomain(G)
    # `g` sees `bufC`-shaped input on every call (see `gradient!` below), so it can be
    # preallocated for that shape right away instead of paying its own scratch
    # allocation (if any) on every solver iteration.
    g = preallocate(g, bufC)
    return PrecomposeNonlinear{typeof(g), T, typeof(bufD), typeof(bufC)}(g, G, bufD, bufC, bufC2)
end

is_smooth(f::PrecomposeNonlinear) = is_smooth(f.g)

function (f::PrecomposeNonlinear)(x)
    return f.g(f.G * x)
end

function gradient(f::PrecomposeNonlinear, x::ArrayPartition)
    y = zero(x)
    fy = gradient!(y, f, x)
    return y, fy
end

# ArrayPartition <: AbstractArray, so this one method covers both the single-array
# and the multi-variable (ArrayPartition) cases.
function gradient!(y::D, f::PrecomposeNonlinear{P, T, D, C}, x::D) where {P, T, D <: AbstractArray, C}
    mul!(f.bufC, f.G, x)
    v = gradient!(f.bufC2, f.g, f.bufC)
    J = Jacobian(f.G, x)
    y = mul!(y, J', f.bufC2)
    return v
end

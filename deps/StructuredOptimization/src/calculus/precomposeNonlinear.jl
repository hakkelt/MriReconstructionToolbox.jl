import ..ProximalOperators: gradient!, gradient, preallocate # this can be removed when moved to Prox

export PrecomposeNonlinear

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

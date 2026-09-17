# Regularize

export Regularize

"""
    Regularize(f, ρ=1.0, a=0.0)

Given function `f`, and optional parameters `ρ` (positive) and `a`, return
```math
g(x) = f(x) + \\tfrac{ρ}{2}\\|x-a\\|².
```
Parameter `a` can be either an array or a scalar, in which case it is subtracted component-wise from `x` in the above expression.
"""
struct Regularize{T, S, A, B}
    f::T
    rho::S
    a::A
    buf::B
    function Regularize{T,S,A,B}(f::T, rho::S, a::A, buf::B) where {T, S, A, B}
        if rho <= 0.0
            error("parameter `ρ` must be positive")
        else
            new(f, rho, a, buf)
        end
    end
end

Regularize{T,S,A}(f::T, rho::S, a::A) where {T, S, A} =
    Regularize{T, S, A, Nothing}(f, rho, a, nothing)

regularize(f::T, rho::S, a::A; buf=nothing) where {T, S, A} =
    Regularize{T, S, A, typeof(buf)}(f, rho, a, buf)

preallocate(g::Regularize, x) = regularize(
    preallocate(g.f, x), g.rho, g.a;
    buf = (sig = input_signature(x), z = similar(x)),
)

# ‖y - a‖², with `a` either an array or a scalar broadcast over `y`
@inline sqrdist(y, a::Number) = sum(v -> abs2(v - a), y)
@inline sqrdist(y, a::AbstractArray) = normdiff2(y, a)

# `gr2 .* (x .+ gr .* a)`, written into the buffer when there is one.
@inline regularize_arg(g::Regularize, x, gr, gr2) = _regularize_arg(g, g.buf, x, gr, gr2)
@inline _regularize_arg(g, ::Nothing, x, gr, gr2) = gr2 .* (x .+ gr .* g.a)
@inline function _regularize_arg(g, buf, x, gr, gr2)
    check_signature(buf.sig, x, g)
    buf.z .= gr2 .* (x .+ gr .* g.a)
    return buf.z
end

is_separable(::Type{<:Regularize{T}}) where T = is_separable(T)
is_proximable(::Type{<:Regularize{T}}) where T = is_proximable(T)
is_convex(::Type{<:Regularize{T}}) where T = is_convex(T)
is_smooth(::Type{<:Regularize{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Regularize{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Regularize{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Regularize}) = true

Regularize(f::T, rho::S=1, a::A=0) where {T, S, A} = Regularize{T, S, A}(f, rho, a)

function (g::Regularize)(x)
    return g.f(x) + g.rho/2*sqrdist(x, g.a)
end

function gradient!(y, g::Regularize, x)
    v = gradient!(y, g.f, x)
    y .+= g.rho .* (x .- g.a)
    return v + g.rho / 2 * sqrdist(x, g.a)
end

function prox!(y, g::Regularize, x, gamma)
    R = real(eltype(x))
    gr = g.rho * gamma
    gr2 = R(1) ./ (R(1) .+ gr)
    v = prox!(y, g.f, regularize_arg(g, x, gr, gr2), gr2 .* gamma)
    return v + g.rho / R(2) * sqrdist(y, g.a)
end

function prox_naive(g::Regularize, x, gamma)
    R = real(eltype(x))
    y, v = prox_naive(g.f, x./(R(1) .+ gamma.*g.rho) .+ g.a./(R(1)./(gamma.*g.rho) .+ R(1)), gamma./(R(1) .+ gamma.*g.rho))
    return y, v + g.rho/R(2)*norm(y .- g.a)^2
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:Regularize{T}}) where T = device_tier(T)

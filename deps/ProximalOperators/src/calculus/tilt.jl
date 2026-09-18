# Tilting (addition with affine function)

export Tilt

"""
    Tilt(f, a, b=0.0)

Given function `f`, an array `a` and a constant `b` (optional), return function
```math
g(x) = f(x) + \\langle a, x \\rangle + b.
```
"""
struct Tilt{T, S, R, B}
    f::T
    a::S
    b::R
    buf::B
end

Tilt(f::T, a::S, b::R; buf=nothing) where {T, S, R} =
    Tilt{T, S, R, typeof(buf)}(f, a, b, buf)

preallocate(g::Tilt, x) = Tilt(
    preallocate(g.f, x), g.a, g.b;
    buf = (sig = input_signature(x), z = similar(x)),
)

# `x .- gamma .* a`, written into the buffer when there is one.
@inline tilt_arg(g::Tilt, x, gamma) = _tilt_arg(g, g.buf, x, gamma)
@inline _tilt_arg(g, ::Nothing, x, gamma) = x .- gamma .* g.a
@inline function _tilt_arg(g, buf, x, gamma)
    check_signature(buf.sig, x, g)
    buf.z .= x .- gamma .* g.a
    return buf.z
end

is_separable(::Type{<:Tilt{T}}) where T = is_separable(T)
is_proximable(::Type{<:Tilt{T}}) where T = is_proximable(T)
is_convex(::Type{<:Tilt{T}}) where T = is_convex(T)
is_singleton_indicator(::Type{<:Tilt{T}}) where T = is_singleton_indicator(T)
is_smooth(::Type{<:Tilt{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Tilt{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Tilt{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Tilt{T}}) where T = is_strongly_convex(T)

Tilt(f::T, a::S) where {T, S} = Tilt(f, a, real(eltype(S))(0))

function (g::Tilt)(x)
    return g.f(x) + real(dot(g.a, x)) + g.b
end

function prox!(y, g::Tilt, x, gamma)
    v = prox!(y, g.f, tilt_arg(g, x, gamma), gamma)
    return v + real(dot(g.a, y)) + g.b
end

function prox_naive(g::Tilt, x, gamma)
    y, v = prox_naive(g.f, x .- gamma .* g.a, gamma)
    return y, v + real(dot(g.a, y)) + g.b
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:Tilt{T}}) where T = device_tier(T)

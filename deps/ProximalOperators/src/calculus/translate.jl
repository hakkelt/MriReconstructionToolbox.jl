export Translate

"""
    Translate(f, b)

Return the translated function
```math
g(x) = f(x + b)
```
"""
struct Translate{T, V, B}
    f::T
    b::V
    buf::B
end

Translate(f::T, b::V; buf=nothing) where {T, V} = Translate{T, V, typeof(buf)}(f, b, buf)

function preallocate(g::Translate, x)
    z = x .+ g.b
    return Translate(
        preallocate(g.f, z), g.b; buf = (sig = input_signature(x), z = z)
    )
end

# `x + b`, written into the buffer when there is one.
@inline translate_arg(g::Translate, x) = _translate_arg(g, g.buf, x)
@inline _translate_arg(g, ::Nothing, x) = x .+ g.b
@inline function _translate_arg(g, buf, x)
    check_signature(buf.sig, x, g)
    buf.z .= x .+ g.b
    return buf.z
end

is_separable(::Type{<:Translate{T}}) where T = is_separable(T)
is_proximable(::Type{<:Translate{T}}) where T = is_proximable(T)
is_convex(::Type{<:Translate{T}}) where T = is_convex(T)
is_set_indicator(::Type{<:Translate{T}}) where T = is_set_indicator(T)
is_singleton_indicator(::Type{<:Translate{T}}) where T = is_singleton_indicator(T)
is_cone_indicator(::Type{<:Translate{T}}) where T = is_cone_indicator(T)
is_affine_indicator(::Type{<:Translate{T}}) where T = is_affine_indicator(T)
is_smooth(::Type{<:Translate{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Translate{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Translate{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Translate{T}}) where T = is_strongly_convex(T)

function (g::Translate)(x)
    return g.f(x .+ g.b)
end

function gradient!(y, g::Translate, x)
    v = gradient!(y, g.f, translate_arg(g, x))
    return v
end

function prox!(y, g::Translate, x, gamma)
    v = prox!(y, g.f, translate_arg(g, x), gamma)
    y .-= g.b
    return v
end

function prox_naive(g::Translate, x, gamma)
    y, v = prox_naive(g.f, x .+ g.b, gamma)
    return y - g.b, v
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:Translate{T}}) where T = device_tier(T)

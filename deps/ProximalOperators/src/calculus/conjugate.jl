# Conjugate

export Conjugate

"""
    Conjugate(f)

Return the convex conjugate (also known as Fenchel conjugate, or Fenchel-Legendre transform) of function `f`, that is
```math
f^*(x) = \\sup_y \\{ \\langle y, x \\rangle - f(y) \\}.
```
"""
struct Conjugate{T, B}
    f::T
    buf::B
    function Conjugate{T, B}(f::T, buf::B) where {T, B}
        if is_convex(f) == false
            error("`f` must be convex")
        end
        new(f, buf)
    end
end

is_proximable(::Type{<:Conjugate{T}}) where T = is_proximable(T)
is_convex(::Type{<:Conjugate{T}}) where T = true
is_cone_indicator(::Type{<:Conjugate{T}}) where T = is_cone_indicator(T) && is_convex(T)
is_smooth(::Type{<:Conjugate{T}}) where T = is_strongly_convex(T)
is_strongly_convex(::Type{<:Conjugate{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Conjugate{T}}) where T = is_generalized_quadratic(T)
is_set_indicator(::Type{<:Conjugate{T}}) where T = is_convex(T) && is_support(T)
is_positively_homogeneous(::Type{<:Conjugate{T}}) where T = is_convex(T) && is_set_indicator(T)

Conjugate(f::T; buf=nothing) where T = Conjugate{T, typeof(buf)}(f, buf)

Conjugate{T}(f::T) where T = Conjugate{T, Nothing}(f, nothing)

# `prox!` evaluates the inner `prox!` at `x/gamma`, which lives in the same space
# as `x`, so the inner operator gets the same prototype.
preallocate(g::Conjugate, x::AbstractArray) = Conjugate(
    preallocate(g.f, x); buf = (sig = input_signature(x), xg = similar(x))
)

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

# `x/gamma`, written into the buffer when there is one. The non-preallocated
# branch keeps the original expression, so it works for any input type for
# which `/` is defined and allocates exactly as much as before.
@inline conjugate_arg(g::Conjugate, x, gamma) = _conjugate_arg(g, g.buf, x, gamma)
@inline _conjugate_arg(g, ::Nothing, x, gamma) = x / gamma
@inline function _conjugate_arg(g, buf, x, gamma)
    check_signature(buf.sig, x, g)
    buf.xg .= x ./ gamma
    return buf.xg
end

function prox!(y, g::Conjugate, x, gamma)
    # Moreau identity
    v = prox!(y, g.f, conjugate_arg(g, x, gamma), 1/gamma)
    if is_set_indicator(g)
        v = real(eltype(x))(0)
    else
        v = real(dot(x, y)) - gamma * real(dot(y, y)) - v
    end
    y .= x .- gamma .* y
    return v
end

# naive implementation

function prox_naive(g::Conjugate, x, gamma)
    y, v = prox_naive(g.f, x/gamma, 1/gamma)
    return x - gamma * y, if is_set_indicator(g) real(eltype(x))(0) else real(dot(x, y)) - gamma * real(dot(y, y)) - v end
end

# TODO: hard-code conjugation rules? E.g. precompose/epicompose

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:Conjugate{T}}) where T = device_tier(T)

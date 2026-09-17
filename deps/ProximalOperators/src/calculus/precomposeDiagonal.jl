# Precompose with diagonal scaling and translation

export PrecomposeDiagonal

"""
    PrecomposeDiagonal(f, a, b)

Return the function
```math
g(x) = f(\\mathrm{diag}(a)x + b)
```
Function ``f`` must be convex and separable, or `a` must be a scalar, for the
`prox` of ``g`` to be computable. Parametes `a` and `b` can be arrays of
multiple dimensions, according to the shape/size of the input `x` that will be
provided to the function: the way the above expression for ``g`` should be
thought of, is `g(x) = f(a.*x + b)`.
"""
struct PrecomposeDiagonal{T, R, S, B}
    f::T
    a::R
    b::S
    buf::B
    function PrecomposeDiagonal{T,R,S,B}(f::T, a::R, b::S, buf::B) where {T, R, S, B}
        if R <: AbstractArray && !(is_convex(f) && is_separable(f))
            error("`f` must be convex and separable since `a` is of type $(R)")
        end
        if any(a == 0)
            error("elements of `a` must be nonzero")
        else
            new(f, a, b, buf)
        end
    end
end

PrecomposeDiagonal{T,R,S}(f::T, a::R, b::S) where {T, R, S} =
    PrecomposeDiagonal{T, R, S, Nothing}(f, a, b, nothing)

is_separable(::Type{<:PrecomposeDiagonal{T}}) where T = is_separable(T)
is_proximable(::Type{<:PrecomposeDiagonal{T}}) where T = is_proximable(T)
is_convex(::Type{<:PrecomposeDiagonal{T}}) where T = is_convex(T)
is_set_indicator(::Type{<:PrecomposeDiagonal{T}}) where T = is_set_indicator(T)
is_singleton_indicator(::Type{<:PrecomposeDiagonal{T}}) where T = is_singleton_indicator(T)
is_cone_indicator(::Type{<:PrecomposeDiagonal{T}}) where T = is_cone_indicator(T)
is_affine_indicator(::Type{<:PrecomposeDiagonal{T}}) where T = is_affine_indicator(T)
is_smooth(::Type{<:PrecomposeDiagonal{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:PrecomposeDiagonal{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:PrecomposeDiagonal{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:PrecomposeDiagonal{T}}) where T = is_strongly_convex(T)

PrecomposeDiagonal(f::T, a::S=1, b::S=0) where {T, S <: Real} = PrecomposeDiagonal{T, S, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S=0) where {T, R <: AbstractArray, S <: Real} = PrecomposeDiagonal{T, R, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S) where {T, R <: Union{AbstractArray, Real}, S <: AbstractArray} = PrecomposeDiagonal{T, R, S}(f, a, b)

precompose_diagonal(f::T, a::R, b::S; buf=nothing) where {T, R, S} =
    PrecomposeDiagonal{T, R, S, typeof(buf)}(f, a, b, buf)

function preallocate(g::PrecomposeDiagonal, x)
    z = g.a .* x .+ g.b
    # the step size passed to the inner prox is an array only when `a` is one
    agam = g.a isa AbstractArray ? similar(z, real(eltype(z))) : nothing
    return precompose_diagonal(
        preallocate(g.f, z), g.a, g.b;
        buf = (sig = input_signature(x), z = z, agam = agam),
    )
end

# `a .* x .+ b`, written into the buffer when there is one.
@inline diagonal_arg(g::PrecomposeDiagonal, x) = _diagonal_arg(g, g.buf, x)
@inline _diagonal_arg(g, ::Nothing, x) = g.a .* x .+ g.b
@inline function _diagonal_arg(g, buf, x)
    check_signature(buf.sig, x, g)
    buf.z .= g.a .* x .+ g.b
    return buf.z
end

# `(a .* a) .* gamma`: only allocates, and only needs a buffer, when `a` is an array.
@inline diagonal_gamma(g::PrecomposeDiagonal, gamma) = _diagonal_gamma(g, g.buf, gamma)
@inline _diagonal_gamma(g, ::Nothing, gamma) = (g.a .* g.a) .* gamma
@inline _diagonal_gamma(g, buf, gamma) = _diagonal_gamma_buf(g, buf.agam, gamma)
@inline _diagonal_gamma_buf(g, ::Nothing, gamma) = (g.a .* g.a) .* gamma
@inline function _diagonal_gamma_buf(g, agam, gamma)
    agam .= (g.a .* g.a) .* gamma
    return agam
end

function (g::PrecomposeDiagonal)(x)
    return g.f(g.a .* x .+ g.b)
end

function gradient!(y, g::PrecomposeDiagonal, x)
    v = gradient!(y, g.f, diagonal_arg(g, x))
    y .*= g.a
    return v
end

function prox!(y, g::PrecomposeDiagonal, x, gamma)
    v = prox!(y, g.f, diagonal_arg(g, x), diagonal_gamma(g, gamma))
    y .-= g.b
    y ./= g.a
    return v
end

function prox_naive(g::PrecomposeDiagonal, x, gamma)
    z = g.a .* x .+ g.b
    y, fy = prox_naive(g.f, z, (g.a .* g.a) .* gamma)
    return (y .- g.b)./g.a, fy
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:PrecomposeDiagonal{T}}) where T = device_tier(T)

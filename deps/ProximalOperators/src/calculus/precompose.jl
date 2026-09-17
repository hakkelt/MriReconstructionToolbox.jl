# Precompose with a linear mapping + translation (= affine)

export Precompose

"""
    Precompose(f, L, μ, b)

Return the function
```math
g(x) = f(Lx + b)
```
where ``f`` is a convex function and ``L`` is a linear mapping: this must
satisfy ``LL^* = μI`` for ``μ > 0``. Furthermore, either ``f`` is separable or
parameter `μ` is a scalar, for the `prox` of ``g`` to be computable.

Parameter `L` defines ``L`` through the `mul!` method. Therefore `L` can be an
`AbstractMatrix` for example, but not necessarily.

In this case, `prox` and `prox!` are computed according to Prop. 24.14 in
Bauschke, Combettes "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd edition, 2016. The same result is Prop. 23.32 in the 1st edition
of the same book.
"""
struct Precompose{T, M, U, V, B}
    f::T
    L::M
    mu::U
    b::V
    buf::B
    function Precompose{T, M, U, V, B}(f::T, L::M, mu::U, b::V, buf::B) where {T, M, U, V, B}
        if !is_convex(f)
            error("f must be convex")
        end
        if any(mu .<= 0)
            error("elements of μ must be positive")
        end
        new(f, L, mu, b, buf)
    end
end

is_proximable(::Type{<:Precompose{T}}) where T = is_proximable(T)
is_convex(::Type{<:Precompose{T}}) where T = is_convex(T)
is_set_indicator(::Type{<:Precompose{T}}) where T = is_set_indicator(T)
is_singleton_indicator(::Type{<:Precompose{T}}) where T = is_singleton_indicator(T)
is_cone_indicator(::Type{<:Precompose{T}}) where T = is_cone_indicator(T)
is_affine_indicator(::Type{<:Precompose{T}}) where T = is_affine_indicator(T)
is_smooth(::Type{<:Precompose{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Precompose{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Precompose{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Precompose{T}}) where T = is_strongly_convex(T)

Precompose(f::T, L::M, mu::U, b::V; buf=nothing) where {T, M, U, V} =
    Precompose{T, M, U, V, typeof(buf)}(f, L, mu, b, buf)

Precompose{T, M, U, V}(f::T, L::M, mu::U, b::V) where {T, M, U, V} =
    Precompose{T, M, U, V, Nothing}(f, L, mu, b, nothing)

Precompose(f::T, L::M, mu::U) where {T, M, U} = Precompose(f, L, mu, 0)

function preallocate(g::Precompose, x)
    res = g.L * x
    if g.b != 0
        res = res .+ g.b
    end
    return Precompose(
        preallocate(g.f, res), g.L, g.mu, g.b;
        buf = (sig = input_signature(x), res = res, out = similar(res)),
    )
end

function (g::Precompose)(x)
    return g.f(g.L * x .+ g.b)
end

# Returns the buffers with `res` already set to `L*x` (`+ b` when `b != 0`). Unlike the
# generic `get_buffers`, this avoids recomputing `L*x` on the non-preallocated path, and
# keeps the zero-`b` case free of the translation add the un-preallocated path already
# skipped.
@inline precompose_buffers(g::Precompose, x) = _precompose_buffers(g, g.buf, x)

@inline function _precompose_buffers(g::Precompose, ::Nothing, x)
    res = g.L * x
    if g.b != 0
        res = res .+ g.b
    end
    return (sig = nothing, res = res, out = similar(res))
end

@inline function _precompose_buffers(g::Precompose, buf, x)
    check_signature(buf.sig, x, g)
    mul!(buf.res, g.L, x)
    if g.b != 0
        buf.res .+= g.b
    end
    return buf
end

function gradient!(y, g::Precompose, x)
    b = precompose_buffers(g, x)
    v = gradient!(b.out, g.f, b.res)
    mul!(y, adjoint(g.L), b.out)
    return v
end

function prox!(y, g::Precompose, x, gamma)
    # See Prop. 24.14 in Bauschke, Combettes
    # "Convex Analysis and Monotone Operator Theory in Hilbert Spaces",
    # 2nd ed., 2016.
    #
    # The same result is Prop. 23.32 in the 1st ed. of the same book.
    #
    # This case has an additional translation: if f(x) = h(x + b) then
    #     prox_f(x) = prox_h(x + b) - b
    # Then one can apply the above mentioned result to g(x) = f(Lx).
    #
    b = precompose_buffers(g, x)
    v = prox!(b.out, g.f, b.res, g.mu.*gamma)
    b.out .-= b.res
    b.out ./= g.mu
    mul!(y, adjoint(g.L), b.out)
    y .+= x
    return v
end

function prox_naive(g::Precompose, x, gamma)
    res = g.L*x .+ g.b
    proxres, v = prox_naive(g.f, res, g.mu .* gamma)
    y = x + g.L'*((proxres .- res)./g.mu)
    return y, v
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:Precompose{T}}) where T = device_tier(T)

# indicator of the zero cone

export IndZero

"""
    IndZero(; threaded=true)

Return the indicator function of the set containing the origin, the "zero cone".

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndZero{Th} end

IndZero(; threaded::Bool=true) = IndZero{threaded}()

is_separable(f::Type{<:IndZero}) = true
is_convex(f::Type{<:IndZero}) = true
is_singleton_indicator(f::Type{<:IndZero}) = true
is_cone_indicator(f::Type{<:IndZero}) = true
is_affine_indicator(f::Type{<:IndZero}) = true

@threadable IndZero{Th} Arithmetic

function (f::IndZero)(x)
    C = eltype(x)
    return all_satisfy(f, xk -> xk == C(0), x) ? real(C)(0) : real(C)(Inf)
end

function prox!(y, f::IndZero, x, gamma)
    T = eltype(y)
    map_prox!(f, y, _ -> T(0), x)
    return real(eltype(x))(0)
end

function prox_naive(::IndZero, x, gamma)
    return zero(x), real(eltype(x))(0)
end

# indicator of nonpositive orthant

export IndNonpositive

"""
    IndNonpositive(; threaded=true)

Return the indicator of the nonpositive orthant
```math
C = \\{ x : x \\leq 0 \\}.
```

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndNonpositive{Th} end

IndNonpositive(; threaded::Bool=true) = IndNonpositive{threaded}()

is_separable(f::Type{<:IndNonpositive}) = true
is_convex(f::Type{<:IndNonpositive}) = true
is_cone_indicator(f::Type{<:IndNonpositive}) = true

@threadable IndNonpositive{Th} Arithmetic

function (f::IndNonpositive)(x)
    R = eltype(x)
    return all_satisfy(f, xk -> xk <= 0, x) ? R(0) : R(Inf)
end

function prox!(y, f::IndNonpositive, x, gamma)
    R = eltype(x)
    map_prox!(f, y, xk -> min(xk, R(0)), x)
    return R(0)
end

function prox_naive(::IndNonpositive, x, gamma)
    R = eltype(x)
    y = min.(R(0), x)
    return y, R(0)
end

# indicator of nonnegative orthant

export IndNonnegative

"""
    IndNonnegative(; threaded=true)

Return the indicator of the nonnegative orthant
```math
C = \\{ x : x \\geq 0 \\}.
```

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndNonnegative{Th} end

IndNonnegative(; threaded::Bool=true) = IndNonnegative{threaded}()

is_separable(f::Type{<:IndNonnegative}) = true
is_convex(f::Type{<:IndNonnegative}) = true
is_cone_indicator(f::Type{<:IndNonnegative}) = true

@threadable IndNonnegative{Th} Arithmetic

function (f::IndNonnegative)(x)
    R = eltype(x)
    return all_satisfy(f, xk -> xk >= 0, x) ? R(0) : R(Inf)
end

function prox!(y, f::IndNonnegative, x, gamma)
    R = eltype(x)
    map_prox!(f, y, xk -> max(xk, R(0)), x)
    return R(0)
end

function prox_naive(::IndNonnegative, x, gamma)
    R = eltype(x)
    y = max.(R(0), x)
    return y, R(0)
end

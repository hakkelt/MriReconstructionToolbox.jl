# indicator of the Cartesian product of real binary sets

export IndBinary

"""
    IndBinary(low, up; threaded=true)

Return the indicator function of the set
```math
S = \\{ x : x_i = low_i\\ \\text{or}\\ x_i = up_i \\},
```
Parameters `low` and `up` can be either scalars or arrays of the same dimension as the space.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndBinary{T, S, Th}
    low::T
    high::S
end

IndBinary(low::T, high::S; threaded::Bool=true) where {T, S} =
    IndBinary{T, S, threaded}(low, high)

is_set_indicator(f::Type{<:IndBinary}) = true

@threadable IndBinary{<:Any, <:Any, Th} Arithmetic

IndBinary() = IndBinary(0, 1)

IndBinary_low(f::IndBinary{<: Number}, i) = f.low
IndBinary_low(f::IndBinary, i) = f.low[i]
IndBinary_high(f::IndBinary{<:Any, <: Number}, i) = f.high
IndBinary_high(f::IndBinary, i) = f.high[i]

function (f::IndBinary)(x)
    R = real(eltype(x))
    ok = all_satisfy_idx(f, (k, xk) -> xk == IndBinary_low(f, k) || xk == IndBinary_high(f, k), x)
    return ok ? R(0) : R(Inf)
end

function prox!(y, f::IndBinary, x, gamma)
    T = eltype(y)
    map_prox_idx!(f, y, function (k, xk)
        low = T(IndBinary_low(f, k))
        high = T(IndBinary_high(f, k))
        abs(xk - low) < abs(xk - high) ? low : high
    end, x)
    return real(eltype(x))(0)
end

function prox_naive(f::IndBinary, x, gamma)
    distlow = abs.(x .- f.low)
    disthigh = abs.(x .- f.high)
    indlow = distlow .< disthigh
    indhigh = distlow .>= disthigh
    y = f.low.*indlow + f.high.*indhigh
    return y, real(eltype(x))(0)
end

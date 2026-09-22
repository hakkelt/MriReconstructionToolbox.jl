# indicator of a generic box

export IndBox, IndBallLinf

"""
    IndBox(low, up; threaded=true)

Return the indicator function of the box
```math
S = \\{ x : low \\leq x \\leq up \\}.
```
Parameters `low` and `up` can be either scalars or arrays of the same dimension as the space: they must satisfy `low <= up`, and are allowed to take values `-Inf` and `+Inf` to indicate unbounded coordinates.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndBox{T, S, Th}
    lb::T
    ub::S
    function IndBox{T,S,Th}(lb::T, ub::S) where {T, S, Th}
        if !(eltype(lb) <: Real && eltype(ub) <: Real)
            error("`lb` and `ub` must be real")
        end
        if any(lb .> ub)
            error("`lb` and `ub` must satisfy `lb <= ub`")
        else
            new(lb, ub)
        end
    end
end

is_separable(f::Type{<:IndBox}) = true
is_convex(f::Type{<:IndBox}) = true
is_set_indicator(f::Type{<:IndBox}) = true

@threadable IndBox{<:Any, <:Any, Th} Arithmetic

compatible_bounds(::Real, ::Real) = true
compatible_bounds(::Real, ::AbstractArray) = true
compatible_bounds(::AbstractArray, ::Real) = true
compatible_bounds(lb::AbstractArray, ub::AbstractArray) = size(lb) == size(ub)

IndBox(lb, ub; threaded::Bool=true) = if compatible_bounds(lb, ub)
    IndBox{typeof(lb), typeof(ub), threaded}(lb, ub)
else
    error("bounds must have the same dimensions, or at least one of them be scalar")
end

function (f::IndBox)(x)
    R = eltype(x)
    lb, ub = f.lb, f.ub
    # the early `return R(Inf)` became an `all`-reduction: an exit from inside the loop
    # cannot live in a threaded or device kernel
    inside = all_satisfy_idx(f, (k, xk) -> get_kth_elem(lb, k) <= xk <= get_kth_elem(ub, k), x)
    return inside ? R(0) : R(Inf)
end

function prox!(y, f::IndBox, x, gamma)
    lb, ub = f.lb, f.ub
    map_prox_idx!(f, y, (k, xk) -> clamp(xk, get_kth_elem(lb, k), get_kth_elem(ub, k)), x)
    return eltype(x)(0)
end

"""
**Indicator of a ``L_∞`` norm ball**

    IndBallLinf(r=1.0)

Return the indicator function of the set
```math
S = \\{ x : \\max (|x_i|) \\leq r \\}.
```
Parameter `r` must be positive.
"""
IndBallLinf(r::R=1; threaded::Bool=true) where R = IndBox(-r, r; threaded)

function prox_naive(f::IndBox, x, gamma)
    y = min.(f.ub, max.(f.lb, x))
    return y, eltype(x)(0)
end

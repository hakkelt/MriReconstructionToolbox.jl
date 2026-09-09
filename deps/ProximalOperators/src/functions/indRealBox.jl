# indicator of a real box, after discarding the imaginary part

export IndRealBox

"""
    IndRealBox(low, up)

For a **complex** input, return the indicator of the set of complex numbers whose imaginary
part is zero and whose real part lies in `[low, up]`,
```math
S = \\{ x \\in \\mathbb{C} : \\mathrm{Im}(x) = 0,\\ low \\leq \\mathrm{Re}(x) \\leq up \\}.
```
`low` and `up` can be scalars or arrays of the same dimension as the space, and must satisfy
`low <= up`; `-Inf`/`+Inf` indicate an unbounded coordinate. The projection zeros the imaginary
part and clamps the real part to `[low, up]`, following the same real-projection-then-bound
convention as [`IndRealNonnegative`](@ref). For a real-valued input, use [`IndBox`](@ref)
instead.
"""
struct IndRealBox{T,S}
    lb::T
    ub::S
    function IndRealBox{T,S}(lb::T, ub::S) where {T,S}
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

is_separable(f::Type{<:IndRealBox}) = true
is_convex(f::Type{<:IndRealBox}) = true
is_set_indicator(f::Type{<:IndRealBox}) = true

IndRealBox(lb, ub) =
    if compatible_bounds(lb, ub)
        IndRealBox{typeof(lb),typeof(ub)}(lb, ub)
    else
        error("bounds must have the same dimensions, or at least one of them be scalar")
    end

function (f::IndRealBox)(x::AbstractArray{<:Complex})
    R = real(eltype(x))
    for k in eachindex(x)
        xk = x[k]
        if imag(xk) != 0 || real(xk) < get_kth_elem(f.lb, k) || real(xk) > get_kth_elem(f.ub, k)
            return R(Inf)
        end
    end
    return R(0)
end

function prox!(y, f::IndRealBox, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    for k in eachindex(x)
        r = clamp(real(x[k]), get_kth_elem(f.lb, k), get_kth_elem(f.ub, k))
        y[k] = complex(r, R(0))
    end
    return R(0)
end

function prox_naive(f::IndRealBox, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    y = complex.(min.(f.ub, max.(f.lb, real.(x))), R(0))
    return y, R(0)
end

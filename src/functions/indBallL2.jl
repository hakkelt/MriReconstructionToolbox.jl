# indicator of the L2 norm ball with given radius

export IndBallL2

"""
    IndBallL2(r=1.0; threaded=true)

Return the indicator function of the Euclidean ball
```math
S = \\{ x : \\|x\\| \\leq r \\},
```
where ``\\|\\cdot\\|`` is the ``L_2`` (Euclidean) norm. Parameter `r` must be positive.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct IndBallL2{R, Th}
    r::R
    function IndBallL2{R, Th}(r::R) where {R, Th}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::Type{<:IndBallL2}) = true
is_set_indicator(f::Type{<:IndBallL2}) = true

@threadable IndBallL2{<:Any, Th} MemoryBound

IndBallL2(r::R=1; threaded::Bool=true) where R = IndBallL2{R, threaded}(r)

function (f::IndBallL2)(x)
    R = real(eltype(x))
    if isapprox_le(norm(x), f.r, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndBallL2, x, gamma)
    R = real(eltype(x))
    scal = f.r/norm(x)
    if scal > 1
        y .= x
        return R(0)
    end
    map_prox!(f, y, xk -> scal * xk, x)
    return R(0)
end

function prox_naive(f::IndBallL2, x, gamma)
    normx = norm(x)
    if normx > f.r
        y = (f.r/normx)*x
    else
        y = x
    end
    return y, real(eltype(x))(0)
end

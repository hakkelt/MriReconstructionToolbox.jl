# indicator of the ball of matrices with (at most) a given rank

using LinearAlgebra
using TSVD

export IndBallRank

"""
    IndBallRank(r=1)

Return the indicator function of the set of matrices of rank at most `r`:
```math
S = \\{ X : \\mathrm{rank}(X) \\leq r \\},
```
Parameter `r` must be a positive integer.
"""
struct IndBallRank{I, B}
    r::I
    buf::B
    function IndBallRank{I, B}(r::I, buf::B) where {I, B}
        return if r <= 0
            error("parameter r must be a positive integer")
        else
            new(r, buf)
        end
    end
end

is_set_indicator(f::Type{<:IndBallRank}) = true
is_proximable(f::Type{<:IndBallRank}) = true

IndBallRank(r::I = 1; buf = nothing) where {I} = IndBallRank{I, typeof(buf)}(r, buf)
IndBallRank{I}(r::I) where {I} = IndBallRank{I, Nothing}(r, nothing)

# `tsvd` has no in-place entry point, so it keeps allocating its factors on
# every call; only the product buffer can be preallocated here.
preallocate(f::IndBallRank, x::AbstractMatrix) = IndBallRank(
    f.r; buf = (sig = input_signature(x), M = similar(x, f.r, size(x, 2)))
)

function (f::IndBallRank)(x)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        return R(0)
    end
    U, S, V = tsvd(x, f.r + 1)
    # the tolerance in the following line should be customizable
    if S[end] / S[1] <= 1.0e-7
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndBallRank, x, gamma)
    R = real(eltype(x))
    check_input(f, x)
    maxr = minimum(size(x))
    if maxr <= f.r
        y .= x
        return R(0)
    end
    b = get_buffers(f, x)
    U, S, V = tsvd(x, f.r)
    # TODO: the order of the following matrix products should depend on the shape of x
    M = b.M
    M .= S .* V'
    mul!(y, U, M)
    return R(0)
end

function prox_naive(f::IndBallRank, x, gamma)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        y = x
        return y, R(0)
    end
    F = svd(x)
    y = F.U[:, 1:f.r] * (Diagonal(F.S[1:f.r]) * F.V[:, 1:f.r]')
    return y, R(0)
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:IndBallRank}) = :device_lapack

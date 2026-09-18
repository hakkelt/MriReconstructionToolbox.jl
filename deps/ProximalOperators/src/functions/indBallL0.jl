# indicator of the L0 norm ball with given (integer) radius

export IndBallL0

"""
    IndBallL0(r=1)

Return the indicator function of the ``L_0`` pseudo-norm ball
```math
S = \\{ x : \\mathrm{nnz}(x) \\leq r \\}.
```
Parameter `r` must be a positive integer.
"""
struct IndBallL0{I}
    r::I
    function IndBallL0{I}(r::I) where {I}
        if r <= 0
            error("parameter r must be a positive integer")
        else
            new(r)
        end
    end
end

is_set_indicator(f::Type{<:IndBallL0}) = true

IndBallL0(r::I) where {I} = IndBallL0{I}(r)

function (f::IndBallL0)(x)
    R = real(eltype(x))
    # a `sum` over a 0/1 map rather than `count`, which is not available on every backend
    if sum(xi -> iszero(xi) ? 0 : 1, x) > f.r
        return R(Inf)
    end
    return R(0)
end

function _get_top_k_abs_indices(x::AbstractVector, k)
    range = firstindex(x):(firstindex(x) + k - 1)
    return partialsortperm(x, range, by=abs, rev=true)
end

_get_top_k_abs_indices(x, k) = _get_top_k_abs_indices(x[:], k)

# `partialsortperm` needs a sortable, scalar-indexable array. On storage that is neither --
# a device array, or any array type the package does not recognise -- the same selection is
# made by bisecting the threshold that leaves `r` entries above it, which is built from
# reductions and a single broadcast. Where the r-th and (r+1)-th magnitudes are equal the
# two paths may keep a different subset, just as `partialsortperm` itself is unspecified on
# ties; both results satisfy `nnz ≤ r`.
function prox!(y, f::IndBallL0, x, gamma)
    T = eltype(x)
    if is_cpu_storage(typeof(x))
        p = _get_top_k_abs_indices(x, f.r)
        # The surviving entries are read out before `y` is cleared, so that `prox!(x, f, x, gamma)` (an in-place
        # step, which callers are allowed to take) does not zero them out before they are copied.
        kept = [x[i] for i in p]
        y .= T(0)
        for i in eachindex(p)
            y[p[i]] = kept[i]
        end
    else
        # The broadcast reads `x` and writes `y` elementwise, so it is in-place safe as it stands.
        tau = kth_largest_abs(x, f.r, real(T))
        y .= ifelse.(abs.(x) .> tau, x, zero(T))
    end
    return real(T)(0)
end

function prox_naive(f::IndBallL0, x, gamma)
    T = eltype(x)
    p = sortperm(abs.(x)[:], rev=true)
    y = similar(x)
    y[p[begin:begin+f.r-1]] .= x[p[begin:begin+f.r-1]]
    y[p[begin+f.r:end]] .= T(0)
    return y, real(T)(0)
end

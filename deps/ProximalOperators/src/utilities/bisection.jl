# Threshold bisection.
#
# Several projections reduce to "find the threshold τ that makes a monotone quantity hit a
# target, then apply it elementwise":
#
#   simplex       Σ max(xᵢ - τ, 0) = a          y = max(x - τ, 0)
#   L1 ball       Σ max(|xᵢ| - τ, 0) = r        y = sign(x)·max(|x| - τ, 0)
#   top-k         #{i : |xᵢ| > τ} = k           keep the entries above τ
#
# On the CPU these are best solved by a sort or by Condat's algorithm, which are exact and
# single-pass. Neither can run on a device: a sort needs a device sort (GPUArrays ships no
# generic one) and Condat is inherently sequential, pushing and popping stacks one element
# at a time.
#
# Bisection needs none of that. Every step is a `sum`, a `count`, a `min`/`max` or a
# broadcast -- exactly the operations every GPU array library implements generically -- so
# the same code runs on any storage, with no scalar indexing anywhere and no allocation.
#
# It is a *second* algorithm, not a replacement: it costs ~50 full passes over the data
# where Condat costs one, so CPU arrays keep the existing exact algorithm and only storage
# the package cannot index element-by-element takes this path. See `is_cpu_storage`.

# Bisection converges one bit per iteration, so this bound is reached only if the interval
# never collapses; `Float64` needs ~60 and `Float32` ~30, and the loop exits on width.
const BISECTION_MAXIT = 200

"""
    bisect_bracket(phi, lo, hi, target) -> (lo, hi)

Narrow `[lo, hi]` around the point where the non-increasing `phi` crosses `target`, keeping
the invariant `phi(lo) > target ≥ phi(hi)`. `lo` must start with `phi(lo) > target` and `hi`
with `phi(hi) ≤ target`.

The bracket rather than a single number is returned because which end to take depends on the
caller. For a continuous `phi` -- the simplex and `L1`-ball thresholds -- the two ends are
within a rounding error of each other and the midpoint is the natural answer. For a *step*
`phi` such as a count they are not interchangeable: only the `hi` end is guaranteed to
satisfy `phi ≤ target`, and taking the midpoint could select one element too many.

Halving stops when the bracket is narrower than the floating-point spacing at its own
magnitude, so the answer is as sharp as `R` permits rather than as sharp as a fixed
iteration count happens to leave it.
"""
@inline function bisect_bracket(phi::F, lo::R, hi::R, target::R) where {F, R <: Real}
    for _ in 1:BISECTION_MAXIT
        hi - lo <= (one(R) + abs(hi)) * eps(R) && break
        mid = lo + (hi - lo) / 2
        # a midpoint that does not move would loop forever
        (mid <= lo || mid >= hi) && break
        if phi(mid) > target
            lo = mid
        else
            hi = mid
        end
    end
    return lo, hi
end

"""
    bisect_threshold(phi, lo, hi, target) -> τ

The midpoint of [`bisect_bracket`](@ref); the right answer for a continuous `phi`.
"""
@inline function bisect_threshold(phi::F, lo::R, hi::R, target::R) where {F, R <: Real}
    lo, hi = bisect_bracket(phi, lo, hi, target)
    return lo + (hi - lo) / 2
end

"""
    simplex_threshold(x, a, ::Type{R}) -> τ

The `τ` for which `Σ max(xᵢ - τ, 0) = a`, so that `max.(x .- τ, 0)` is the projection of `x`
onto `{y ≥ 0, Σ y = a}`.

The bracket is exact rather than heuristic: at `τ = max(x)` the sum is `0 ≤ a`, and at
`τ = min(x) - a/n` every term is at least `a/n`, so the sum is at least `a`.
"""
function simplex_threshold(x, a, ::Type{R}) where {R <: Real}
    n = length(x)
    lo = R(minimum(x)) - R(a) / n
    hi = R(maximum(x))
    return bisect_threshold(t -> sum(xi -> max(R(xi) - t, zero(R)), x), lo, hi, R(a))
end

"""
    simplex_proj_bisect!(y, a, x)

Project `x` onto the simplex `{y ≥ 0, Σ y = a}` using [`simplex_threshold`](@ref). The
storage-generic counterpart of `simplex_proj_condat!`.
"""
function simplex_proj_bisect!(y, a, x)
    R = real(eltype(x))
    tau = simplex_threshold(x, a, R)
    y .= max.(x .- tau, zero(R))
    return y
end

"""
    l1_ball_threshold(x, r, ::Type{R}) -> τ

The `τ` for which `Σ max(|xᵢ| - τ, 0) = r`. Soft-thresholding `x` at `τ` projects it onto
the `L1` ball of radius `r`.
"""
function l1_ball_threshold(x, r, ::Type{R}) where {R <: Real}
    lo = zero(R)
    hi = R(maximum(abs, x))
    return bisect_threshold(t -> sum(xi -> max(abs(xi) - t, zero(R)), x), lo, hi, R(r))
end

"""
    kth_largest_abs(x, k, ::Type{R}) -> τ

A threshold separating the `k` largest-magnitude entries of `x` from the rest: `τ` is such
that `#{i : |xᵢ| > τ}` is as close to `k` as the data allows, approached from above so that
no more than `k` entries survive a `> τ` test.

This replaces `partialsortperm` on storage that cannot be sorted. Ties are resolved the way
a stable selection would resolve them only in the sense that *some* `k` of the tied entries
are kept; which ones is unspecified, exactly as for `partialsortperm` on equal keys.
"""
function kth_largest_abs(x, k::Integer, ::Type{R}) where {R <: Real}
    k <= 0 && return R(Inf)
    k >= length(x) && return R(-Inf)
    lo = zero(R)
    hi = R(maximum(abs, x))
    # `#{|xᵢ| > t}` is a non-increasing step function of `t`, so the same bisection applies
    # with the count as φ. It crosses `k` at the (k+1)-th largest magnitude, which is
    # exactly the threshold that leaves the top `k` strictly above it -- and the `hi` end is
    # the one whose count is guaranteed not to exceed `k`.
    _, hi = bisect_bracket(t -> R(count_above_abs(x, t)), lo, hi, R(k))
    return hi
end

"""
    kth_largest(x, k, ::Type{R}) -> τ

As [`kth_largest_abs`](@ref), but by value rather than by magnitude: `#{i : xᵢ > τ} ≤ k`,
with `τ` as low as that allows. Used for the sum of the `k` largest entries.
"""
function kth_largest(x, k::Integer, ::Type{R}) where {R <: Real}
    k <= 0 && return R(Inf)
    k >= length(x) && return R(-Inf)
    lo = R(minimum(x)) - one(R)
    hi = R(maximum(x))
    _, hi = bisect_bracket(t -> R(count_above(x, t)), lo, hi, R(k))
    return hi
end

# `count(f, x)` is not uniformly available across array backends; `sum` over a 0/1 map is,
# and reduces on the device.
@inline count_above_abs(x, t) = sum(xi -> abs(xi) > t ? 1 : 0, x)
@inline count_above(x, t) = sum(xi -> xi > t ? 1 : 0, x)

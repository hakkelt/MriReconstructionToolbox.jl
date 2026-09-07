export ProximalAverage

"""
    ProximalAverage(f_1, ..., f_k)
    ProximalAverage(fs, weights)

Given proximable functions `f_1` to `f_k` and convex weights `w_1` to `w_k` (non-negative, summing to
one), return their *proximal average*: the convex function whose Moreau envelope is the weighted
average of the individual Moreau envelopes. Its proximal mapping is exactly the weighted average of
the individual proximal mappings,

```math
\\mathrm{prox}_{\\gamma \\bar f}(x) = \\sum_j w_j \\, \\mathrm{prox}_{\\gamma f_j}(x),
```

which is what this type implements. See Bauschke, Goebel, Lucet & Wang, *The proximal average: basic
theory*, SIAM J. Optim. 19(2), 766-785 (2008).

The first form uses uniform weights `1/k`. In the second form `fs` is a tuple of functions and
`weights` any real vector or tuple of the same length.

Note that ``\\bar f`` is close to, but not equal to, the weighted average ``\\sum_j w_j f_j``, and it
has no closed form in general. Calling a `ProximalAverage` evaluates that weighted average instead,
so callers needing an exact objective value should not rely on it.

This is the calculus rule to reach for when a sum of proximable terms has no joint prox — several
transforms of the same variable, for instance — and splitting the problem is not an option.
"""
struct ProximalAverage{T, W}
    fs::T
    weights::W
    function ProximalAverage(fs::T, weights::W) where {T <: Tuple, W}
        length(fs) == length(weights) || throw(ArgumentError("one weight is required per function"))
        isempty(fs) && throw(ArgumentError("ProximalAverage needs at least one function"))
        all(w -> w >= 0, weights) || throw(ArgumentError("weights must be non-negative"))
        sum(weights) ≈ 1 || throw(ArgumentError("weights must sum to 1, got $(sum(weights))"))
        return new{T, W}(fs, weights)
    end
end

function ProximalAverage(fs...)
    k = length(fs)
    return ProximalAverage(fs, fill(1 / k, k))
end

component_types(::Type{ProximalAverage{T, W}}) where {T, W} = fieldtypes(T)

# A convex combination of convex functions is convex, and the average is smooth only in degenerate
# cases, so the conservative answer is the right one for smoothness.
@generated is_convex(::Type{T}) where {T <: ProximalAverage} =
    return all(is_convex, component_types(T)) ? :(true) : :(false)
is_smooth(::Type{<:ProximalAverage}) = false
is_separable(::Type{<:ProximalAverage}) = false

(g::ProximalAverage)(x) = sum(w * f(x) for (w, f) in zip(g.weights, g.fs))

function prox!(y, g::ProximalAverage, x, gamma)
    # Callers are allowed to take an in-place step (`prox!(x, g, x, gamma)`), and every component
    # needs to see the original `x`, so keep a copy of the input whenever it aliases the output.
    input = y === x ? copy(x) : x
    fill!(y, zero(eltype(y)))
    buffer = similar(y)
    for (w, f) in zip(g.weights, g.fs)
        prox!(buffer, f, input, gamma)
        y .+= w .* buffer
    end
    # The contract is that `prox!` returns the function's value *at the point it wrote*, so evaluate
    # at `y`. Averaging the per-component values at their own prox points would return a different
    # (and, by convexity, strictly smaller) number whenever the components disagree.
    return g(y)
end

function prox_naive(g::ProximalAverage, x, gamma)
    y = zero(x)
    for (w, f) in zip(g.weights, g.fs)
        y_f, _ = prox_naive(f, x, gamma)
        y = y .+ w .* y_f
    end
    return y, g(y)
end

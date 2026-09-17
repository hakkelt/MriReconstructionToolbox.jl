# Separable sum, using tuples of arrays as variables

export SeparableSum

"""
    SeparableSum(f_1, ..., f_k; threaded=true)

Given functions `f_1` to `f_k`, return their separable sum, that is
```math
g(x_1, ..., x_k) = \\sum_{i=1}^k f_i(x_i).
```
The object `g` constructed in this way can be evaluated at `Tuple`s of length
`k`. Likewise, the `prox` and `prox!` methods for `g` operate with
(input and output) `Tuple`s of length `k`.

Example:

    f = SeparableSum(NormL1(), NuclearNorm()); # separable sum of two functions
    x = randn(10); # some random vector
    Y = randn(20, 30); # some random matrix
    f_xY = f((x, Y)); # evaluates f at (x, Y)
    (u, V), f_uV = prox(f, (x, Y), 1.3); # computes prox at (x, Y)

With `threaded = true` (the default) the blocks are proxed in parallel, but only when there
are at least `$(MIN_BLOCKS_FOR_PARALLEL)` of them: the block count is a compile-time tuple
length, so for the common two- or three-block sum the decision folds away and no task is
ever spawned. See [`is_threaded`](@ref).
"""
struct SeparableSum{T, Th}
    fs::T
end

SeparableSum(fs::Tuple; threaded::Bool=true) = SeparableSum{typeof(fs), threaded}(fs)
SeparableSum(fs::Vararg; threaded::Bool=true) = SeparableSum((fs...,); threaded)

@threadable SeparableSum{<:Any, Th} BlockParallel

component_types(::Type{<:SeparableSum{T}}) where T = fieldtypes(T)

# no scratch space of its own: each block is proxed in place
preallocate(g::SeparableSum{<:Any, Th}, xs::Tuple) where Th =
    SeparableSum(map(preallocate, g.fs, xs); threaded = Th)

@generated is_proximable(::Type{T}) where T <: SeparableSum = return all(is_proximable, component_types(T)) ? true : false
@generated is_convex(::Type{T}) where T <: SeparableSum = return all(is_convex, component_types(T)) ? true : false
@generated is_set_indicator(::Type{T}) where T <: SeparableSum = return all(is_set_indicator, component_types(T)) ? true : false
@generated is_singleton_indicator(::Type{T}) where T <: SeparableSum = return all(is_singleton_indicator, component_types(T)) ? true : false
@generated is_cone_indicator(::Type{T}) where T <: SeparableSum = return all(is_cone_indicator, component_types(T)) ? true : false
@generated is_affine_indicator(::Type{T}) where T <: SeparableSum = return all(is_affine_indicator, component_types(T)) ? true : false
@generated is_smooth(::Type{T}) where T <: SeparableSum = return all(is_smooth, component_types(T)) ? true : false
@generated is_locally_smooth(::Type{T}) where T <: SeparableSum = return all(is_locally_smooth, component_types(T)) ? true : false
@generated is_generalized_quadratic(::Type{T}) where T <: SeparableSum = return all(is_generalized_quadratic, component_types(T)) ? true : false
@generated is_strongly_convex(::Type{T}) where T <: SeparableSum = return all(is_strongly_convex, component_types(T)) ? true : false

(g::SeparableSum)(xs::Tuple) = sum(f(x) for (f, x) in zip(g.fs, xs))

# The blocks of a `SeparableSum` have different types, so this is a heterogeneous tuple and
# not a loop `@batch` can take: one task per block is spawned instead. The region runs with
# the thread budget restricted, so a block that calls BLAS shares the budget with its
# siblings rather than each opening a full pool.
#
# The tasks are spawned *inside* the restricted region, not before it. A budget scope is
# decided when it is entered, so a block that starts while nothing is restricted opens a
# full Polyester pool of its own, and all the blocks together then oversubscribe the
# machine: measured on 8 blocks of 2^17 elements over 4 threads, spawning first cost
# 1957us against 574us for the same work run serially, and spawning inside the region
# turns that into a speedup.
#
# Each child is also rebuilt with `unthreaded` before it is spawned: `Polyester`'s pool is
# all-or-nothing, so a child that still permits threading would try (and fail, expensively)
# to claim a pool this block loop already holds. See `unthreaded` in
# `src/utilities/execution.jl`.
@inline function _prox_blocks!(ys::Tuple, fs::Tuple, xs::Tuple, gammas, R::Type)
    fs = map(unthreaded, fs)
    return NestedThreading.with_restricted_threads() do
        tasks = map((y, f, x, gamma) -> Threads.@spawn(prox!(y, f, x, gamma)), ys, fs, xs, gammas)
        sum(t -> fetch(t)::R, tasks)
    end
end

@inline _block_work(xs::Tuple) = sum(length, xs)

function prox!(ys::Tuple, g::SeparableSum, xs::Tuple, gamma::Number)
    if should_thread_blocks(g, length(xs), _block_work(xs))
        R = real(eltype(first(xs)))
        return _prox_blocks!(ys, g.fs, xs, map(_ -> gamma, xs), R)
    end
    return sum(prox!(y, f, x, gamma) for (y, f, x) in zip(ys, g.fs, xs))
end

function prox!(ys::Tuple, g::SeparableSum, xs::Tuple, gammas::Tuple)
    if should_thread_blocks(g, length(xs), _block_work(xs))
        R = real(eltype(first(xs)))
        return _prox_blocks!(ys, g.fs, xs, gammas, R)
    end
    return sum(prox!(y, f, x, gamma) for (y, f, x, gamma) in zip(ys, g.fs, xs, gammas))
end

function prox(g::SeparableSum, xs::Tuple, gamma=1)
    ys = similar.(xs)
    fys = prox!(ys, g, xs, gamma)
    return ys, fys
end

gradient!(grads::Tuple, g::SeparableSum, xs::Tuple) = sum(gradient!(grad, f, x) for (grad, f, x) in zip(grads, g.fs, xs))

function gradient(g::SeparableSum, xs::Tuple)
    ys = similar.(xs)
    fxs = gradient!(ys, g, xs)
    return ys, fxs
end

function prox_naive(f::SeparableSum, xs::Tuple, gamma)
    fys = real(eltype(xs[1]))(0)
    ys = []
    for k in eachindex(xs)
        y, fy = prox_naive(f.fs[k], xs[k], typeof(gamma) <: Number ? gamma : gamma[k])
        fys += fy
        append!(ys, [y])
    end
    return Tuple(ys), fys
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{T}) where T <: SeparableSum = _combine_tiers(map(device_tier, component_types(T))...)

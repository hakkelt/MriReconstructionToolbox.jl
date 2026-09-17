# Shared elementwise kernels.
#
# Most of `src/functions/` has one of two shapes: write `y` elementwise from `x`, and
# reduce a function of the result to a scalar (the value `prox!` returns); or just reduce,
# for a call operator. Routing them through two helpers keeps the threading and GPU
# stories in one place instead of repeated across fifty methods, and is what lets a single
# file in `ext/GpuExt` give the whole separable family device support.
#
# The CPU body stays a *fused* single pass. That is not incidental: the existing methods
# are written as loops precisely because writing `y` and accumulating its norm in one sweep
# halves the memory traffic, and `benchmark/threading_sweep.jl` measures the two-pass
# alternative losing that class outright. The GPU body is deliberately two passes, because
# there the count of kernel launches matters and bandwidth does not.
#
# Transcendental operators do *not* use these helpers: LoopVectorization cannot see through
# an opaque `h`, so those kernels splice their loop body in literally via
# `@transcendental_loop`. See the comment above the macros in `execution.jl`.

"""
    map_reduce_prox!(f, y, h, g, x) -> value

Write `y[i] = h(x[i])` and return `sum(g, y)`, fused into one pass on the CPU and expressed
as a broadcast plus a `mapreduce` on the GPU. Threading follows [`should_thread`](@ref).
"""
@inline map_reduce_prox!(f, y, h::H, g::G, x) where {H, G} =
    _map_reduce_prox!(execution_strategy(f, x), y, h, g, x)

@inline function _map_reduce_prox!(strategy::Strategy, y, h::H, g::G, x) where {H, G}
    acc = real(eltype(y))(0)
    @elementwise_loop strategy reduction = ((+, acc),) for i in eachindex(x, y)
        yi = h(x[i])
        y[i] = yi
        acc += g(yi)
    end
    return acc
end

"""
    map_reduce_prox_idx!(f, y, h, g, x) -> value

As [`map_reduce_prox!`](@ref), but `h` and `g` receive the index as well as the value, for
kernels whose parameter varies per element: a weighted `NormL1`, an array-valued `gamma`,
or an `IndBox` whose bounds come from `get_kth_elem`.
"""
@inline map_reduce_prox_idx!(f, y, h::H, g::G, x) where {H, G} =
    _map_reduce_prox_idx!(execution_strategy(f, x), y, h, g, x)

@inline function _map_reduce_prox_idx!(strategy::Strategy, y, h::H, g::G, x) where {H, G}
    acc = real(eltype(y))(0)
    @elementwise_loop strategy reduction = ((+, acc),) for i in eachindex(x, y)
        yi = h(i, x[i])
        y[i] = yi
        acc += g(i, yi)
    end
    return acc
end

"""
    map_prox!(f, y, h, x) -> y

Write `y[i] = h(x[i])`, with no reduction.
"""
@inline map_prox!(f, y, h::H, x) where {H} = _map_prox!(execution_strategy(f, x), y, h, x)

@inline function _map_prox!(strategy::Strategy, y, h::H, x) where {H}
    @elementwise_loop strategy for i in eachindex(x, y)
        y[i] = h(x[i])
    end
    return y
end

@inline map_prox_idx!(f, y, h::H, x) where {H} =
    _map_prox_idx!(execution_strategy(f, x), y, h, x)

@inline function _map_prox_idx!(strategy::Strategy, y, h::H, x) where {H}
    @elementwise_loop strategy for i in eachindex(x, y)
        y[i] = h(i, x[i])
    end
    return y
end

"""
    reduce_call(f, g, x) -> value

Return `sum(g, x)`. The call-operator counterpart of [`map_reduce_prox!`](@ref).
"""
@inline reduce_call(f, g::G, x) where {G} = _reduce_call(execution_strategy(f, x), g, x)

@inline function _reduce_call(strategy::Strategy, g::G, x) where {G}
    acc = real(eltype(x))(0)
    @elementwise_loop strategy reduction = ((+, acc),) for i in eachindex(x)
        acc += g(x[i])
    end
    return acc
end

@inline reduce_call_idx(f, g::G, x) where {G} =
    _reduce_call_idx(execution_strategy(f, x), g, x)

@inline function _reduce_call_idx(strategy::Strategy, g::G, x) where {G}
    acc = real(eltype(x))(0)
    @elementwise_loop strategy reduction = ((+, acc),) for i in eachindex(x)
        acc += g(i, x[i])
    end
    return acc
end

# Divergent early exits -- `return +Inf` from inside the loop, as the indicator functions
# do -- cannot sit inside `@batch` or a GPU kernel. They become reductions instead, which is
# the same rewrite that makes them device-clean.

"""
    all_satisfy(f, p, x) -> Bool

Whether `p(x[i])` holds for every element. Replaces a loop that returns early on the first
violation; indicators use it to decide between `0` and `+Inf`.
"""
@inline all_satisfy(f, p::P, x) where {P} = _all_satisfy(execution_strategy(f, x), p, x)

@inline function _all_satisfy(strategy::Strategy, p::P, x) where {P}
    acc = true
    @elementwise_loop strategy reduction = ((&, acc),) for i in eachindex(x)
        acc &= p(x[i])
    end
    # `&` reductions come back as `UInt8` from the vectorised backends
    return !iszero(acc)
end

@inline all_satisfy_idx(f, p::P, x) where {P} =
    _all_satisfy_idx(execution_strategy(f, x), p, x)

@inline function _all_satisfy_idx(strategy::Strategy, p::P, x) where {P}
    acc = true
    @elementwise_loop strategy reduction = ((&, acc),) for i in eachindex(x)
        acc &= p(i, x[i])
    end
    return !iszero(acc)
end

"""
    map_reduce2_prox!(f, y, h, g1, g2, x) -> (v1, v2)

Write `y[i] = h(x[i])` and return both `sum(g1, y)` and `sum(g2, y)` from the same pass.

`ElasticNet` needs the squared norm and the 1-norm of the same result; computing them in two
separate sweeps would read `y` twice for no reason on the host. On a device the reductions
are separate anyway, which is why the GPU method is allowed to be two `mapreduce`s.
"""
@inline map_reduce2_prox!(f, y, h::H, g1::G1, g2::G2, x) where {H, G1, G2} =
    _map_reduce2_prox!(execution_strategy(f, x), y, h, g1, g2, x)

@inline function _map_reduce2_prox!(strategy::Strategy, y, h::H, g1::G1, g2::G2, x) where {H, G1, G2}
    R = real(eltype(y))
    acc1 = R(0)
    acc2 = R(0)
    @elementwise_loop strategy reduction = ((+, acc1), (+, acc2)) for i in eachindex(x, y)
        yi = h(x[i])
        y[i] = yi
        acc1 += g1(yi)
        acc2 += g2(yi)
    end
    return acc1, acc2
end

@inline map_reduce2_prox_idx!(f, y, h::H, g1::G1, g2::G2, x) where {H, G1, G2} =
    _map_reduce2_prox_idx!(execution_strategy(f, x), y, h, g1, g2, x)

@inline function _map_reduce2_prox_idx!(strategy::Strategy, y, h::H, g1::G1, g2::G2, x) where {H, G1, G2}
    R = real(eltype(y))
    acc1 = R(0)
    acc2 = R(0)
    @elementwise_loop strategy reduction = ((+, acc1), (+, acc2)) for i in eachindex(x, y)
        yi = h(i, x[i])
        y[i] = yi
        acc1 += g1(yi)
        acc2 += g2(yi)
    end
    return acc1, acc2
end

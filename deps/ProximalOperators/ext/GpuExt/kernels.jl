# The device forms of the shared kernels.
#
# The CPU bodies fuse the write and the reduction into one pass, because halving the memory
# traffic is what makes them fast and `benchmark/threading_sweep.jl` measures the two-pass
# alternative losing outright. On a device the trade is the other way round: bandwidth is
# ample and the count of kernel launches is what costs, so a broadcast followed by a
# `mapreduce` -- two passes, two launches, no scalar indexing -- is the right shape.

@inline function ProximalOperators._map_reduce_prox!(
    ::Strategy, y::AbstractGPUArray, h::H, g::G, x::AbstractGPUArray
) where {H, G}
    y .= h.(x)
    return sum(g, y)
end

@inline function ProximalOperators._map_reduce_prox_idx!(
    ::Strategy, y::AbstractGPUArray, h::H, g::G, x::AbstractGPUArray
) where {H, G}
    idx = eachindex(x)
    y .= h.(idx, x)
    return sum(g.(idx, y))
end

@inline function ProximalOperators._map_prox!(
    ::Strategy, y::AbstractGPUArray, h::H, x::AbstractGPUArray
) where {H}
    y .= h.(x)
    return y
end

@inline function ProximalOperators._map_prox_idx!(
    ::Strategy, y::AbstractGPUArray, h::H, x::AbstractGPUArray
) where {H}
    y .= h.(eachindex(x), x)
    return y
end

@inline ProximalOperators._reduce_call(::Strategy, g::G, x::AbstractGPUArray) where {G} =
    sum(g, x)

@inline ProximalOperators._reduce_call_idx(::Strategy, g::G, x::AbstractGPUArray) where {G} =
    sum(g.(eachindex(x), x))

# `all` over a device array is a reduction, which is exactly why the early-exit loops in the
# indicator functions were rewritten as predicates in the first place.
@inline ProximalOperators._all_satisfy(::Strategy, p::P, x::AbstractGPUArray) where {P} =
    all(p, x)

@inline ProximalOperators._all_satisfy_idx(::Strategy, p::P, x::AbstractGPUArray) where {P} =
    all(p.(eachindex(x), x))

@inline function ProximalOperators._map_reduce2_prox!(
    ::Strategy, y::AbstractGPUArray, h::H, g1::G1, g2::G2, x::AbstractGPUArray
) where {H, G1, G2}
    y .= h.(x)
    return sum(g1, y), sum(g2, y)
end

@inline function ProximalOperators._map_reduce2_prox_idx!(
    ::Strategy, y::AbstractGPUArray, h::H, g1::G1, g2::G2, x::AbstractGPUArray
) where {H, G1, G2}
    y .= h.(eachindex(x), x)
    return sum(g1, y), sum(g2, y)
end

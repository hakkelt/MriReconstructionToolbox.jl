module MriReconstructionToolboxGPUExt

using MriReconstructionToolbox
using MriReconstructionToolbox: ProximalOperators
using GPUArrays: AbstractGPUArray
using KernelAbstractions

const Strategy = ProximalOperators.Strategy

# GPU support for the vendored ProximalOperators submodule.
#
# Most of what makes proximal operators work on device arrays is *not* in here. The kernels in
# `deps/ProximalOperators/src/utilities/kernels.jl` are written so their bodies are expressible
# as broadcasts, the selection operators gained a reduction-based second algorithm in
# `deps/ProximalOperators/src/utilities/bisection.jl`, and both choose their path from
# `is_cpu_storage`, which is conservative by default. The result is that a device array already
# takes the right branch everywhere before this extension is loaded at all.
#
# What this extension adds is what genuinely needs the GPU packages in scope -- unlike
# `RecursiveArrayToolsExt`, which the submodule inlines unconditionally because
# `RecursiveArrayTools` is already always available in `MriReconstructionToolbox`, GPUArrays and
# KernelAbstractions are not, so this stays a real, optional extension at MRT's own top level
# rather than a hard dependency baked into the vendored submodule:
#
#   this file's "properties" section  states that an `AbstractGPUArray` is not CPU storage --
#                                      true by default, but worth saying explicitly rather than
#                                      relying on the fallback
#   "kernels" section                 replaces the fused CPU loop with a broadcast plus a
#                                      `mapreduce`, which is the right shape on a device even
#                                      though it costs a second pass
#   "guards" section                  rejects mixed host/device argument pairs with a message
#                                      that names both, instead of letting them fail deep inside
#                                      a broadcast

# --- properties ---

ProximalOperators.is_cpu_storage(::Type{<:AbstractGPUArray}) = false

# --- kernels ---
#
# The device forms of the shared kernels. The CPU bodies fuse the write and the reduction into
# one pass, because halving the memory traffic is what makes them fast on CPU. On a device the
# trade is the other way round: bandwidth is ample and the count of kernel launches is what
# costs, so a broadcast followed by a `mapreduce` -- two passes, two launches, no scalar
# indexing -- is the right shape.

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

# --- guards ---
#
# Passing a host `y` with a device `x` is always a mistake, but left alone it surfaces as a
# scalar-indexing error or a failed broadcast several frames deep inside an array package,
# saying nothing about which argument was wrong.
#
# The check sits on the kernel helpers rather than on `prox!`. That is a dispatch constraint,
# not a preference: a method `prox!(y::AbstractArray, f, x::AbstractGPUArray, γ)` is more
# specific than every operator's own `prox!` in its storage arguments and less specific in `f`,
# so Julia rightly calls the pair ambiguous -- for *every* operator, including the correct
# device-to-device call. The helpers take no operator argument, so guarding them is
# unambiguous, and since the whole separable family routes through them the message lands where
# the mistake actually shows up.
#
# This is the only rejection this extension adds. No operator refuses to run because its input
# is on a device -- everything either runs there or is served by a host round-trip -- so
# mismatched storage is the one thing left that can be an error.

@noinline function _mixed_storage(y, x)
    throw(ArgumentError(
        "mixed storage: the output is $(typeof(y)) but the input is $(typeof(x)). Both \
must live in the same memory. Move one of them -- `Array(x)` to bring the input to the \
host, or the device array constructor to send the output to the device."
    ))
end

for fn in (:_map_reduce_prox!, :_map_reduce_prox_idx!)
    @eval begin
        ProximalOperators.$fn(::Strategy, y::AbstractArray, h, g, x::AbstractGPUArray) =
            _mixed_storage(y, x)
        ProximalOperators.$fn(::Strategy, y::AbstractGPUArray, h, g, x::AbstractArray) =
            _mixed_storage(y, x)
    end
end

for fn in (:_map_prox!, :_map_prox_idx!)
    @eval begin
        ProximalOperators.$fn(::Strategy, y::AbstractArray, h, x::AbstractGPUArray) =
            _mixed_storage(y, x)
        ProximalOperators.$fn(::Strategy, y::AbstractGPUArray, h, x::AbstractArray) =
            _mixed_storage(y, x)
    end
end

for fn in (:_map_reduce2_prox!, :_map_reduce2_prox_idx!)
    @eval begin
        ProximalOperators.$fn(::Strategy, y::AbstractArray, h, g1, g2, x::AbstractGPUArray) =
            _mixed_storage(y, x)
        ProximalOperators.$fn(::Strategy, y::AbstractGPUArray, h, g1, g2, x::AbstractArray) =
            _mixed_storage(y, x)
    end
end

end # module MriReconstructionToolboxGPUExt

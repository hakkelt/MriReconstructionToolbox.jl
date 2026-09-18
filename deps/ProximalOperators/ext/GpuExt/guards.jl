# Mixed host/device arguments.
#
# Passing a host `y` with a device `x` is always a mistake, but left alone it surfaces as a
# scalar-indexing error or a failed broadcast several frames deep inside an array package,
# saying nothing about which argument was wrong.
#
# The check sits on the kernel helpers rather than on `prox!`. That is a dispatch
# constraint, not a preference: a method `prox!(y::AbstractArray, f, x::AbstractGPUArray, γ)`
# is more specific than every operator's own `prox!` in its storage arguments and less
# specific in `f`, so Julia rightly calls the pair ambiguous -- for *every* operator, including
# the correct device-to-device call. The helpers take no operator argument, so guarding them
# is unambiguous, and since the whole separable family routes through them the message lands
# where the mistake actually shows up.
#
# This is the only rejection in the package. No operator refuses to run because its input is
# on a device -- everything either runs there or is served by a host round-trip -- so
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

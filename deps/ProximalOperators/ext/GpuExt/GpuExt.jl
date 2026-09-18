# GPU support.
#
# Most of what makes this package work on device arrays is *not* in here. The kernels in
# `src/utilities/kernels.jl` are written so their bodies are expressible as broadcasts, the
# selection operators gained a reduction-based second algorithm in `src/utilities/bisection.jl`,
# and both choose their path from `is_cpu_storage`, which is conservative by default. The
# result is that a device array already takes the right branch everywhere before this
# extension is loaded at all.
#
# What the extension adds is what genuinely needs the GPU packages in scope:
#
#   properties.jl  states that an `AbstractGPUArray` is not CPU storage -- true by default,
#                  but worth saying explicitly rather than relying on the fallback
#   kernels.jl     replaces the fused CPU loop with a broadcast plus a `mapreduce`, which is
#                  the right shape on a device even though it costs a second pass
#   guards.jl      rejects mixed host/device argument pairs with a message that names both,
#                  instead of letting them fail deep inside a broadcast
module GpuExt

using GPUArrays: AbstractGPUArray
using KernelAbstractions

using ProximalOperators
using ProximalOperators: Strategy

include("properties.jl")
include("kernels.jl")
include("guards.jl")

end # module

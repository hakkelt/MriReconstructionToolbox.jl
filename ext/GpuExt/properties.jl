# Device storage is never threaded by Julia: the work is already parallel, and a Polyester
# loop over a device array would serialise it through scalar indexing.
#
# `is_cpu_storage` already answers `false` for any array type it does not recognise, so this
# is not what makes device arrays work -- it makes the reason explicit, and it keeps the
# answer correct for a GPU array type that someone later wraps in something the fallback
# would otherwise have to guess about.
ProximalOperators.is_cpu_storage(::Type{<:AbstractGPUArray}) = false

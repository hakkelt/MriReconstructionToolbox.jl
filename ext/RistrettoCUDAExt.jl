module RistrettoCUDAExt

using Ristretto: Ristretto
using CUDA: CuArray, CUFFT

# ProximalOperators is a submodule of Ristretto, so its CUDA extension is included
# here, in a module that binds the package it extends.
module ProximalOperatorsCUDA
    using Ristretto: ProximalOperators
    include(joinpath(@__DIR__, "..", "deps", "ProximalOperators", "ext", "ProximalOperatorsCUDAExt.jl"))
end

# ─── FFT plans back to the cache ─────────────────────────────────────────────────────────────
#
# A cuFFT plan's handle returns to CUDA.jl's handle cache when the plan is finalized, and from
# there the next plan of the same shape takes it in microseconds. An operator is a tree of
# structs, tuples and arrays of operators; every plan found in it is finalized once.

function Ristretto._release_device_plans!(op, ::CuArray)
    _release_plans!(Base.IdSet{Any}(), op)
    return nothing
end

_release_plans!(seen, p::CUFFT.CuFFTPlan) = (p in seen || (push!(seen, p); finalize(p)); nothing)
_release_plans!(seen, ::Union{Number, Symbol, AbstractString, Type, Function, Module, Task, Base.AbstractLock, Nothing}) = nothing
# A slot never assigned (a cache allocated with `undef`, the storage behind a `Dict`) holds nothing.
function _release_plans!(seen, x::AbstractArray)
    isbitstype(eltype(x)) && return nothing
    for i in eachindex(x)
        isassigned(x, i) && _release_plans!(seen, x[i])
    end
    return nothing
end
function _release_plans!(seen, x)
    if ismutable(x)
        x in seen && return nothing
        push!(seen, x)
    end
    for i in 1:nfields(x)
        isdefined(x, i) && _release_plans!(seen, getfield(x, i))
    end
    return nothing
end

end # module RistrettoCUDAExt

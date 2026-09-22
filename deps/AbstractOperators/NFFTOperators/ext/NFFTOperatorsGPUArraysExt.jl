module NFFTOperatorsGPUArraysExt

using NFFTOperators
import NFFTOperators: _nfft_plan, _nfft_adapt, NFFTOp
import NFFTOperators.NFFT: NFFT
using GPUArrays
using Adapt

"""
    _nfft_plan(array_type, trajectory, image_size, threaded; kwargs...)

GPU override: creates a GPU NFFT plan via `NFFT.plan_nfft(array_type, ...)`.
The trajectory must be a CPU `Matrix{T}`; only the computation buffers are on GPU.
Threading is ignored for GPU plans (GPU parallelism is used instead).
"""
function _nfft_plan(
        array_type::Type{<:AbstractGPUArray},
        trajectory::AbstractArray{T},
        image_size,
        threaded;
        kwargs...,
    ) where {T}
    traj = Matrix{T}(reshape(trajectory, size(trajectory, 1), :))
    N = image_size
    return NFFT.plan_nfft(array_type, traj, N; kwargs...)
end

"""
    _nfft_adapt(array_type, arr)

GPU override: adapts a CPU array to the target GPU array type using Adapt.jl.
"""
_nfft_adapt(array_type::Type{<:AbstractGPUArray}, arr::AbstractArray) = adapt(array_type, arr)

end # module NFFTOperatorsGPUArraysExt

module NFFTOperatorsGPUArraysExt

using ..NFFTOperators
import ..NFFTOperators: _nfft_plan, _nfft_adapt, NFFTOp, _toeplitz_kernel, _nodes, _first_plan
import ..NFFTOperators.NFFT: NFFT
import ..NFFTOperators: FFTW
using GPUArrays
using Adapt
using LinearAlgebra: mul!

"""
    _nfft_plan(array_type, trajectory, image_size, threaded; batch = 1, frames = 1, kwargs...)

GPU override: NFFT.jl's GPU plan for `array_type`, one plan for `batch × frames` images, the
nodes of frame `f` being the `f`-th of `frames` equal groups of `trajectory`'s samples. The
trajectory must be a CPU array; the plan moves what it needs to the device. Threading is ignored
for GPU plans (GPU parallelism is used instead), and so are FFTW's planner flags.
"""
function _nfft_plan(
        array_type::Type{<:AbstractGPUArray},
        trajectory::AbstractArray{T},
        image_size,
        threaded;
        batch = 1,
        frames = 1,
        fftflags = nothing,
        kwargs...,
    ) where {T}
    k = Matrix{T}(reshape(trajectory, size(trajectory, 1), :))
    return NFFT.plan_nfft(NFFT.backend(), array_type, k, image_size; batch, frames, kwargs...)
end

"""
    _nfft_adapt(array_type, arr)

GPU override: adapts a CPU array to the target GPU array type using Adapt.jl.
"""
_nfft_adapt(array_type::Type{<:AbstractGPUArray}, arr::AbstractArray) = adapt(array_type, arr)

# The Toeplitz kernels of all frames, computed on the device by one plan on the
# twice-oversampled grid. The FFT plans made here are finalized before returning: a device
# library may hand a plan's resources back only then.
function _toeplitz_kernel(
        op::NFFTOp{T, D, N, M, P, <:AbstractGPUArray}, shape_ext, frames
    ) where {T, D, N, M, P}
    params = _first_plan(op).params
    λ = similar(op.ksp_buffer, (shape_ext..., 1, frames))
    array_type = Base.typename(typeof(λ)).wrapper
    p = NFFT.plan_nfft(NFFT.backend(), array_type, _nodes(op), shape_ext; m = params.m, σ = params.σ, frames)
    mul!(λ, adjoint(p), Complex{T}.(vec(op.dcf)))
    finalize(p.forwardFFT)
    finalize(p.backwardFFT)
    λ = circshift(reshape(λ, shape_ext..., frames), (shape_ext .÷ 2..., 0))
    fft_plan = FFTW.plan_fft!(λ, 1:D)
    fft_plan * λ
    finalize(fft_plan)
    λ ./= T(prod(shape_ext))
    return λ
end

end # module NFFTOperatorsGPUArraysExt

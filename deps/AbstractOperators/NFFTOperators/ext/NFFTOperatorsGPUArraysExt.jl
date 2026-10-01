module NFFTOperatorsGPUArraysExt

using ..NFFTOperators
import ..NFFTOperators:
    _nfft_plan, _batched_nfft_plan, _nfft_adapt, NFFTOp, BatchedNFFTOp, _get_normal_op, NfftNormalOp
import ..NFFTOperators.NFFT: NFFT
import ..NFFTOperators: FFTW
import ..NFFTOperators.AbstractOperators: allocate_in_domain, domain_array_type
using GPUArrays
using Adapt
using LinearAlgebra: mul!

"""
    _nfft_plan(array_type, trajectory, image_size, threaded; kwargs...)

GPU override: NFFT.jl's GPU plan for `array_type`. The trajectory must be a CPU array; the plan
moves what it needs to the device. Threading is ignored for GPU plans (GPU parallelism is used
instead), and so are FFTW's planner flags.
"""
function _nfft_plan(
        array_type::Type{<:AbstractGPUArray},
        trajectory::AbstractArray{T},
        image_size,
        threaded;
        fftflags = nothing,
        kwargs...,
    ) where {T}
    traj = Matrix{T}(reshape(trajectory, size(trajectory, 1), :))
    return NFFT.plan_nfft(NFFT.backend(), array_type, traj, image_size; kwargs...)
end

function _batched_nfft_plan(
        array_type::Type{<:AbstractGPUArray}, k::Matrix, image_size, batch, frames; fftflags = nothing, kwargs...
    )
    return NFFT.plan_nfft(NFFT.backend(), array_type, k, image_size; batch, frames, kwargs...)
end

"""
    _nfft_adapt(array_type, arr)

GPU override: adapts a CPU array to the target GPU array type using Adapt.jl.
"""
_nfft_adapt(array_type::Type{<:AbstractGPUArray}, arr::AbstractArray) = adapt(array_type, arr)

"""
    _get_normal_op(op::NFFTOp) (GPU)

GPU override of the Toeplitz normal operator. Its kernel `λ`, the adjoint NFFT of the density
compensation on the twice-oversampled grid, is computed on the device with a device plan for that
grid, and the two FFTs of the embedding are planned on the device buffer, without FFTW's planner
flags. The host plan and adjoint this replaced took 15 ms of a 21 ms build for a 2D radial
trajectory of 16384 samples on one thread.

The inverse FFT is an unnormalized `bfft!`, its `1/length` folded into `λ`: the normalization of
`inv(fftplan)` is a BLAS `scal` that, on CUDA, also allocates and uploads its scalar on every
application.
"""
function _get_normal_op(
        op::NFFTOp{T, D, P, K},
    ) where {T, D, P <: NFFT.AbstractNFFTPlan{T, D}, K <: AbstractGPUArray{Complex{T}}}
    shape = op.plan.N
    buf = allocate_in_domain(op, (2 .* shape)...)
    λ = _toeplitz_kernel(op.plan, buf, Complex{T}.(vec(op.dcf)), 1)
    fill!(buf, 0)
    return NfftNormalOp(domain_array_type(op), shape, λ, FFTW.plan_fft!(buf), FFTW.plan_bfft!(buf), buf, op.threaded)
end

"""
    _get_normal_op(op::BatchedNFFTOp)

The Toeplitz normal operator of a batched NFFT: each frame has its own kernel `λ`, the adjoint
NFFT of its density compensation on the twice-oversampled grid, computed by one plan over all
frames, and the embedding's FFTs transform the whole stack at once.
"""
function _get_normal_op(op::BatchedNFFTOp{T, D, M}) where {T, D, M}
    shape = op.plan.N
    shape_ext = 2 .* shape
    stack_dims = op.dim_in[(D + 1):end]
    frame_dims = stack_dims[(end - op.nframe + 1):end]
    λ_frames = similar(op.ksp_buffer, shape_ext..., 1, op.plan.frames)
    λ = _toeplitz_kernel(op.plan, λ_frames, Complex{T}.(vec(op.dcf)), op.plan.frames)
    λ = reshape(λ, shape_ext..., map(_ -> 1, stack_dims[1:(end - op.nframe)])..., frame_dims...)
    buf = similar(op.ksp_buffer, shape_ext..., stack_dims...)
    fill!(buf, 0)
    return NfftNormalOp(
        domain_array_type(op), op.dim_in, λ, FFTW.plan_fft!(buf, 1:D), FFTW.plan_bfft!(buf, 1:D), buf, false
    )
end

# `λ` for `frames` frames of `plan`'s nodes, each the FFT of the adjoint NFFT of `w` on the
# twice-oversampled grid, centred, and divided by the grid size for the unnormalized inverse.
# `buf` is `(2N..., [1, frames])` scratch.
function _toeplitz_kernel(plan, buf, w, frames)
    T = real(eltype(buf))
    shape = plan.N
    D = length(shape)
    array_type = Base.typename(typeof(buf)).wrapper
    p = NFFT.plan_nfft(
        NFFT.backend(), array_type, plan.k, 2 .* shape; m = plan.params.m, σ = plan.params.σ, frames
    )
    mul!(buf, adjoint(p), w)
    shifted = circshift(buf, (shape..., ntuple(_ -> 0, ndims(buf) - D)...))
    λ = FFTW.fft(shifted, 1:D)
    λ ./= T(prod(2 .* shape))
    return λ
end

end # module NFFTOperatorsGPUArraysExt

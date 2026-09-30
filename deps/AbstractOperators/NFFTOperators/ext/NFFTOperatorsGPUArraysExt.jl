module NFFTOperatorsGPUArraysExt

using ..NFFTOperators
import ..NFFTOperators: _nfft_plan, _nfft_adapt, NFFTOp, _get_normal_op, NfftNormalOp
import ..NFFTOperators.NFFT: NFFT
import ..NFFTOperators: FFTW
import ..NFFTOperators.AbstractOperators: allocate_in_domain, domain_array_type
using GPUArrays
using Adapt
using LinearAlgebra: mul!

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

"""
    _get_normal_op(op::NFFTOp) (GPU)

GPU override of the Toeplitz normal operator. Its kernel `λ` is computed once, on the host, with
the same oversampled CPU NFFT plan the CPU method uses (a GPU NFFT plan has no blocking or
polynomial precomputation to choose), and moved to the device. What runs on every application --
the two FFTs of the embedding -- is planned on the device buffer, without FFTW's planner flags.
"""
function _get_normal_op(
        op::NFFTOp{T, D, P, K},
    ) where {T, D, P <: NFFT.AbstractNFFTPlan{T, D}, K <: AbstractGPUArray{Complex{T}}}
    shape = op.plan.N
    shape_ext = 2 .* shape

    p = NFFT.plan_nfft(
        Matrix(op.plan.k),
        shape_ext;
        m = op.plan.params.m,
        σ = op.plan.params.σ,
        precompute = NFFT.POLYNOMIAL,
        fftflags = FFTW.ESTIMATE,
        blocking = true,
    )
    buf_host = zeros(Complex{T}, shape_ext)
    mul!(buf_host, adjoint(p), Vector{Complex{T}}(vec(Array(op.dcf))))
    λ_host = FFTW.fft(FFTW.fftshift(buf_host))

    buf = allocate_in_domain(op, shape_ext...)
    fill!(buf, 0)
    fftplan = FFTW.plan_fft!(buf)
    λ = copyto!(similar(buf), λ_host)

    return NfftNormalOp(domain_array_type(op), shape, λ, fftplan, inv(fftplan), buf, op.threaded)
end

end # module NFFTOperatorsGPUArraysExt

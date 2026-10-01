module NFFTOperatorsGPUArraysExt

using NFFTOperators
import NFFTOperators: _nfft_plan, _nfft_adapt, NFFTOp, _get_normal_op, NfftNormalOp
import NFFTOperators.NFFT: NFFT
import NFFTOperators: FFTW
import NFFTOperators.AbstractOperators: allocate_in_domain, domain_array_type
using GPUArrays
using Adapt
using LinearAlgebra: mul!
using SparseArrays: spzeros

"""
    _nfft_plan(array_type, trajectory, image_size, threaded; kwargs...)

GPU override: creates a GPU NFFT plan with [`_device_nfft_plan`](@ref).
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
    return _device_nfft_plan(array_type, traj, image_size; kwargs...)
end

"""
    _device_nfft_plan(array_type, k, N; kwargs...)

The plan `NFFT.plan_nfft(array_type, k, N; kwargs...)` returns, with its interpolation matrix
computed on the device instead of the host.

A GPU NFFT plan always stores the full sparse interpolation matrix: one window value per sample
and grid neighbour, `(2m)^D` per sample. NFFT.jl evaluates those on the host, one sample after
another, and uploads the result; for a 2D radial trajectory of 16384 samples (`m = 4`) that took
15.4 ms of the plan's 18.7 ms on one thread. Here each axis's neighbour indices and window values
are a broadcast over `(2m, samples)` on the device, and their tensor products are one more, which
takes 1.2 ms (Quadro RTX 6000). Indices are equal and values agree to a few ulps.

Only the Kaiser-Bessel window, NFFT.jl's default, is built this way; another window is left to
`NFFT.plan_nfft`.
"""
function _device_nfft_plan(
        array_type::Type{<:AbstractGPUArray}, k::Matrix{T}, N::NTuple{D, Int}; fftflags = nothing, kwargs...
    ) where {T, D}
    params, N, NOut, J, Ñ, dims = NFFT.initParams(k, N, 1:D; kwargs...)
    params.window === :kaiser_bessel || return NFFT.plan_nfft(array_type, k, N; kwargs...)
    # The deconvolution tables, without the host interpolation matrix `FULL` would compute.
    params.storeDeconvolutionIdx = true
    params.precompute = NFFT.POLYNOMIAL
    _, _, windowHatInvLUT, deconvolveIdx, _ = NFFT.precomputation(k, N, Ñ, params)
    params.precompute = NFFT.FULL

    tmpVec = adapt(array_type, zeros(Complex{T}, Ñ))
    FP = FFTW.plan_fft!(tmpVec, dims)
    BP = FFTW.plan_bfft!(tmpVec, dims)
    tmpVecHat = adapt(array_type, zeros(Complex{T}, N))
    deconvIdx = Int32.(adapt(array_type, deconvolveIdx))
    winHatInvLUT = Complex{T}.(adapt(array_type, windowHatInvLUT[1]))
    B = _device_interpolation_matrix(array_type, k, Ñ, params.m, params.σ)

    Plan = Base.get_extension(NFFT, :NFFTGPUArraysExt).GPU_NFFTPlan
    return Plan{
        T, D, typeof(tmpVec), typeof(deconvIdx), typeof(FP), typeof(BP), typeof(winHatInvLUT), typeof(B),
    }(
        N, NOut, J, k, Ñ, dims, params, FP, BP, tmpVec, tmpVecHat, deconvIdx, Vector{T}(undef, 0),
        winHatInvLUT, B,
    )
end

# NFFT.jl's `precomputeB` with `precompute = FULL`: column `j` of the `prod(Ñ) × J` matrix holds
# the `(2m)^D` grid neighbours of sample `j`, the first axis running fastest.
function _device_interpolation_matrix(array_type, k::Matrix{T}, Ñ::NTuple{D, Int}, m::Int, σ::T) where {T, D}
    J = size(k, 2)
    L = 2m
    nodes = ntuple(d -> _window_nodes(array_type, k[d, :], Ñ[d], m, σ), D)
    along(d, a) = reshape(a, ntuple(e -> e == d ? L : 1, D)..., J)
    stride = ntuple(d -> prod(Ñ[1:(d - 1)]), D)
    rows = reduce((a, b) -> a .+ b, ntuple(d -> along(d, (nodes[d][1] .- 1) .* stride[d]), D)) .+ 1
    vals = reduce((a, b) -> a .* b, ntuple(d -> along(d, nodes[d][2]), D))
    colptr = adapt(array_type, Int32.(1:(L^D):(L^D * J + 1)))
    CSC = Base.typename(typeof(adapt(array_type, spzeros(Complex{T}, Int32, 1, 1)))).wrapper
    return CSC(colptr, Int32.(vec(rows)), Complex{T}.(vec(vals)), (prod(Ñ), J))
end

# The `2m` grid neighbours of each sample along one axis: wrapped 1-based grid indices and
# window values, each `(2m, J)`.
function _window_nodes(array_type, k_d::Vector{T}, Ñ::Int, m::Int, σ::T) where {T}
    kscale = reshape(adapt(array_type, k_d), 1, :) .* T(Ñ)
    l = adapt(array_type, reshape(collect(0:(2m - 1)), :, 1))
    off = floor.(Int, kscale) .- m .+ 1
    idx = rem.(l .+ off .+ Ñ, Ñ) .+ 1
    win = NFFT.window_kaiser_bessel.((kscale .- l .- off) ./ T(Ñ), Ñ, m, σ)
    return idx, win
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
    shape_ext = 2 .* shape
    array_type = Base.typename(K).wrapper

    p = _device_nfft_plan(array_type, Matrix(op.plan.k), shape_ext; m = op.plan.params.m, σ = op.plan.params.σ)
    buf = allocate_in_domain(op, shape_ext...)
    mul!(buf, adjoint(p), Complex{T}.(vec(op.dcf)))
    λ = FFTW.fft(circshift(buf, shape))
    λ ./= length(λ)

    fill!(buf, 0)
    fftplan = FFTW.plan_fft!(buf)
    return NfftNormalOp(domain_array_type(op), shape, λ, fftplan, FFTW.plan_bfft!(buf), buf, op.threaded)
end

end # module NFFTOperatorsGPUArraysExt

const RealOrComplex = Union{T, Complex{T}} where {T <: Real}

"""
    GPU_NFFTPlan

NFFT plan whose transforms run on a GPU array backend.

The interpolation onto the oversampled grid is stored as two `(J, (2m)^D)` device matrices:
`rows[j, l]` is the linear grid index of neighbour `l` of sample `j` and `vals[j, l]` its window
value. Convolution and its transpose are kernels over the samples that read these tables, so a
transform makes no host round trip.

A plan can transform `batch × frames` arrays at once (keywords `batch` and `frames` of
`plan_nfft`): the input is `(N..., batch, frames)`, the output `(J, batch, frames)`, and the
nodes `k` hold `frames` consecutive groups of `J` samples, group `f` being the trajectory of
frame `f`. All `batch` arrays of a frame share its trajectory. One FFT plan transforms all of
them, and each step of the transform is a single kernel launch whatever their number.
"""
mutable struct GPU_NFFTPlan{T,D, arrTc <: AbstractGPUArray{Complex{T}}, vecI <: AbstractGPUVector{Int32}, FP, BP, INV <: AbstractGPUVector{Complex{T}}, IM <: AbstractGPUMatrix{Int32}, VM <: AbstractGPUMatrix{Complex{T}}} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  NOut::NTuple{1,Int64}
  J::Int64
  k::Matrix{T}
  Ñ::NTuple{D,Int64}
  dims::UnitRange{Int64}
  params::NFFTParams{T}
  forwardFFT::FP
  backwardFFT::BP
  tmpVec::arrTc
  deconvolveIdx::vecI
  windowHatInvLUT::INV
  rows::IM
  vals::VM
  batch::Int64
  frames::Int64
end

function AbstractNFFTs.plan_nfft(::NFFTBackend, arr::Type{<:AbstractGPUArray}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
  timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = GPU_NFFTPlan(arr, k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function GPU_NFFTPlan(arr, k::Matrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                 fftflags=nothing, batch::Integer=1, frames::Integer=1, kwargs...) where {T,D}

    if dims != 1:D
      error("GPU NFFT does not work along directions right now!")
    end
    (batch >= 1 && frames >= 1) || throw(ArgumentError("batch and frames must be positive, got $batch and $frames"))
    size(k, 2) % frames == 0 ||
      throw(ArgumentError("the $(size(k, 2)) nodes do not split into $frames frames of equal size"))

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)
    params.storeDeconvolutionIdx = true # GPU_NFFT only works this way

    # The Kaiser-Bessel window is evaluated on the device; any other window through the host
    # tables of `precompute = FULL`.
    on_device = params.window === :kaiser_bessel
    params.precompute = on_device ? NFFT.POLYNOMIAL : NFFT.FULL
    _, _, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(k, N[dims_], Ñ[dims_], params)
    params.precompute = NFFT.FULL
    rows, vals = if on_device
      _device_interpolation(arr, k, Ñ, params.m, params.σ)
    else
      L = nnz(B) ÷ J
      (adapt(arr, Int32.(permutedims(reshape(rowvals(B), L, J)))),
       adapt(arr, Complex{T}.(permutedims(reshape(nonzeros(B), L, J)))))
    end

    batched = batch * frames > 1
    tmpVec = similar(adapt(arr, Complex{T}[]), batched ? (Ñ..., batch, frames) : Ñ)
    fill!(tmpVec, zero(Complex{T}))
    FP = plan_fft!(tmpVec, dims_)
    BP = plan_bfft!(tmpVec, dims_)

    deconvIdx = adapt(arr, Int32.(deconvolveIdx))
    winHatInvLUT = adapt(arr, Complex{T}.(windowHatInvLUT[1]))

    GPU_NFFTPlan{T,D, typeof(tmpVec), typeof(deconvIdx), typeof(FP), typeof(BP), typeof(winHatInvLUT), typeof(rows), typeof(vals)}(
      N, NOut, J ÷ frames, k, Ñ, dims_, params, FP, BP, tmpVec, deconvIdx, winHatInvLUT, rows, vals, batch, frames)
end

# The interpolation tables of `precompute = FULL` for the Kaiser-Bessel window: per axis, the
# `2m` wrapped grid neighbours of each sample and their window values as one broadcast over
# `(J, 2m)`, and the tensor products of the axes as one more. Neighbour `l` runs over the first
# axis fastest.
function _device_interpolation(arr, k::Matrix{T}, Ñ::NTuple{D,Int}, m::Int, σ::T) where {T,D}
  J = size(k, 2)
  L = 2m
  nodes = ntuple(d -> _window_nodes(arr, k[d, :], Ñ[d], m, σ), D)
  along(d, a) = reshape(a, J, ntuple(e -> e == d ? L : 1, D)...)
  stride = ntuple(d -> prod(Ñ[1:(d - 1)]), D)
  rows = reduce((a, b) -> a .+ b, ntuple(d -> along(d, (nodes[d][1] .- 1) .* stride[d]), D)) .+ 1
  vals = reduce((a, b) -> a .* b, ntuple(d -> along(d, nodes[d][2]), D))
  return Int32.(reshape(rows, J, L^D)), Complex{T}.(reshape(vals, J, L^D))
end

# The `2m` grid neighbours of each sample along one axis: wrapped 1-based grid indices and
# window values, each `(J, 2m)`.
function _window_nodes(arr, k_d::Vector{T}, Ñ::Int, m::Int, σ::T) where {T}
  kscale = adapt(arr, k_d) .* T(Ñ)
  l = adapt(arr, reshape(collect(0:(2m - 1)), 1, :))
  off = floor.(Int, kscale) .- m .+ 1
  idx = rem.(l .+ off .+ Ñ, Ñ) .+ 1
  win = NFFT.window_kaiser_bessel.((kscale .- l .- off) ./ T(Ñ), Ñ, m, σ)
  return idx, win
end

_nbatch(p::GPU_NFFTPlan) = p.batch * p.frames
_batched(p::GPU_NFFTPlan) = _nbatch(p) > 1

AbstractNFFTs.size_in(p::GPU_NFFTPlan) = _batched(p) ? (p.N..., p.batch, p.frames) : p.N
AbstractNFFTs.size_out(p::GPU_NFFTPlan) = _batched(p) ? (p.J, p.batch, p.frames) : p.NOut

# The kernels view every array as `(elements per transform, transforms)`, so arrays of any
# shape with the right length are accepted.
function _check_sizes(p::GPU_NFFTPlan, f, fHat)
  nb = _nbatch(p)
  (length(f) == prod(p.N) * nb && length(fHat) == p.J * nb) ||
    throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
  return nothing
end

# ─── Kernels ──────────────────────────────────────────────────────────────────
#
# Arrays are viewed as `(elements per transform, transforms)`; transform `b` belongs to frame
# `(b - 1) ÷ batch + 1`, whose samples are rows `(frame - 1) * J .+ (1:J)` of `rows` and `vals`.
# The convolution kernels run one thread per sample and transform. Consecutive threads take
# consecutive samples of one transform, so their table reads are contiguous and their grid
# accesses close to each other.

_weight(::Type{<:Real}, v) = real(v)
_weight(::Type{<:Complex}, v) = v

@kernel function _deconvolve_kernel!(g, @Const(f), @Const(idx), @Const(winv))
  i, b = @index(Global, NTuple)
  @inbounds g[idx[i], b] = f[i, b] * winv[i]
end

@kernel function _deconvolve_transpose_kernel!(f, @Const(g), @Const(idx), @Const(winv))
  i, b = @index(Global, NTuple)
  @inbounds f[i, b] = g[idx[i], b] * winv[i]
end

@kernel function _convolve_kernel!(fHat, @Const(g), @Const(rows), @Const(vals), batch)
  j, b = @index(Global, NTuple)
  col = j + ((b - 1) ÷ batch) * size(fHat, 1)
  acc = zero(eltype(fHat))
  for l in 1:size(rows, 2)
    @inbounds acc += _weight(eltype(fHat), vals[col, l]) * g[rows[col, l], b]
  end
  @inbounds fHat[j, b] = acc
end

# Grid point `r` of a complex grid is elements `2r - 1` and `2r` of its real view.
@kernel function _convolve_transpose_kernel!(gr, @Const(fHat), @Const(rows), @Const(vals), batch)
  j, b = @index(Global, NTuple)
  col = j + ((b - 1) ÷ batch) * size(fHat, 1)
  @inbounds v = fHat[j, b]
  for l in 1:size(rows, 2)
    @inbounds w = vals[col, l] * v
    @inbounds r = 2 * rows[col, l]
    KernelAbstractions.@atomic gr[r - 1, b] += real(w)
    KernelAbstractions.@atomic gr[r, b] += imag(w)
  end
end

@kernel function _convolve_transpose_real_kernel!(g, @Const(fHat), @Const(rows), @Const(vals), batch)
  j, b = @index(Global, NTuple)
  col = j + ((b - 1) ÷ batch) * size(fHat, 1)
  @inbounds v = fHat[j, b]
  for l in 1:size(rows, 2)
    @inbounds KernelAbstractions.@atomic g[rows[col, l], b] += real(vals[col, l]) * v
  end
end

_as_matrix(a, n) = reshape(a, n, length(a) ÷ n)

function _launch(kernel, out, ndrange, args...)
  kernel(get_backend(out))(out, args...; ndrange)
  return nothing
end

function AbstractNFFTs.convolve!(p::GPU_NFFTPlan, g::AbstractGPUArray, fHat::AbstractGPUArray)
  fHat_ = _as_matrix(fHat, p.J)
  _launch(_convolve_kernel!, fHat_, size(fHat_), _as_matrix(g, prod(p.Ñ)), p.rows, p.vals, p.batch)
  return fHat
end

function AbstractNFFTs.convolve_transpose!(p::GPU_NFFTPlan, fHat::AbstractGPUArray, g::AbstractGPUArray)
  fill!(g, zero(eltype(g)))
  fHat_ = _as_matrix(fHat, p.J)
  if eltype(g) <: Complex
    gr = _as_matrix(reinterpret(real(eltype(g)), g), 2 * prod(p.Ñ))
    _launch(_convolve_transpose_kernel!, gr, size(fHat_), fHat_, p.rows, p.vals, p.batch)
  else
    g_ = _as_matrix(g, prod(p.Ñ))
    _launch(_convolve_transpose_real_kernel!, g_, size(fHat_), fHat_, p.rows, p.vals, p.batch)
  end
  return g
end

function AbstractNFFTs.deconvolve!(p::GPU_NFFTPlan, f::AbstractGPUArray, g::AbstractGPUArray)
  f_ = _as_matrix(f, prod(p.N))
  _launch(_deconvolve_kernel!, _as_matrix(g, prod(p.Ñ)), size(f_), f_, p.deconvolveIdx, p.windowHatInvLUT)
  return g
end

function AbstractNFFTs.deconvolve_transpose!(p::GPU_NFFTPlan, g::AbstractGPUArray, f::AbstractGPUArray)
  f_ = _as_matrix(f, prod(p.N))
  _launch(_deconvolve_transpose_kernel!, f_, size(f_), _as_matrix(g, prod(p.Ñ)), p.deconvolveIdx, p.windowHatInvLUT)
  return f
end

"""  in-place NFFT on the GPU"""
function LinearAlgebra.mul!(fHat::AbstractGPUArray, p::GPU_NFFTPlan{T}, f::AbstractGPUArray;
                          verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}
    _check_sizes(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed deconvolve!(p, f, p.tmpVec)
    t2 = @elapsed p.forwardFFT * p.tmpVec
    t3 = @elapsed convolve!(p, p.tmpVec, fHat)
    if verbose
        @info "Timing: deconv=$t1 fft=$t2 conv=$t3"
    end
    if timing != nothing
      timing.conv = t3
      timing.fft = t2
      timing.deconv = t1
    end

    return fHat
end

"""  in-place adjoint NFFT on the GPU"""
function LinearAlgebra.mul!(f::AbstractGPUArray, pl::Adjoint{Complex{T},<:GPU_NFFTPlan{T}}, fHat::AbstractGPUArray;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}
    p = pl.parent
    _check_sizes(p, f, fHat)

    t1 = @elapsed convolve_transpose!(p, fHat, p.tmpVec)
    t2 = @elapsed p.backwardFFT * p.tmpVec
    t3 = @elapsed deconvolve_transpose!(p, p.tmpVec, f)
    if verbose
        @info "Timing: conv=$t1 fft=$t2 deconv=$t3"
    end
    if timing != nothing
      timing.conv_adjoint = t1
      timing.fft_adjoint = t2
      timing.deconv_adjoint = t3
    end

    return f
end

module MriReconstructionToolboxCUDAExt

# Batched singular value operations through CUSOLVER's batched Jacobi solvers, in place of the
# generic GPU path, which moves every slice's Gram matrix to the host for its eigendecomposition.
# Median ms on an A100 (ComplexF32, slices = LLR blocks × frames):
#
# | m × n × slices | host eigen | Gram + heevjBatched | gesvdjBatched |
# |----------------|-----------:|--------------------:|--------------:|
# | 16 × 10 × 1024 |      30.6  |                1.27 |          2.00 |
# | 16 × 30 × 1024 |      69.6  |               14.4  |          2.96 |
# | 16 × 30 × 4096 |     276    |               47.2  |         10.3  |
# | 64 × 10 × 256  |       5.26 |                1.12 |             - |
# | 64 × 30 × 1024 |     190    |               12.5  |             - |
# | 64 × 30 × 16384|    2932    |              203    |             - |
#
# Both solvers take matrices of at most 32 × 32: `gesvdjBatched` the slice itself, so it serves
# slices that small, and `heevjBatched` the Gram matrix of the slice's short side, so it serves
# every slice with a side of at most 32 — for a low-rank block stack, the frame count. Slices with
# both sides longer fall through to the generic GPU method. `gesvdaStridedBatched` has no size limit
# but was 3-5x slower than the Gram route wherever both applied, and `svd!` per slice 30-250x.

using MriReconstructionToolbox: MriReconstructionToolbox as MRT
using CUDA: CuArray, CUBLAS, CUSOLVER, CUFFT
using GPUArrays: AbstractGPUArray

const _CusolverEltype = Union{Float32, Float64, ComplexF32, ComplexF64}
const _JACOBI_MAX = 32

_shrink(s, τ) = ifelse(s > τ, 1 - τ / s, zero(s))

_eigen_batched!(jobz, uplo, G::CuArray{<:Complex}) = CUSOLVER.heevjBatched!(jobz, uplo, G)
_eigen_batched!(jobz, uplo, G::CuArray{<:Real}) = CUSOLVER.syevjBatched!(jobz, uplo, G)

function MRT._batched_svt!(A::CuArray{T, 3}, τ) where {T <: _CusolverEltype}
    m, n, _ = size(A)
    R = real(T)
    if m <= _JACOBI_MAX && n <= _JACOBI_MAX
        return _svt_gesvdj!(A, R(τ))
    elseif min(m, n) <= _JACOBI_MAX
        return _svt_gram!(A, R(τ))
    end
    return @invoke MRT._batched_svt!(A::AbstractGPUArray{T, 3}, τ)
end

function MRT._batched_nuclear_norm(A::CuArray{T, 3}) where {T <: _CusolverEltype}
    min(size(A, 1), size(A, 2)) <= _JACOBI_MAX ||
        return @invoke MRT._batched_nuclear_norm(A::AbstractGPUArray{T, 3})
    w = _eigen_batched!('N', 'U', _short_side_gram(A))
    return real(T)(sum(sqrt.(max.(w, 0))))
end

function _svt_gesvdj!(A, τ)
    m, n, nb = size(A)
    k = min(m, n)
    U, S, V = CUSOLVER.gesvdj!('V', copy(A))
    σ = max.(S .- τ, 0)
    Uσ = U[:, 1:k, :] .* reshape(σ, 1, k, nb)
    CUBLAS.gemm_strided_batched!('N', 'C', one(eltype(A)), Uσ, V[:, 1:k, :], zero(eltype(A)), A)
    return sum(σ)
end

# The Gram matrix of each slice's short side: `Aᴴ A` for a tall slice, `A Aᴴ` for a wide one.
function _short_side_gram(A)
    m, n, nb = size(A)
    tall = m >= n
    p = tall ? n : m
    G = similar(A, p, p, nb)
    CUBLAS.gemm_strided_batched!(tall ? 'C' : 'N', tall ? 'N' : 'C', one(eltype(A)), A, A, zero(eltype(A)), G)
    return G
end

# `Aᴴ A = V S² Vᴴ` gives `svt(A) = A V diag(max(1 - τ/s, 0)) Vᴴ`, and `A Aᴴ = U S² Uᴴ` the same
# weight matrix applied from the left.
function _svt_gram!(A, τ)
    m, n, nb = size(A)
    tall = m >= n
    p = tall ? n : m
    w, V = _eigen_batched!('V', 'U', _short_side_gram(A))
    s = sqrt.(max.(w, 0))
    total = sum(max.(s .- τ, 0))
    Vw = V .* reshape(_shrink.(s, τ), 1, p, nb)
    W = similar(A, p, p, nb)
    CUBLAS.gemm_strided_batched!('N', 'C', one(eltype(A)), Vw, V, zero(eltype(A)), W)
    Y = similar(A)
    if tall
        CUBLAS.gemm_strided_batched!('N', 'N', one(eltype(A)), A, W, zero(eltype(A)), Y)
    else
        CUBLAS.gemm_strided_batched!('N', 'N', one(eltype(A)), W, A, zero(eltype(A)), Y)
    end
    copyto!(A, Y)
    return total
end

# ─── FFT plans back to the cache ─────────────────────────────────────────────────────────────
#
# A cuFFT plan's handle returns to CUDA.jl's handle cache when the plan is finalized, and from
# there the next plan of the same shape takes it in microseconds. An operator is a tree of
# structs, tuples and arrays of operators; every plan found in it is finalized once.

function MRT._release_device_plans!(op)
    _release_plans!(Base.IdSet{Any}(), op)
    return nothing
end

_release_plans!(seen, p::CUFFT.CuFFTPlan) = (p in seen || (push!(seen, p); finalize(p)); nothing)
_release_plans!(seen, ::Union{Number, Symbol, AbstractString, Type, Function, Module, Task, Base.AbstractLock, Nothing}) = nothing
function _release_plans!(seen, x::AbstractArray)
    isbitstype(eltype(x)) && return nothing
    foreach(e -> _release_plans!(seen, e), x)
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

end # module MriReconstructionToolboxCUDAExt

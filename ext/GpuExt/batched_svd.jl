# Batched singular value operations for `BatchedNuclearNorm`.
#
# The slices of a low-rank block stack are tall and thin (voxels × frames), so each slice's
# singular value thresholding is done through its small Gram matrix: `Aᴴ A = V S² Vᴴ`, and
# `svt(A) = A V diag(max(1 - τ/s, 0)) Vᴴ`. The Gram matrices and the product with the weight
# matrices are two batched kernels on the device; the eigendecompositions of the `n × n` Gram
# matrices, `n` the short side, run on the host in double precision, which only moves
# `n² × nslices` numbers each way.

using LinearAlgebra: Diagonal, Hermitian, eigen

@kernel function _gram_kernel!(G, @Const(A))
    i, j, b = @index(Global, NTuple)
    acc = zero(eltype(G))
    for k in axes(A, 1)
        acc += conj(A[k, i, b]) * A[k, j, b]
    end
    G[i, j, b] = acc
end

@kernel function _right_multiply_kernel!(Y, @Const(A), @Const(W))
    k, j, b = @index(Global, NTuple)
    acc = zero(eltype(Y))
    for i in axes(W, 1)
        acc += A[k, i, b] * W[i, j, b]
    end
    Y[k, j, b] = acc
end

function _batched_gram(A::AbstractGPUArray{T, 3}) where {T}
    n = size(A, 2)
    G = similar(A, n, n, size(A, 3))
    backend = get_backend(A)
    _gram_kernel!(backend)(G, A; ndrange = size(G))
    return Array{complex(Float64)}(Array(G))
end

# The singular values of every slice, from its host Gram matrix, and the eigenvectors.
function _slice_spectrum(G::Array, b)
    E = eigen(Hermitian(view(G, :, :, b)))
    return sqrt.(max.(E.values, 0.0)), E.vectors
end

# Slices wider than they are tall are handled through their adjoint, whose thresholding is the
# adjoint of theirs.
_tall(A) = size(A, 1) >= size(A, 2) ? A : conj.(permutedims(A, (2, 1, 3)))

function ProximalOperators.batched_svt!(A::AbstractGPUArray{T, 3}, τ) where {T}
    size(A, 1) >= size(A, 2) || return _svt_through_adjoint!(A, τ)
    G = _batched_gram(A)
    n, nb = size(G, 1), size(G, 3)
    W = Array{complex(Float64)}(undef, n, n, nb)
    total = 0.0
    for b in 1:nb
        s, V = _slice_spectrum(G, b)
        w = map(si -> si > τ ? 1 - τ / si : 0.0, s)
        W[:, :, b] = V * Diagonal(w) * V'
        total += sum(si -> max(si - τ, 0.0), s)
    end
    Wd = copyto!(similar(A, n, n, nb), convert(Array{T}, T <: Real ? real.(W) : W))
    Y = similar(A)
    _right_multiply_kernel!(get_backend(A))(Y, A, Wd; ndrange = size(Y))
    copyto!(A, Y)
    return real(T)(total)
end

function _svt_through_adjoint!(A, τ)
    At = _tall(A)
    total = ProximalOperators.batched_svt!(At, τ)
    A .= conj.(permutedims(At, (2, 1, 3)))
    return total
end

function ProximalOperators.batched_nuclear_norm(A::AbstractGPUArray{T, 3}) where {T}
    G = _batched_gram(_tall(A))
    return real(T)(sum(b -> sum(first(_slice_spectrum(G, b))), axes(G, 3); init = 0.0))
end

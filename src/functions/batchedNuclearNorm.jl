# sum of the nuclear norms of the slices of a 3-D array (times a constant)

export BatchedNuclearNorm

"""
    BatchedNuclearNorm(λ=1)

Return the sum of the nuclear norms of the slices `X[:, :, k]` of a three-dimensional array
```math
f(X) = λ ∑_k \\|X_{:,:,k}\\|_*,
```
where `λ` is a nonnegative parameter. Its proximal mapping thresholds the singular values of
every slice independently.

A stack of many small matrices is what block low-rank models produce, and the slices are what
the device methods batch over: on a GPU array the Gram matrices of all slices are formed in one
kernel and decomposed together on the host, and with CUDA loaded, CUSOLVER's batched Jacobi
solvers do it on the device for slices with a side of at most 32.
"""
struct BatchedNuclearNorm{R}
    lambda::R
    function BatchedNuclearNorm{R}(lambda::R) where {R}
        lambda < 0 && error("parameter λ must be nonnegative")
        return new(lambda)
    end
end

BatchedNuclearNorm(lambda::R = 1) where {R} = BatchedNuclearNorm{R}(lambda)

is_convex(::Type{<:BatchedNuclearNorm}) = true
is_positively_homogeneous(::Type{<:BatchedNuclearNorm}) = true

(f::BatchedNuclearNorm)(X::AbstractArray{<:Any, 3}) = f.lambda * batched_nuclear_norm(X)

function prox!(Y::AbstractArray{<:Any, 3}, f::BatchedNuclearNorm, X::AbstractArray{<:Any, 3}, gamma)
    Y === X || copyto!(Y, X)
    return f.lambda * batched_svt!(Y, f.lambda * gamma)
end

function prox_naive(f::BatchedNuclearNorm, X, gamma)
    Y = copy(X)
    return Y, f.lambda * batched_svt!(Y, f.lambda * gamma)
end

"""
    batched_svt!(A::AbstractArray{T,3}, τ) -> Real

Replace every slice `A[:, :, k]` with its singular value thresholding at `τ` (soft-threshold the
singular values, keep the vectors) and return the sum of the thresholded singular values. The
device extensions batch over the slices; this method loops over them.
"""
function batched_svt!(A::AbstractArray{T, 3}, τ) where {T}
    R = real(T)
    total = zero(R)
    for k in axes(A, 3)
        M = view(A, :, :, k)
        F = svd!(Matrix(M))
        σ = max.(zero(R), F.S .- R(τ))
        copyto!(M, F.U * Diagonal(σ) * F.Vt)
        total += sum(σ)
    end
    return total
end

"""
    batched_nuclear_norm(A::AbstractArray{T,3}) -> Real

`∑ₖ ‖A[:, :, k]‖_*`, batched over the slices by the device extensions.
"""
batched_nuclear_norm(A::AbstractArray{T, 3}) where {T} =
    sum(k -> sum(svdvals!(Matrix(view(A, :, :, k)))), axes(A, 3); init = zero(real(T)))

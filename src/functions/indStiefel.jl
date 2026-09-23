# indicator of the Stiefel manifold

export IndStiefel

@doc raw"""
    IndStiefel()

Return the indicator of the Stiefel manifold
```math
S_{n,p} = \left\{ X \in \mathbb{F}^{n \times p} : X^*X = I \right\}.
```
where ``\mathbb{F}`` is the real or complex field, and parameters ``n`` and ``p``
are inferred from the matrix provided as input.
"""
struct IndStiefel{B}
    buf::B
end

IndStiefel() = IndStiefel{Nothing}(nothing)

is_set_indicator(f::Type{<:IndStiefel}) = true

# `svd` copies its argument internally; `svd!` on an owned copy saves that copy.
preallocate(f::IndStiefel, X::AbstractMatrix) = IndStiefel(
    (sig = input_signature(X), X_copy = similar(X))
)

function (::IndStiefel)(X)
    R = real(eltype(X))
    F = with_factorization_threads(() -> svd(X), X)
    if all(F.S .≈ R(1))
        return R(0)
    end
    return R(Inf)
end

function prox!(Y, f::IndStiefel, X, gamma)
    # Falls back to the host when the storage has no device factorization. The question is
    # asked of the method table rather than of the array type: `svd!` exists for a `CuMatrix`
    # and this then runs natively on the device, while a backend that has not implemented it
    # is served correctly by a round-trip instead of failing.
    runs_natively(svd!, X) || return host_prox!(Y, f, X, gamma)
    R = real(eltype(X))
    n, p = size(X)
    b = get_buffers(f, X)
    b.X_copy .= X
    F = with_factorization_threads(() -> svd!(b.X_copy), X)
    U_sliced = view(F.U, :, 1:p)
    mul!(Y, U_sliced, F.Vt)
    return R(0)
end

function prox_naive(::IndStiefel, X, gamma)
    R = real(eltype(X))
    n, p = size(X)
    F = svd(X)
    Y = F.U[:, 1:p] * F.Vt
    return Y, R(0)
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:IndStiefel}) = :device_lapack

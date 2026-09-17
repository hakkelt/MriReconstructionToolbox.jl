# nuclear Norm (times a constant)

export NuclearNorm

"""
    NuclearNorm(λ=1)

Return the nuclear norm
```math
f(X) = \\|X\\|_* = λ ∑_i σ_i(X),
```
where `λ` is a positive parameter and ``σ_i(X)`` is ``i``-th singular value of matrix ``X``.
"""
struct NuclearNorm{R, B}
    lambda::R
    buf::B
    function NuclearNorm{R, B}(lambda::R, buf::B) where {R, B}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda, buf)
        end
    end
end

is_convex(f::Type{<:NuclearNorm}) = true

NuclearNorm(lambda::R=1; buf=nothing) where {R} = NuclearNorm{R, typeof(buf)}(lambda, buf)
NuclearNorm{R}(lambda::R) where {R} = NuclearNorm{R, Nothing}(lambda, nothing)

function preallocate(f::NuclearNorm, X::AbstractMatrix)
    R = real(eltype(X))
    k = minimum(size(X))
    return NuclearNorm(f.lambda; buf = (
        sig = input_signature(X),
        X_copy = similar(X),
        S_thresh = similar(X, R, k),
        M = similar(X, k, size(X, 2)),
    ))
end

function (f::NuclearNorm)(X)
    runs_natively(svd!, X) || return host_call(f, X)
    F = svd(X)
    return f.lambda * sum(F.S)
end

function prox!(Y, f::NuclearNorm, X, gamma)
    # Falls back to the host when the storage has no device factorization. The question is
    # asked of the method table rather than of the array type: `svd!` exists for a `CuMatrix`
    # and this then runs natively on the device, while a backend that has not implemented it
    # is served correctly by a round-trip instead of failing.
    runs_natively(svd!, X) || return host_prox!(Y, f, X, gamma)
    R = real(eltype(X))
    b = get_buffers(f, X)
    b.X_copy .= X
    F = svd!(b.X_copy)
    S_thresh = b.S_thresh
    S_thresh .= max.(R(0), F.S .- f.lambda*gamma)
    rankY = findfirst(iszero, S_thresh)
    if rankY === nothing
        rankY = minimum(size(X))
    end
    Vt_thresh = view(F.Vt, 1:rankY, :)
    U_thresh = view(F.U, :, 1:rankY)
    # TODO: the order of the following matrix products should depend on the shape of x
    M = view(b.M, 1:rankY, :)
    M .= view(S_thresh, 1:rankY) .* Vt_thresh
    mul!(Y, U_thresh, M)
    return f.lambda * sum(S_thresh)
end

function prox_naive(f::NuclearNorm, X, gamma)
    F = svd(X)
    S = max.(0, F.S .- f.lambda*gamma)
    Y = F.U * (Diagonal(S) * F.Vt)
    return Y, f.lambda * sum(S)
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:NuclearNorm}) = :device_lapack

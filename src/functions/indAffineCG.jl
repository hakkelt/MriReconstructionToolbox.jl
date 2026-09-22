# indicator of an affine set defined by a matrix-free linear operator

export IndAffineCG

"""
    IndAffineCG(A, b; maxit=50, tol=1e-6, AAc_diag=nothing)

Indicator function of the affine set
```math
S = \\{x : \\mathcal{A}x = b\\},
```
where `A` is a **matrix-free** linear operator: any object supporting `A * x` and `A' * y`, with
`A * x` the same size as `b`. This is the counterpart of [`IndAffine`](@ref) for operators that
have no matrix representation — a Fourier transform, a subsampling mask, a stack of the two — where
neither the QR factorization of `IndAffineDirect` nor the row-by-row sweep of `IndAffineIterative`
is available.

The proximal mapping is the orthogonal projection onto ``S``,
```math
\\mathrm{proj}(x) = x - \\mathcal{A}^* (\\mathcal{A}\\mathcal{A}^*)^{-1} (\\mathcal{A}x - b),
```
in which ``(\\mathcal{A}\\mathcal{A}^*)v = r`` is solved by Conjugate Gradient, at most `maxit`
iterations, to a relative residual of `tol`. `A` is assumed to have full row rank; if it does not,
CG converges to a least-squares solution and the result is the projection onto the nearest
consistent set.

`tol` doubles as the tolerance of the feasibility test in `f(x)`: the indicator returns `0` when
``\\|\\mathcal{A}x - b\\| \\le`` `tol` and `Inf` otherwise.

Pass `AAc_diag` when ``\\mathcal{A}\\mathcal{A}^*`` is known to be diagonal — a scalar for a multiple
of the identity, or an array of its diagonal entries broadcastable against `b`. The projection is
then evaluated in closed form and no CG iteration is run, which is much cheaper. Entries at or below
`eps` are treated as zero, i.e. the corresponding residual component is dropped rather than divided
by.
"""
struct IndAffineCG{M, V, R <: Real, D} <: IndAffine
    A::M
    b::V
    maxit::Int
    tol::R
    AAc_diag::D
end

function IndAffineCG(A, b; maxit::Int = 50, tol::Real = 1.0e-6, AAc_diag = nothing)
    return IndAffineCG(A, b, maxit, float(tol), AAc_diag)
end

is_proximable(::Type{<:IndAffineCG}) = true
# The indicator of an affine set is convex, and (being an indicator) never smooth. `IndAffine` itself
# leaves `is_convex` at its `false` default, so state it here.
is_convex(::Type{<:IndAffineCG}) = true
is_smooth(::Type{<:IndAffineCG}) = false

function (f::IndAffineCG)(x)
    R = real(eltype(x))
    residual = f.A * x - f.b
    return norm(residual) <= f.tol ? R(0) : R(Inf)
end

"""
    _cg_solve_AAc(A, r; maxit, tol)

Solve ``(\\mathcal{A}\\mathcal{A}^*) v = r`` by Conjugate Gradient, using only `A * ·` and `A' * ·`.
The system matrix is Hermitian positive semi-definite by construction, so CG applies; a non-positive
curvature direction means the operator is rank-deficient along it and the iteration stops there.
"""
function _cg_solve_AAc(A, r::AbstractArray{T}; maxit::Int = 50, tol::Real = 1.0e-6) where {T}
    v = zeros(T, size(r))
    norm_r = norm(r)
    norm_r <= tol && return v
    p = copy(r)
    res = copy(r)
    rsold = real(dot(res, res))
    for _ in 1:maxit
        Ap = A' * p
        Hp = A * Ap
        pHp = real(dot(p, Hp))
        if pHp <= eps(real(T))
            break
        end
        alpha = rsold / pHp
        v .+= alpha .* p
        res .-= alpha .* Hp
        rsnew = real(dot(res, res))
        if sqrt(rsnew) / norm_r <= tol
            break
        end
        p .= res .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return v
end

_solve_AAc(f::IndAffineCG, r) = _solve_AAc(f.AAc_diag, f, r)

_solve_AAc(::Nothing, f::IndAffineCG, r) = _cg_solve_AAc(f.A, r; maxit = f.maxit, tol = f.tol)

# A zero diagonal entry means the operator has no component along that direction: drop the residual
# there rather than dividing by zero.
_solve_AAc(d::Number, ::IndAffineCG, r) =
    abs(d) > eps(real(float(d))) ? r ./ d : zero(r)

_solve_AAc(d::AbstractArray, ::IndAffineCG, r) =
    ifelse.(abs.(d) .> eps(real(eltype(d))), r ./ d, zero(r))

function prox!(y, f::IndAffineCG, x, gamma)
    R = real(eltype(x))
    r = f.A * x - f.b
    if norm(r) <= f.tol
        copyto!(y, x)
        return R(0)
    end
    v = _solve_AAc(f, r)
    copyto!(y, x .- f.A' * v)
    return R(0)
end

function prox_naive(f::IndAffineCG, x, gamma)
    y = similar(x)
    fy = prox!(y, f, x, gamma)
    return y, fy
end

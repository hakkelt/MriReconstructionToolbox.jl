"""
	HardConsistencyProx(𝒜, y, inner_maxit, inner_tol)

Proximable indicator function representing the hard data consistency constraint ``\\{x \\mid \\mathcal{A}x = y\\}``.
The proximal operator computes the orthogonal projection:
``\\operatorname{proj}(x) = x - \\mathcal{A}^* (\\mathcal{A} \\mathcal{A}^*)^{-1} (\\mathcal{A}x - y)``.

When `is_AAc_diagonal(𝒜)` is true (e.g. single-coil Cartesian or a `KSpaceToImage` signal model), ``(\\mathcal{A} \\mathcal{A}^*)^{-1}``
is evaluated directly in closed form via `diag_AAc(𝒜)`. Otherwise, ``(\\mathcal{A} \\mathcal{A}^*) v = r`` is solved
iteratively using Conjugate Gradient up to `inner_maxit` iterations and tolerance `inner_tol`.
"""
struct HardConsistencyProx{Op, Y, R <: Real}
    𝒜::Op
    y::Y
    inner_maxit::Int
    inner_tol::R
end

ProximalCore.is_convex(::Type{<:HardConsistencyProx}) = true
ProximalCore.is_smooth(::Type{<:HardConsistencyProx}) = false
ProximalCore.is_set_indicator(::Type{<:HardConsistencyProx}) = true

function (f::HardConsistencyProx)(x)
    residual = f.𝒜 * x - f.y
    return norm(residual) <= f.inner_tol ? real(eltype(x))(0) : real(eltype(x))(Inf)
end

function _cg_solve_AAc(𝒜, r::AbstractArray{T}; inner_maxit::Int = 50, inner_tol::Real = 1.0e-6) where {T}
    v = zeros(T, size(r))
    norm_r = norm(r)
    norm_r <= inner_tol && return v
    p = copy(r)
    res = copy(r)
    rsold = real(dot(res, res))
    for _ in 1:inner_maxit
        Ap = 𝒜' * p
        Hp = 𝒜 * Ap
        pHp = real(dot(p, Hp))
        if pHp <= eps(real(T))
            break
        end
        alpha = rsold / pHp
        v .+= alpha .* p
        res .-= alpha .* Hp
        rsnew = real(dot(res, res))
        if sqrt(rsnew) / norm_r <= inner_tol
            break
        end
        p .= res .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return v
end

function _project_hard_consistency(𝒜, y, x, inner_maxit::Int, inner_tol::Real)
    r = 𝒜 * x - y
    if norm(r) <= inner_tol
        return copy(x)
    end
    if is_AAc_diagonal(𝒜)
        d = diag_AAc(𝒜)
        v = if d isa Real
            abs(d) > eps(typeof(d)) ? r ./ d : zero(r)
        else
            ifelse.(abs.(d) .> eps(real(eltype(d))), r ./ d, zero(r))
        end
    else
        v = _cg_solve_AAc(𝒜, r; inner_maxit, inner_tol)
    end
    corr = 𝒜' * v
    return x .- corr
end

function ProximalCore.prox!(out, f::HardConsistencyProx, x, gamma = 1)
    proj = _project_hard_consistency(f.𝒜, f.y, x, f.inner_maxit, f.inner_tol)
    copyto!(out, proj)
    return real(eltype(x))(0)
end

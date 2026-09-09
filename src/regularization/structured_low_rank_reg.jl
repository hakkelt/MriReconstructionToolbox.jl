"""
	HankelLowRankProx(λ, max_rank, H, invmult, nbatch, threaded)

Proximable function behind [`StructuredLowRank`](@ref). It is applied to the multi-channel
k-space variable directly (identity operator), the way `BlockNuclearNorm` is applied to the
image.

Each prox evaluation is one Cadzow step per batch slab: lift the slab into the block-Hankel
matrix `M = 𝓗 * kᵦ`, low-rank it (singular-value soft-thresholding for the penalty form, hard
rank truncation for the constraint form), then de-Hankelize by the multiplicity-weighted
adjoint `kᵦ ← (𝓗ᴴ M̂) ./ mult`. Because `mult = diag(𝓗ᴴ𝓗)`, the weighted adjoint is a left
inverse of `𝓗`, so this is the exact prox when `𝓗` is treated as a weighted tight frame — the
standard SAKE / LORAKS / ALOHA update.

Like `BlockNuclearNorm`, this stays in MRT rather than going upstream (`NAMING.md`
rule 7.2): it is defined against the `(k-space grid…, channels, batch)` layout throughout, so
there is no layout-free core to lift into `ProximalOperators`. The generic piece — the
block-Hankel lift itself — did go upstream, as `AbstractOperators.Hankel`.

The `RANK` type parameter selects the form: `true` is the hard rank constraint (an indicator
function, non-convex), `false` the nuclear-norm penalty (convex).
"""
struct HankelLowRankProx{RANK, R <: Real, H <: Hankel, A <: AbstractArray}
    λ::R
    max_rank::Int
    H::H
    invmult::A          # real, size (gridsize..., nchannels); 1 ./ diag(𝓗ᴴ𝓗)
    nbatch::Int
    threaded::Bool
end

ProximalCore.is_convex(::Type{<:HankelLowRankProx{false}}) = true
ProximalCore.is_convex(::Type{<:HankelLowRankProx{true}}) = false
ProximalCore.is_positively_homogeneous(::Type{<:HankelLowRankProx}) = true
ProximalCore.is_smooth(::Type{<:HankelLowRankProx}) = false
ProximalCore.is_separable(::Type{<:HankelLowRankProx}) = false

_hlrp_slab_size(f::HankelLowRankProx) = (f.H.gridsize..., f.H.nchannels)

# Relative tolerance for deciding whether a lifted slab is numerically of rank ≤ max_rank.
# Matches `ProximalOperators.IndBallRank`, so the constraint form of `StructuredLowRank` and
# `RankLimit` call a matrix low-rank by the same yardstick.
const _HANKEL_RANK_RTOL = 1.0e-7

function (f::HankelLowRankProx{RANK})(x) where {RANK}
    xr = reshape(x, _hlrp_slab_size(f)..., f.nbatch)
    R = real(eltype(x))
    value = R(0)
    for b in 1:f.nbatch
        σ = svdvals!(f.H * collect(selectdim(xr, ndims(xr), b)))
        if RANK
            # Indicator of {k : rank(𝓗 k) ≤ max_rank}: 0 inside the set, Inf outside. The
            # Cadzow prox below is a projection of the *lifted* matrix, not of `k` itself
            # (a truncated matrix is generally not in the range of 𝓗), so an iterate is
            # normally infeasible and this correctly reports Inf. That is what makes the
            # constraint form a heuristic — see the warning in `StructuredLowRank`.
            length(σ) <= f.max_rank && continue
            σ[f.max_rank + 1] / σ[1] <= R(_HANKEL_RANK_RTOL) || return R(Inf)
        else
            value += sum(σ)
        end
    end
    return f.λ * value
end

function ProximalCore.prox!(y, f::HankelLowRankProx{RANK}, x, gamma) where {RANK}
    slab = _hlrp_slab_size(f)
    xr = reshape(x, slab..., f.nbatch)
    yr = reshape(y, slab..., f.nbatch)
    R = real(eltype(x))
    threshold = RANK ? R(0) : R(f.λ * gamma)
    partial = zeros(R, f.nbatch)
    if f.threaded
        @budgeted_threads for b in 1:f.nbatch
            partial[b] = _hlrp_prox_slab!(yr, xr, f, b, threshold, Val(RANK))
        end
    else
        for b in 1:f.nbatch
            partial[b] = _hlrp_prox_slab!(yr, xr, f, b, threshold, Val(RANK))
        end
    end
    # The constraint form is an indicator: its value at the projected point is 0.
    return RANK ? R(0) : f.λ * sum(partial)
end

function _hlrp_prox_slab!(yr, xr, f::HankelLowRankProx, b::Int, threshold, ::Val{RANK}) where {RANK}
    R = real(eltype(xr))
    xb = collect(selectdim(xr, ndims(xr), b))
    M = f.H * xb
    F = svd!(M)
    if RANK
        r = min(f.max_rank, length(F.S))
        @inbounds for i in (r + 1):length(F.S)
            F.S[i] = 0
        end
        nucval = sum(@view F.S[1:r])
    else
        F.S .= max.(R(0), F.S .- threshold)
        nucval = sum(F.S)
    end
    lmul!(Diagonal(F.S), F.Vt)
    mul!(M, F.U, F.Vt)
    yb = selectdim(yr, ndims(yr), b)
    mul!(xb, f.H', M)             # reuse xb as the (gridsize..., nchannels) adjoint buffer
    @. yb = xb * f.invmult
    return nucval
end

"""
	StructuredLowRank(; λ=nothing, max_rank=nothing, window, structure=:c, batch_dims=nothing)

Calibrationless structured low-rank k-space regularization — the SAKE / LORAKS-C family.
Exactly one of `λ` or `max_rank` must be given.

The multi-channel k-space variable is lifted into a block-Hankel matrix built from every
sliding window of size `window` over the k-space encoding dimensions, with the coil axis
stacked as extra columns (the "SAKE" layout), and low rank of that matrix is promoted. That
matrix is low-rank because the coil images obey linear-predictability relations in k-space —
which is the same structure GRAPPA and SPIRiT exploit, except that here it is estimated from
the undersampled data itself rather than from a calibration region. No sensitivity maps and
no ACS lines are needed.

# Arguments
- `λ`: penalty form. The term is `λ ⋅ ‖𝓗 k‖_*`, the nuclear norm of the lifted matrix. This is
  the convex relaxation used by LORAKS-C.
- `max_rank`: constraint form. The lifted matrix is required to have rank at most `max_rank`.
  This is SAKE's hard rank constraint, imposed by a Cadzow truncation step.
- `window`: sliding-window size over the k-space encoding dims, a 2- or 3-tuple. Typically
  `(5, 5)` or `(6, 6)` in 2D and `(4, 4, 4)` in 3D.
- `structure`: only `:c` (plain block-Hankel) is currently implemented.
- `batch_dims`: (optional) dimension name(s) solved independently (e.g. `:time`); they are not
  coupled by the Hankel embedding. Matching is by name, so this only has an effect when the
  image is a `NamedDimsArray`.

!!! warning "The `max_rank` form is non-convex"
    `rank(𝓗 k) ≤ r` (SAKE) is a projection onto a **non-convex** set, and the projection is
    applied to the lifted matrix rather than to `k` itself. Any splitting algorithm applied to
    it — `ADMM`, `DouglasRachford` — is therefore a heuristic with no convergence guarantee: it
    may converge to a non-global fixed point or not converge at all, and the result depends on
    the initial estimate. This mirrors the original Cadzow-style alternating-projection
    algorithm of Shin et al. (2014). The `λ` form (LORAKS-C) is convex.

# Notes
- `λ` and `max_rank` are mutually exclusive; passing both, or neither, is an `ArgumentError`
  (the same penalty/constraint pairing as [`L0Image`](@ref), `NAMING.md` rule 1.3).
- The optimization variable is the full multi-channel k-space, so pair the term with
  `signal_model = KSpaceToImage(...)`, the way `SPIRiT(; iterative = true)` does.
- The prox is one Cadzow step per batch slab: lift, low-rank, then de-Hankelize with the
  multiplicity-weighted adjoint (the internal `HankelLowRankProx`). Per-iteration cost is one
  economy SVD of a `prod(gridsize .- window .+ 1) × (prod(window) * ncoils)` matrix per slab,
  so keep `window` small.
- Only the plain block-Hankel structure is implemented. The LORAKS S-matrix (which also
  imposes conjugate symmetry) and ALOHA's transform-domain weighting are not available.

# References
- Shin, P. J., et al. (2014). *Calibrationless parallel imaging reconstruction based on
  structured low-rank matrix completion.* Magn Reson Med, 72(4), 959-970. — the `max_rank` form.
- Haldar, J. P. (2014). *Low-rank modeling of local k-space neighborhoods (LORAKS) for
  constrained MRI.* IEEE Trans Med Imaging, 33(3), 668-681. — the `λ` form.
"""
struct StructuredLowRank{T, N} <: Regularization
    λ::Union{T, Nothing}
    max_rank::Union{Int, Nothing}
    window::NTuple{N, Int}
    structure::Symbol
    batch_dims::Union{Nothing, Tuple}
    function StructuredLowRank(;
            λ::Union{Real, Nothing} = nothing, max_rank::Union{Integer, Nothing} = nothing,
            window, structure::Symbol = :c, batch_dims = nothing,
        )
        @argcheck (λ === nothing) != (max_rank === nothing) "StructuredLowRank requires exactly one of `λ` or `max_rank` (they are mutually exclusive), got λ=$(repr(λ)), max_rank=$(repr(max_rank))"
        λ !== nothing && @argcheck λ >= 0 "λ must be non-negative"
        max_rank !== nothing && @argcheck max_rank > 0 "max_rank must be positive"
        @argcheck structure === :c "only structure = :c is currently implemented"
        @argcheck length(window) in (2, 3) "window must be a 2- or 3-tuple"
        @argcheck all(window .> 0) "window sizes must be positive"
        bd = batch_dims === nothing ? nothing : Tuple(batch_dims)
        T = λ === nothing ? Float64 : typeof(λ)
        w = NTuple{length(window), Int}(window)
        return new{T, length(window)}(λ, max_rank === nothing ? nothing : Int(max_rank), w, structure, bd)
    end
end

# The prox acts on the k-space variable itself; the block-Hankel lift lives inside it.
get_operator(::StructuredLowRank, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

# The Hankel embedding couples every k-space encoding sample and every coil, so only
# explicitly declared batch dims stay separable when the problem is split into tasks.
function get_affected_dims(reg::StructuredLowRank, ::Nothing, image_dims)
    reg.batch_dims === nothing && return Tuple(image_dims)
    return Tuple(d for d in image_dims if !(d in reg.batch_dims))
end

# The nuclear norm is homogeneous of degree 1, so λ scales linearly. A rank constraint is
# scale-invariant (rank(factor * M) == rank(M)), so it needs no correction.
function scale_regularization(reg::StructuredLowRank, factor::Real)
    reg.λ === nothing && return reg
    return StructuredLowRank(;
        λ = reg.λ * factor, window = reg.window, structure = reg.structure, batch_dims = reg.batch_dims,
    )
end

function _build_hankel_prox(reg::StructuredLowRank{T0, N}, x_val, ::Type{T}) where {T0, N, T}
    @argcheck ndims(x_val) >= N + 1 "StructuredLowRank needs at least $(N + 1) dims (k-space grid + channels), got $(ndims(x_val))"
    gridsize = NTuple{N, Int}(size(x_val)[1:N])
    @argcheck all(reg.window .<= gridsize) "window $(reg.window) exceeds the k-space grid size $gridsize"
    nchannels = size(x_val, N + 1)
    nbatch = prod(size(x_val)[(N + 2):end]; init = 1)
    H = Hankel(T, gridsize, reg.window; nchannels, channels = true)
    invmult = one(real(T)) ./ real(T).(AbstractOperators.diag_AcA(H))
    return H, invmult, nbatch
end

function materialize(reg::StructuredLowRank, x::Variable{T}; threaded::Bool) where {T}
    H, invmult, nbatch = _build_hankel_prox(reg, ~x, T)
    op = get_operator(reg, ~x; threaded)
    R = real(T)
    if reg.λ !== nothing
        λ = R(reg.λ)
        f = HankelLowRankProx{false, R, typeof(H), typeof(invmult)}(λ, 0, H, invmult, nbatch, threaded)
        repr = @sprintf "%g ⋅ ‖𝓗 * %s‖_*" λ get_name(x)
    else
        f = HankelLowRankProx{true, R, typeof(H), typeof(invmult)}(one(R), reg.max_rank, H, invmult, nbatch, threaded)
        repr = @sprintf "rank(𝓗 * %s) ≤ %d" get_name(x) reg.max_rank
    end
    return StructuredOptimization.Term(1, f, op * x, repr)
end

# Prox is one economy SVD per batch slab: level-3 BLAS, worth threading. See `uses_blas3`.
uses_blas3(::StructuredLowRank) = true

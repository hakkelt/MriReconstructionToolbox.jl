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

With a transform-domain weight `w` (ALOHA), the lift is `M = 𝓗 * (w ⊙ kᵦ)` and the same
least-squares de-Hankelization becomes `kᵦ ← conj(w) ⊙ (𝓗ᴴ M̂) ./ (|w|² ⊙ mult)`, since
`(𝓗∘diag(w))ᴴ(𝓗∘diag(w)) = diag(|w|² ⊙ mult)`. Samples where `w` vanishes (the DC sample of a
first-difference weight, for instance) are not constrained by the weighted term at all, so the
prox leaves them untouched; `keep` marks them and `invmult` is zero there.

Like `BlockNuclearNorm`, this stays in MRT rather than going upstream (`NAMING.md`
rule 7.2): it is defined against the `(k-space grid…, channels, batch)` layout throughout, so
there is no layout-free core to lift into `ProximalOperators`. The generic piece — the
block-Hankel lift itself — did go upstream, as `AbstractOperators.Hankel`.

The `RANK` type parameter selects the form: `true` is the hard rank constraint (an indicator
function, non-convex), `false` the nuclear-norm penalty (convex).
"""
struct HankelLowRankProx{RANK, R <: Real, H <: Hankel, A <: AbstractArray, W, K}
    λ::R
    max_rank::Int
    H::H
    invmult::A          # size (gridsize..., nchannels); 1 ./ diag(𝓗ᴴ𝓗), or conj(w) ./ (|w|² ⊙ mult)
    nbatch::Int
    threaded::Bool
    w::W                # nothing, or the ALOHA transform-domain weight on the k-space grid
    keep::K             # nothing, or `true` where `w` vanishes and the prox is the identity
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
        σ = svdvals!(f.H * _hlrp_weighted(f.w, collect(selectdim(xr, ndims(xr), b))))
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

# The ALOHA lift `w ⊙ k`. `nothing` is the unweighted SAKE / LORAKS-C case and returns the slab
# itself, so that path allocates and copies exactly what it did before weights existed.
_hlrp_weighted(::Nothing, xb) = xb
_hlrp_weighted(w, xb) = w .* xb

# `keep` marks the samples the weighted term does not constrain (`w == 0`), where the prox is the
# identity rather than zero.
_hlrp_restore!(yb, xb, ::Nothing) = yb
function _hlrp_restore!(yb, xb, keep)
    @. yb += keep * xb
    return yb
end

function _hlrp_prox_slab!(yr, xr, f::HankelLowRankProx, b::Int, threshold, ::Val{RANK}) where {RANK}
    R = real(eltype(xr))
    xb = collect(selectdim(xr, ndims(xr), b))
    buffer = _hlrp_weighted(f.w, xb)
    M = f.H * buffer
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
    # `buffer` is `xb` itself when unweighted, and the separate `w ⊙ xb` array when weighted, so
    # in both cases it is a scratch array of the right shape for the adjoint -- but `xb` must
    # survive it in the weighted case, which is why the restore below reads `xb`, not `buffer`.
    mul!(buffer, f.H', M)
    @. yb = buffer * f.invmult
    _hlrp_restore!(yb, xb, f.keep)
    return nucval
end

"""
	StructuredLowRank(; λ=nothing, max_rank=nothing, window, structure=:c, weights=nothing, batch_dims=nothing)

Calibrationless structured low-rank k-space regularization — the SAKE / LORAKS-C / ALOHA family.
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
- `weights`: (optional) ALOHA's transform-domain weighting. `nothing` (the default) is plain
  SAKE / LORAKS-C. Otherwise the lift is applied to `w ⊙ k` for each weight `w`, and several
  weights are combined by a [`ProximalAverage`](@ref) the way [`MultiScaleLowRank`](@ref)
  combines its scales. Accepts:
  - `:tv` — one first-difference weight per k-space encoding dimension, the annihilating filter
    implied by a piecewise-constant (total-variation-sparse) image.
  - `:wavelet` — the pyramidal form: the same first differences taken at step 1 and step 2, per
    dimension (the Haar detail bands of the first two scales), giving `2 * length(window)` terms.
  - an array on the k-space encoding grid — a custom weight.
  - a tuple or vector of such arrays — a custom pyramid.
  The built-in models assume MRT's default centered k-space layout (DC at `N ÷ 2 + 1`); pass an
  explicit array when the DC sits elsewhere (`shifted_kspace_dims`).
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
- ALOHA (`weights`) exploits a second structure on top of the coil relations: if a transform of
  the image is sparse, the correspondingly weighted k-space is annihilated by a small filter, so
  `𝓗(w ⊙ k)` is low-rank even for a single channel. The weighted terms cost one SVD each, so
  `weights = :wavelet` is `2 * length(window)` times the work of the unweighted term.
- Only the plain block-Hankel structure is implemented. The LORAKS S-matrix (which also imposes
  conjugate symmetry) and the G-matrix are not available.

# References
- Shin, P. J., et al. (2014). *Calibrationless parallel imaging reconstruction based on
  structured low-rank matrix completion.* Magn Reson Med, 72(4), 959-970. — the `max_rank` form.
- Haldar, J. P. (2014). *Low-rank modeling of local k-space neighborhoods (LORAKS) for
  constrained MRI.* IEEE Trans Med Imaging, 33(3), 668-681. — the `λ` form.
- Jin, K. H., Lee, D., & Ye, J. C. (2016). *A general framework for compressed sensing and
  parallel MRI using annihilating filter based low-rank Hankel matrix.* IEEE Trans Comput
  Imaging, 2(4), 480-495. — ALOHA, the `weights` argument.
"""
struct StructuredLowRank{T, N, W} <: Regularization
    λ::Union{T, Nothing}
    max_rank::Union{Int, Nothing}
    window::NTuple{N, Int}
    structure::Symbol
    weights::W
    batch_dims::Union{Nothing, Tuple}
    function StructuredLowRank(;
            λ::Union{Real, Nothing} = nothing, max_rank::Union{Integer, Nothing} = nothing,
            window, structure::Symbol = :c, weights = nothing, batch_dims = nothing,
        )
        @argcheck (λ === nothing) != (max_rank === nothing) "StructuredLowRank requires exactly one of `λ` or `max_rank` (they are mutually exclusive), got λ=$(repr(λ)), max_rank=$(repr(max_rank))"
        λ !== nothing && @argcheck λ >= 0 "λ must be non-negative"
        max_rank !== nothing && @argcheck max_rank > 0 "max_rank must be positive"
        @argcheck structure === :c "only structure = :c is currently implemented"
        @argcheck length(window) in (2, 3) "window must be a 2- or 3-tuple"
        @argcheck all(window .> 0) "window sizes must be positive"
        _check_slr_weights(weights, length(window))
        bd = batch_dims === nothing ? nothing : Tuple(batch_dims)
        T = λ === nothing ? Float64 : typeof(λ)
        w = NTuple{length(window), Int}(window)
        return new{T, length(window), typeof(weights)}(
            λ, max_rank === nothing ? nothing : Int(max_rank), w, structure, weights, bd
        )
    end
end

const _SLR_WEIGHT_MODELS = (:tv, :wavelet)

_check_slr_weights(::Nothing, ::Int) = nothing
function _check_slr_weights(weights::Symbol, ::Int)
    return @argcheck weights in _SLR_WEIGHT_MODELS "weights must be one of $(_SLR_WEIGHT_MODELS), an array on the k-space grid, or a collection of such arrays, got :$weights"
end
function _check_slr_weights(weights::AbstractArray{<:Number}, N::Int)
    return @argcheck ndims(weights) == N "a weight array must have one dimension per k-space encoding dimension ($N), got $(ndims(weights))"
end
function _check_slr_weights(weights, N::Int)
    @argcheck weights isa Union{Tuple, AbstractVector} "weights must be nothing, one of $(_SLR_WEIGHT_MODELS), an array on the k-space grid, or a collection of such arrays"
    @argcheck !isempty(weights) "a weight collection must contain at least one weight"
    for w in weights
        @argcheck w isa AbstractArray{<:Number} "every entry of a weight collection must be an array"
        _check_slr_weights(w, N)
    end
    return nothing
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
        λ = reg.λ * factor, window = reg.window, structure = reg.structure,
        weights = reg.weights, batch_dims = reg.batch_dims,
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

"""
	_slr_difference_weight(T, gridsize, d, h)

The DFT symbol of the finite difference `x[n] - x[n - h]` along dimension `d`, on the centered
k-space grid (DC at `N ÷ 2 + 1`): `w(k) = 1 - exp(-2πi h k / N)`. This is the annihilating filter
implied by sparsity of that difference, which is what ALOHA weights the Hankel lift with — `h = 1`
is the total-variation model, `h = 2ˢ⁻¹` the Haar detail band of scale `s`.

Returned with a trailing singleton axis so it broadcasts over the channel dimension.
"""
function _slr_difference_weight(::Type{T}, gridsize::NTuple{N, Int}, d::Int, h::Int) where {T, N}
    n = gridsize[d]
    line = [one(T) - cispi(T(-2 * h * (i - 1 - n ÷ 2) // n)) for i in 1:n]
    w = similar(line, (gridsize..., 1))
    for idx in CartesianIndices(gridsize)
        w[idx, 1] = line[idx[d]]
    end
    return w
end

_slr_reshape_weight(::Type{T}, w::AbstractArray, gridsize::NTuple{N, Int}) where {T, N} =
    reshape(T.(w), (gridsize..., 1))

# The ALOHA weights as concrete arrays on the k-space grid. `nothing` keeps the unweighted
# SAKE / LORAKS-C term; anything else yields one weighted term per entry, combined by
# `ProximalAverage` in `materialize` exactly as `MultiScaleLowRank` combines its scales.
_slr_weights(::StructuredLowRank{T0, N0, Nothing}, ::NTuple{N, Int}, ::Type{T}) where {T0, N0, N, T} = nothing
function _slr_weights(reg::StructuredLowRank, gridsize::NTuple{N, Int}, ::Type{T}) where {N, T}
    reg.weights isa Symbol || return Tuple(_slr_reshape_weight(T, w, gridsize) for w in _slr_weight_collection(reg.weights))
    steps = reg.weights === :tv ? (1,) : (1, 2)
    @argcheck all(gridsize .> maximum(steps)) "the k-space grid $gridsize is too small for the :$(reg.weights) weight model"
    # The first entry is the approximation band -- an all-ones weight, i.e. the plain unweighted
    # LORAKS-C term. It is part of the model, not a hedge: a difference weight vanishes at DC, so
    # the detail bands alone say nothing about the low frequencies that carry most of the energy,
    # and a pyramid without its approximation band reconstructs measurably worse than plain
    # LORAKS-C. The detail bands are what ALOHA adds on top.
    return (
        fill(one(T), (gridsize..., 1)),
        (_slr_difference_weight(T, gridsize, d, h) for d in 1:N for h in steps)...,
    )
end

_slr_weight_collection(w::AbstractArray{<:Number}) = (w,)
_slr_weight_collection(w) = Tuple(w)

# `(𝓗∘diag(w))ᴴ(𝓗∘diag(w)) = diag(|w|² ⊙ mult)`, so the least-squares de-Hankelization factor is
# `conj(w) ./ (|w|² ⊙ mult)`. Where `w` vanishes the weighted term says nothing about that sample,
# so the factor is zero and `keep` hands the sample through unchanged.
function _slr_weighted_factors(w, invmult, ::Type{T}) where {T}
    R = real(T)
    w2 = abs2.(w)
    tol = R(eps(R)) * maximum(w2)
    live = w2 .> tol
    factor = @. ifelse(live, conj(w) * invmult / ifelse(live, w2, one(R)), zero(T))
    keep = T.(.!live)
    return factor, keep
end

function _hankel_low_rank_prox(::Val{RANK}, λ, max_rank, H, invmult, nbatch, threaded, w, keep) where {RANK}
    return HankelLowRankProx{RANK, typeof(λ), typeof(H), typeof(invmult), typeof(w), typeof(keep)}(
        λ, max_rank, H, invmult, nbatch, threaded, w, keep
    )
end

function materialize(reg::StructuredLowRank, x::Variable{T}; threaded::Bool) where {T}
    H, invmult, nbatch = _build_hankel_prox(reg, ~x, T)
    op = get_operator(reg, ~x; threaded)
    R = real(T)
    penalty = reg.λ !== nothing
    λ = penalty ? R(reg.λ) : one(R)
    max_rank = penalty ? 0 : reg.max_rank
    form = Val(!penalty)
    ws = _slr_weights(reg, H.gridsize, T)
    if ws === nothing
        f = _hankel_low_rank_prox(form, λ, max_rank, H, invmult, nbatch, threaded, nothing, nothing)
        repr = penalty ?
            (@sprintf "%g ⋅ ‖𝓗 * %s‖_*" λ get_name(x)) :
            (@sprintf "rank(𝓗 * %s) ≤ %d" get_name(x) max_rank)
    else
        fs = Tuple(
            begin
                    factor, keep = _slr_weighted_factors(w, invmult, T)
                    _hankel_low_rank_prox(form, λ, max_rank, H, factor, nbatch, threaded, w, keep)
                end for w in ws
        )
        f = length(fs) == 1 ? fs[1] : ProximalAverage(fs, fill(R(1 / length(fs)), length(fs)))
        model = reg.weights isa Symbol ? ":$(reg.weights)" : "custom"
        repr = penalty ?
            (@sprintf "avg_{w ∈ %s} %g ⋅ ‖𝓗(w ⊙ %s)‖_*" model λ get_name(x)) :
            (@sprintf "avg_{w ∈ %s} rank(𝓗(w ⊙ %s)) ≤ %d" model get_name(x) max_rank)
    end
    return StructuredOptimization.Term(1, f, op * x, repr)
end

# Prox is one economy SVD per batch slab: level-3 BLAS, worth threading. See `uses_blas3`.
uses_blas3(::StructuredLowRank) = true

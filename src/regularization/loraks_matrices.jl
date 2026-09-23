"""
	LoraksLift(gridsize, ksize, nchannels, structure; center = nothing)

Row and column geometry of the two phase-constrained LORAKS lifts — Haldar's S-matrix and
G-matrix (IEEE TMI 33(3):668-681, 2014, Eqs. 17-22) — used by [`StructuredLowRank`](@ref) with
`structure = :s` or `:g`.

Both lifts read the k-space slab `(grid…, channels)` twice: once at the sliding-window
neighbourhoods `k(ν - p)` that already make up the plain block-Hankel (C) matrix, and once on
the *reflected* side of k-space. The reflection is about the DC sample, because that is where
conjugate symmetry of a real-valued image lives: a real image has `k(-ν) = conj(k(ν))`, and a
smoothly varying image phase leaves an approximate version of that relation in place. The S
matrix is

    S = [ Sr₊ - Sr₋   -Si₊ + Si₋
          Si₊ + Si₋    Sr₊ + Sr₋ ] ∈ ℝ^(2K × 2NᵣL)

where `Sr±`/`Si±` hold the real and imaginary parts of `k(±ν⁽ᵏ⁾ - pₘ)` (Eqs. 22, 3-6). The G
matrix is

    G = [ -gr   Gr   -Gi
           gi   Gi    Gr ] ∈ ℝ^(2K × (2NᵣL + L))

with `Gr`/`Gi` the real and imaginary parts of the same neighbourhoods and `g` the single
reflected sample `k(-ν⁽ᵏ⁾)` per row (Eqs. 17-21). `G` is the weaker of the two: the paper's own
analysis is that it is rank-deficient but not necessarily *low* rank unless the image support is
limited as well, which is why P-LORAKS carries the S matrix forward and not this one. The
`g` column is one per channel here — the papers write `G` down for a single channel only, so the
multi-channel form follows P-LORAKS' channel stacking.

The reflection is taken **modulo the grid**, `ν ↦ -ν (mod N)`, which is the exact conjugate
symmetry of the DFT the encoding operator actually applies. Haldar restricts the rows instead to
the neighbourhoods whose reflection stays inside an odd, zero-symmetric grid; that choice was
measured here and rejected — it yields the same numerical rank (34 of 50 on a limited-support
real phantom, with an exact null space either way) but leaves 295 of 1024 k-space samples out of
the lift entirely, where the modular form constrains all of them. It also makes `center` a real
parameter rather than an assumption: the lift is identical for a centred k-space and for the same
data with DC at index 1.

Both lifts are only **real**-linear: they split k-space into real and imaginary parts and
recombine them with fixed signs, so `L(i ⋅ k) ≠ i ⋅ L(k)`. That is why they stay here rather than
going upstream next to `AbstractOperators.Hankel` (`NAMING.md` rule 7.3) — they are not
`LinearOperator`s in that package's sense, and a DC-centred reflection is MRI content, not a
generic sliding-window embedding. What they do share with `Hankel` is the property the Cadzow
prox needs: `LᴴL` is the real diagonal `_loraks_multiplicity`, because the sign pattern above
makes every cross term between the two sides of k-space cancel
(`‖S(k)‖_F² = 2 Σ (|k(ν - p)|² + |k(-ν - p)|²)`). The multiplicity-weighted adjoint is therefore
an exact left inverse, exactly as for the unweighted C matrix.
"""
struct LoraksLift{N}
    gridsize::NTuple{N, Int}
    ksize::NTuple{N, Int}
    nwin::NTuple{N, Int}
    nchannels::Int
    center::NTuple{N, Int}
end

function LoraksLift(
        gridsize::NTuple{N, Int}, ksize::NTuple{N, Int}, nchannels::Int, structure::Symbol;
        center::Union{Nothing, NTuple{N, Int}} = nothing,
    ) where {N}
    @argcheck structure in (:s, :g) "LoraksLift: structure must be :s or :g, got :$structure"
    @argcheck all(1 .<= ksize .<= gridsize) "LoraksLift: the window $ksize does not fit the grid $gridsize"
    @argcheck nchannels >= 1 "LoraksLift: nchannels must be positive"
    c = center === nothing ? gridsize .÷ 2 .+ 1 : center
    @argcheck all(1 .<= c .<= gridsize) "the k-space centre $c is outside the grid $gridsize"
    return LoraksLift{N}(gridsize, ksize, gridsize .- ksize .+ 1, nchannels, c)
end

_loraks_rows(lift::LoraksLift) = CartesianIndices(lift.nwin)
_loraks_nrows(lift::LoraksLift) = prod(lift.nwin)

# The neighbourhood blocks carry `prod(ksize)` columns per channel in both lifts; `:g` prepends
# one reflected-DC column per channel.
_loraks_nblock(lift::LoraksLift) = prod(lift.ksize) * lift.nchannels
_loraks_ncols(lift::LoraksLift, ::Val{:s}) = 2 * _loraks_nblock(lift)
_loraks_ncols(lift::LoraksLift, ::Val{:g}) = 2 * _loraks_nblock(lift) + lift.nchannels

_loraks_matrix_size(lift::LoraksLift, form::Val) =
    (2 * _loraks_nrows(lift), _loraks_ncols(lift, form))

# `-ν (mod N)`: the index holding the conjugate-symmetry partner of `idx` about DC.
@inline function _loraks_reflect(lift::LoraksLift{N}, idx::CartesianIndex{N}) where {N}
    return CartesianIndex(mod1.(2 .* lift.center .- Tuple(idx), lift.gridsize))
end

# Lifts ------------------------------------------------------------------------------------

function _loraks_lift!(M::AbstractMatrix, lift::LoraksLift{N}, x::AbstractArray, ::Val{:s}) where {N}
    rows = _loraks_rows(lift)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(lift.ksize)
    nrow = length(rows)
    nblock = _loraks_nblock(lift)
    @inbounds for c in 1:lift.nchannels
        xc = selectdim(x, N + 1, c)
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col = coloff + jk
            jw = 0
            for wi in rows
                jw += 1
                plus = wi + ko - shift
                zp = xc[plus]
                zm = xc[_loraks_reflect(lift, plus - 2 * (ko - shift))]
                rp, ip = real(zp), imag(zp)
                rm, im_ = real(zm), imag(zm)
                M[jw, col] = rp - rm
                M[jw, col + nblock] = im_ - ip
                M[jw + nrow, col] = ip + im_
                M[jw + nrow, col + nblock] = rp + rm
            end
        end
    end
    return M
end

function _loraks_lift!(M::AbstractMatrix, lift::LoraksLift{N}, x::AbstractArray, ::Val{:g}) where {N}
    rows = _loraks_rows(lift)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(lift.ksize)
    nrow = length(rows)
    nch = lift.nchannels
    nblock = _loraks_nblock(lift)
    @inbounds for c in 1:nch
        xc = selectdim(x, N + 1, c)
        # the reflected-DC column: one sample per row, not one neighbourhood
        jw = 0
        for wi in rows
            jw += 1
            z = xc[_loraks_reflect(lift, wi)]
            M[jw, c] = -real(z)
            M[jw + nrow, c] = imag(z)
        end
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col_r = nch + coloff + jk
            col_i = nch + nblock + coloff + jk
            jw = 0
            for wi in rows
                jw += 1
                z = xc[wi + ko - shift]
                rz, iz = real(z), imag(z)
                M[jw, col_r] = rz
                M[jw, col_i] = -iz
                M[jw + nrow, col_r] = iz
                M[jw + nrow, col_i] = rz
            end
        end
    end
    return M
end

# Adjoints ---------------------------------------------------------------------------------
#
# The real adjoint of the lift, read off the sign pattern: with the four S-matrix blocks named
# `d₁₁ … d₂₂`, the contribution to the sample at `ν - p` is `(d₁₁ + d₂₂) + i(d₂₁ - d₁₂)` and the
# contribution to its reflection `-ν - p` is `(d₂₂ - d₁₁) + i(d₁₂ + d₂₁)`.

function _loraks_unlift!(y::AbstractArray, lift::LoraksLift{N}, M::AbstractMatrix, ::Val{:s}) where {N}
    T = eltype(y)
    fill!(y, zero(T))
    rows = _loraks_rows(lift)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(lift.ksize)
    nrow = length(rows)
    nblock = _loraks_nblock(lift)
    @inbounds for c in 1:lift.nchannels
        yc = selectdim(y, N + 1, c)
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col = coloff + jk
            jw = 0
            for wi in rows
                jw += 1
                d11 = M[jw, col]
                d12 = M[jw, col + nblock]
                d21 = M[jw + nrow, col]
                d22 = M[jw + nrow, col + nblock]
                plus = wi + ko - shift
                yc[plus] += T(d11 + d22, d21 - d12)
                yc[_loraks_reflect(lift, plus - 2 * (ko - shift))] += T(d22 - d11, d12 + d21)
            end
        end
    end
    return y
end

function _loraks_unlift!(y::AbstractArray, lift::LoraksLift{N}, M::AbstractMatrix, ::Val{:g}) where {N}
    T = eltype(y)
    fill!(y, zero(T))
    rows = _loraks_rows(lift)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    prodk = prod(lift.ksize)
    nrow = length(rows)
    nch = lift.nchannels
    nblock = _loraks_nblock(lift)
    @inbounds for c in 1:nch
        yc = selectdim(y, N + 1, c)
        jw = 0
        for wi in rows
            jw += 1
            yc[_loraks_reflect(lift, wi)] += T(-M[jw, c], M[jw + nrow, c])
        end
        coloff = (c - 1) * prodk
        jk = 0
        for ko in koff
            jk += 1
            col_r = nch + coloff + jk
            col_i = nch + nblock + coloff + jk
            jw = 0
            for wi in rows
                jw += 1
                yc[wi + ko - shift] += T(
                    M[jw, col_r] + M[jw + nrow, col_i],
                    M[jw + nrow, col_r] - M[jw, col_i],
                )
            end
        end
    end
    return y
end

# Multiplicity -----------------------------------------------------------------------------
#
# `diag(LᴴL)`: the number of squared copies of each sample the lift holds. The S matrix carries
# every neighbourhood sample twice on each side of k-space; the G matrix carries the
# neighbourhood twice and the reflected-DC sample once. Every sample is reached by at least one
# window, so the counts are strictly positive and the weighted adjoint never has to guard against
# a zero. `test_reg_structured_low_rank.jl` pins `unlift(lift(x)) == mult .* x` against them.

function _loraks_multiplicity(::Type{R}, lift::LoraksLift{N}, ::Val{:s}) where {R, N}
    m = zeros(R, lift.gridsize..., lift.nchannels)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    @inbounds for c in 1:lift.nchannels
        mc = selectdim(m, N + 1, c)
        for ko in koff, wi in _loraks_rows(lift)
            plus = wi + ko - shift
            mc[plus] += 2
            mc[_loraks_reflect(lift, plus - 2 * (ko - shift))] += 2
        end
    end
    return m
end

function _loraks_multiplicity(::Type{R}, lift::LoraksLift{N}, ::Val{:g}) where {R, N}
    m = zeros(R, lift.gridsize..., lift.nchannels)
    koff = CartesianIndices(lift.ksize)
    shift = oneunit(eltype(koff))
    @inbounds for c in 1:lift.nchannels
        mc = selectdim(m, N + 1, c)
        for wi in _loraks_rows(lift)
            mc[_loraks_reflect(lift, wi)] += 1
        end
        for ko in koff, wi in _loraks_rows(lift)
            mc[wi + ko - shift] += 2
        end
    end
    return m
end

"""
	LoraksLowRankProx(λ, max_rank, lift, invmult, nbatch, threaded)

Proximal function behind `StructuredLowRank(; structure = :s)` and `:g`, the phase-constrained
LORAKS matrices. It is the same Cadzow step as [`HankelLowRankProx`](@ref) — lift the slab,
low-rank the lifted matrix, de-lift by the multiplicity-weighted adjoint — with
[`LoraksLift`](@ref) in place of the block-Hankel lift and a **real** SVD, because both LORAKS
phase matrices are real by construction.

`STRUCT` is `:s` or `:g`; `RANK` selects the form, `true` the hard rank constraint (a non-convex
indicator), `false` the nuclear-norm penalty.
"""
struct LoraksLowRankProx{RANK, STRUCT, R <: Real, L <: LoraksLift, A <: AbstractArray}
    λ::R
    max_rank::Int
    lift::L
    invmult::A          # 1 ./ diag(LᴴL)
    nbatch::Int
    threaded::Bool
end

ProximalCore.is_convex(::Type{<:LoraksLowRankProx{false}}) = true
ProximalCore.is_convex(::Type{<:LoraksLowRankProx{true}}) = false
ProximalCore.is_positively_homogeneous(::Type{<:LoraksLowRankProx}) = true
ProximalCore.is_smooth(::Type{<:LoraksLowRankProx}) = false
ProximalCore.is_separable(::Type{<:LoraksLowRankProx}) = false

_llrp_slab_size(f::LoraksLowRankProx) = (f.lift.gridsize..., f.lift.nchannels)
_llrp_form(::LoraksLowRankProx{RANK, STRUCT}) where {RANK, STRUCT} = Val(STRUCT)

function _llrp_matrix(f::LoraksLowRankProx, xb::AbstractArray)
    form = _llrp_form(f)
    M = Array{real(eltype(xb))}(undef, _loraks_matrix_size(f.lift, form)...)
    return _loraks_lift!(M, f.lift, xb, form)
end

function (f::LoraksLowRankProx{RANK})(x) where {RANK}
    xr = reshape(x, _llrp_slab_size(f)..., f.nbatch)
    R = real(eltype(x))
    value = R(0)
    for b in 1:f.nbatch
        σ = svdvals!(_llrp_matrix(f, selectdim(xr, ndims(xr), b)))
        if RANK
            # Indicator of {k : rank(L k) ≤ max_rank}, with the same yardstick as
            # `HankelLowRankProx`: the lifted matrix is normally infeasible after a Cadzow
            # step, which is what makes the constraint form a heuristic.
            length(σ) <= f.max_rank && continue
            σ[f.max_rank + 1] / σ[1] <= R(_HANKEL_RANK_RTOL) || return R(Inf)
        else
            value += sum(σ)
        end
    end
    return f.λ * value
end

function ProximalCore.prox!(y, f::LoraksLowRankProx{RANK}, x, gamma) where {RANK}
    slab = _llrp_slab_size(f)
    xr = reshape(x, slab..., f.nbatch)
    yr = reshape(y, slab..., f.nbatch)
    R = real(eltype(x))
    threshold = RANK ? R(0) : R(f.λ * gamma)
    partial = zeros(R, f.nbatch)
    if f.threaded
        @budgeted_threads for b in 1:f.nbatch
            partial[b] = _llrp_prox_slab!(yr, xr, f, b, threshold, Val(RANK))
        end
    else
        for b in 1:f.nbatch
            partial[b] = _llrp_prox_slab!(yr, xr, f, b, threshold, Val(RANK))
        end
    end
    # The constraint form is an indicator: its value at the projected point is 0.
    return RANK ? R(0) : f.λ * sum(partial)
end

function _llrp_prox_slab!(yr, xr, f::LoraksLowRankProx, b::Int, threshold, ::Val{RANK}) where {RANK}
    R = real(eltype(xr))
    xb = selectdim(xr, ndims(xr), b)
    M = _llrp_matrix(f, xb)
    F = ProximalOperators.with_factorization_threads(() -> svd!(M), M)
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
    _loraks_unlift!(yb, f.lift, M, _llrp_form(f))
    @. yb *= f.invmult
    return nucval
end

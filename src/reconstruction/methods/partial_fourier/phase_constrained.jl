"""
    PhaseConstrained <: AbstractDirectMethod

Phase-constrained reconstruction for partial Fourier MRI (Margosian et al. 1986).
"""
struct PhaseConstrained{C <: CoilCombination} <: AbstractDirectMethod
    coil_combination::C
    function PhaseConstrained(; coil_combination::CoilCombination = AdjointSensitivity())
        return new{typeof(coil_combination)}(coil_combination)
    end
end

# Phase-constrained partial-Fourier reconstruction (Margosian et al. 1986): estimate the
# low-resolution phase φ from the symmetric centre, then solve the real-valued least-squares
# problem  min_{m ∈ ℝ}  Σ_c ‖ 𝒫 ℱ ( s_c · e^{iφ_c} · m ) − y_c ‖²  by conjugate gradient on the
# normal equations. With sensitivity maps a single real image is recovered; without them each
# coil is solved independently and combined by root-sum-of-squares.
function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::PhaseConstrained)
    ksp = _get_full_kspace(acq)
    ℱ = _cartesian_fourier_op(acq, ksp)
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    R = real(eltype(ksp))

    band = partial_fourier_band(acq)
    N = band.total_size
    W_sym = zeros(R, N)
    W_sym[band.symmetric_range] .= one(R)
    w_shape = ntuple(i -> i == band.dim ? N : 1, ndims(ksp))
    eiϕ = cis.(angle.(_direct_ifft(ℱ, ksp .* reshape(W_sym, w_shape))))

    mask = to_displayable_mask(acq.subsampling, spatial_sz)
    mask_nd = reshape(mask, size(mask)..., ntuple(_ -> 1, ndims(ksp) - 2)...)

    c_dim = _pf_coil_dim(acq)
    has_sens = !isnothing(acq.sensitivity_maps)
    s = has_sens ? unname(acq.sensitivity_maps) : nothing
    scale = sqrt(prod(spatial_sz))

    fwd = m -> begin
        coilwise = has_sens ? (s .* eiϕ .* m) : (eiϕ .* m)
        mask_nd .* (_direct_fft(ℱ, coilwise) ./ scale)
    end
    adj = r -> begin
        img = _direct_ifft(ℱ, mask_nd .* r) .* scale
        img = has_sens ? sum(conj.(s) .* conj.(eiϕ) .* img; dims = c_dim) : (conj.(eiϕ) .* img)
        real.(img)
    end

    b = adj(ksp)
    m = zero(b)
    rk = b - adj(fwd(m))
    p = copy(rk)
    rs_old = sum(abs2, rk)
    for _ in 1:25
        Ap = adj(fwd(p))
        α = rs_old / max(real(sum(conj.(p) .* Ap)), eps(R))
        m = m .+ α .* p
        rk = rk .- α .* Ap
        rs_new = sum(abs2, rk)
        rs_new < 1.0e-12 * length(b) && break
        p = rk .+ (rs_new / rs_old) .* p
        rs_old = rs_new
    end

    if c_dim == 0
        return _pf_finalize(acq, complex.(m), false, 0)
    end
    # `m` still carries a singleton coil axis at `c_dim`; `_pf_finalize` drops it.
    img_out = has_sens ? complex.(m) : complex.(sqrt.(sum(abs2, m; dims = c_dim)))
    return _pf_finalize(acq, img_out, true, c_dim)
end

"""
    PartialFourierFilter

Abstract type representing k-space weighting filters for Homodyne reconstruction.
"""
abstract type PartialFourierFilter end

"""
    LinearRamp <: PartialFourierFilter

Smooth linear transition ramp across the symmetric partial Fourier overlap band.
"""
struct LinearRamp <: PartialFourierFilter end

"""
    StepRamp <: PartialFourierFilter

Step transition filter (hard cutoff) across the partial Fourier transition boundary.
"""
struct StepRamp <: PartialFourierFilter end

"""
    partial_fourier_band(acq::CartesianAcquisitionInfo)

Analyzes the subsampling mask of a Cartesian acquisition to identify the asymmetric
partial Fourier acquisition dimension and returns a named tuple:
`(dim, symmetric_range, acquired_range, total_size)`.
"""
function partial_fourier_band(acq::CartesianAcquisitionInfo)
    @argcheck !isnothing(acq.subsampling) "Acquisition has no subsampling mask"
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    mask = to_displayable_mask(acq.subsampling, spatial_sz)

    # Check for asymmetry along dimension 1 (kx) or dimension 2 (ky)
    proj_y = vec(any(mask; dims = 1))
    proj_x = vec(any(mask; dims = 2))

    y_acq = findall(proj_y)
    x_acq = findall(proj_x)

    dim, acq_idx, N = if length(y_acq) < spatial_sz[2] && length(x_acq) == spatial_sz[1]
        (2, y_acq, spatial_sz[2])
    elseif length(x_acq) < spatial_sz[1] && length(y_acq) == spatial_sz[2]
        (1, x_acq, spatial_sz[1])
    else
        (2, y_acq, spatial_sz[2])
    end

    k_min = first(acq_idx)
    k_max = last(acq_idx)
    k_center = N ÷ 2 + 1

    half_band = min(k_center - k_min, k_max - k_center)
    sym_range = (k_center - half_band):(k_center + half_band)

    return (
        dim = dim,
        symmetric_range = sym_range,
        acquired_range = k_min:k_max,
        total_size = N,
    )
end

"""
    Homodyne{F <: PartialFourierFilter, C <: CoilCombination} <: AbstractDirectMethod

Direct Homodyne reconstruction (Noll et al. 1991) for partial Fourier acquisitions.
Weights k-space with an asymmetric Homodyne filter and restores low-frequency phase.

# Fields
- `filter`: Filter profile across the symmetric band (`LinearRamp()` or `StepRamp()`).
- `coil_combination`: Coil combination method (`AdjointSensitivity()` or `RootSumSquares()`).
"""
struct Homodyne{F <: PartialFourierFilter, C <: CoilCombination} <: AbstractDirectMethod
    filter::F
    coil_combination::C
    function Homodyne(;
            filter::PartialFourierFilter = LinearRamp(),
            coil_combination::CoilCombination = AdjointSensitivity(),
        )
        return new{typeof(filter), typeof(coil_combination)}(filter, coil_combination)
    end
end

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

"""
    POCS <: AbstractDirectMethod

Projection Onto Convex Sets (Haacke et al. 1991) for partial Fourier image reconstruction.
Alternates between data consistency in acquired k-space and phase consistency in image space.
"""
struct POCS{C <: CoilCombination} <: AbstractDirectMethod
    maxit::Int
    tol::Float64
    coil_combination::C
    function POCS(;
            maxit::Int = 20,
            tol::Real = 1.0e-4,
            coil_combination::CoilCombination = AdjointSensitivity(),
        )
        return new{typeof(coil_combination)}(maxit, Float64(tol), coil_combination)
    end
end

"""
    _pf_coil_dim(acq)

Integer position of the coil axis in the (full) k-space array, or `0` when there is none.
Resolved from dimension names when the k-space is a `NamedDimsArray`, else assumed to be
axis 3 for arrays with a third dimension.
"""
function _pf_coil_dim(acq::CartesianAcquisitionInfo)
    if acq.kspace_data isa NamedDimsArray
        idx = findfirst(==(:coil), dimnames(acq.kspace_data))
        return isnothing(idx) ? 0 : Int(idx)
    end
    return ndims(acq.kspace_data) >= 3 ? 3 : 0
end

function _pf_finalize(acq::CartesianAcquisitionInfo, img_out, coil_reduced::Bool, c_dim::Int)
    if coil_reduced && c_dim > 0
        img_out = dropdims(img_out; dims = c_dim)
    end
    if acq.kspace_data isa NamedDimsArray
        img_dims = get_image_dims(acq)
        out_dims = coil_reduced ? filter(!=(:coil), img_dims) : img_dims
        return NamedDimsArray{out_dims}(unname(img_out))
    end
    return img_out
end

function _get_full_kspace(acq::CartesianAcquisitionInfo)
    raw_ksp = unname(acq.kspace_data)
    if isnothing(acq.subsampling)
        return copy(raw_ksp)
    end
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    mask = to_displayable_mask(acq.subsampling, spatial_sz)

    trailing_dims = size(raw_ksp)[3:end]
    full_ksp = zeros(eltype(raw_ksp), spatial_sz..., trailing_dims...)

    if size(raw_ksp, 1) == count(mask)
        full_ksp[mask, :] = reshape(raw_ksp, count(mask), :)
    elseif ndims(raw_ksp) >= 2 && size(raw_ksp, 2) == count(any(mask; dims = 1)) && size(raw_ksp, 1) == spatial_sz[1]
        acq_y = findall(vec(any(mask; dims = 1)))
        full_ksp[:, acq_y, :] .= reshape(raw_ksp, spatial_sz[1], length(acq_y), :)
    else
        full_ksp = copy(raw_ksp)
    end
    return full_ksp
end

# Implement direct reconstruction methods for Homodyne, PhaseConstrained, and POCS

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::Homodyne)
    ksp = _get_full_kspace(acq)
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])

    band = partial_fourier_band(acq)
    dim = band.dim
    sym_range = band.symmetric_range
    acq_range = band.acquired_range
    N = band.total_size

    # Build 1D Homodyne weight vector
    W_1d = zeros(Float32, N)
    if method.filter isa LinearRamp
        for k in 1:N
            if k in sym_range
                frac = Float32(k - first(sym_range)) / Float32(length(sym_range) - 1)
                W_1d[k] = 2.0f0 * (first(acq_range) == 1 ? (1.0f0 - frac) : frac)
            elseif k in acq_range
                W_1d[k] = 2.0f0
            else
                W_1d[k] = 0.0f0
            end
        end
    else
        for k in 1:N
            if k in sym_range
                W_1d[k] = 1.0f0
            elseif k in acq_range
                W_1d[k] = 2.0f0
            else
                W_1d[k] = 0.0f0
            end
        end
    end

    # Low-pass filter for phase estimate
    W_sym = zeros(Float32, N)
    W_sym[sym_range] .= 1.0f0

    w_shape = ntuple(i -> i == dim ? N : 1, ndims(ksp))
    W_mat = reshape(W_1d, w_shape)
    W_sym_mat = reshape(W_sym, w_shape)

    c_dim = _pf_coil_dim(acq)
    coil_reduced = !isnothing(acq.sensitivity_maps)

    # 1. Estimate phase
    ksp_sym = ksp .* W_sym_mat
    f_dims = (1, 2)
    lowres_coil = ifft(ifftshift(ksp_sym, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    lowres_combined = if coil_reduced
        sens = unname(acq.sensitivity_maps)
        sum(lowres_coil .* conj.(sens); dims = c_dim)
    else
        lowres_coil
    end
    phase_est = angle.(lowres_combined)

    # 2. Homodyne weighted inverse FFT
    ksp_hom = ksp .* W_mat
    img_coil = ifft(ifftshift(ksp_hom, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    img_combined = if coil_reduced
        sens = unname(acq.sensitivity_maps)
        sum(img_coil .* conj.(sens); dims = c_dim)
    else
        img_coil
    end

    # 3. Demodulate phase and take real part
    img_out = complex.(real.(img_combined .* cis.(-phase_est)))

    return _pf_finalize(acq, img_out, coil_reduced, c_dim)
end

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::POCS)
    ksp = _get_full_kspace(acq)
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    f_dims = (1, 2)

    band = partial_fourier_band(acq)
    dim = band.dim
    sym_range = band.symmetric_range
    N = band.total_size

    W_sym = zeros(Float32, N)
    W_sym[sym_range] .= 1.0f0
    w_shape = ntuple(i -> i == dim ? N : 1, ndims(ksp))
    ksp_sym = ksp .* reshape(W_sym, w_shape)

    # Initial phase estimate from symmetric ACS
    lowres_coil = ifft(ifftshift(ksp_sym, f_dims), f_dims)
    phase_est = angle.(lowres_coil)

    # POCS iteration on multi-coil k-space
    ksp_pocs = copy(ksp)
    mask = to_displayable_mask(acq.subsampling, spatial_sz)
    mask_nd = reshape(mask, size(mask)..., fill(1, ndims(ksp) - 2)...)

    for iter in 1:(method.maxit)
        img_coil = ifft(ifftshift(ksp_pocs, f_dims), f_dims)
        img_constrained = abs.(img_coil) .* cis.(phase_est)
        ksp_updated = fftshift(fft(img_constrained, f_dims), f_dims)
        ksp_pocs = ifelse.(mask_nd, ksp, ksp_updated)
    end

    c_dim = _pf_coil_dim(acq)
    coil_reduced = !isnothing(acq.sensitivity_maps)
    final_coil_imgs = ifft(ifftshift(ksp_pocs, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    img_out = if coil_reduced
        sens = unname(acq.sensitivity_maps)
        sum(final_coil_imgs .* conj.(sens); dims = c_dim)
    else
        final_coil_imgs
    end

    return _pf_finalize(acq, img_out, coil_reduced, c_dim)
end

# Phase-constrained partial-Fourier reconstruction (Margosian et al. 1986): estimate the
# low-resolution phase φ from the symmetric centre, then solve the real-valued least-squares
# problem  min_{m ∈ ℝ}  Σ_c ‖ 𝒫 ℱ ( s_c · e^{iφ_c} · m ) − y_c ‖²  by conjugate gradient on the
# normal equations. With sensitivity maps a single real image is recovered; without them each
# coil is solved independently and combined by root-sum-of-squares.
function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::PhaseConstrained)
    ksp = _get_full_kspace(acq)
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    f_dims = (1, 2)
    R = real(eltype(ksp))

    band = partial_fourier_band(acq)
    N = band.total_size
    W_sym = zeros(R, N)
    W_sym[band.symmetric_range] .= one(R)
    w_shape = ntuple(i -> i == band.dim ? N : 1, ndims(ksp))
    eiϕ = cis.(angle.(ifft(ifftshift(ksp .* reshape(W_sym, w_shape), f_dims), f_dims)))

    mask = to_displayable_mask(acq.subsampling, spatial_sz)
    mask_nd = reshape(mask, size(mask)..., ntuple(_ -> 1, ndims(ksp) - 2)...)

    c_dim = _pf_coil_dim(acq)
    has_sens = !isnothing(acq.sensitivity_maps)
    s = has_sens ? unname(acq.sensitivity_maps) : nothing
    scale = sqrt(prod(spatial_sz))

    fwd = m -> begin
        coilwise = has_sens ? (s .* eiϕ .* m) : (eiϕ .* m)
        mask_nd .* (fftshift(fft(coilwise, f_dims), f_dims) ./ scale)
    end
    adj = r -> begin
        img = ifft(ifftshift(mask_nd .* r, f_dims), f_dims) .* scale
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

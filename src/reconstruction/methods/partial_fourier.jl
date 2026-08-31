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

    # 1. Estimate phase
    ksp_sym = ksp .* W_sym_mat
    f_dims = (1, 2)
    lowres_coil = ifft(ifftshift(ksp_sym, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    lowres_combined = if !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        c_dim = ndims(ksp) >= 3 ? 3 : 1
        sum(lowres_coil .* conj.(sens); dims = c_dim)
    else
        lowres_coil
    end
    phase_est = angle.(lowres_combined)

    # 2. Homodyne weighted inverse FFT
    ksp_hom = ksp .* W_mat
    img_coil = ifft(ifftshift(ksp_hom, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    img_combined = if !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        c_dim = ndims(ksp) >= 3 ? 3 : 1
        sum(img_coil .* conj.(sens); dims = c_dim)
    else
        img_coil
    end

    # 3. Demodulate phase and take real part
    img_out = real.(img_combined .* cis.(-phase_est))
    img_out = complex.(img_out)

    if acq.kspace_data isa NamedDimsArray
        img_dims = get_image_dims(acq)
        out_dims = isnothing(acq.sensitivity_maps) ? img_dims : filter(!=(:coil), img_dims)
        out_arr = reshape(img_out, ntuple(i -> size(img_out, i), length(out_dims)))
        return NamedDimsArray{out_dims}(out_arr)
    else
        return img_out
    end
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

    final_coil_imgs = ifft(ifftshift(ksp_pocs, f_dims), f_dims) .* sqrt(prod(spatial_sz))
    img_out = if !isnothing(acq.sensitivity_maps)
        sens = unname(acq.sensitivity_maps)
        c_dim = ndims(ksp) >= 3 ? 3 : 1
        sum(final_coil_imgs .* conj.(sens); dims = c_dim)
    else
        final_coil_imgs
    end

    if acq.kspace_data isa NamedDimsArray
        img_dims = get_image_dims(acq)
        out_dims = isnothing(acq.sensitivity_maps) ? img_dims : filter(!=(:coil), img_dims)
        out_arr = reshape(img_out, ntuple(i -> size(img_out, i), length(out_dims)))
        return NamedDimsArray{out_dims}(out_arr)
    else
        return img_out
    end
end

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

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::Homodyne; progress = nothing)
    ksp = _get_full_kspace(acq)
    ℱ = _cartesian_fourier_op(acq, ksp)
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
    lowres_coil = _direct_ifft(ℱ, ksp_sym) .* sqrt(prod(spatial_sz))
    lowres_combined = if coil_reduced
        sens = unname(acq.sensitivity_maps)
        sum(lowres_coil .* conj.(sens); dims = c_dim)
    else
        lowres_coil
    end
    phase_est = angle.(lowres_combined)

    # 2. Homodyne weighted inverse FFT
    ksp_hom = ksp .* W_mat
    img_coil = _direct_ifft(ℱ, ksp_hom) .* sqrt(prod(spatial_sz))
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

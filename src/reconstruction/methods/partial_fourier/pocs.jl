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

function _direct_reconstruct(acq::CartesianAcquisitionInfo, method::POCS)
    ksp = _get_full_kspace(acq)
    ℱ = _cartesian_fourier_op(acq, ksp)
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])

    band = partial_fourier_band(acq)
    dim = band.dim
    sym_range = band.symmetric_range
    N = band.total_size

    W_sym = zeros(Float32, N)
    W_sym[sym_range] .= 1.0f0
    w_shape = ntuple(i -> i == dim ? N : 1, ndims(ksp))
    ksp_sym = ksp .* reshape(W_sym, w_shape)

    # Initial phase estimate from symmetric ACS
    lowres_coil = _direct_ifft(ℱ, ksp_sym)
    phase_est = angle.(lowres_coil)

    # POCS iteration on multi-coil k-space
    ksp_pocs = copy(ksp)
    mask = to_displayable_mask(acq.subsampling, spatial_sz)
    mask_nd = reshape(mask, size(mask)..., fill(1, ndims(ksp) - 2)...)

    for iter in 1:(method.maxit)
        img_coil = _direct_ifft(ℱ, ksp_pocs)
        img_constrained = abs.(img_coil) .* cis.(phase_est)
        ksp_updated = _direct_fft(ℱ, img_constrained)
        ksp_pocs = ifelse.(mask_nd, ksp, ksp_updated)
    end

    c_dim = _pf_coil_dim(acq)
    coil_reduced = !isnothing(acq.sensitivity_maps)
    final_coil_imgs = _direct_ifft(ℱ, ksp_pocs) .* sqrt(prod(spatial_sz))
    img_out = if coil_reduced
        sens = unname(acq.sensitivity_maps)
        sum(final_coil_imgs .* conj.(sens); dims = c_dim)
    else
        final_coil_imgs
    end

    return _pf_finalize(acq, img_out, coil_reduced, c_dim)
end

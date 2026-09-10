function _direct_reconstruct_components(𝒜, acq_data, method::ReconstructionMethod, config; scale_override=nothing)
    @step "Getting initial estimate" config begin
        x̂ = 𝒜' * _measurement(acq_data.kspace_data)
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded=config.threaded)' * _measurement(acq_data.kspace_data)
    else
        x̂
    end
    scale = _resolve_scale(acq_data, scale_input, config, scale_override)
    # `𝒜'y` is only on the image's scale when `𝒜'𝒜 ≈ I`; a raw FFT/NFFT is not. `L`, already
    # needed as the algorithm's step-size estimate (below, or by the caller when this warm start
    # feeds a later `_iterative_reconstruct_core` call), makes `x̂/‖𝒜‖²` (one Landweber step) the
    # scale-correct warm start at no extra cost. Computed from the pre-rescale `x̂` so `scale`
    # above stays exactly as before.
    L = _warm_start_needs_operator_norm(method) ? _operator_norm_for_stepsize(𝒜, method, config) : nothing
    isnothing(L) || (x̂ = _scale_x0(x̂, L^2))
    return x̂, scale, L
end

# The scale is either imposed by the caller (task splitting uses one shared scale for every slice),
# derived from the direct estimate, or absent; a zero estimate would blow up the scaled problem, so it
# falls back to no scaling.
function _resolve_scale(acq_data, x̂, config, scale_override)
    if !isnothing(scale_override)
        scale = scale_override
        log_message(config.verbosity, @sprintf("Using scaling factor: %g", scale))
    elseif config.scaling != NoScaling()
        @step "Computing scaling factor" config begin
            scale = get_scale(config.scaling, acq_data, x̂)
        end
        if scale == 0
            log_message(config.verbosity, "Warning: Computed scale is zero, defaulting to scale=1.0")
            scale = 1
        end
        log_message(config.verbosity, @sprintf("Using scaling factor: %g", scale))
    else
        scale = 1
    end
    return real(eltype(x̂))(scale)
end

"""
    _direct_coil_dim(acq::CartesianAcquisitionInfo)

Integer position of the coil axis in `acq.kspace_data`, or `0` when there is none. Resolved from
dimension names when the k-space is a `NamedDimsArray`, else assumed to immediately follow the
spatial dimensions (`3` for 2D, `4` for 3D) -- unlike `_pf_coil_dim`, which hardcodes `3` and is
only ever used by the 2D-only partial-Fourier methods.
"""
function _direct_coil_dim(acq::CartesianAcquisitionInfo)
    if _has_dimnames(acq.kspace_data)
        idx = findfirst(==(:coil), dimnames(acq.kspace_data))
        return isnothing(idx) ? 0 : Int(idx)
    end
    spatial_dims = acq.is3D ? 3 : 2
    return ndims(acq.kspace_data) > spatial_dims ? spatial_dims + 1 : 0
end

"""
    _direct_reconstruct_coil_combined(acq_data, method::DirectReconstruction, 𝒜)

`DirectReconstruction`'s own coil combination: `𝒜' * kspace_data` bakes in `AdjointSensitivity`
combination whenever sensitivity maps are present (via `_compose_with_sensitivity`) and cannot
express `RootSumSquares` or `NoCoilCombination`, so those are dispatched explicitly here, the same
way the shared `_kspace_to_image` helper (used by GRAPPA/SPIRiT/`KSpaceToImage`) does -- but from a
freshly-built sensitivity-free encoding operator rather than assuming `kspace_data` is already a
complete (zero-filled) grid, since `DirectReconstruction` also runs on subsampled data. Only
implemented for Cartesian acquisitions; non-Cartesian acquisitions keep the `𝒜' * kspace_data`
behavior regardless of `coil_combination` (see the other method below).
"""
function _direct_reconstruct_coil_combined(acq_data::CartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    smaps = acq_data.sensitivity_maps
    if isnothing(smaps) && method.coil_combination isa AdjointSensitivity
        # Nothing to combine with under the default combination; preserve the historical
        # behavior of returning the bare per-coil (or single-channel) adjoint image.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end
    c_dim = _direct_coil_dim(acq_data)
    if c_dim == 0
        # No coil axis at all: nothing for any combination choice to do.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end

    # `𝒜` always bakes sensitivity composition in when `smaps` is present
    # (`_compose_with_sensitivity`); rebuild the bare (sensitivity-free) encoding operator so
    # per-coil images stay correctly zero-filled/gridded even for a Cartesian-subsampled
    # acquisition, then dispatch the combination explicitly.
    ℬ = isnothing(smaps) ? 𝒜 : get_encoding_operator(CartesianAcquisitionInfo(acq_data; sensitivity_maps=nothing))
    coil_imgs = unname(ℬ' * _measurement(acq_data.kspace_data))

    img_out, coil_reduced = if method.coil_combination isa AdjointSensitivity
        @argcheck !isnothing(smaps) "AdjointSensitivity coil combination requires sensitivity maps."
        sum(coil_imgs .* conj.(unname(smaps)); dims=c_dim), true
    elseif method.coil_combination isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims=c_dim)), true
    elseif method.coil_combination isa NoCoilCombination
        coil_imgs, false
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(method.coil_combination))"))
    end
    return _pf_finalize(acq_data, img_out, coil_reduced, c_dim)
end
function _direct_reconstruct_coil_combined(acq_data::NonCartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    @argcheck method.coil_combination isa AdjointSensitivity "RootSumSquares/NoCoilCombination for DirectReconstruction is only implemented for Cartesian acquisitions; use AdjointSensitivity (the default)"
    return 𝒜' * _measurement(acq_data.kspace_data)
end

function _direct_reconstruct(𝒜, acq_data, x₀, method::ReconstructionMethod, config; scale_override=nothing)
    direct_recon_only = method isa DirectMethod
    if !isnothing(x₀) && direct_recon_only
        log_message(
            config.verbosity,
            "Warning: Initial guess x₀ is ignored when no regularization is specified.",
        )
        x₀ = nothing
    end
    is_default_iterative_adjoint = false
    if isnothing(x₀)
        @step (direct_recon_only ? "Reconstructing image" : "Getting initial estimate") config begin
            if method isa DirectReconstruction
                x₀ = _direct_reconstruct_coil_combined(acq_data, method, 𝒜)
            elseif !(method isa DirectMethod)
                x₀ = 𝒜' * _measurement(acq_data.kspace_data)
                is_default_iterative_adjoint = true
            else
                x₀ = _direct_reconstruct(
                    acq_data, method; progress=progress_tick(config.verbosity)
                )
            end
        end
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded=config.threaded)' * _measurement(acq_data.kspace_data)
    else
        x₀
    end
    scale = _resolve_scale(acq_data, scale_input, config, scale_override)
    # Same fix as `_direct_reconstruct_components`, restricted to the case that actually produced
    # a fresh default adjoint here (not a caller-supplied x₀, and not a pure direct method's own
    # reconstruction, which is already correctly scaled).
    L = nothing
    if is_default_iterative_adjoint && _warm_start_needs_operator_norm(method)
        L = _operator_norm_for_stepsize(𝒜, method, config)
        x₀ = _scale_x0(x₀, L^2)
    end
    return x₀, scale, L
end

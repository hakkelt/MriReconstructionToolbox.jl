function _direct_reconstruct_components(𝒜, acq_data, method::ReconstructionMethod, config; scale_override = nothing)
    @step "Getting initial estimate" config begin
        x̂ = 𝒜' * _measurement(acq_data.kspace_data)
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded = config.threaded)' * _measurement(acq_data.kspace_data)
    else
        x̂
    end
    scale = _resolve_scale(acq_data, scale_input, config, scale_override)
    # Computed from the pre-rescale `x̂`, so `scale` above stays exactly as before.
    x̂, L = _scale_default_warm_start(𝒜, x̂, method, config)
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
    lower(method::DirectReconstruction, acq::AcquisitionInfo)

Resolve the default (`nothing`) coil combination against the acquisition. Without sensitivity maps
the coil axis is not part of the signal model at all: it is a batch dimension, and the
reconstruction is one independent image per channel, so `NoCoilCombination` is what actually
happens and what the method should say it does. With maps, the default is the SNR-optimal
`AdjointSensitivity`.
"""
function lower(method::DirectReconstruction{Nothing}, acq::AcquisitionInfo)
    combination = if isnothing(acq.sensitivity_maps) && _direct_coil_dim(acq) != 0
        NoCoilCombination()
    else
        AdjointSensitivity()
    end
    return DirectReconstruction(combination)
end

"""
    check_applicable(method::DirectReconstruction, acq::AcquisitionInfo)

An *explicit* `AdjointSensitivity` or `RootSumSquares` on an acquisition without sensitivity maps
is an error: the coil axis is a batch dimension there (see [`lower`](@ref)), so neither combination
ever sees the channels together, and returning the uncombined per-coil images instead would be a
differently shaped result than the same call on the same data *with* maps.
"""
function check_applicable(method::DirectReconstruction, acq::AcquisitionInfo)
    if isnothing(acq.sensitivity_maps) && _direct_coil_dim(acq) != 0 &&
            !(method.coil_combination isa Union{Nothing, NoCoilCombination})
        throw(
            ArgumentError(
                "$(nameof(typeof(method.coil_combination))) coil combination needs sensitivity maps, and this " *
                    "acquisition carries none -- its coil axis is a batch dimension, reconstructed one channel at a " *
                    "time. Drop the argument (or pass NoCoilCombination()) to get the per-channel images, or attach " *
                    "sensitivity maps (see estimate_sensitivities) to combine them."
            )
        )
    end
    return nothing
end

"""
    _direct_reconstruct_coil_combined(acq_data, method::DirectReconstruction, 𝒜)

`DirectReconstruction`'s own coil combination: `𝒜' * kspace_data` bakes in `AdjointSensitivity`
combination whenever sensitivity maps are present (via `_compose_with_sensitivity`) and cannot
express `RootSumSquares` or `NoCoilCombination`, so those are dispatched explicitly here, the same
way the shared `_kspace_to_image` helper (used by GRAPPA/SPIRiT/`KSpaceToImage`) does -- but from a
freshly-built sensitivity-free encoding operator rather than assuming `kspace_data` is already a
complete (zero-filled) grid, since `DirectReconstruction` also runs on subsampled data. The
non-Cartesian method below does the same from the gridding adjoint.
"""
function _direct_reconstruct_coil_combined(acq_data::CartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    smaps = acq_data.sensitivity_maps
    if isnothing(smaps) && method.coil_combination isa AdjointSensitivity
        # `check_applicable` has already rejected this at the top level whenever the acquisition
        # has a coil axis; what reaches here is single-channel data, where the default combination
        # is the bare adjoint.
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
    ℬ = isnothing(smaps) ? 𝒜 : get_encoding_operator(CartesianAcquisitionInfo(acq_data; sensitivity_maps = nothing))
    coil_imgs = unname(ℬ' * _measurement(acq_data.kspace_data))

    img_out, coil_reduced = if method.coil_combination isa AdjointSensitivity
        @argcheck !isnothing(smaps) "AdjointSensitivity coil combination requires sensitivity maps."
        sum(coil_imgs .* conj.(unname(smaps)); dims = c_dim), true
    elseif method.coil_combination isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims = c_dim)), true
    elseif method.coil_combination isa NoCoilCombination
        coil_imgs, false
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(method.coil_combination))"))
    end
    return _pf_finalize(acq_data, img_out, coil_reduced, c_dim)
end
"""
    _direct_coil_dim(acq::NonCartesianAcquisitionInfo)

Integer position of the coil axis in the *image* a sensitivity-free gridding adjoint produces, or
`0` when the acquisition is single-channel. The non-Cartesian adjoint maps samples to the image
grid, so the coil axis sits directly after the spatial dimensions rather than wherever it lives in
the (sample-indexed) k-space array.
"""
function _direct_coil_dim(acq::NonCartesianAcquisitionInfo)
    isnothing(acq.kspace_data) && return 0
    if _has_dimnames(acq.kspace_data)
        :coil ∈ dimnames(acq.kspace_data) || return 0
    else
        sample_dims = ndims(acq.trajectory) - 1
        ndims(acq.kspace_data) > sample_dims || return 0
    end
    return length(acq.image_size) + 1
end

function _direct_reconstruct_coil_combined(acq_data::NonCartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    smaps = acq_data.sensitivity_maps
    c_dim = _direct_coil_dim(acq_data)
    if c_dim == 0 || (method.coil_combination isa AdjointSensitivity)
        # Single-channel data, or the combination `𝒜'` already performs: nothing extra to do.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end
    # As in the Cartesian case: `𝒜` composes the sensitivity operator in whenever maps are
    # present, so the per-coil gridded images need a freshly built, sensitivity-free operator.
    ℬ = isnothing(smaps) ? 𝒜 : get_encoding_operator(AcquisitionInfo(acq_data; sensitivity_maps = nothing))
    coil_imgs = unname(ℬ' * _measurement(acq_data.kspace_data))
    img_out, coil_reduced = if method.coil_combination isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims = c_dim)), true
    elseif method.coil_combination isa NoCoilCombination
        coil_imgs, false
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(method.coil_combination))"))
    end
    return _pf_finalize(acq_data, img_out, coil_reduced, c_dim)
end

function _direct_reconstruct(𝒜, acq_data, x₀, method::ReconstructionMethod, config; scale_override = nothing)
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
                    acq_data, method; progress = progress_tick(config.verbosity)
                )
            end
        end
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded = config.threaded)' * _measurement(acq_data.kspace_data)
    else
        x₀
    end
    scale = _resolve_scale(acq_data, scale_input, config, scale_override)
    # Only the case that actually produced a fresh default adjoint here is rescaled: not a
    # caller-supplied x₀, and not a pure direct method's own reconstruction, which is already
    # correctly scaled.
    x₀, L = is_default_iterative_adjoint ?
        _scale_default_warm_start(𝒜, x₀, method, config) : (x₀, nothing)
    return x₀, scale, L
end

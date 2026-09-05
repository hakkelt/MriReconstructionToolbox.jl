function _direct_reconstruct_components(𝒜, acq_data, method::AbstractReconstructionMethod, config; scale_override = nothing)
    @step "Getting initial estimate" config begin
        x̂ = 𝒜' * acq_data.kspace_data
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded = config.threaded)' * acq_data.kspace_data
    else
        x̂
    end
    return x̂, _resolve_scale(acq_data, scale_input, config, scale_override)
end

# The scale is either imposed by the caller (decomposition uses one shared scale for every slice),
# derived from the direct estimate, or absent; a zero estimate would blow up the scaled problem, so it
# falls back to no scaling.
function _resolve_scale(acq_data, x̂, config, scale_override)
    if !isnothing(scale_override)
        scale = scale_override
        log_message(config.verbosity, @sprintf("Using scaling factor: %g", scale))
    elseif config.normalization != NoScaling()
        @step "Computing scaling factor" config begin
            scale = get_scale(config.normalization, acq_data, x̂)
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

function _direct_reconstruct(𝒜, acq_data, x₀, method::AbstractReconstructionMethod, config; scale_override = nothing)
    direct_recon_only = method isa AbstractDirectMethod
    if !isnothing(x₀) && direct_recon_only
        log_message(
            config.verbosity,
            "Warning: Initial guess x₀ is ignored when no regularization is specified.",
        )
        x₀ = nothing
    end
    if isnothing(x₀)
        @step (direct_recon_only ? "Reconstructing image" : "Getting initial estimate") config begin
            if method isa DirectReconstruction || !(method isa AbstractDirectMethod)
                x₀ = 𝒜' * acq_data.kspace_data
            else
                x₀ = _direct_reconstruct(
                    acq_data, method; progress = progress_tick(config.verbosity)
                )
            end
        end
    end
    scale_input = if method isa IterativeReconstruction && method.signal_model !== nothing
        get_encoding_operator(acq_data; threaded = config.threaded)' * acq_data.kspace_data
    else
        x₀
    end
    return x₀, _resolve_scale(acq_data, scale_input, config, scale_override)
end

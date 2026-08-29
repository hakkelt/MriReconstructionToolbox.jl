"""
	reconstruct(
		acq_data::AcquisitionInfo,
		[regularization::Union{Regularization, Tuple{Vararg{Regularization}}}],
		[algorithm];
		kwargs...)

Performs MRI reconstruction from k-space data using the specified regularization, and optimization algorithm.

# Positional arguments
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data, sensitivity maps, and other parameters.
- `regularization::Union{Regularization, Tuple{Vararg{Regularization}}}`: The regularization term(s) to use (default is no regularization).
- `algorithm`: The optimization algorithm(s) to use (default is `(CG(), CGNR(), FISTA(), ADMM())`; the applicable one is selected based on the model).

# Keyword arguments
- `x₀::Union{Nothing,AbstractArray}=nothing`: Optional initial guess for the image (default is 𝒜' * y).
- `normalization::Normalization = BartScaling()`: scaling applied to operators/data (see also `NoScaling`, `MeasurementBasedScaling`, `FixedScaling`)
- `tol::Float64 = 1e-4`: stopping tolerance for iterative algorithms
- `maxit::Int = 100`: maximum iterations for the chosen solver
- `freq::Union{Nothing,Int} = nothing`: progress print frequency (iterations)
- `verbose::Bool = true`: enable/disable logging output
- `threaded::Bool = (Threads.nthreads() > 1)`: enable threaded execution when available
- `exact_opnorm::Bool = false`: use exact operator norm for stepsize estimation
- `decomposition_executor::Union{Nothing,ReconstructionExecutor} = nothing`: override executor for decomposition
- `disable_inverse_scale_output::Bool = false`: skip rescaling the final output
- `disable_normalop_optimization::Bool = false`: disable normal-operator optimizations
- `disable_problem_decomposition::Bool = false`: disable automatic problem decomposition
- `disable_operator_normalization::Bool = false`: disable operator normalization
- `printfunc::Function = println`: custom logging function

# Returns
- The reconstructed image (NamedDimsArray if input is NamedDimsArray, otherwise standard Array).

# Notes

## Normalop Optimization

- If `disable_normalop_optimization` is false, the function uses `normalop_ls` for efficiency when appropriate.
- It should not change the result of the reconstruction, but it changes the reported value of consistency term.
- To understand this, one can think of the optimization problem as minimizing
`|| 𝒜*x - y ||_2^2 + Σ R_i(x)`, where `|| 𝒜*x - y ||_2^2` is the consistency term, `R_i` are the regularization
terms, `𝒜` is the encoding operator, and `y` is the k-space data.
- When normalop optimization is disabled, the reported value of `f(x)` is exactly the consistency term
`|| 𝒜*x - y ||_2^2`. When enabled, it exploits the fact that `∇f(x) = 𝒜'*(𝒜*x - y) = 𝒜'*𝒜*x - 𝒜'*y`,
and usually there exists an optimized operator for `𝒜'*𝒜`. Therefore, it computes `f(x)` as `|| 𝒜'*𝒜*x - 𝒜'*y ||_2^2`, which leads to the same result, but with
potentially improved efficiency.

## Problem Decomposition

- If `disable_problem_decomposition` is false, the function automatically decomposes the reconstruction problem
over batch dimensions of the image that are not affected by the Fourier transform or regularization terms.
- E.g., for a 3D+t acquisition with no regularization, the reconstruction is decomposed over the time dimension,
and each 3D volume is reconstructed independently.
- This can significantly speed up the reconstruction when multiple CPU cores are available. Usually, this leads
to better resource utilization and faster overall reconstruction times, but is useful to disable for debug purposes.

"""
function reconstruct(
        acq_data::AcquisitionInfo,
        regularization::Union{Regularization, Component, Tuple{Vararg{Union{Regularization, Component}}}} = (),
        algorithm = (CG(), CGNR(), FISTA(), ADMM());
        x₀::Union{Nothing, AbstractArray, Tuple, NamedTuple} = nothing,
        kwargs...,
    )
    config = construct_config(kwargs)
    t_start = time()
    regularization = ensure_tuple(regularization)
    x = if !isempty(regularization) && any(r -> r isa Component, regularization)
        @argcheck all(r -> r isa Component, regularization) "Cannot mix bare regularization terms with `Component`s; wrap loose regularization terms in a `Component`."
        components = regularization
        check_components(components)
        _reconstruct_dispatch_components(acq_data, components, algorithm, x₀, config)
    else
        @argcheck isnothing(x₀) || x₀ isa AbstractArray "x₀ must be a plain array unless reconstructing with `Component`s."
        _reconstruct_dispatch(acq_data, regularization, algorithm, x₀, config)
    end
    t_end = time()
    config.verbose && config.printfunc("Total time: ", format_time(t_end - t_start))
    return x
end

function _reconstruct_dispatch(acq_data, regularization, algorithm, x₀, config)
    decomposition_plan = get_problem_decomposition_plan(acq_data, regularization, config)
    x = if isnothing(decomposition_plan)
        reconstruction_result = nothing
        @conditionally_enable_threading config.threaded begin
            reconstruction_result = _reconstruct(acq_data, regularization, algorithm, x₀, config)
        end
        first(reconstruction_result)
    else
        if !isnothing(x₀)
            @argcheck size(x₀) == decomposition_plan.image_size "Size of x₀ ($(size(x₀))) must match the image size ($(decomposition_plan.image_size))"
        end
        result = if regularization == ()
            # Direct reconstruction needs no scaling; keep slices identical to the
            # non-decomposed result instead of normalizing each slice separately.
            config = Config(config; normalization = NoScaling())
            execute(decomposition_plan, acq_data, config) do idx, local_acq, local_conf
                local_x₀ = isnothing(x₀) ? nothing : get_x₀_slice(x₀, decomposition_plan, idx)
                _reconstruct(local_acq, regularization, algorithm, local_x₀, local_conf)
            end
        else
            # Each slice's regularization strength is scale-dependent, so each slice is
            # first solved with its own scale to size λ correctly (via scale_regularization),
            # then the actual solve and the final image use one shared scale across all
            # slices so the output intensities are consistent slice-to-slice.
            execute_regularized(decomposition_plan, acq_data, config, regularization, algorithm, x₀)
        end
        if acq_data.kspace_data isa NamedDimsArray
            result = NamedDimsArray{get_image_dims(acq_data)}(unname(result))
        end
        result
    end
    return x
end

function _reconstruct(acq_data, regularization, algorithm, x₀, config; scale_override = nothing)
    # Construct encoding operator
    @step "Constructing encoding operator" config begin
        𝒜 = get_encoding_operator(
            acq_data; threaded = config.threaded, fast_planning = regularization == ()
        )
    end

    # Direct reconstruction
    x̂, scale = _direct_reconstruct(𝒜, acq_data, x₀, regularization, config; scale_override)

    if regularization == ()
        # No regularization, return direct reconstruction
        if scale != 1 && config.disable_inverse_scale_output
            @step "Scaling image" config begin
                x̂ ./= scale
            end
        end
    else
        # Iterative reconstruction with regularization
        x̂ = _iterative_reconstruct(
            𝒜, acq_data, x̂, scale, regularization, algorithm, config
        )
    end

    return x̂, scale
end

function _reconstruct_dispatch_components(acq_data, components, algorithm, x₀, config)
    decomposition_plan = get_problem_decomposition_plan(acq_data, components, config)
    img = if isnothing(decomposition_plan)
        result = nothing
        @conditionally_enable_threading config.threaded begin
            result = _reconstruct_components(acq_data, components, algorithm, x₀, config)
        end
        first(result)
    else
        if !isnothing(x₀)
            check_x₀_components_size(x₀, components, decomposition_plan.image_size)
        end
        execute_regularized_components(decomposition_plan, acq_data, config, components, algorithm, x₀)
    end
    if acq_data.kspace_data isa NamedDimsArray && !(total(img) isa NamedDimsArray)
        img_dimnames = get_image_dims(acq_data)
        img = DecomposedImage(
            NamedDimsArray{img_dimnames}(unname(total(img))),
            NamedTuple{keys(img.components)}(
                map(c -> NamedDimsArray{img_dimnames}(unname(c)), values(img.components))
            ),
        )
    end
    return img
end

function check_x₀_components_size(x₀, components, image_size)
    x₀ isa Union{Tuple, NamedTuple} ||
        throw(ArgumentError("x₀ for image decomposition must be `nothing`, a Tuple, or a NamedTuple of per-component arrays."))
    if x₀ isa Tuple
        @argcheck length(x₀) == length(components) "x₀ tuple must have one entry per component ($(length(components))), got $(length(x₀))."
    end
    for x in values(x₀)
        @argcheck size(x) == image_size "Size of x₀ ($(size(x))) must match the image size ($image_size)"
    end
    return nothing
end

function _reconstruct_components(acq_data, components, algorithm, x₀, config; scale_override = nothing)
    @step "Constructing encoding operator" config begin
        𝒜 = get_encoding_operator(acq_data; threaded = config.threaded, fast_planning = false)
    end
    x̂, scale = _direct_reconstruct_components(𝒜, acq_data, config; scale_override)
    x₀s = get_component_x0s(components, x̂, x₀)
    img = _iterative_reconstruct_components(𝒜, acq_data, x₀s, scale, components, algorithm, config)
    return img, scale
end

function _direct_reconstruct_components(𝒜, acq_data, config; scale_override = nothing)
    @step "Getting initial estimate" config begin
        x̂ = 𝒜' * acq_data.kspace_data
    end
    if !isnothing(scale_override)
        scale = scale_override
        config.verbose && config.printfunc(@sprintf("Using scaling factor: %g", scale))
    elseif config.normalization != NoScaling()
        @step "Computing scaling factor" config begin
            scale = get_scale(config.normalization, acq_data, x̂)
        end
        if scale == 0
            config.verbose &&
                config.printfunc("Warning: Computed scale is zero, defaulting to scale=1.0")
            scale = 1
        end
        config.verbose && config.printfunc(@sprintf("Using scaling factor: %g", scale))
    else
        scale = 1
    end
    return x̂, real(eltype(x̂))(scale)
end

# Component initialisation: the first component gets the direct-recon estimate x̂
# (standard L+S/RPCA warm start), the rest start at zero. `x₀` (nothing / Tuple /
# NamedTuple of per-component arrays) overrides this default per component.
function get_component_x0s(components, x̂, ::Nothing)
    n = length(components)
    return ntuple(i -> i == 1 ? copy(unname(x̂)) : zero(unname(x̂)), n)
end

function get_component_x0s(components, x̂, x₀::Tuple)
    n = length(components)
    @argcheck length(x₀) == n "x₀ tuple must have one entry per component ($n), got $(length(x₀))."
    return ntuple(n) do i
        @argcheck size(x₀[i]) == size(x̂) "x₀ component size $(size(x₀[i])) must match the image size $(size(x̂))."
        unname(x₀[i])
    end
end

function get_component_x0s(components, x̂, x₀::NamedTuple)
    return ntuple(length(components)) do i
        name = components[i].name
        if haskey(x₀, name)
            x = x₀[name]
            @argcheck size(x) == size(x̂) "x₀ component size $(size(x)) must match the image size $(size(x̂))."
            unname(x)
        elseif i == 1
            copy(unname(x̂))
        else
            zero(unname(x̂))
        end
    end
end

function _iterative_reconstruct_components(𝒜, acq_data, x₀s, scale, components, algorithm, config)
    if scale != 1
        @step "Scaling k-space data" config begin
            acq_data = AcquisitionInfo(acq_data; kspace_data = acq_data.kspace_data ./ scale)
            x₀s = map(x -> x ./ scale, x₀s)
        end
    end
    if !config.disable_operator_normalization
        @step "Normalizing encoding operator" config begin
            𝒜 = normalize_op(𝒜, config.exact_opnorm)
        end
    end
    @step "Building optimization model" config begin
        model, vars, _auxiliaries = build_model(
            unname(𝒜), unname(acq_data.kspace_data), components;
            threaded = config.threaded, x₀s,
        )
    end
    @printing_step "Reconstructing image" config begin
        if isnothing(config.freq)
            freq = config.verbose ? get_reasonable_freq(config.maxit) : -1
        else
            freq = config.freq
        end
        ϵ = eps(real(eltype(x₀s[1])))
        tol = config.tol == 0 ? 0 : max(ϵ * 10, config.tol * maximum(x -> maximum(abs, x), x₀s))
        stop =
            (iter, state) -> ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
        display =
            (it, alg, iter, state) ->
        ProximalAlgorithms.default_display(it, alg, iter, state, config.printfunc)
        algorithm = patch_algorithm_with_default_values(algorithm, length(components))
        verbose = freq != -1
        solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
        xs = map(v -> copy(~v), vars)
    end
    if !config.disable_inverse_scale_output && scale != 1
        @step "Inverse scaling image" config begin
            xs = map(x -> x .* scale, xs)
        end
    end
    total_x = reduce(+, xs)
    if acq_data.kspace_data isa NamedDimsArray
        img_dimnames = dimnames(𝒜, 2)
        total_x = NamedDimsArray{img_dimnames}(total_x)
        xs = map(x -> NamedDimsArray{img_dimnames}(x), xs)
    end
    names = map(c -> c.name, components)
    return DecomposedImage(total_x, NamedTuple{names}(xs))
end

function _direct_reconstruct(𝒜, acq_data, x₀, regularization, config; scale_override = nothing)
    direct_recon_only = regularization == ()
    if !isnothing(x₀) && direct_recon_only
        config.verbose && config.printfunc(
            "Warning: Initial guess x₀ is ignored when no regularization is specified."
        )
        x₀ = nothing
    end
    if isnothing(x₀)
        @step (direct_recon_only ? "Reconstructing image" : "Getting initial estimate") config begin
            x₀ = 𝒜' * acq_data.kspace_data
        end
    end
    if !isnothing(scale_override)
        scale = scale_override
        config.verbose && config.printfunc(@sprintf("Using scaling factor: %g", scale))
    elseif config.normalization != NoScaling()
        @step "Computing scaling factor" config begin
            scale = get_scale(config.normalization, acq_data, x₀)
        end
        if scale == 0
            config.verbose &&
                config.printfunc("Warning: Computed scale is zero, defaulting to scale=1.0")
            scale = 1
        end
        config.verbose && config.printfunc(@sprintf("Using scaling factor: %g", scale))
    else
        scale = 1
    end
    return x₀, real(eltype(x₀))(scale)
end

function _iterative_reconstruct(𝒜, acq_data, x₀, scale, regularization, algorithm, config)
    if scale != 1
        @step "Scaling k-space data" config begin
            acq_data = AcquisitionInfo(acq_data; kspace_data = acq_data.kspace_data ./ scale)
            # Solver iterates in scaled units, so warm start and tolerance must match.
            x₀ = x₀ ./ scale
        end
    end
    if !config.disable_operator_normalization
        @step "Normalizing encoding operator" config begin
            𝒜 = normalize_op(𝒜, config.exact_opnorm)
        end
    end
    @step "Building optimization model" config begin
        model, x_var, _auxiliaries = build_model_with_variables(
            unname(𝒜),
            unname(acq_data.kspace_data),
            regularization;
            threaded = config.threaded,
            x₀,
            disable_normalop_optimization = config.disable_normalop_optimization,
        )
    end
    @printing_step "Reconstructing image" config begin
        if isnothing(config.freq)
            freq = config.verbose ? get_reasonable_freq(config.maxit) : -1
        else
            freq = config.freq
        end
        ϵ = eps(real(eltype(x₀)))
        tol = config.tol == 0 ? 0 : max(ϵ * 10, config.tol * maximum(abs, x₀))
        stop =
            (iter, state) -> ProximalAlgorithms.default_stopping_criterion(tol, iter, state)
        display =
            (it, alg, iter, state) ->
        ProximalAlgorithms.default_display(it, alg, iter, state, config.printfunc)
        algorithm = patch_algorithm_with_default_values(algorithm)
        verbose = freq != -1
        solve(model, algorithm; stop, maxit = config.maxit, freq, verbose, display)
        # Read the solution from the image variable itself: once a regularization contributes auxiliary
        # variables (e.g. total generalized variation), the solver returns them alongside the image and its
        # ordering is not something to depend on.
        x = copy(~x_var)
    end
    if !config.disable_inverse_scale_output && scale != 1
        @step "Inverse scaling image" config begin
            x .*= scale
        end
    end
    if acq_data.kspace_data isa NamedDimsArray
        img_dimnames = dimnames(𝒜, 2)
        x = NamedDimsArray{img_dimnames}(x)
    end
    return x
end

function get_reasonable_freq(maxit)
    reasonable_freqs = [1, 5, 10, 20, 50, 100]
    freq_i = findfirst(x -> x >= maxit ÷ 20, reasonable_freqs)
    return isnothing(freq_i) ? 100 : reasonable_freqs[freq_i]
end

function patch_algorithm_with_default_values(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{T}, n_components::Int = 1
    ) where {
        T <: Union{
            ProximalAlgorithms.ForwardBackwardIteration,
            ProximalAlgorithms.FastForwardBackwardIteration,
        },
    }
    if :Lf ∉ keys(algorithm.kwargs)
        # For n components sharing the same operator 𝒜, the data term is
        # ‖𝒜*(x₁+…+xₙ) - y‖²; its Lipschitz constant is n (not 1) when 𝒜 is
        # normalized to unit norm, since ‖[𝒜 … 𝒜]‖ = √n‖𝒜‖.
        return ProximalAlgorithms.override_parameters(algorithm; Lf = n_components)
    else
        return algorithm
    end
end

function patch_algorithm_with_default_values(
        algorithm::ProximalAlgorithms.IterativeAlgorithm{ProximalAlgorithms.ADMMIteration}, n_components::Int = 1
    )
    # `cg_maxit` is capped well below ADMM's own default of 100 because the inner CG is warm-started
    # from the previous outer iterate, so a short solve per outer step is enough.
    #
    # Neither `rho` nor `cg_tol` is defaulted here, and both omissions are deliberate.
    #
    # `cg_tol` is derived by `ADMM` as `min(1e-2, tol * 100)`, keeping the inner solve tighter than
    # the outer stopping tolerance. Pinning any constant would break that coupling and cap the
    # reachable accuracy whenever a caller tightens `tol`.
    #
    # `rho` is left to ADMM's adaptive `SpectralRadiusApproximationPenalty`, which converges far
    # faster than any fixed penalty on the problems this package builds: on a 2x-undersampled
    # L1Wavelet + TV reconstruction the adaptive penalty reaches 0.035 relative error within 100
    # iterations, while a fixed `rho = 1` needs ~2000 iterations to match it. Terms whose ADMM
    # behaviour is sensitive to the penalty should document a tuned `rho` of their own rather than
    # have one imposed on every caller here.
    defaults = (cg_maxit = 10,)
    missing_keys = filter(k -> k ∉ keys(algorithm.kwargs), keys(defaults))
    isempty(missing_keys) && return algorithm
    return ProximalAlgorithms.override_parameters(algorithm; NamedTuple{missing_keys}(defaults)...)
end

function patch_algorithm_with_default_values(algorithm::ProximalAlgorithms.IterativeAlgorithm, n_components::Int = 1)
    return algorithm
end

function patch_algorithm_with_default_values(algorithm::Tuple, n_components::Int = 1)
    return map(a -> patch_algorithm_with_default_values(a, n_components), algorithm)
end

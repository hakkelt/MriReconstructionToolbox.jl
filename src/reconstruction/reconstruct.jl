"""
	reconstruct(
		acq_data::AcquisitionInfo,
		[method::AbstractReconstructionMethod = DirectReconstruction()];
		[x₀], kwargs...)

Performs MRI reconstruction from k-space data using the specified reconstruction method.

# Arguments
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data, sensitivity maps, and other parameters.
- `method::AbstractReconstructionMethod = DirectReconstruction()`: The reconstruction method (e.g. `DirectReconstruction()`, `IterativeReconstruction(...)`).

# Keyword arguments
- `x₀::Union{Nothing,AbstractArray,Tuple,NamedTuple}=nothing`: Optional initial guess for the image (default is 𝒜' * y).
- `normalization::Normalization = BartScaling()`: scaling applied to operators/data (see also `NoScaling`, `MeasurementBasedScaling`, `FixedScaling`)
- `tol::Float64 = 1e-4`: stopping tolerance for iterative algorithms
- `maxit::Int = 100`: maximum iterations for the chosen solver
- `freq::Union{Nothing,Int} = nothing`: progress print frequency (iterations)
- `verbose::Bool = true`: enable/disable logging output
- `threaded::Bool = (Threads.nthreads() > 1)`: enable threaded execution when available
- `decomposition_executor::Union{Nothing,ReconstructionExecutor} = nothing`: override executor for decomposition
- `disable_inverse_scale_output::Bool = false`: skip rescaling the final output
- `disable_problem_decomposition::Bool = false`: disable automatic problem decomposition
- `printfunc::Function = println`: custom logging function

# Returns
- The reconstructed image (NamedDimsArray if input is NamedDimsArray, otherwise standard Array).
"""
function reconstruct(
        acq_data::AcquisitionInfo,
        method::AbstractReconstructionMethod = DirectReconstruction();
        x₀::Union{Nothing, AbstractArray, Tuple, NamedTuple} = nothing,
        kwargs...,
    )
    config = construct_config(kwargs)
    t_start = time()
    method = lower(method)
    check_applicable(method, acq_data)
    x = _reconstruct_dispatch(acq_data, method, x₀, config)
    t_end = time()
    config.verbose && config.printfunc("Total time: ", format_time(t_end - t_start))
    return x
end

function _reconstruct_dispatch(acq_data, method::DirectReconstruction, x₀, config)
    @argcheck isnothing(x₀) || x₀ isa AbstractArray "x₀ must be a plain array unless reconstructing with `Component`s."
    return _reconstruct_dispatch_plain(acq_data, method, x₀, config)
end

function _reconstruct_dispatch(acq_data, method::IterativeReconstruction, x₀, config)
    if method.regularization isa Tuple{Component, Vararg{Component}}
        check_components(method.regularization)
        return _reconstruct_dispatch_components(acq_data, method, x₀, config)
    else
        @argcheck isnothing(x₀) || x₀ isa AbstractArray "x₀ must be a plain array unless reconstructing with `Component`s."
        return _reconstruct_dispatch_plain(acq_data, method, x₀, config)
    end
end

function _reconstruct_dispatch_plain(acq_data, method::AbstractReconstructionMethod, x₀, config)
    decomposition_plan = get_problem_decomposition_plan(acq_data, method, config)
    x = if isnothing(decomposition_plan)
        reconstruction_result = nothing
        @conditionally_enable_threading config.threaded begin
            reconstruction_result = _reconstruct(acq_data, method, x₀, config)
        end
        first(reconstruction_result)
    else
        if !isnothing(x₀)
            @argcheck size(x₀) == decomposition_plan.image_size "Size of x₀ ($(size(x₀))) must match the image size ($(decomposition_plan.image_size))"
        end
        result = if method isa AbstractDirectMethod
            # Direct reconstruction needs no scaling; keep slices identical to the
            # non-decomposed result instead of normalizing each slice separately.
            config = Config(config; normalization = NoScaling())
            execute(decomposition_plan, acq_data, config) do idx, local_acq, local_conf
                local_x₀ = isnothing(x₀) ? nothing : get_x₀_slice(x₀, decomposition_plan, idx)
                _reconstruct(local_acq, method, local_x₀, local_conf)
            end
        else
            # Each slice's regularization strength is scale-dependent, so each slice is
            # first solved with its own scale to size λ correctly (via scale_regularization),
            # then the actual solve and the final image use one shared scale across all
            # slices so the output intensities are consistent slice-to-slice.
            execute_regularized(decomposition_plan, acq_data, config, method, x₀)
        end
        if acq_data.kspace_data isa NamedDimsArray
            result = NamedDimsArray{get_image_dims(acq_data)}(unname(result))
        end
        result
    end
    return x
end

function _reconstruct(
        acq_data, method::AbstractReconstructionMethod, x₀, config;
        scale_override = nothing, 𝒜 = nothing,
    )
    fast_planning = method isa DirectReconstruction
    if isnothing(𝒜)
        @step "Constructing encoding operator" config begin
            𝒜 = get_encoding_operator(
                acq_data; threaded = config.threaded, fast_planning
            )
        end
    end

    # Direct reconstruction / estimate
    x̂, scale = _direct_reconstruct(𝒜, acq_data, x₀, method, config; scale_override)

    if method isa AbstractDirectMethod
        # No regularization, return direct reconstruction
        if scale != 1 && config.disable_inverse_scale_output
            @step "Scaling image" config begin
                x̂ ./= scale
            end
        end
    elseif method isa IterativeReconstruction
        # Iterative reconstruction with regularization
        bound_regs = bind_dimensions(method.regularization, get_image_dims(acq_data))
        build = (𝒜, y; x₀) -> build_model_with_variables(
            𝒜, y, bound_regs;
            threaded = config.threaded, x₀,
            disable_normalop_optimization = method.disable_normalop_optimization,
            fidelity = method.fidelity,
        )
        x̂ = _iterative_reconstruct_core(𝒜, acq_data, x̂, scale, method, config; build)
        if acq_data.kspace_data isa NamedDimsArray
            x̂ = NamedDimsArray{dimnames(𝒜, 2)}(x̂)
        end
    end

    return x̂, scale
end

function _reconstruct_dispatch_components(acq_data, method::IterativeReconstruction, x₀, config)
    decomposition_plan = get_problem_decomposition_plan(acq_data, method, config)
    components = method.regularization
    img = if isnothing(decomposition_plan)
        # The decomposition branch below validates x₀ against the plan's image size; this branch has
        # no plan, so it validates against the acquisition's own image size. Both must check, or a
        # mistyped component name is only caught when the problem happens to be decomposed.
        if !isnothing(x₀)
            check_x₀_components_size(x₀, components, get_image_size(acq_data))
        end
        result = nothing
        @conditionally_enable_threading config.threaded begin
            result = _reconstruct_components(acq_data, method, x₀, config)
        end
        first(result)
    else
        if !isnothing(x₀)
            check_x₀_components_size(x₀, components, decomposition_plan.image_size)
        end
        execute_regularized_components(decomposition_plan, acq_data, config, method, x₀)
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

function _reconstruct_components(
        acq_data, method::IterativeReconstruction, x₀, config;
        scale_override = nothing, x₀s = nothing, 𝒜 = nothing,
    )
    components = bind_dimensions(method.regularization, get_image_dims(acq_data))
    if isnothing(𝒜)
        @step "Constructing encoding operator" config begin
            𝒜 = get_encoding_operator(acq_data; threaded = config.threaded, fast_planning = false)
        end
    end
    # `x₀s` lets a caller that has already formed the per-component initial guesses skip the adjoint
    # that would produce them. The decomposition path computes them in its first phase to derive the
    # per-slice scales, and without this would recompute 𝒜'y per slice only to discard it.
    scale = if isnothing(x₀s)
        x̂, s = _direct_reconstruct_components(𝒜, acq_data, config; scale_override)
        x₀s = get_component_x0s(components, x̂, x₀)
        s
    else
        @argcheck !isnothing(scale_override) "scale_override is required when x₀s is supplied."
        scale_override
    end
    build = (𝒜, y; x₀) -> build_model(
        𝒜, y, components;
        threaded = config.threaded, x₀s = x₀,
        fidelity = method.fidelity,
    )
    xs = _iterative_reconstruct_core(𝒜, acq_data, x₀s, scale, method, config; build)
    total_x = broadcast(+, xs...)
    if acq_data.kspace_data isa NamedDimsArray
        img_dimnames = dimnames(𝒜, 2)
        total_x = NamedDimsArray{img_dimnames}(total_x)
        xs = map(x -> NamedDimsArray{img_dimnames}(x), xs)
    end
    names = map(c -> c.name, components)
    img = DecomposedImage(total_x, NamedTuple{names}(xs))
    return img, scale
end

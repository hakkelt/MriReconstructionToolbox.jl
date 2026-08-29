struct ProblemDecompositionPlan{N, M, K, L}
    image_size::NTuple{N, Int}
    image_batch_dims::NTuple{M, Int}
    kspace_size::NTuple{K, Int}
    kspace_batch_dims::NTuple{L, Int}
    slices_sensitivity_maps::Bool
end

abstract type ReconstructionExecutor end
struct SequentialExecutor <: ReconstructionExecutor end
struct MultiThreadingExecutor <: ReconstructionExecutor end

function get_problem_decomposition_plan(acq_data, regularization, config)
    if config.disable_problem_decomposition
        return nothing
    elseif acq_data isa NonCartesianAcquisitionInfo
        return nothing
    end

    # Determine which image dimensions can be used for problem decomposition
    image_dims = get_image_dims(acq_data)
    image_batch_dims = collect(get_nonfourier_image_dims(acq_data))
    for reg in regularization
        affected_dims = get_affected_dims(reg, acq_data, image_dims)
        image_batch_dims = setdiff(image_batch_dims, affected_dims)
    end
    image_batch_dims = tuple(image_batch_dims...)

    if image_batch_dims == () # no batch dimensions, no decomposition
        return nothing
    end

    image_size = get_image_size(acq_data)
    if image_batch_dims[1] isa Symbol # convert to indices
        image_batch_dims = tuple(findall(in(image_batch_dims), collect(image_dims))...)
    end

    kspace_size = size(acq_data.kspace_data)

    # Calculate how the image batch dimensions map to k-space batch dimensions
    kspace_fourier_dims = get_fourier_kspace_dims(acq_data)
    image_fourier_dims = get_fourier_image_dims(acq_data)
    dim_index_offset = length(image_fourier_dims) - length(kspace_fourier_dims)
    if !isnothing(acq_data.sensitivity_maps)
        dim_index_offset -= 1 # account for coil dimension in k-space
    end
    kspace_batch_dims = tuple((d - dim_index_offset for d in image_batch_dims)...)

    slices_sensitivity_maps = (
        !isnothing(acq_data.sensitivity_maps) &&
            ndims(acq_data.sensitivity_maps) == 4 && # if true, the third dimension of the image must be the slice dimension
            3 ∈ image_batch_dims
    )

    return ProblemDecompositionPlan(
        image_size,
        image_batch_dims,
        kspace_size,
        kspace_batch_dims,
        slices_sensitivity_maps,
    )
end

function execute(f::Function, plan, acq_data, config)
    executor = suggest_executor(plan, config)
    return execute(f, plan, acq_data, config, executor)
end

function execute(f::Function, plan, acq_data, config, executor::ReconstructionExecutor)
    maybe_print_decomposition_info(plan, config)
    batch_sizes = plan.image_size[collect(plan.image_batch_dims)]
    results = Array{AbstractArray}(undef, batch_sizes)
    scales = Array{real(eltype(acq_data.kspace_data))}(undef, batch_sizes)
    run_slices!(results, scales, f, plan, acq_data, config, executor)
    maybe_rescale_results!(results, scales, config)
    return stack_image_slices(results, plan, Val(config.threaded))
end

function run_slices!(results, scales, f, plan, acq_data, config, executor::ReconstructionExecutor)
    slices = collect(get_slices(plan, acq_data))
    # A multi-threading executor already occupies the threads with whole slices, so the work inside a
    # slice runs sequentially there.
    slice_threaded = executor isa MultiThreadingExecutor ? false : config.threaded
    for_each_item!(slices, config, executor) do (idx, id, local_acq)
        r, s = execute_single_slice(f, idx, id, local_acq, config; threaded = slice_threaded)
        results[idx] = r
        scales[idx] = s
    end
    return nothing
end

# Regularized decomposition: regularization strength (λ) is scale-dependent, so each slice
# must be normalized before the regularization term is applied. But if each slice used its own
# scale for the final output too, slice-to-slice intensity would vary with noisy per-slice scale
# estimates instead of the true (similar) signal levels. So: estimate each slice's own scale first
# (phase 1), then solve every slice using one shared `global_scale` for both k-space data and the
# final image - giving uniform output - while compensating λ per slice by `scale_i / global_scale`
# so the regularization behaves as if that slice had been normalized by its own scale (see
# `scale_regularization`).
function execute_regularized(plan, acq_data, config, regularization, algorithm, x₀)
    prepare = function (idx, local_acq, local_conf)
        local_x₀ = isnothing(x₀) ? nothing : get_x₀_slice(x₀, plan, idx)
        # Planned properly (not `fast_planning`), because this same operator is reused for the
        # iterative solve in phase 2 below -- otherwise phase 2 would plan an equivalent operator
        # again from scratch.
        𝒜 = get_encoding_operator(local_acq; threaded = false, fast_planning = false)
        warm_start, scale = _direct_reconstruct(𝒜, local_acq, local_x₀, regularization, local_conf)
        return warm_start, scale, 𝒜
    end
    solve_slice = function (local_acq, warm_start, ratio, global_scale, local_conf, 𝒜)
        local_reg = map(r -> scale_regularization(r, ratio), regularization)
        result, _ = _reconstruct(
            local_acq, local_reg, algorithm, warm_start, local_conf; scale_override = global_scale, 𝒜
        )
        return result
    end
    return execute_two_phase(plan, acq_data, config, prepare, solve_slice)
end

# Same two-phase scheme as `execute_regularized`, for a component (multi-variable)
# reconstruction: phase 1 gets each slice's own scale from a plain direct estimate,
# phase 2 solves every slice under one shared `global_scale`, with each component's
# regularization compensated by `scale_i / global_scale` (`scale_regularization`).
function execute_regularized_components(plan, acq_data, config, components, algorithm, x₀)
    prepare = function (idx, local_acq, local_conf)
        local_x₀ = isnothing(x₀) ? nothing : slice_x₀_components(x₀, plan, idx)
        𝒜 = get_encoding_operator(local_acq; threaded = false, fast_planning = false)
        x̂, scale = _direct_reconstruct_components(𝒜, local_acq, local_conf)
        return get_component_x0s(components, x̂, local_x₀), scale, 𝒜
    end
    solve_slice = function (local_acq, x₀s, ratio, global_scale, local_conf, 𝒜)
        local_components = map(c -> scale_regularization(c, ratio), components)
        result, _ = _reconstruct_components(
            local_acq, local_components, algorithm, nothing, local_conf;
            scale_override = global_scale, x₀s, 𝒜,
        )
        return result
    end
    return execute_two_phase(plan, acq_data, config, prepare, solve_slice)
end

# Shared skeleton of the two-phase scheme described above. `prepare(idx, local_acq, local_conf)` returns
# `(warm_start, scale, 𝒜)` for one slice -- `𝒜` is the fully-planned encoding operator phase 1 already
# had to build to get the warm start, cached here so phase 2 does not plan an equivalent one again;
# `solve(local_acq, warm_start, ratio, global_scale, local_conf, 𝒜)` solves that slice under the shared
# scale, with its regularization compensated by `ratio`, reusing that cached operator.
function execute_two_phase(plan, acq_data, config, prepare::Function, solve::Function)
    executor = suggest_executor(plan, config)
    maybe_print_decomposition_info(plan, config)
    batch_sizes = plan.image_size[collect(plan.image_batch_dims)]
    slices = collect(get_slices(plan, acq_data))

    slice_threaded = executor isa MultiThreadingExecutor ? false : config.threaded
    slice_config = (id) -> Config(
        config;
        verbose = false, printfunc = (s...) -> config.printfunc("[$id] ", s...),
        freq = -1, threaded = slice_threaded,
    )

    # `prelim`'s element type isn't known until `prepare` actually runs (it depends on the acquisition
    # and warm-start array types), so the first slice is run outside the (possibly threaded) loop to
    # learn it; `prelim` is then allocated concretely instead of as `Array{Any}`, keeping the phase-2
    # unpacking below type-stable.
    first_idx, first_id, first_local_acq = slices[1]
    first_warm_start, first_scale, first_𝒜 = prepare(first_idx, first_local_acq, slice_config(first_id))
    first_prelim = (first_id, first_local_acq, first_warm_start, first_scale, first_𝒜)
    prelim = Array{typeof(first_prelim)}(undef, batch_sizes)
    prelim[first_idx] = first_prelim
    for_each_item!(@view(slices[2:end]), config, executor) do (idx, id, local_acq)
        warm_start, scale, 𝒜 = prepare(idx, local_acq, slice_config(id))
        prelim[idx] = (id, local_acq, warm_start, scale, 𝒜)
    end

    global_scale = robust_global_scale(vec(map(p -> p[4], prelim)))
    config.verbose && config.printfunc(
        @sprintf("Using shared scaling factor across slices: %g", global_scale)
    )

    results = Array{AbstractArray}(undef, batch_sizes)
    indices = vec(collect(CartesianIndices(batch_sizes)))
    for_each_item!(indices, config, executor) do idx
        id, local_acq, warm_start, scale, 𝒜 = prelim[idx]
        ratio = safe_scale_ratio(scale, global_scale)
        results[idx] = solve(local_acq, warm_start, ratio, global_scale, slice_config(id), 𝒜)
    end

    return stack_image_slices(results, plan, Val(config.threaded))
end

function slice_x₀_components(x₀::Tuple, plan, idx)
    return map(x -> get_x₀_slice(x, plan, idx), x₀)
end

function slice_x₀_components(x₀::NamedTuple, plan, idx)
    return NamedTuple{keys(x₀)}(map(x -> get_x₀_slice(x, plan, idx), values(x₀)))
end

function for_each_item!(f!::Function, items, config, ::SequentialExecutor)
    threaded = config.threaded
    @conditionally_enable_threading threaded for item in items
        f!(item)
    end
    return nothing
end

function for_each_item!(f!::Function, items, config, ::MultiThreadingExecutor)
    @budgeted_threads for item in items
        f!(item)
    end
    return nothing
end

function robust_global_scale(scales)
    nonzero = filter(!iszero, scales)
    return isempty(nonzero) ? one(eltype(scales)) : median(nonzero)
end

function safe_scale_ratio(scale, global_scale)
    # Guard against a slice whose own scale estimate is zero (or negligible relative to the
    # rest of the slices, e.g. an empty/noise-only slice): shrinking λ towards zero there would
    # leave that slice's noise essentially unregularized, so fall back to no correction instead.
    ratio = scale / global_scale
    return abs(ratio) < 1.0e-6 ? one(ratio) : ratio
end

# Helper functions

function map_dims_to_strs(sizes, batch_dims)
    return map(d -> d[1] ∈ batch_dims ? "_$(d[2])_" : string(d[2]), enumerate(sizes))
end

function Base.show(io::IO, plan::ProblemDecompositionPlan)
    print(io, "ProblemDecompositionPlan{")
    img_size_strs = map_dims_to_strs(plan.image_size, plan.image_batch_dims)
    print(io, "image_size=(", join(img_size_strs, ", "), "), ")
    ksp_size_strs = map_dims_to_strs(plan.kspace_size, plan.kspace_batch_dims)
    return print(io, "kspace_size=(", join(ksp_size_strs, ", "), ")}")
end

function Base.length(plan::ProblemDecompositionPlan)
    return prod(plan.image_size[collect(plan.image_batch_dims)])
end

function maybe_print_decomposition_info(plan, config)
    return if config.verbose
        batch_dims = plan.image_batch_dims
        batch_size = plan.image_size[collect(batch_dims)]
        if length(batch_dims) == 1
            msg_part = "dimension $(batch_dims[1]) with size $(batch_size[1])"
        else
            msg_part = "dimensions $batch_dims with sizes $batch_size"
        end
        config.printfunc("Decomposing problem over $msg_part")
    end
end

function get_slice_id(plan, idx, slice_idx_widths)
    id_parts = []
    counter = 1
    for d in eachindex(plan.image_size)
        if d in plan.image_batch_dims
            idx_str = @sprintf("%*s", slice_idx_widths[counter], string(idx[counter]))
            push!(id_parts, idx_str)
            counter += 1
        else
            push!(id_parts, ":")
        end
    end
    return "[" * join(id_parts, ", ") * "]"
end

function get_slices(plan, acq_data)
    indices = CartesianIndices(plan.kspace_size[collect(plan.kspace_batch_dims)])
    get_index_max_width = d -> length(string(size(acq_data.kspace_data, d)))
    slice_idx_widths = map(get_index_max_width, plan.kspace_batch_dims)
    slice_ids = map(idx -> get_slice_id(plan, idx, slice_idx_widths), indices)
    slices = eachslice(acq_data.kspace_data; dims = plan.kspace_batch_dims)
    ssm = plan.slices_sensitivity_maps
    local_acq = (
        get_acquisition_info_slice(acq_data, idx, ksp_slice, ssm) for
            (idx, ksp_slice) in zip(indices, slices)
    )
    return zip(indices, slice_ids, local_acq)
end

function get_acquisition_info_slice(acq_info::CartesianAcquisitionInfo, idx, kspace_data, slice_sensitivity_maps)
    if slice_sensitivity_maps
        sensitivity_maps = @view acq_info.sensitivity_maps[:, :, :, idx[1]]
        return CartesianAcquisitionInfo(acq_info; kspace_data, sensitivity_maps)
    else
        return CartesianAcquisitionInfo(acq_info; kspace_data)
    end
end

# `results` holds slices of whatever the per-slice reconstruction returned, so the element type is
# abstract; dispatch on one element rather than testing its type here.
function stack_image_slices(results, plan, threaded::Val)
    return stack_slices_like(first(results), results, plan, threaded)
end

function stack_slices_like(::AbstractArray, results, plan, threaded::Val)
    return stack_plain_image_slices(results, plan, threaded)
end

function stack_slices_like(::DecomposedImage, results, plan, threaded::Val)
    return stack_decomposed_image_slices(results, plan, threaded)
end

function stack_plain_image_slices(results, plan, ::Val{false})
    full_image = similar(unname(results[1]), plan.image_size)
    for (output_slice, result) in
        zip(eachslice(full_image; dims = plan.image_batch_dims), results)
        output_slice .= unname(result)
    end
    return full_image
end

function stack_plain_image_slices(results, plan, ::Val{true})
    full_image = similar(unname(results[1]), plan.image_size)
    extended_results = collect(
        zip(eachslice(full_image; dims = plan.image_batch_dims), results)
    )
    @threads for (output_slice, result) in extended_results
        output_slice .= unname(result)
    end
    return full_image
end

function stack_decomposed_image_slices(results, plan, threaded::Val)
    total_image = stack_plain_image_slices(map(total, results), plan, threaded)
    names = keys(first(results).components)
    comps = NamedTuple{names}(
        Tuple(
            stack_plain_image_slices(map(r -> r.components[name], results), plan, threaded)
                for name in names
        )
    )
    return DecomposedImage(total_image, comps)
end

function execute_single_slice(f::Function, idx, id, local_acq, config; kwargs...)
    if config.verbose
        freq = isnothing(config.freq) ? 0 : config.freq
    else
        freq = -1
    end
    printfunc = (s...) -> config.printfunc("[$id] ", s...)
    local_conf = Config(
        config;
        verbose = false, printfunc, freq, disable_inverse_scale_output = true, kwargs...,
    )
    return f(idx, local_acq, local_conf)
end

function get_x₀_slice(x₀, plan, idx)
    slicer = ntuple(length(plan.image_size)) do d
        i = findfirst(==(d), plan.image_batch_dims)
        isnothing(i) ? Colon() : idx[i]
    end
    return @view unname(x₀)[slicer...]
end

function maybe_rescale_results!(results, scales, config)
    return if !config.disable_inverse_scale_output
        median_scale = median(scales)
        @threads for i in eachindex(results)
            _rescale_result!(results[i], median_scale)
        end
        config.verbose && median_scale != 1 &&
            config.printfunc("Rescaled output by median scale factor $median_scale")
    end
end

_rescale_result!(x::AbstractArray, factor) = (x .*= factor)
_rescale_result!(x::DecomposedImage, factor) = rescale!(x, factor)

function suggest_executor(plan, config)
    if !isnothing(config.decomposition_executor)
        return config.decomposition_executor
    elseif config.threaded && length(plan) > nthreads()
        return MultiThreadingExecutor()
    else
        return SequentialExecutor()
    end
end

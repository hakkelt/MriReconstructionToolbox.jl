struct ProblemDecompositionPlan{N, M, K, L}
    variable_size::NTuple{N, Int}
    variable_batch_dims::NTuple{M, Int}
    kspace_size::NTuple{K, Int}
    kspace_batch_dims::NTuple{L, Int}
    slices_sensitivity_maps::Bool
    # Size of the *reconstructed image* per non-batch layout. Equals `variable_size` unless a
    # shape-changing `signal_model` (e.g. `TemporalBasis` with K < Nt) is in play, in which case
    # the signal-model-affected dims carry their expanded image size here. Used only to allocate
    # the merged output in `stack_*_image_slices`.
    output_size::NTuple{N, Int}
end

function get_problem_decomposition_plan(acq_data, method::AbstractReconstructionMethod, config)
    if config.disable_problem_decomposition
        return nothing
    elseif acq_data isa NonCartesianAcquisitionInfo
        return nothing
    end

    # Determine which image/variable dimensions can be used for problem decomposition
    image_dims = get_image_dims(acq_data)
    variable_batch_dims = collect(get_nonfourier_image_dims(acq_data))
    if method isa IterativeReconstruction
        for reg in method.regularization
            affected_dims = get_affected_dims(reg, acq_data, image_dims)
            variable_batch_dims = setdiff(variable_batch_dims, affected_dims)
        end
        if method.signal_model !== nothing
            affected_dims = get_affected_dims(method.signal_model, acq_data, image_dims)
            variable_batch_dims = setdiff(variable_batch_dims, affected_dims)
        end
    end
    variable_batch_dims = tuple(variable_batch_dims...)

    if variable_batch_dims == () # no batch dimensions, no decomposition
        return nothing
    end

    var_size = variable_size(method, acq_data)
    if variable_batch_dims[1] isa Symbol # convert to indices
        variable_batch_dims = tuple(findall(in(variable_batch_dims), collect(image_dims))...)
    end

    # The merged output has the reconstructed *image* size; it differs from `var_size` only
    # where a shape-changing signal model expands a variable dimension (e.g. subspace K -> Nt).
    out_size = var_size
    if method isa IterativeReconstruction && method.signal_model !== nothing
        img_size = get_image_size(acq_data)
        aff = get_affected_dims(method.signal_model, acq_data, image_dims)
        aff_idx = findall(in(aff), collect(image_dims))
        out_size = ntuple(d -> d in aff_idx ? img_size[d] : var_size[d], length(var_size))
    end

    kspace_size = size(acq_data.kspace_data)

    # Calculate how the variable batch dimensions map to k-space batch dimensions
    kspace_fourier_dims = get_fourier_kspace_dims(acq_data)
    image_fourier_dims = get_fourier_image_dims(acq_data)
    dim_index_offset = length(image_fourier_dims) - length(kspace_fourier_dims)
    if !isnothing(acq_data.sensitivity_maps)
        dim_index_offset -= 1 # account for coil dimension in k-space
    end
    kspace_batch_dims = tuple((d - dim_index_offset for d in variable_batch_dims)...)

    slices_sensitivity_maps = (
        !isnothing(acq_data.sensitivity_maps) &&
            ndims(acq_data.sensitivity_maps) == 4 && # if true, the third dimension of the image must be the slice dimension
            3 ∈ variable_batch_dims
    )

    return ProblemDecompositionPlan(
        var_size,
        variable_batch_dims,
        kspace_size,
        kspace_batch_dims,
        slices_sensitivity_maps,
        out_size,
    )
end

function map_dims_to_strs(sizes, batch_dims)
    return map(d -> d[1] ∈ batch_dims ? "_$(d[2])_" : string(d[2]), enumerate(sizes))
end

function Base.show(io::IO, plan::ProblemDecompositionPlan)
    print(io, "ProblemDecompositionPlan{")
    img_size_strs = map_dims_to_strs(plan.variable_size, plan.variable_batch_dims)
    print(io, "variable_size=(", join(img_size_strs, ", "), "), ")
    ksp_size_strs = map_dims_to_strs(plan.kspace_size, plan.kspace_batch_dims)
    return print(io, "kspace_size=(", join(ksp_size_strs, ", "), ")}")
end

function Base.length(plan::ProblemDecompositionPlan)
    return prod(plan.variable_size[collect(plan.variable_batch_dims)])
end

function maybe_print_decomposition_info(plan, config)
    return if config.verbose
        batch_dims = plan.variable_batch_dims
        batch_size = plan.variable_size[collect(batch_dims)]
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
    for d in eachindex(plan.variable_size)
        if d in plan.variable_batch_dims
            idx_str = @sprintf("%*s", slice_idx_widths[counter], string(idx[counter]))
            push!(id_parts, idx_str)
            counter += 1
        else
            push!(id_parts, ":")
        end
    end
    return "[" * join(id_parts, ", ") * "]"
end

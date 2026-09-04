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

function slice_x₀_components(x₀::Tuple, plan, idx)
    return map(x -> get_x₀_slice(x, plan, idx), x₀)
end

function slice_x₀_components(x₀::NamedTuple, plan, idx)
    return NamedTuple{keys(x₀)}(map(x -> get_x₀_slice(x, plan, idx), values(x₀)))
end

function get_x₀_slice(x₀, plan, idx)
    slicer = ntuple(length(plan.variable_size)) do d
        i = findfirst(==(d), plan.variable_batch_dims)
        isnothing(i) ? Colon() : idx[i]
    end
    return @view unname(x₀)[slicer...]
end

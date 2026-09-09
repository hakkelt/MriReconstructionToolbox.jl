function get_slices(plan, acq_data)
    indices = CartesianIndices(plan.kspace_size[collect(plan.kspace_batch_dims)])
    get_index_max_width = d -> length(string(size(acq_data.kspace_data, d)))
    slice_idx_widths = map(get_index_max_width, plan.kspace_batch_dims)
    slice_ids = map(idx -> get_slice_id(plan, idx, slice_idx_widths), indices)
    slices = eachslice(acq_data.kspace_data; dims = plan.kspace_batch_dims)
    ssm = plan.slices_sensitivity_maps
    local_acq = (
        get_acquisition_info_slice(acq_data, plan, idx, ksp_slice, ssm) for
            (idx, ksp_slice) in zip(indices, slices)
    )
    return zip(indices, slice_ids, local_acq)
end

function get_acquisition_info_slice(acq_info::CartesianAcquisitionInfo, plan, idx, kspace_data, slice_sensitivity_maps)
    subsampling = slice_subsampling(acq_info.subsampling, acq_info.kspace_data, plan.kspace_batch_dims, idx)
    if slice_sensitivity_maps
        sensitivity_maps = @view acq_info.sensitivity_maps[:, :, :, idx[1]]
        return CartesianAcquisitionInfo(acq_info; kspace_data, sensitivity_maps, subsampling)
    else
        return CartesianAcquisitionInfo(acq_info; kspace_data, subsampling)
    end
end

# A subsampling spec shared by every batch element (`nothing`, or the per-axis tuple `(:, mask)`)
# is handed to each task unchanged.
slice_subsampling(subsampling, kspace_data, kspace_batch_dims, idx) = subsampling
slice_subsampling(subsampling::AbstractArray{Bool}, kspace_data, kspace_batch_dims, idx) = subsampling

# ... but `subsampling` may instead be an *array of specs*, one per batch element — a different ky
# mask per frame, say. Each task must then get the specs belonging to its own slice, not the whole
# array; otherwise the k-space it holds and the pattern that produced it no longer agree.
function slice_subsampling(subsampling::AbstractArray, kspace_data, kspace_batch_dims, idx)
    # Which k-space dimensions the spec array spans: the same alignment `get_subsampling_operator`
    # performs, against the non-Fourier k-space dimensions (coil and batch).
    fourier_dims = _get_subsampled_dims_count(subsampling)
    nonfourier = size(kspace_data)[(fourier_dims + 1):end]
    n = ndims(subsampling)
    start = findfirst(
        s -> nonfourier[s:(s + n - 1)] == size(subsampling),
        1:(length(nonfourier) - n + 1),
    )
    isnothing(start) && return subsampling
    slicer = ntuple(n) do j
        kspace_dim = fourier_dims + start - 1 + j
        position = findfirst(==(kspace_dim), kspace_batch_dims)
        return isnothing(position) ? Colon() : idx[position]
    end
    sliced = subsampling[slicer...]
    return sliced isa AbstractArray ? sliced : _normalize_subsampling(sliced)
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

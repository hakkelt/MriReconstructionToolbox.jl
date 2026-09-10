"""
    PartialFourierFilter

Abstract type representing k-space weighting filters for Homodyne reconstruction.
"""
abstract type PartialFourierFilter end

"""
    LinearRamp <: PartialFourierFilter

Smooth linear transition ramp across the symmetric partial Fourier overlap band.
"""
struct LinearRamp <: PartialFourierFilter end

"""
    StepRamp <: PartialFourierFilter

Step transition filter (hard cutoff) across the partial Fourier transition boundary.
"""
struct StepRamp <: PartialFourierFilter end

"""
    partial_fourier_band(acq::CartesianAcquisitionInfo)

Analyzes the subsampling mask of a Cartesian acquisition to identify the asymmetric
partial Fourier acquisition dimension and returns a named tuple:
`(dim, symmetric_range, acquired_range, total_size)`.
"""
function partial_fourier_band(acq::CartesianAcquisitionInfo)
    _reject_partitioned(acq.kspace_data, "partial-Fourier band detection")
    @argcheck !isnothing(acq.subsampling) "Acquisition has no subsampling mask"
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    mask = to_displayable_mask(acq.subsampling, spatial_sz)

    # Check for asymmetry along dimension 1 (kx) or dimension 2 (ky)
    proj_y = vec(any(mask; dims = 1))
    proj_x = vec(any(mask; dims = 2))

    y_acq = findall(proj_y)
    x_acq = findall(proj_x)

    dim, acq_idx, N = if length(y_acq) < spatial_sz[2] && length(x_acq) == spatial_sz[1]
        (2, y_acq, spatial_sz[2])
    elseif length(x_acq) < spatial_sz[1] && length(y_acq) == spatial_sz[2]
        (1, x_acq, spatial_sz[1])
    else
        (2, y_acq, spatial_sz[2])
    end

    k_min = first(acq_idx)
    k_max = last(acq_idx)
    k_center = N ÷ 2 + 1

    half_band = min(k_center - k_min, k_max - k_center)
    sym_range = (k_center - half_band):(k_center + half_band)

    return (
        dim = dim,
        symmetric_range = sym_range,
        acquired_range = k_min:k_max,
        total_size = N,
    )
end

"""
    _pf_coil_dim(acq)

Integer position of the coil axis in the (full) k-space array, or `0` when there is none.
Resolved from dimension names when the k-space is a `NamedDimsArray`, else assumed to be
axis 3 for arrays with a third dimension.
"""
function _pf_coil_dim(acq::CartesianAcquisitionInfo)
    if acq.kspace_data isa NamedDimsArray
        idx = findfirst(==(:coil), dimnames(acq.kspace_data))
        return isnothing(idx) ? 0 : Int(idx)
    end
    return ndims(acq.kspace_data) >= 3 ? 3 : 0
end

function _pf_finalize(acq::CartesianAcquisitionInfo, img_out, coil_reduced::Bool, c_dim::Int)
    if coil_reduced && c_dim > 0
        img_out = dropdims(img_out; dims = c_dim)
    end
    if acq.kspace_data isa NamedDimsArray
        # `get_image_dims` describes the *combined* image, so it carries a `:coil` axis only when
        # the acquisition has no sensitivity maps (nothing consumed the coil axis). Both ends have
        # to be reconciled here: drop `:coil` when the channels were combined, and re-insert it at
        # `c_dim` when they were not but the maps had already removed it from the image dims.
        img_dims = get_image_dims(acq)
        out_dims = if coil_reduced
            filter(!=(:coil), img_dims)
        elseif :coil in img_dims || c_dim == 0
            img_dims
        else
            (img_dims[1:(c_dim - 1)]..., :coil, img_dims[c_dim:end]...)
        end
        return NamedDimsArray{out_dims}(unname(img_out))
    end
    return img_out
end

function _get_full_kspace(acq::CartesianAcquisitionInfo)
    raw_ksp = unname(acq.kspace_data)
    if isnothing(acq.subsampling)
        return copy(raw_ksp)
    end
    img_sz = get_image_size(acq)
    spatial_sz = (img_sz[1], img_sz[2])
    mask = to_displayable_mask(acq.subsampling, spatial_sz)

    trailing_dims = if size(raw_ksp, 1) == count(mask)
        size(raw_ksp)[2:end]
    else
        size(raw_ksp)[3:end]
    end
    full_ksp = zeros(eltype(raw_ksp), spatial_sz..., trailing_dims...)

    if size(raw_ksp, 1) == count(mask)
        full_ksp_flat = reshape(full_ksp, prod(spatial_sz), :)
        full_ksp_flat[vec(mask), :] .= reshape(raw_ksp, count(mask), :)
    elseif ndims(raw_ksp) >= 2 && size(raw_ksp, 2) == count(any(mask; dims = 1)) && size(raw_ksp, 1) == spatial_sz[1]
        acq_y = findall(vec(any(mask; dims = 1)))
        full_ksp[:, acq_y, :] .= reshape(raw_ksp, spatial_sz[1], length(acq_y), :)
    else
        full_ksp = copy(raw_ksp)
    end
    return full_ksp
end

"""
    simulate_acquisition(image, acq_info::CartesianAcquisitionInfo)

Simulate MRI k-space acquisition from a given image using the specified acquisition parameters.

# Arguments
- `image`: The input image to be transformed into k-space data. Can be a standard array or a `NamedDimsArray`.
- `acq_info::CartesianAcquisitionInfo`: Acquisition settings for Cartesian encoding.

# Returns
- An updated acquisition object with the simulated k-space data stored in `kspace_data`.
"""
function simulate_acquisition(image, acq_info::CartesianAcquisitionInfo)
    ksp_size = get_kspace_size(image, acq_info)
    ksp = similar(image, complex(eltype(image)), ksp_size)
    if image isa NamedDimsArray
        if acq_info.is3D && isnothing(acq_info.sensitivity_maps)
            full_ksp_dims = (:kx, :ky, :kz, dimnames(image)[4:end]...)
        elseif acq_info.is3D
            full_ksp_dims = (:kx, :ky, :kz, :coil, dimnames(image)[4:end]...)
        elseif isnothing(acq_info.sensitivity_maps)
            full_ksp_dims = (:kx, :ky, dimnames(image)[3:end]...)
        else
            full_ksp_dims = (:kx, :ky, :coil, dimnames(image)[3:end]...)
        end
        if isnothing(acq_info.subsampling)
            ksp_dims = full_ksp_dims
        else
            ksp_dims = _get_dimnames_from_subsampling(
                full_ksp_dims,
                acq_info.image_size,
                acq_info.subsampling,
            )
        end
        ksp = NamedDimsArray{ksp_dims}(NamedDims.unname(ksp))
    end
    if !isnothing(acq_info.sensitivity_maps)
        if acq_info.is3D
            @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
            @argcheck size(image)[1:3] == size(acq_info.sensitivity_maps)[1:3] "image spatial dimensions must match sensitivity maps spatial dimensions for 3D acquisition"
        else
            if ndims(acq_info.sensitivity_maps) == 4
                @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 2D multislice acquisition"
                @argcheck size(image)[1:3] == size(acq_info.sensitivity_maps)[[1, 2, 4]] "image spatial dimensions must match sensitivity maps spatial dimensions for 2D acquisition"
            else
                @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
                @argcheck size(image)[1:2] == size(acq_info.sensitivity_maps)[1:2] "image spatial dimensions must match sensitivity maps spatial dimensions for 2D acquisition"
            end
        end
    end
    acq_info = CartesianAcquisitionInfo(acq_info; kspace_data = ksp)
    E = get_encoding_operator(acq_info)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return acq_info
end

"""
    simulate_acquisition(image, acq_info::NonCartesianAcquisitionInfo)

Simulate MRI k-space acquisition from a given image using the specified non-Cartesian
acquisition parameters (trajectory, and optionally sensitivity maps).

# Arguments
- `image`: The input image to be transformed into k-space data. Can be a standard array or a `NamedDimsArray`.
- `acq_info::NonCartesianAcquisitionInfo`: Acquisition settings for non-Cartesian encoding.

# Returns
- An updated acquisition object with the simulated k-space data stored in `kspace_data`.
"""
function simulate_acquisition(image, acq_info::NonCartesianAcquisitionInfo)
    if acq_info.is3D
        @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
        @argcheck size(image)[1:3] == acq_info.image_size "image spatial dimensions must match image_size"
    else
        @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        @argcheck size(image)[1:2] == acq_info.image_size "image spatial dimensions must match image_size"
    end
    if !isnothing(acq_info.sensitivity_maps)
        spatial_dims = acq_info.is3D ? 3 : 2
        @argcheck size(image)[1:spatial_dims] == size(acq_info.sensitivity_maps)[1:spatial_dims] "image spatial dimensions must match sensitivity maps spatial dimensions"
    end

    sample_dims = size(acq_info.trajectory)[2:end]
    ncoil = isnothing(acq_info.sensitivity_maps) ? () : (size(acq_info.sensitivity_maps)[end],)
    ksp_size = (sample_dims..., ncoil...)
    ksp = similar(image, Complex{eltype(acq_info.trajectory)}, ksp_size)
    if image isa NamedDimsArray && acq_info.trajectory isa NamedDimsArray
        sample_dimnames = dimnames(acq_info.trajectory)[2:end]
        coil_dimnames = isnothing(acq_info.sensitivity_maps) ? () : (:coil,)
        ksp = NamedDimsArray{(sample_dimnames..., coil_dimnames...)}(NamedDims.unname(ksp))
    end

    acq_info = NonCartesianAcquisitionInfo(acq_info; kspace_data = ksp)
    E = get_encoding_operator(acq_info)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return acq_info
end

function get_kspace_size(image, acq_info::CartesianAcquisitionInfo)
    if isnothing(acq_info.subsampling) && isnothing(acq_info.sensitivity_maps)
        return size(image)
    elseif acq_info.is3D
        @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
        transformed_size = get_transformed_size(image, acq_info)
        if !isnothing(acq_info.sensitivity_maps)
            return (transformed_size..., size(acq_info.sensitivity_maps, 4), size(image)[4:end]...)
        else
            return (transformed_size..., size(image)[4:end]...)
        end
    elseif !isnothing(acq_info.sensitivity_maps)
        if ndims(acq_info.sensitivity_maps) == 4
            @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 2D multislice acquisition"
        else
            @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        end
        transformed_size = get_transformed_size(image, acq_info)
        return (transformed_size..., size(acq_info.sensitivity_maps, 3), size(image)[3:end]...)
    else
        @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        transformed_size = get_transformed_size(image, acq_info)
        return (transformed_size..., size(image)[3:end]...)
    end
end

function get_transformed_size(image, acq_info::CartesianAcquisitionInfo)
    if isnothing(acq_info.subsampling)
        return acq_info.image_size
    elseif acq_info.subsampling isa AbstractArray
        spatial_dims = acq_info.is3D ? 3 : 2
        spreading_dims = ndims(acq_info.subsampling)
        @argcheck ndims(image) >= spatial_dims + spreading_dims "image must provide one trailing dimension per subsampling pattern dimension"

        first_index = first(CartesianIndices(acq_info.subsampling))
        sample_img = @view image[fill(:, spatial_dims)..., Tuple(first_index)..., fill(1, ndims(image) - spatial_dims - spreading_dims)...]
        sample_subsampling = acq_info.subsampling[first_index]
        Base.checkbounds(sample_img, sample_subsampling...)
        return size(@view(sample_img[sample_subsampling...]))
    else
        if acq_info.is3D
            single_img = @view image[:, :, :, ones(Int, ndims(image) - 3)...]
        else
            single_img = @view image[:, :, ones(Int, ndims(image) - 2)...]
        end
        Base.checkbounds(single_img, acq_info.subsampling...)
        return size(@view(single_img[acq_info.subsampling...]))
    end
end

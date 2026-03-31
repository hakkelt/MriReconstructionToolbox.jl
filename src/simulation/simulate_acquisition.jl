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
    ksp = similar(image, ksp_size)
    if image isa NamedDimsArray
        if acq_info.is3D && isnothing(acq_info.sensitivity_maps)
            ksp_dims = (:kx, :ky, :kz, dimnames(image)[4:end]...)
        elseif acq_info.is3D
            ksp_dims = (:kx, :ky, :kz, :coil, dimnames(image)[4:end]...)
        elseif isnothing(acq_info.sensitivity_maps)
            ksp_dims = (:kx, :ky, dimnames(image)[3:end]...)
        else
            ksp_dims = (:kx, :ky, :coil, dimnames(image)[3:end]...)
        end
        ksp = NamedDimsArray{ksp_dims}(ksp)
    end
    if !isnothing(acq_info.sensitivity_maps)
        if acq_info.is3D
            @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
            @argcheck size(image)[1:3] == size(acq_info.sensitivity_maps)[1:3] "image spatial dimensions must match sensitivity maps spatial dimensions for 3D acquisition"
        else
            if ndims(acq_info.sensitivity_maps) == 4
                @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 2D multislice acquisition"
                @argcheck size(image)[1:3] == size(acq_info.sensitivity_maps)[[1,2,4]] "image spatial dimensions must match sensitivity maps spatial dimensions for 2D acquisition"
            else
                @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
                @argcheck size(image)[1:2] == size(acq_info.sensitivity_maps)[1:2] "image spatial dimensions must match sensitivity maps spatial dimensions for 2D acquisition"
            end
        end
    end
    acq_info = CartesianAcquisitionInfo(acq_info; kspace_data=ksp)
    E = get_encoding_operator(acq_info)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return acq_info
end

function simulate_acquisition(image, acq_info::NonCartesianAcquisitionInfo)
    error("Non-Cartesian acquisition simulation is not implemented yet")
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
    else
        if acq_info.is3D
            single_img = @view image[:,:,:,ones(Int,ndims(image)-3)...]
        else
            single_img = @view image[:,:,ones(Int,ndims(image)-2)...]
        end
        Base.checkbounds(single_img, acq_info.subsampling...)
        return size(@view(single_img[acq_info.subsampling...]))
    end
end

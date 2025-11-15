function simulate_acquisition(image, acq_info)
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
    acq_info = AcquisitionInfo(acq_info, kspace_data=ksp)
    E = get_encoding_operator(acq_info)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return acq_info
end

function get_kspace_size(image, acq_info)
    if isnothing(acq_info.subsampling)
        return size(image)
    elseif acq_info.is3D
        @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
        single_img = @view image[:,:,:,ones(Int,ndims(image)-3)...]
        Base.checkbounds(single_img, acq_info.subsampling...)
        subsampled_size = size(@view(single_img[acq_info.subsampling...]))
        if !isnothing(acq_info.sensitivity_maps)
            return (subsampled_size..., size(acq_info.sensitivity_maps, 4), size(image)[4:end]...)
        else
            return (subsampled_size..., size(image)[4:end]...)
        end
    elseif !isnothing(acq_info.sensitivity_maps)
        if ndims(acq_info.sensitivity_maps) == 4
            @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 2D multislice acquisition"
        else
            @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        end
        single_img = @view image[:,:,ones(Int,ndims(image)-2)...]
        Base.checkbounds(single_img, acq_info.subsampling...)
        subsampled_size = size(@view(single_img[acq_info.subsampling...]))
        return (subsampled_size..., size(acq_info.sensitivity_maps, 3), size(image)[3:end]...)
    else
        @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        single_img = @view image[:,:,ones(Int,ndims(image)-2)...]
        Base.checkbounds(single_img, acq_info.subsampling...)
        subsampled_size = size(@view(single_img[acq_info.subsampling...]))
        return (subsampled_size..., size(image)[3:end]...)
    end
end

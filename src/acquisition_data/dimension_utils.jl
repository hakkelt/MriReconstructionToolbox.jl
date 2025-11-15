function get_image_size(info::AcquisitionInfo)
    @argcheck !isnothing(info.kspace_data) "kspace_data must be provided to infer output dimensions"
    transform_dims_count = isnothing(info.subsampling) ? (info.is3D ? 3 : 2) : length(info.subsampling)
	batch_dims_start = isnothing(info.sensitivity_maps) ? transform_dims_count + 1 : transform_dims_count + 2
    batch_dims = size(info.kspace_data)[batch_dims_start:end]
    return (info.image_size..., batch_dims...)
end

function get_time_dim(time_dim, image_dims)
    if isnothing(time_dim)
        if image_dims[1] isa Symbol
            time_dim = findfirst(==(:time), image_dims)
			@argcheck !isnothing(time_dim) "Dimension :time not found in image_dims ($(image_dims))"
        else
            throw(ArgumentError("time_dim must be specified if kspace data is not a NamedDimsArray"))
        end
    elseif time_dim isa Symbol
        @argcheck image_dims[1] isa Symbol "when time_dim is a Symbol, kspace data must be a NamedDimsArray"
        time_dim = findfirst(==(time_dim), image_dims)
        @argcheck !isnothing(time_dim) "Dimension $(time_dim) not found in image_dims ($(image_dims))"
    end
	@assert 1 <= time_dim <= length(image_dims) "time_dim out of bounds"
	return time_dim
end

function get_fourier_kspace_dims(acq_info::AcquisitionInfo)
    @argcheck !isnothing(acq_info.kspace_data) "kspace_data must be provided in AcquisitionInfo to determine Fourier transformed dimensions"
    if acq_info.kspace_data isa NamedDimsArray
        if isnothing(acq_info.subsampling)
            return dimnames(acq_info.kspace_data)[1:(acq_info.is3D ? 3 : 2)]
        else
            return dimnames(acq_info.kspace_data)[1:length(acq_info.subsampling)]
        end
    else
        if isnothing(acq_info.subsampling)
            return 1:(acq_info.is3D ? 3 : 2)
        else
            return 1:length(acq_info.subsampling)
        end
    end
end

function get_fourier_image_dims(acq_info::AcquisitionInfo)
	@argcheck !isnothing(acq_info.kspace_data) "kspace_data must be provided in AcquisitionInfo to determine Fourier transformed dimensions"
	if acq_info.kspace_data isa NamedDimsArray
		return acq_info.is3D ? (:x, :y, :z) : (:x, :y)
	else
		return 1:(acq_info.is3D ? 3 : 2)
	end
end

function get_nonfourier_image_dims(acq_info::AcquisitionInfo)
    @argcheck !isnothing(acq_info.kspace_data) "kspace_data must be provided in AcquisitionInfo to determine non Fourier transformed dimensions"
    image_dims = get_image_dims(acq_info)
    fourier_image_dims = get_fourier_image_dims(acq_info)
    skipped_dims_count = length(fourier_image_dims)
	return image_dims[skipped_dims_count+1:end]
end

function get_nonfourier_kspace_dims(acq_info::AcquisitionInfo)
	@argcheck !isnothing(acq_info.kspace_data) "kspace_data must be provided in AcquisitionInfo to determine non Fourier transformed dimensions"
	ksp = acq_info.kspace_data
	if isnothing(acq_info.subsampling)
		skipped_dims_count = acq_info.is3D ? 3 : 2
	else
		skipped_dims_count = length(acq_info.subsampling)
	end
	if !isnothing(acq_info.sensitivity_maps)
		skipped_dims_count += 1
	end
	if ksp isa NamedDimsArray
		return dimnames(ksp)[(skipped_dims_count+1):end]
	else
		batch_dim_count = ndims(ksp) - skipped_dims_count
		@assert batch_dim_count >= 0 "kspace_data has fewer dimensions than expected"
		return (skipped_dims_count+1):(skipped_dims_count + batch_dim_count)
	end
end

function get_image_dims(acq_data::AcquisitionInfo)
	@argcheck !isnothing(acq_data.kspace_data) "kspace_data must be provided in AcquisitionInfo to determine output dimensions"
	ksp = acq_data.kspace_data
	spacial_dims = get_fourier_image_dims(acq_data)
	nonspacial_dims = get_nonfourier_kspace_dims(acq_data)
	if ksp isa NamedDimsArray
		return (spacial_dims..., nonspacial_dims...)
	else
		return 1:(length(spacial_dims) + length(nonspacial_dims))
	end
end

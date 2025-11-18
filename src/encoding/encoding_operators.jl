"""
Main encoding operators for MRI reconstruction.

This module provides the primary interface for creating MRI encoding operators
that model the complete data acquisition process, including Fourier transforms,
sensitivity map encoding, and subsampling patterns.
"""

"""
    get_encoding_operator(info::AcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
    get_encoding_operator(ksp, is3D::Bool; sensitivity_maps=nothing, image_size=nothing, subsampling=nothing, threaded=true, fast_planning=false)
    get_encoding_operator(ksp::NamedDimsArray; sensitivity_maps=nothing, image_size=nothing, subsampling=nothing, threaded=true, fast_planning=false)

Create the main MRI encoding operator for data acquisition modeling.

# Arguments for AcquisitionInfo method
- `info::AcquisitionInfo`: Contains k-space data, sensitivity maps, image size, subsampling pattern, and other acquisition parameters.

# Arguments for raw k-space method
- `ksp`: K-space data array
- `is3D::Bool`: Whether the acquisition is 3D
- `sensitivity_maps`: Coil sensitivity maps (optional)
- `image_size`: Image size tuple (optional)
- `subsampling`: Subsampling pattern (optional)

# Arguments for NamedDimsArray method
- `ksp::NamedDimsArray`: K-space data with named dimensions
- `sensitivity_maps`: Coil sensitivity maps (optional, can be NamedDimsArray)
- `image_size`: Image size tuple (optional, Tuple{Int,Int} or Tuple{Int,Int,Int})
- `subsampling`: Subsampling pattern (optional, 2D or 3D pattern)

# Common keyword arguments
- `threaded::Bool=true`: Whether to use multi-threading for operator construction and FFTs.
- `fast_planning::Bool=false`: Whether to use fast FFTW planning (reduces setup time, may affect performance).

# Returns
- Encoding operator modeling the full MRI acquisition process, including Fourier transform, sensitivity map encoding, and subsampling (if present).

# Details
This function constructs the composite encoding operator E that models the MRI data acquisition pipeline:
1. Applies sensitivity map encoding (if provided)
2. Applies Fourier transform (subsampled if a subsampling pattern is present)
3. Returns the composed operator E = F * S or E = F

If no sensitivity maps are provided, only the Fourier/subsampled Fourier operator is returned.
"""
function get_encoding_operator(info::AcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
	@argcheck !isnothing(info.kspace_data) "The provided AcquisitionInfo does not contain k-space data, which is required to build the encoding operator."
	has_subs = !isnothing(info.subsampling)
	ℱ = has_subs ? get_subsampled_fourier_operator(info; threaded, fast_planning) : get_fourier_operator(info; threaded, fast_planning)
	smaps = info.sensitivity_maps
	𝒜 = if isnothing(smaps)
		ℱ
	elseif smaps isa NamedDimsArray
		batch_dims_size = size(ℱ, 2)[ndims(smaps)+1:end]
		batch_dim_names = dimnames(ℱ, 2)[ndims(smaps)+1:end]
		batch_dims = NamedTuple{batch_dim_names}(batch_dims_size)
		𝒮 = get_sensitivity_map_operator(smaps; batch_dims, threaded)
		ℱ * 𝒮
	else
		batch_dims_start = ndims(smaps) + 1
		batch_dims = size(ℱ, 2)[batch_dims_start:end]
		𝒮 = get_sensitivity_map_operator(smaps, info.is3D; batch_dims, threaded)
		ℱ * 𝒮
	end
	return 𝒜 # Normalize operator to have norm 1
end

function get_encoding_operator(
	ksp,
	is3D::Bool;
	sensitivity_maps=nothing,
	image_size=nothing,
	subsampling=nothing,
	threaded::Bool=true,
	fast_planning::Bool=false,
)
	info = AcquisitionInfo(ksp; is3D, sensitivity_maps, image_size, subsampling)
	return get_encoding_operator(info; threaded, fast_planning)
end

function get_encoding_operator(
	ksp::NamedDimsArray;
	sensitivity_maps::Union{<:NamedDimsArray,Nothing}=nothing,
	image_size::Union{Tuple{Int,Int},Tuple{Int,Int,Int},Nothing}=nothing,
	subsampling::Union{<:_2D_subsampling_type,<:_3D_subsampling_type,Nothing}=nothing,
	threaded::Bool=true,
	fast_planning::Bool=false,
)
	info = AcquisitionInfo(ksp; sensitivity_maps, image_size, subsampling)
	return get_encoding_operator(info; threaded, fast_planning)
end

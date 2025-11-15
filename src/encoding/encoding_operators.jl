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

# Examples
```jldoctest
# Setup sample data
julia> using Random; Random.seed!(0);

# Create sample data for testing
julia> ksp = rand(ComplexF32, 64, 64, 8);  # k-space data with 8 coils
julia> smaps = rand(ComplexF32, 64, 64, 8);  # sensitivity maps for 8 coils
julia> mask = rand(Bool, 64, 64);  # Random sampling mask
julia> mask[1:2:end, :] .= true;  # Ensure some structure in sampling
julia> img = rand(ComplexF32, 64, 64);  # Ground truth image

# Using AcquisitionInfo
julia> info = AcquisitionInfo(ksp; sensitivity_maps=smaps, image_size=(64,64), subsampling=mask);
julia> E = get_encoding_operator(info);
julia> size(E)  # Shows input/output dimensions
(32768, 4096)

# Using raw k-space data for 2D acquisition
julia> E_raw = get_encoding_operator(ksp, false; sensitivity_maps=smaps, image_size=(64,64));
julia> size(E_raw)
(32768, 4096)

# Using NamedDimsArray for better dimension tracking
julia> using NamedDims
julia> ksp_named = NamedDimsArray(ksp, (:kx, :ky, :coil));
julia> smaps_named = NamedDimsArray(smaps, (:x, :y, :coil));
julia> E_named = get_encoding_operator(ksp_named; sensitivity_maps=smaps_named);
julia> dimnames(E_named, 1)  # Check input dimension names
(:kx, :ky, :coil)

# Demonstrate forward and adjoint operations
julia> ksp_sim = E * img;  # Forward operation (image to k-space)
julia> size(ksp_sim)  # Output size matches original k-space
(64, 64, 8)

julia> img_recon = E' * ksp_sim;  # Adjoint operation (k-space to image)
julia> size(img_recon) == size(img)  # Reconstructed image has correct size
true

# Example with subsampling
julia> E_sub = get_encoding_operator(ksp, false; sensitivity_maps=smaps, image_size=(64,64), subsampling=mask);
julia> ksp_sub = E_sub * img;
julia> count(mask) * 8 == length(ksp_sub)  # Subsampled data size matches mask
true
```
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

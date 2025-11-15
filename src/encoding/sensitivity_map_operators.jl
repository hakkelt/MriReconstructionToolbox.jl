"""
Sensitivity map operators for parallel MRI reconstruction.

This module provides functions for creating operators that model the sensitivity
patterns of multiple receiver coils in parallel MRI. These operators transform
single-coil images into multi-coil images and vice versa.
"""

"""
	get_sensitivity_map_operator(sensitivity_maps, [is3D]; batch_dims=nothing, threaded=true)

Create a sensitivity map operator for parallel MRI reconstruction.

# Arguments
- `sensitivity_maps`: Sensitivity maps for each coil (NamedDimsArray or AbstractArray)
- `is3D::Bool`: Whether the sensitivity maps are 3D (required for AbstractArray; auto-detected for NamedDimsArray)
- `batch_dims`: Batch dimensions specification (NamedTuple for NamedDimsArray, Tuple for AbstractArray, optional)
- `threaded::Bool=true`: Whether to use multi-threading for operations

# Returns
- Sensitivity map operator (NamedDimsOp for NamedDimsArray inputs, Compose for AbstractArray inputs)

# Method Variants
- **NamedDimsArray**: Automatically detects 2D vs 3D from dimension names, uses NamedTuple for batch_dims
- **AbstractArray**: Requires explicit `is3D` parameter, uses Tuple for batch_dims

# Required dimension names (NamedDimsArray method)
- `:x`: First spatial dimension (required, must be first dimension)
- `:y`: Second spatial dimension (required, must be second dimension)
- `:z`: Third spatial dimension (optional, must be third dimension if present)
- `:coil`: Coil dimension (required, must be last dimension)

# Array shapes (AbstractArray method)
- **2D**: `(nx, ny, ncoils)` or `(nx, ny, ncoils, batch_dims...)`
- **3D**: `(nx, ny, nz, ncoils)` or `(nx, ny, nz, ncoils, batch_dims...)`

# Details
The operator multiplies single-coil images by the sensitivity maps to produce
multi-coil images. The adjoint operation combines multi-coil images using the
conjugate of the sensitivity maps.

Forward operation: `multi_coil_images = S * single_coil_image`
Adjoint operation: `combined_image = S' * multi_coil_images`

The operator is constructed as:
1. `DiagOp(sensitivity_maps)`: Element-wise multiplication with sensitivity maps
2. `BroadCast(Eye(dummy_img), size(sensitivity_maps))`: Broadcasting identity for image dimensions
3. `BatchOp(...)`: If batch dimensions are present, wrap in batch operator

# Examples
```jldoctest
julia> using NamedDims

julia> smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8));

julia> S = get_sensitivity_map_operator(smaps);

julia> img = rand(ComplexF32, 64, 64);

julia> multi_coil = S * img;

julia> size(multi_coil)
(64, 64, 8)

julia> smaps_array = rand(ComplexF32, 64, 64, 8);

julia> S_array = get_sensitivity_map_operator(smaps_array, false);

julia> img_recon = S_array' * multi_coil;

julia> size(img_recon)
(64, 64)
```
"""
function get_sensitivity_map_operator(
	sensitivity_maps::NamedDimsArray;
	batch_dims::Union{NamedTuple,Nothing}=nothing,
	threaded::Bool=true,
)
	dn = dimnames(sensitivity_maps)
	@assert :x ∈ dn "sensitivity maps array must have a dimension named :x for Cartesian data"
	@assert dn[1] == :x "sensitivity maps array must have the first dimension named :x"
	@assert :y ∈ dn "sensitivity maps array must have a dimension named :ky for Cartesian data"
	@assert dn[2] == :y "sensitivity maps array must have the second dimension named :y"
	@assert :coil ∈ dn "sensitivity maps array must have a dimension named :coil for sensitivity maps array"
	@assert dn[end] == :coil "sensitivity maps array must have the last dimension named :coil"
	is3D = :z ∈ dn
	if is3D
		@assert dn[3] == :z "sensitivity maps array must have the third dimension named :z for 3D data"
	end
	tuple_batch_dims = isnothing(batch_dims) ? nothing : Tuple(batch_dims)
	𝒮 = get_sensitivity_map_operator(
		parent(sensitivity_maps), is3D; batch_dims=tuple_batch_dims, threaded
	)
	if isnothing(batch_dims) || isempty(batch_dims)
		return NamedDimsOp{dn[1:(end - 1)],dn}(𝒮)
	else
		codomain_dims = (dn..., keys(batch_dims)...)
		domain_dims = filter(dn -> dn != :coil, codomain_dims)
		return NamedDimsOp{domain_dims,codomain_dims}(𝒮)
	end
end

function get_sensitivity_map_operator(info::AcquisitionInfo; threaded::Bool=true)
	smaps = info.sensitivity_maps
	@argcheck !isnothing(smaps) "sensitivity_maps must be provided in AcquisitionInfo"
	if smaps isa NamedDimsArray
		# Derive batch dims from k-space/fourier layout if possible
		return get_sensitivity_map_operator(smaps; threaded)
	else
		return get_sensitivity_map_operator(smaps, info.is3D; threaded)
	end
end

function get_sensitivity_map_operator(
	sensitivity_maps::AbstractArray,
	is3D::Bool;
	batch_dims::Union{Tuple,Nothing}=nothing,
	threaded::Bool=true,
)
	if is3D
		@argcheck ndims(sensitivity_maps) == 4 "sensitivity maps array must be a 4D array for 3D data"
		I = Eye(@view(sensitivity_maps[:, :, :, 1]))
	elseif ndims(sensitivity_maps) == 4 # 2D multislice
		dummy_img = @view sensitivity_maps[:, :, 1, :]
		nx = size(sensitivity_maps, 1)
		ny = size(sensitivity_maps, 2)
		nz = size(sensitivity_maps, 4)
		I = reshape(Eye(dummy_img), nx, ny, 1, nz)
	else
		@argcheck ndims(sensitivity_maps) == 3 "sensitivity maps array must be a 3D array for 2D data"
		I = Eye(@view(sensitivity_maps[:, :, 1]))
	end
	if isnothing(batch_dims) || isempty(batch_dims)
		D = DiagOp(sensitivity_maps; threaded)
		B = BroadCast(I, size(sensitivity_maps); threaded)
		return D * B
	else
		inner_threaded = threaded && prod(batch_dims) < nthreads() ÷ 2
		D = DiagOp(sensitivity_maps; threaded=inner_threaded)
		B = BroadCast(I, size(sensitivity_maps); threaded=inner_threaded)
		return BatchOp(D * B, batch_dims; threaded)
	end
end

"""
	_check_smaps(ksp, smaps)

Internal function to validate compatibility between k-space data and sensitivity maps.

# Arguments
- `ksp`: K-space data array
- `smaps`: Sensitivity maps array

# Validation checks
1. Element types must match between ksp and smaps
2. Sensitivity maps must be 3D or 4D array
3. If ksp has named dimensions, it must include a `:coil` dimension when smaps are provided

# Throws
- `ArgumentError`: If any validation check fails
"""
function _check_smaps(ksp, smaps)
	@argcheck eltype(ksp) == eltype(smaps) "k-space array and sensitivity maps array must have the same element type"
	@argcheck 3 ≤ ndims(smaps) ≤ 4 "sensitivity maps array must be a 3D or 4D array"
	if ksp isa NamedDimsArray && :coil ∉ dimnames(ksp)
		@argcheck false "k-space array must have a dimension named :coil when sensitivity maps are provided"
	end
end

"""
Subsampling operators for accelerated MRI reconstruction.

This module provides functions for creating operators that model undersampled
k-space acquisition patterns. These operators select subsets of k-space data
according to various sampling strategies used in compressed sensing MRI.
"""

"""
	get_subsampled_fourier_operator(info::AcquisitionInfo)
    get_subsampled_fourier_operator(subsampled_ksp, img_size, subsampling; shifted_kspace_dims=(), shifted_image_dims=(), threaded=true, fast_planning=false)

Create a combined Fourier transform and subsampling operator.

# Arguments when using explicit parameters
- `subsampled_ksp`: Subsampled k-space data array
- `img_size`: Size of the full image (2 or 3 element tuple)
- `subsampling`: Subsampling pattern used to generate `subsampled_ksp`
- `shifted_kspace_dims::Tuple=()`: Dimensions in k‑space where the DC is at the first index (useful for pre‑shifted data)
- `shifted_image_dims::Tuple=()`: Image dimensions requiring fft shift (equivalent to an kspace-domain sign-alternation)
- `threaded::Bool=true`: Whether to use multi-threading
- `fast_planning::Bool=false`: Whether to use fast FFTW planning

# Returns
- Composed operator `Γ * ℱ` where `ℱ` is Fourier transform and `Γ` is subsampling

# Method Variants
- **Explicit parameters**: Requires `subsampled_ksp`, `img_size`, and `subsampling`
- **AcquisitionInfo**: Extracts necessary parameters from the `AcquisitionInfo` struct

# Details
This function creates an operator that:
1. Takes an image as input
2. Applies the Fourier transform to get full k-space
3. Applies subsampling to get the observed k-space data

The adjoint operation reconstructs an image from subsampled k-space data.
"""
function get_subsampled_fourier_operator(
	subsampled_ksp, img_size, subsampling;
	shifted_kspace_dims::Tuple=(),
	shifted_image_dims::Tuple=(),
	threaded::Bool=true,
	fast_planning::Bool=false
)
	ksp, Γ = _build_subsampling_context(subsampled_ksp, img_size, subsampling)
	is3D = length(img_size) == 3
	ℱ = ksp isa NamedDimsArray ?
		get_fourier_operator(ksp; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning) :
		get_fourier_operator(ksp, is3D; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning)
	return Γ * ℱ
end

function get_subsampled_fourier_operator(info::AcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
	@argcheck !isnothing(info.kspace_data) "The provided AcquisitionInfo does not contain k-space data, which is required to build the subsampled Fourier operator."
	@argcheck !isnothing(info.subsampling) "AcquisitionInfo must include a subsampling pattern to build a subsampled Fourier operator."
	return get_subsampled_fourier_operator(info.kspace_data, info.image_size, info.subsampling; shifted_kspace_dims=info.shifted_kspace_dims, shifted_image_dims=info.shifted_image_dims, threaded, fast_planning)
end


"""
	get_subsampling_operator(subsampled_ksp, img_size, subsampling)
	get_subsampling_operator(info::AcquisitionInfo)

Create the subsampling operator Γ that maps full k-space to a given
subsampled layout. This is useful when you need Γ separately or want to
compose it with other operators manually. For a combined `Γ * ℱ` operator,
use `get_subsampled_fourier_operator`.

# Arguments
- `subsampled_ksp`: Subsampled k-space array (Array or NamedDimsArray).
  Used to determine batch dimensions and, for named arrays, validate
  dimension names.
- `img_size`: Full image size as `(nx, ny)` or `(nx, ny, nz)`.
- `subsampling`: Subsampling pattern that produced `subsampled_ksp`.
  Supports boolean masks and tuples mixing `Colon`, boolean masks, and ranges.
- `info::AcquisitionInfo`: Alternative API that takes configuration from a
  validated `AcquisitionInfo` (must contain `image_size` and `subsampling`).

# Returns
- `Γ`: A `GetIndex` or `BatchOp{GetIndex}`. For NamedDims inputs, a
  `NamedDimsOp` wrapping the un-named Γ is returned to preserve
  dimension names.

# Details
- For NamedDims input, validates that `dimnames(subsampled_ksp)` matches the
  names implied by the subsampling pattern and the full k-space layout.
- Batch dimensions (beyond the spatial dims) are preserved; Γ becomes a batch
  operator when needed.

# Examples
```julia
using MriReconstructionToolbox

# 2D mask, array input
ksp_full = rand(ComplexF32, 64, 64, 8)
mask = rand(Bool, 64, 64)
ksp_sub = ksp_full[mask, :]
Γ = get_subsampling_operator(ksp_sub, (64, 64), mask)

# 2D NamedDims input
using NamedDims
ksp_nd = NamedDimsArray{(:kxy, :coil)}(ksp_sub)
Γ_nd = get_subsampling_operator(ksp_nd, (64, 64), mask)

# Via AcquisitionInfo
info = AcquisitionInfo(ksp_sub; is3D=false, image_size=(64, 64), subsampling=mask)
Γ_info = get_subsampling_operator(info)
```
"""
function get_subsampling_operator(subsampled_ksp, img_size, subsampling)
	_, Γ = _build_subsampling_context(subsampled_ksp, img_size, subsampling)
	return Γ
end

function get_subsampling_operator(acq_info::AcquisitionInfo)
	@argcheck !isnothing(acq_info.subsampling) "AcquisitionInfo must include a subsampling pattern"
	return get_subsampling_operator(acq_info.kspace_data, acq_info.image_size, acq_info.subsampling)
end

# -------- Type definitions for different subsampling patterns --------

const _1D_subsampling_type = Union{
	Colon,OrdinalRange{Int},AbstractArray{Bool,1},AbstractVector{Int}
}

const _2D_subsampling_type = Union{
	Tuple{<:AbstractArray{Bool,2}},
	Tuple{<:AbstractVector{Int}},
	Tuple{<:AbstractVector{CartesianIndex{2}}},
	Tuple{<:_1D_subsampling_type,<:_1D_subsampling_type},
}

const _3D_subsampling_type = Union{
	Tuple{<:AbstractArray{Bool,3}},
	Tuple{<:AbstractVector{Int}},
	Tuple{<:AbstractVector{CartesianIndex{3}}},
	Tuple{<:_1D_subsampling_type,<:AbstractArray{Bool,2}},
	Tuple{<:_1D_subsampling_type,<:AbstractVector{Int}},
	Tuple{<:_1D_subsampling_type,<:AbstractVector{CartesianIndex{2}}},
	Tuple{<:_1D_subsampling_type,<:_1D_subsampling_type,<:_1D_subsampling_type},
}

# -------- Internal helper functions --------

function _check_ksp_dimnames(ksp_dimnames, subs::Nothing, is3D, img_size)
	@argcheck :kx ∈ ksp_dimnames && :ky ∈ ksp_dimnames "k-space must have :kx and :ky dimensions"
	@argcheck ksp_dimnames[1] == :kx && ksp_dimnames[2] == :ky "first dims must be :kx, :ky"
	if is3D
		@argcheck :kz ∈ ksp_dimnames && ksp_dimnames[3] == :kz "third dim must be :kz for 3D"
	end
end

function _check_ksp_dimnames(ksp_dimnames, subs::_2D_subsampling_type, is3D, img_size)
	if subs isa Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type}
		@argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
		@argcheck :ky ∈ ksp_dimnames "k-space must have :ky dimension"
	else
		@argcheck :kxy ∈ ksp_dimnames "k-space must have :kxy dimension"
	end
	@argcheck length(img_size) == 2 "image_size must be length 2 for 2D subsampling"
	@argcheck !is3D "is3D must be false for 2D subsampling"
	if :coil ∈ ksp_dimnames
		@argcheck ksp_dimnames[3] == :coil "third dim must be :coil for 2D subsampling"
	end
	if :z ∈ ksp_dimnames && :coil ∈ ksp_dimnames
		@argcheck ksp_dimnames[4] == :z "fourth dim must be :z for 2D subsampling if :coil is present"
	elseif :z ∈ ksp_dimnames
		@argcheck ksp_dimnames[3] == :z "third dim must be :z for 2D subsampling if :coil is not present"
	end
end

function _check_ksp_dimnames(ksp_dimnames, subs, is3D, img_size)
	if subs isa Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type, <:_1D_subsampling_type}
		@argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
		@argcheck :ky ∈ ksp_dimnames "k-space must have :ky dimension"
		@argcheck :kz ∈ ksp_dimnames "k-space must have :kz dimension"
		@argcheck length(img_size) == 3 "image_size must be length 3 for 3D subsampling"
		@argcheck is3D "is3D must be true for 3D subsampling"
	elseif subs isa Tuple{<:_2D_subsampling_type, <:_1D_subsampling_type}
		@argcheck :kxy ∈ ksp_dimnames "k-space must have :kxy dimension"
		@argcheck :kz ∈ ksp_dimnames "k-space must have :kz dimension"
		@argcheck length(img_size) == 3 "image_size must be length 3 for 3D subsampling"
		@argcheck is3D "is3D must be true for 3D subsampling"
	elseif subs isa Tuple{<:_1D_subsampling_type, <:_2D_subsampling_type}
		@argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
		@argcheck :kyz ∈ ksp_dimnames "k-space must have :kyz dimension"
		@argcheck length(img_size) == 3 "image_size must be length 3 for 3D subsampling"
		@argcheck is3D "is3D must be true for 3D subsampling"
	else
		@argcheck :kxyz ∈ ksp_dimnames "k-space must have :kxyz dimension"
	end
	@argcheck length(img_size) == 3 "image_size must be length 3 for 3D subsampling"
	@argcheck is3D "is3D must be true for 3D subsampling"
	if :coil ∈ ksp_dimnames
		@argcheck ksp_dimnames[4] == :coil "fourth dim must be :coil for 3D subsampling"
	end
	@argcheck !(:z ∈ ksp_dimnames) "3D subsampling cannot have :z dimension"
end

function _get_dimnames_from_subsampling(ksp_dimnames, subsampling::_2D_subsampling_type)
	if length(subsampling) == 2
		return (:kx, :ky, ksp_dimnames[3:end]...)
	else
		return (:kxy, ksp_dimnames[3:end]...)
	end
end

function _get_dimnames_from_subsampling(ksp_dimnames, subsampling::_3D_subsampling_type)
	if length(subsampling) == 3
		return (:kx, :ky, :kz, ksp_dimnames[4:end]...)
	elseif length(subsampling) == 2 && subsampling[1] isa _1D_subsampling_type
		return (:kx, :kyz, ksp_dimnames[4:end]...)
	elseif length(subsampling) == 2 && subsampling[2] isa _1D_subsampling_type
		return (:kxy, :kz, ksp_dimnames[4:end]...)
	else
		return (:kxyz, ksp_dimnames[4:end]...)
	end
end

function _get_subsampling_operator(ksp, img_size, subsampling::_2D_subsampling_type)
	@argcheck length(img_size) == 2 "img_size must be a 2-element tuple for 2D subsampling"
	@argcheck img_size == size(ksp)[1:2] DimensionMismatch
	if ndims(ksp) > length(img_size)
		batch_dims = size(ksp)[3:end]
		ksp_view = @view ksp[:, :, fill(1, length(batch_dims))...]
		Γ = GetIndex(ksp_view, subsampling)
		return BatchOp(Γ, batch_dims; threaded=true)
	else
		return GetIndex(ksp, subsampling)
	end
end

function _get_subsampling_operator(ksp, img_size, subsampling::_3D_subsampling_type)
	@argcheck length(img_size) == 3 "img_size must be a 3-element tuple for 3D subsampling"
	@argcheck img_size == size(ksp)[1:3] DimensionMismatch
	if ndims(ksp) > length(img_size)
		batch_dims = size(ksp)[4:end]
		ksp_view = @view ksp[:, :, :, fill(1, length(batch_dims))...]
		Γ = GetIndex(ksp_view, subsampling)
		return BatchOp(Γ, batch_dims; threaded=true)
	else
		return GetIndex(ksp, subsampling)
	end
end

function _get_subsampling_operator(ksp, img_size, subsampling::AbstractArray)
	error("currently not implemented")
	# Future implementation would handle arrays of subsampling patterns
	#if all(subs isa _2D_subsampling_type for subs in subsampling)
	#	# subsampling is an array of 2D subsampling masks
	#	@argcheck length(img_size) == 2 "img_size must be a 2-element tuple for 2D subsampling"
	#	@argcheck img_size == size(ksp)[1:2] DimensionMismatch
	#	@argcheck size(subsampling) == size(ksp)[3:2+ndims(subsampling)] DimensionMismatch
	#	batch_dims = size(ksp)[(2 + ndims(subsampling)):end]
	#	@show batch_dims
	#	ksp_view = @view ksp[:, :, fill(1, length(batch_dims))...]
	#	Γᵢ = [_get_subsampling_operator(ksp_view, img_size, subs) for subs in subsampling]
	#	return BatchOp(Γᵢ, batch_dims; threaded=true)
	#elseif all(subs isa _3D_subsampling_type for subs in subsampling)
	#	# subsampling is an array of 3D subsampling masks
	#	@argcheck length(img_size) == 3 "img_size must be a 3-element tuple for 3D subsampling"
	#	@argcheck img_size == size(ksp)[1:3] DimensionMismatch
	#	@argcheck size(subsampling) == size(ksp)[4:3+ndims(subsampling)] DimensionMismatch
	#	batch_dims = size(ksp)[(3 + ndims(subsampling)):end]
	#	ksp_view = @view ksp[:, :, :, fill(1, length(batch_dims))...]
	#	Γᵢ = [_get_subsampling_operator(ksp_view, img_size, subs) for subs in subsampling]
	#	return BatchOp(Γᵢ, batch_dims; threaded=true)
	#else
	#	throw(ArgumentError("Could not handle subsampling type: $(typeof(subsampling))"))
	#end
end

_get_subsampled_dims_count(::AbstractArray{Bool,N}) where {N} = N
_get_subsampled_dims_count(::AbstractVector{Int}) = 1
_get_subsampled_dims_count(::AbstractVector{CartesianIndex{N}}) where {N} = N
_get_subsampled_dims_count(::Colon) = 1
_get_subsampled_dims_count(::OrdinalRange) = 1
_get_subsampled_dims_count(subsampling::Tuple) = sum(_get_subsampled_dims_count.(subsampling))
function _get_subsampled_dims_count(subsampling::AbstractArray)
	dim_counts = _get_subsampled_dims_count.(subsampling)
	if all(==(dim_counts[1]), dim_counts)
		return dim_counts[1]
	else
		return error("Inconsistent subsampling dimensions inferred: $(dim_counts)")
	end
end

_get_img_size_from_subsampling(subsampling::AbstractArray{Bool}, ksp) = size(subsampling)
_get_img_size_from_subsampling(::AbstractVector{<:Integer}, ksp) = nothing
_get_img_size_from_subsampling(::AbstractVector{CartesianIndex{N}}, ksp) where {N} = nothing

function _get_img_size_from_subsampling(subsampling::Tuple, ksp)
	img_size = ()
	dim_counter = 1
	for subs in subsampling
		if subs isa Colon
			img_size = (img_size..., size(ksp, dim_counter)...)
			dim_counter += 1
		elseif subs isa AbstractArray{Bool}
			img_size = (img_size..., size(subs)...)
			dim_counter += ndims(subs)
		else
			return nothing
		end
	end
	return img_size
end

function _get_img_size_from_subsampling(subsampling::AbstractArray, ksp)
	guessed_sizes = tuple([_get_img_size_from_subsampling(subs, ksp) for subs in subsampling]...)
	if any(isnothing, guessed_sizes)
		return nothing
	elseif all(==(guessed_sizes[1]), guessed_sizes)
		return guessed_sizes[1]
	else
		return error("Inconsistent image sizes inferred from subsampling masks: $(guessed_sizes)")
	end
end

function _build_subsampling_context(subsampled_ksp, img_size, subsampling)
	@argcheck 2 ≤ length(img_size) ≤ 3 "img_size must be either length 2 or 3"
	batch_dims_start = length(subsampling) + 1
	ksp_size = (img_size..., size(subsampled_ksp)[batch_dims_start:end]...)
	ksp = similar(subsampled_ksp, ksp_size)
	is3D = length(img_size) == 3
	if subsampled_ksp isa NamedDimsArray
		batch_dim_names = dimnames(subsampled_ksp)[batch_dims_start:end]
		full_dimnames = is3D ?
			(:kx, :ky, :kz, batch_dim_names...) :
			(:kx, :ky, batch_dim_names...)
		expected_subs_dimnames = _get_dimnames_from_subsampling(full_dimnames, subsampling)
		@argcheck dimnames(subsampled_ksp) == expected_subs_dimnames
		ksp = NamedDimsArray{full_dimnames}(unname(ksp))
		Γ_unwrapped = _get_subsampling_operator(unname(ksp), img_size, subsampling)
		D = dimnames(ksp)
		new_dimnames = _get_dimnames_from_subsampling(D, subsampling)
		Γ = NamedDimsOp{D,new_dimnames}(Γ_unwrapped)
	else
		Γ = _get_subsampling_operator(ksp, img_size, subsampling)
	end
	return ksp, Γ
end

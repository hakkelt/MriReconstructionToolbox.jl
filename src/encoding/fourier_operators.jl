"""
Fourier transform operators for MRI reconstruction.

This module provides functions for creating Fourier transform operators that convert
between image space and k-space (frequency domain) representations in MRI data.
"""

"""
	get_fourier_operator(ksp, [is3D], [shifted_kspace_dims], [shifted_image_dims]; threaded=true, fast_planning=false)
	get_fourier_operator(info::AcquisitionInfo; threaded=true, fast_planning=false)

Create a Fourier transform operator for MRI data.

The operator transforms between image space and k-space using the discrete Fourier transform.
For named dimension arrays, automatically detects 2D vs 3D from dimension names.
For regular arrays, specify `is3D` explicitly.

# Arguments with explicit types
- `ksp`: K-space data array (NamedDimsArray or AbstractArray)
- `is3D::Bool`: Whether the data is 3D (required for AbstractArray; optionally auto-detected for NamedDimsArray)
- `shifted_kspace_dims::Tuple`: Dimensions in k-space where the DC is at the first index instead of the center (useful for pre-shifted data, default: empty)
- `shifted_image_dims::Tuple`: Dimensions in image space requiring fftshift (equivalent to an kspace-domain sign-alternation, default: empty)
- `threaded::Bool`: Whether to use multi-threading for FFT operations (default: true)
- `fast_planning::Bool`: If true, use FFTW.ESTIMATE for faster planning (default: false)

# Returns
- Fourier transform operator

# Method Variants
- **NamedDimsArray**: Automatically determines 2D vs 3D from the presence of `:kz` dimension
- **AbstractArray**: Requires explicit `is3D` parameter to determine dimensionality
- **AcquisitionInfo**: Extracts `ksp`, `is3D`, `shifted_kspace_dims`, and `shifted_image_dims` from the `AcquisitionInfo` struct
"""
function get_fourier_operator(info::AcquisitionInfo; threaded::Bool=true, fast_planning::Bool=false)
	@argcheck !isnothing(info.kspace_data) "The provided AcquisitionInfo does not contain k-space data, which is required to build the Fourier operator."
	if isnothing(info.subsampling)
		ksp = info.kspace_data
	else
		ksp = get_full_kspace(info)
	end
	shifted_image_dims = info.shifted_image_dims
	shifted_kspace_dims = info.shifted_kspace_dims
	return get_fourier_operator(
		ksp,
		info.is3D;
		shifted_kspace_dims,
		shifted_image_dims,
		threaded,
		fast_planning,
	)
end

function get_fourier_operator(
	ksp::NamedDimsArray,
	is3D::Bool=(:kz ∈ dimnames(ksp));
	shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
	shifted_image_dims::Union{Tuple,Integer,Symbol}=(),
	threaded::Bool=true,
	fast_planning::Bool=false,
)
	ksp_dimnames = dimnames(ksp)
	@argcheck :kx ∈ ksp_dimnames "k-space array must have a dimension named :kx for Cartesian data"
	@argcheck ksp_dimnames[1] == :kx "k-space array must have the first dimension named :kx"
	@argcheck :ky ∈ ksp_dimnames "k-space array must have a dimension named :ky for Cartesian data"
	@argcheck ksp_dimnames[2] == :ky "k-space array must have the second dimension named :ky"
	@argcheck is3D == (:kz ∈ ksp_dimnames) "is3D does not match presence of :kz dimension in k-space array"
	if is3D
		@argcheck ksp_dimnames[3] == :kz "k-space array must have the third dimension named :kz for 3D data"
		img_dimnames = (:x, :y, :z, ksp_dimnames[4:end]...)
	else
		img_dimnames = (:x, :y, ksp_dimnames[3:end]...)
	end
	shifted_kspace_dims = _normalize_shifted_dims(
		shifted_kspace_dims, is3D, ksp, "shifted_kspace_dims"
	)
	shifted_image_dims = _normalize_shifted_dims(
		shifted_image_dims, is3D, ksp, "shifted_image_dims"
	)
	ℱ = get_fourier_operator(
		parent(ksp), is3D; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning
	)
	return NamedDimsOp{img_dimnames,ksp_dimnames}(ℱ)
end

function get_fourier_operator(
	ksp::AbstractArray,
	is3D::Bool;
	shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
	shifted_image_dims::Union{Tuple,Integer,Symbol}=(),
	threaded::Bool=true,
	fast_planning::Bool=false,
)
	flags = fast_planning ? FFTW.ESTIMATE : FFTW.MEASURE
	num_threads = threaded ? nthreads() : 1
	ksp_dims = is3D ? (1, 2, 3) : (1, 2)
	ℱ = DFT(ksp, ksp_dims; normalization=FFTWOperators.BACKWARD, flags, num_threads)
	kspace_dims_to_shift = tuple([d for d in ksp_dims if d ∉ shifted_kspace_dims]...)
	shifted_kspace_dims = _normalize_shifted_dims(
		shifted_kspace_dims, is3D, ksp, "shifted_kspace_dims"
	)
	shifted_image_dims = _normalize_shifted_dims(
		shifted_image_dims, is3D, ksp, "shifted_image_dims"
	)
	if !isempty(kspace_dims_to_shift) || !isempty(shifted_image_dims)
		# Wrap with shifts if needed
		ℱ = ifftshift_op(
			ℱ; domain_shifts=shifted_image_dims, codomain_shifts=kspace_dims_to_shift
		)
	end
	return ℱ
end

function _normalize_shifted_dims(
	shifted_dims::Union{Tuple,Integer,Symbol},
	is3D::Bool,
	ksp::AbstractArray,
	context::String
)
	if shifted_dims isa Integer || shifted_dims isa Symbol
		shifted_dims = (shifted_dims,)
	end
	for d in shifted_dims
		if d isa Integer
			@argcheck d in (is3D ? (1, 2, 3) : (1, 2)) "$context contains invalid dimension $d for is3D=$is3D"
		else
			@argcheck d isa Symbol "$context contains invalid dimension $d (must be Integer or Symbol)"
			@argcheck (ksp isa NamedDimsArray) "$context with Symbol dimensions requires the kspace data to be a NamedDimsArray"
			@argcheck (d ∈ (is3D ? (:x, :y, :z) : (:x, :y))) "$context contains invalid dimension $d"
		end
	end
	return shifted_dims
end

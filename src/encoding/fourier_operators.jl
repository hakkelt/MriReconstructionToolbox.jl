"""
Fourier transform operators for MRI reconstruction.

This module provides functions for creating Fourier transform operators that convert
between image space and k-space (frequency domain) representations in MRI data.
"""

"""
    get_fourier_operator(ksp, [is3D], [shifted_kspace_dims], [shifted_image_dims]; threaded=true, fast_planning=false)
    get_fourier_operator(info::CartesianAcquisitionInfo; threaded=true, fast_planning=false)
    get_fourier_operator(info::NonCartesianAcquisitionInfo; threaded=true)
    get_fourier_operator(ksp, image_size, trajectory; dcf=nothing, threaded=true)

Create the Fourier encoding operator for MRI data.

This function dispatches on its arguments and returns either a Cartesian
DFT-backed operator or a non-Cartesian NFFT-backed operator. For Cartesian
acquisitions it transforms between image space and regularly sampled k-space.
For non-Cartesian acquisitions it maps images on a Cartesian grid to
trajectory-sampled k-space.

For named-dimension Cartesian arrays, 2D versus 3D is inferred from the
presence of `:kz`. For plain arrays, `is3D` must be provided explicitly.
For non-Cartesian inputs, dispatch is selected by passing
`NonCartesianAcquisitionInfo` or the explicit `(ksp, image_size, trajectory)`
arguments.

# Arguments with explicit Cartesian types
- `ksp`: Cartesian k-space data array (`NamedDimsArray` or `AbstractArray`)
- `is3D::Bool`: Whether the Cartesian data is 3D
- `shifted_kspace_dims`: K-space dimensions where the DC is already at the first index
- `shifted_image_dims`: Image dimensions requiring fftshift / sign alternation
- `threaded::Bool`: Whether to use multi-threading for FFT/NFFT construction
- `fast_planning::Bool`: If true, use FFTW.ESTIMATE for faster DFT planning

# Arguments with explicit non-Cartesian types
- `ksp`: Non-Cartesian k-space data array
- `image_size::Tuple`: Cartesian image grid size used for the NFFT domain
- `trajectory`: Sampling trajectory; its leading dimension stores coordinates
- `dcf`: Optional density compensation factors matching the trajectory sample layout

# Returns
- A Fourier encoding operator backed by `DFT` for Cartesian data or `NFFTOp`
  for non-Cartesian data.

# Method Variants
- **NamedDimsArray (Cartesian)**: infers 2D vs 3D from `:kz`
- **AbstractArray (Cartesian)**: requires explicit `is3D`
- **CartesianAcquisitionInfo**: extracts Cartesian settings from the acquisition struct
- **NonCartesianAcquisitionInfo**: constructs an NFFT-backed operator from trajectory metadata
- **(ksp, image_size, trajectory)**: explicit non-Cartesian constructor
"""
function get_fourier_operator(info::CartesianAcquisitionInfo; threaded::Bool = true, fast_planning::Bool = false)
    @argcheck !isnothing(info.kspace_data) "The provided CartesianAcquisitionInfo does not contain k-space data, which is required to build the Fourier operator."
    if isnothing(info.subsampling)
        ksp = info.kspace_data
    else
        # Only a planning template with the full k-space layout is needed here;
        # avoid materializing the full k-space via an adjoint apply.
        ksp = _full_kspace_template(info.kspace_data, info.image_size, info.subsampling)
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
        is3D::Bool = (:kz ∈ dimnames(ksp));
        shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
        shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
        threaded::Bool = true,
        fast_planning::Bool = false,
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
        shifted_kspace_dims, is3D, ksp, "shifted_kspace_dims", (:kx, :ky, :kz)
    )
    shifted_image_dims = _normalize_shifted_dims(
        shifted_image_dims, is3D, ksp, "shifted_image_dims", (:x, :y, :z)
    )
    ℱ = get_fourier_operator(
        parent(ksp), is3D; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning
    )
    return NamedDimsOp{img_dimnames, ksp_dimnames}(ℱ)
end

function get_fourier_operator(
        ksp::AbstractArray,
        is3D::Bool;
        shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
        shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
        threaded::Bool = true,
        fast_planning::Bool = false,
    )
    flags = fast_planning ? FFTW.ESTIMATE : FFTW.MEASURE
    ksp_dims = is3D ? (1, 2, 3) : (1, 2)
    ℱ = DFT(ksp, ksp_dims; normalization = FFTWOperators.BACKWARD, flags, threaded)
    shifted_kspace_dims = _normalize_shifted_dims(
        shifted_kspace_dims, is3D, ksp, "shifted_kspace_dims", (:kx, :ky, :kz)
    )
    shifted_image_dims = _normalize_shifted_dims(
        shifted_image_dims, is3D, ksp, "shifted_image_dims", (:x, :y, :z)
    )
    kspace_dims_to_shift = tuple([d for d in ksp_dims if d ∉ shifted_kspace_dims]...)
    if !isempty(kspace_dims_to_shift) || !isempty(shifted_image_dims)
        # Wrap with shifts if needed
        ℱ = ifftshift_op(
            ℱ; domain_shifts = shifted_image_dims, codomain_shifts = kspace_dims_to_shift
        )
    end
    return ℱ
end

function get_fourier_operator(info::NonCartesianAcquisitionInfo; threaded::Bool = true)
    @argcheck !isnothing(info.kspace_data) "The provided NonCartesianAcquisitionInfo does not contain k-space data, which is required to build the NFFT operator."
    return get_fourier_operator(
        info.kspace_data,
        info.image_size,
        info.trajectory;
        dcf = info.dcf,
        threaded,
    )
end

function get_fourier_operator(
        ksp::NamedDimsArray,
        image_size::Tuple,
        trajectory::NamedDimsArray;
        dcf = nothing,
        threaded::Bool = true,
    )
    fourier_dims = ndims(trajectory) - 1
    ksp_dimnames = dimnames(ksp)
    traj_dimnames = dimnames(trajectory)
    @argcheck ksp_dimnames[1:fourier_dims] == traj_dimnames[2:end] "k-space dimension names must match trajectory sample dimension names"

    image_dimnames = if length(image_size) == 3
        (:x, :y, :z, ksp_dimnames[(fourier_dims + 1):end]...)
    else
        (:x, :y, ksp_dimnames[(fourier_dims + 1):end]...)
    end
    raw_dcf = dcf isa NamedDimsArray ? parent(dcf) : dcf
    𝒩 = get_fourier_operator(parent(ksp), image_size, parent(trajectory); dcf = raw_dcf, threaded)
    return NamedDimsOp{image_dimnames, ksp_dimnames}(𝒩)
end

function get_fourier_operator(
        ksp::AbstractArray,
        image_size::Tuple,
        trajectory::AbstractArray;
        dcf = nothing,
        threaded::Bool = true,
    )
    fourier_dims = ndims(trajectory) - 1
    batch_dims = size(ksp)[(fourier_dims + 1):end]
    inner_threaded = threaded && isempty(batch_dims)
    𝒩 = if isnothing(dcf)
        NFFTOp(image_size, trajectory; threaded = inner_threaded)
    else
        NFFTOp(image_size, trajectory, dcf; threaded = inner_threaded)
    end
    if isempty(batch_dims)
        return 𝒩
    end
    return BatchOp(𝒩, batch_dims; threaded)
end

"""
    _axis_dft_op(template, dims::Tuple; kspace_shift = false, threaded = true, fast_planning = false)

Bare `BACKWARD`-normalized `DFT` over `dims` of an array shaped like `template`, optionally with an
`fftshift` on the k-space (codomain) side. `op * x` is `fft(x, dims)` (or `fftshift(fft(x, dims), dims)`
with `kspace_shift`); `op' * k` is the matching inverse (`ifft(ifftshift(k, dims), dims)`). Used where
only a subset of axes is transformed (readout-only coil compression, spatial-only sensitivity maps),
so `get_fourier_operator` — which assumes a full Cartesian layout — does not apply.
"""
function _axis_dft_op(
        template::AbstractArray, dims::Tuple;
        kspace_shift::Bool = false, threaded::Bool = true, fast_planning::Bool = false,
    )
    flags = fast_planning ? FFTW.ESTIMATE : FFTW.MEASURE
    ℱ = DFT(template, dims; normalization = FFTWOperators.BACKWARD, flags, num_threads = threaded ? nthreads() : 1)
    return kspace_shift ? fftshift_op(ℱ; codomain_shifts = dims) : ℱ
end

function _normalize_shifted_dims(
        shifted_dims::Union{Tuple, Integer, Symbol},
        is3D::Bool,
        ksp::AbstractArray,
        context::String,
        valid_symbols::Tuple,
    )
    if shifted_dims isa Integer || shifted_dims isa Symbol
        shifted_dims = (shifted_dims,)
    end
    valid_symbols = is3D ? valid_symbols : valid_symbols[1:2]
    return map(shifted_dims) do d
        if d isa Integer
            @argcheck d in (is3D ? (1, 2, 3) : (1, 2)) "$context contains invalid dimension $d for is3D=$is3D"
            Int(d)
        else
            @argcheck d isa Symbol "$context contains invalid dimension $d (must be Integer or Symbol)"
            @argcheck (ksp isa NamedDimsArray) "$context with Symbol dimensions requires the kspace data to be a NamedDimsArray"
            i = findfirst(==(d), valid_symbols)
            @argcheck !isnothing(i) "$context contains invalid dimension $d (valid dimension names: $valid_symbols)"
            i
        end
    end
end

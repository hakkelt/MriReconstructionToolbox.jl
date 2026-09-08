"""
Fourier transform operators for MRI reconstruction.

This module provides functions for creating Fourier transform operators that convert
between image space and k-space (frequency domain) representations in MRI data.
"""

"""
    get_fourier_operator(ksp, [is3D], [shifted_kspace_dims], [shifted_image_dims]; threaded=true, fast_planning=false)
    get_fourier_operator(info::CartesianAcquisitionInfo; threaded=true, fast_planning=false)
    get_fourier_operator(info::NonCartesianAcquisitionInfo; threaded=true, m=nothing, sigma=nothing, precompute=nothing)
    get_fourier_operator(ksp, image_size, trajectory; dcf=nothing, threaded=true, m=nothing, sigma=nothing, precompute=nothing)

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
- `dcf`: Density compensation, forwarded to `NFFTOp` unchanged: `nothing` (the default) applies
  none, so the resulting operator's adjoint is the true adjoint; `:auto` estimates it with
  NFFTOperators' iterative sample density compensation method; an array matching the trajectory
  sample layout is used as given. See `NFFTOp`'s docstring for the full contract and why `:auto`
  or an explicit array makes the adjoint a density-compensated approximate inverse, not the true
  adjoint.
- `m`, `sigma`, `precompute`: NFFT gridding operating point (kernel half-width, oversampling
  factor, `NFFT.PrecomputeFlags`), forwarded to `NFFTOp`/NFFT.jl. Left at `nothing` (the
  default), MRT's own default operating point is used (`DEFAULT_NFFT_M`, `DEFAULT_NFFT_SIGMA`,
  `DEFAULT_NFFT_PRECOMPUTE` -- a lower-accuracy, faster point than NFFT.jl's own default). See
  "Non-Cartesian accuracy / speed trade-off" in `docs/src/high-level/performance.md` for the
  measured accuracy/speed table this default is picked from.

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

function get_fourier_operator(
        info::NonCartesianAcquisitionInfo;
        threaded::Bool = true,
        m::Union{Nothing, Integer} = nothing,
        sigma::Union{Nothing, Real} = nothing,
        precompute = nothing,
    )
    @argcheck !isnothing(info.kspace_data) "The provided NonCartesianAcquisitionInfo does not contain k-space data, which is required to build the NFFT operator."
    return get_fourier_operator(
        info.kspace_data,
        info.image_size,
        info.trajectory;
        dcf = info.dcf,
        threaded, m, sigma, precompute,
    )
end

function get_fourier_operator(
        ksp::NamedDimsArray,
        image_size::Tuple,
        trajectory::NamedDimsArray;
        dcf = nothing,
        threaded::Bool = true,
        m::Union{Nothing, Integer} = nothing,
        sigma::Union{Nothing, Real} = nothing,
        precompute = nothing,
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
    𝒩 = get_fourier_operator(
        parent(ksp), image_size, parent(trajectory); dcf = raw_dcf, threaded, m, sigma, precompute
    )
    return NamedDimsOp{image_dimnames, ksp_dimnames}(𝒩)
end

"""
    get_fourier_operator(ksp::AbstractArray, image_size::Tuple, trajectory::AbstractArray;
                          dcf=nothing, threaded=true, m=nothing, sigma=nothing, precompute=nothing)

Non-Cartesian (NFFT-backed) Fourier operator. `m`, `sigma` (`σ`) and `precompute` expose the
gridding operating point: `m` is the interpolation kernel's half-width, `sigma` its oversampling
factor, `precompute` the `NFFT.PrecomputeFlags` gridding strategy. Leaving them at `nothing` (the
default) uses MRT's own default operating point (`m=4, σ=1.5, precompute=NFFT.POLYNOMIAL` —
`DEFAULT_NFFT_M`/`DEFAULT_NFFT_SIGMA`/`DEFAULT_NFFT_PRECOMPUTE`), chosen for speed at negligible
accuracy cost; pass explicit values for a different point on the accuracy/speed curve, e.g.
NFFT.jl's own higher-accuracy default (`m=5, sigma=2.0`) or MRIReco's faster, less accurate one
(`m=3, sigma=1.25, precompute=NFFT.TENSOR`). See "Non-Cartesian accuracy / speed trade-off" in
`docs/src/high-level/performance.md` for the measured table.

`dcf` (density compensation) is forwarded to `NFFTOp` unchanged: `nothing` (the default) applies
none, so `op'` is the *true* adjoint of `op` (required by anything that assumes the adjoint
relationship, e.g. operator-norm estimation, CG/CGNR); `:auto` estimates it with NFFTOperators'
iterative sample density compensation method; an array is used as given. Both `:auto` and an
explicit array make `op'` a density-compensated approximate inverse instead of the true adjoint —
useful for a quick direct (gridding) reconstruction, wrong as the adjoint fed to an
adjoint-assuming algorithm. See `NFFTOp`'s docstring (`deps/AbstractOperators/NFFTOperators`) for
the full contract.
"""
function get_fourier_operator(
        ksp::AbstractArray,
        image_size::Tuple,
        trajectory::AbstractArray;
        dcf = nothing,
        threaded::Bool = true,
        m::Union{Nothing, Integer} = nothing,
        sigma::Union{Nothing, Real} = nothing,
        precompute = nothing,
    )
    fourier_dims = ndims(trajectory) - 1
    batch_dims = size(ksp)[(fourier_dims + 1):end]
    inner_threaded = threaded && isempty(batch_dims)
    nfft_kwargs = _nfft_operating_point_kwargs(m, sigma, precompute)
    # `dcf` is forwarded to `NFFTOp` as-is: `nothing` (the default) means no density
    # compensation (the true adjoint), `:auto` requests NFFTOperators' own estimator, and an
    # array is used as given. See `NFFTOp`'s docstring for the full contract.
    𝒩 = NFFTOp(image_size, trajectory, dcf; threaded = inner_threaded, nfft_kwargs...)
    if isempty(batch_dims)
        return 𝒩
    end
    return BatchOp(𝒩, batch_dims; threaded)
end

"""
    DEFAULT_NFFT_M, DEFAULT_NFFT_SIGMA, DEFAULT_NFFT_PRECOMPUTE

MRT's own default NFFT gridding operating point, applied whenever `m`/`sigma`/`precompute` are
left at `nothing` on `get_fourier_operator`/`get_encoding_operator`. Measured on a 128×128 radial
phantom (`GeometricMedicalPhantoms`'s Shepp-Logan, 256 samples × 128 spokes), single thread,
interleaved runs (`benchmark/comparison`-style methodology; a single measurement on this shared
node can swing 30-60%, so configs were timed round-robin rather than one after another):

| m | σ | precompute | forward (min/median ms) | adjoint (min/median ms) | forward rel. error |
|---|---|---|---|---|---|
| 5 | 2.00 | POLYNOMIAL (former MRT default = NFFT.jl's own default) | 8.5 / 9.7 | 7.5 / 8.6 | 0 (reference) |
| 4 | 2.00 | POLYNOMIAL | 7.0 / 8.1 | 5.5 / 6.4 | 3.8e-8 |
| **4** | **1.50** | **POLYNOMIAL (new MRT default)** | **4.0 / 4.5** | **4.6 / 5.3** | **2.5e-7** |
| 3 | 2.00 | POLYNOMIAL | 6.0 / 6.8 | 4.3 / 4.9 | 2.4e-6 |
| 3 | 1.50 | POLYNOMIAL | 2.8 / 3.2 | 3.3 / 3.8 | 1.7e-5 |
| 3 | 1.25 | TENSOR (MRIReco's point) | 2.5 / 2.9 | 2.8 / 3.1 | 7.1e-5 |
| 2 | 1.50 | POLYNOMIAL | 2.3 / 2.6 | 2.4 / 2.8 | 7.4e-4 |
| 2 | 1.25 | TENSOR | 2.0 / 2.3 | 1.9 / 2.2 | 2.1e-3 |

`m=4, σ=1.5, POLYNOMIAL` is chosen as the new default: forward relative error against the old
default is 2.5e-7 -- indistinguishable from full accuracy for reconstruction purposes (the direct
gridding reconstruction's NRMSE against the phantom does not move outside run-to-run noise across
this whole table, consistent with `docs/src/high-level/performance.md`'s existing observation
that gridding-adjoint NRMSE barely depends on the operating point) -- while running about 2x
faster on the forward transform and about 1.6x faster on the adjoint than the old default. Every
row below it in the table trades measurably more accuracy for comparatively little extra speed.

Call `get_fourier_operator`/`get_encoding_operator` with explicit `m`, `sigma`, `precompute`
keywords to override this (e.g. to match MRIReco's `m=3, σ=1.25, precompute=NFFT.TENSOR`, or to
go back to the old high-accuracy default with `m=5, sigma=2.0`).
"""
const DEFAULT_NFFT_M = 4
const DEFAULT_NFFT_SIGMA = 1.5
const DEFAULT_NFFT_PRECOMPUTE = NFFT.POLYNOMIAL

# Always forward a concrete operating point: leaving `m`/`sigma`/`precompute` at `nothing` now
# substitutes MRT's own (lower-accuracy, faster) default rather than NFFT.jl's own default -- see
# `DEFAULT_NFFT_M` and friends for the measured justification.
function _nfft_operating_point_kwargs(m, sigma, precompute)
    return (
        m = isnothing(m) ? DEFAULT_NFFT_M : m,
        σ = isnothing(sigma) ? DEFAULT_NFFT_SIGMA : sigma,
        precompute = isnothing(precompute) ? DEFAULT_NFFT_PRECOMPUTE : precompute,
    )
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
    ℱ = DFT(template, dims; normalization = FFTWOperators.BACKWARD, flags, threaded)
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

# AcquisitionInfo

`AcquisitionInfo` is a validated configuration container for MRI acquisition parameters. It centralizes k-space data, sensitivity maps, image dimensions, subsampling patterns, and FFT shift conventions, performing comprehensive validation at construction time to catch configuration errors early.

Benefits:
- **Centralized validation**: Checks dimension compatibility at construction time
- **Clear parameter organization**: Named fields instead of positional arguments
- **Type safety**: Supports both plain arrays and `NamedDimsArray`
- **Reusability**: Pass the same config to multiple functions

## Constructor

```@docs
AcquisitionInfo
```

## Accepted Input Types

### K-space Data

The first argument can be:

1. **Plain `AbstractArray`**: Standard Julia arrays containing k-space data
2. **`NamedDimsArray`**: Arrays with named dimensions for automatic inference
3. **`nothing`**: When setting up acquisition parameters without actual data (e.g., for simulation)

#### Constraints on Dimensions

- The first two dimensions correspond to transformed spatial axes (kx, ky)
- It must be followed by transformed z-axis if 3D encoding is used (kz)
- The next dimension must correspond to coils, if sensitivity maps are provided
- If 2D encoding is used and sensitivity maps are provided, then slice dimensions must be after the coil dimension
- Sensitivity maps must match spatial dimensions of the image:
    - 2D encoding: ``(N_x, N_y, N_c, [N_z])``
    - 3D encoding: ``(N_x, N_y, N_z, N_c)``

```@setup acqinfo
using MriReconstructionToolbox
using NamedDims
```

```@example acqinfo
using MriReconstructionToolbox

# Plain array - requires explicit is3D
ksp_plain = rand(ComplexF32, 64, 64, 8)
AcquisitionInfo(ksp_plain; is3D=false)
```

```@example acqinfo
using NamedDims

# NamedDimsArray - is3D inferred from :kz presence
ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 8)
)
AcquisitionInfo(ksp_named)  # is3D = false (no :kz)
```

```@example acqinfo
# 3D with named dimensions
ksp_3d = NamedDimsArray{(:kx, :ky, :kz, :coil)}(
    rand(ComplexF32, 32, 32, 16, 4)
)
AcquisitionInfo(ksp_3d)  # is3D = true (has :kz)
```

```@example acqinfo
# Without k-space data (for simulation setup)
info4 = AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(128, 128)
)
```

### Sensitivity Maps

Sensitivity maps must match k-space dimensions and element type:

```@example acqinfo
# 2D single-coil sensitivity maps
ksp = rand(ComplexF32, 64, 64, 8)
smaps = rand(ComplexF32, 64, 64, 8)  # (nx, ny, ncoils)

AcquisitionInfo(ksp; is3D=false, sensitivity_maps=smaps)
```

```@example acqinfo
# 3D sensitivity maps
ksp_3d = rand(ComplexF32, 32, 32, 16, 4)
smaps_3d = rand(ComplexF32, 32, 32, 16, 4)  # (nx, ny, nz, ncoils)

AcquisitionInfo(ksp_3d; is3D=true, sensitivity_maps=smaps_3d)
```

```@example acqinfo
# 2D multi-slice with per-slice sensitivity maps
ksp_ms = rand(ComplexF32, 64, 64, 4, 10)  # 4 coils, 10 slices
smaps_ms = rand(ComplexF32, 64, 64, 4, 10)  # (nx, ny, ncoils, nslices)

AcquisitionInfo(ksp_ms; is3D=false, sensitivity_maps=smaps_ms)
```

### Subsampling Patterns

Multiple subsampling formats are supported:

```@example acqinfo
# Boolean mask
mask = rand(Bool, 64, 64)
mask[25:40, 25:40] .= true  # Fully sample center

info_mask = AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(64, 64),
    subsampling=mask
)
```

```@example acqinfo
# Tuple of Colon and mask for Cartesian undersampling
mask_ky = rand(Bool, 64)
mask_ky[28:36] .= true  # Fully sample center lines

AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(64, 64),
    subsampling=(:, mask_ky)  # Fully sample kx, undersample ky
)
```

```@example acqinfo
# 3D subsampling with multiple dimensions
mask_3d = rand(Bool, 32, 32, 16)
mask_3d[13:20, 13:20, 5:12] .= true

AcquisitionInfo(
    nothing;
    is3D=true,
    image_size=(32, 32, 16),
    subsampling=mask_3d
)
```

```@example acqinfo
# One pattern PER BATCH ELEMENT: an array of specs, shaped like the batch dimensions it spans.
# A dynamic acquisition that shifts its ky lines from frame to frame is the usual case — the
# aliasing is then incoherent along time, which is what a temporal or low-rank regularizer needs.
# When every element retains the same number of samples, `kspace_data` stays one dense array.
base = falses(64)
base[1:3:64] .= true
masks = [circshift(base, t - 1) for t in 1:8]   # 8 frames

AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(64, 64),
    subsampling=[(:, m) for m in masks]
)
```

#### Unequal sample counts per frame

A dense `kspace_data` array has one length along its sample axis, so it can only hold a per-frame
pattern when every frame selects the same number of samples. When the counts genuinely differ —
a variable-density dynamic acquisition, rather than a shifted mask of fixed size — the measurement
is held one frame at a time in a [`PartitionedKSpace`](@ref) instead. `simulate_acquisition`
produces one automatically:

```@example acqinfo
counts_vary = [copy(base) for _ in 1:4]
counts_vary[2][2] = true            # one extra line in frame 2
counts_vary[3][[5, 7]] .= true      # two extra in frame 3

acq_ragged = AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(64, 64),
    subsampling=[(:, m) for m in counts_vary]
)
```

The frames' full k-space grids are identical, so the encoding operator is a `VCAT` of one
`GetIndex` per frame — no new operator type — and its codomain is an `ArrayPartition`, one block per
frame. Reconstruction is otherwise unchanged: task splitting hands each frame its own slice and its
own spec, a temporal regularizer keeps the whole partitioned operator, and **`reconstruct` still
returns an ordinary dense array**, because the *image* size is the same for every frame.

`PartitionedKSpace` is deliberately not an `AbstractArray`: dimension `ragged_dim` has no single
size, and `size(ksp)` says so rather than inventing a number. Ask for `size(ksp, d)` of a
non-ragged dimension, or reach the frames through `parts(ksp)`. Preprocessing and analysis that
needs a rectangle — prewhitening, coil compression, sensitivity estimation, `pseudo_replica`,
partial-Fourier band detection, GRAPPA/SPIRiT, the `KSpaceToImage` signal model — throws a clear
error for a partitioned acquisition instead of quietly reading the wrong samples. `add_noise`
works, since noise is well defined per frame.

```@docs
PartitionedKSpace
MriReconstructionToolbox.is_partitioned
```

### FFT Shift Conventions

The provided k-space data is assumed to follow standard FFT conventions (DC at center). Sometimes, data may be pre-shifted (DC at first index) or require image-space shifts. Use `shifted_kspace_dims` and `shifted_image_dims` to specify these dimensions:

```@example acqinfo
# Pre-shifted k-space (DC at first index)
ksp = rand(ComplexF32, 64, 64)

info_shifted = AcquisitionInfo(
    ksp;
    is3D=false,
    shifted_kspace_dims=(1, 2)  # Both dimensions pre-shifted
)
```

```@example acqinfo
# Image-space shifts (equivalent to sign alternation in k-space)
info_img_shift = AcquisitionInfo(
    ksp;
    is3D=false,
    shifted_image_dims=(1,)  # First dimension needs shift
)
```

```@example acqinfo
# Named dimensions for shifts
ksp_named = NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 64, 64))

info_named_shift = AcquisitionInfo(
    ksp_named;
    shifted_kspace_dims=(:kx, :ky)
)
```

### Building from raw ISMRMRD data (`MRIBase.RawAcquisitionData`)

Loading `MRIBase.RawAcquisitionData` (from `MRIFiles.RawAcquisitionData`/`MRITestData.load_raw`) — a
weak dependency: this constructor is available once `MRIBase` is loaded — derives the encoding
matrix, k-space layout, coil dimension, subsampling pattern and Cartesian/non-Cartesian dispatch
directly from the ISMRMRD header, instead of hand-assembling arrays from `raw.profiles`:

```julia
using MRIBase   # loads the MRIBase extension
using MriReconstructionToolbox

info = AcquisitionInfo(raw)  # raw::MRIBase.RawAcquisitionData
```

- **`kspace_data`** is a `NamedDimsArray` with `:kx`, `:ky` (and `:kz` for a true 3D acquisition —
  see below), `:coil`, and one batch dimension per ISMRMRD counter that actually varies across
  profiles: `:z` (slices), `:contrast`, `:time` (cardiac/dynamic phase), `:repetition`, `:set`,
  `:average`, in that order. A counter that never varies contributes no dimension.
- **`is3D`** is true only when `kspace_encode_step_2` actually varies (`encodedSize[3] > 1` alone
  is not enough — a 2D multi-slice acquisition also has `encodedSize[3] > 1`, but its slices come
  back as the `:z` batch dimension instead).
- **`subsampling`** reflects exactly the samples present in `raw.profiles` — `kspace_data` is *not*
  zero-padded to `encodedSize` — reduced to `Colon()` (fully sampled), a range (one contiguous
  block, e.g. an asymmetric-echo/partial-Fourier readout), or a boolean mask (e.g. an accelerated
  ky pattern), same as any other `AcquisitionInfo`.
- **`sensitivity_maps`** is not derivable from raw k-space and must be passed as a keyword if
  needed: `AcquisitionInfo(raw; sensitivity_maps)`.

#### FFT-shift derivation

Every profile's `kspace_encode_step_1`/`kspace_encode_step_2` and readout sample index are counted
from 0, with the true k=0 line/sample at `encoding_limits.center` / `head.center_sample` — which
need not be the geometric middle of the encoded axis (partial-Fourier and asymmetric-echo
acquisitions in particular). Consistent with "FFT Shift Conventions" above (DC at
`N ÷ 2 + 1`), this constructor places every sample at `raw_index - center + N ÷ 2` along its axis
before construction, so the result already satisfies that convention and no `shifted_kspace_dims`
is needed. Naively placing sample/line `i` at raw position `i + 1` (ignoring `center`) is exactly
the bug this avoids: it silently shifts the reconstructed image by `center - N ÷ 2` samples along
the affected axis.

The *image* domain needs the matching half of the convention, and the constructor sets that too:
`shifted_image_dims` is `(:x, :y)`, or `(:x, :y, :z)` when `is3D`. MRT's own default is the
plain-DFT one — image origin at index 1 — which round-trips consistently for k-space that MRT
itself simulated, but is not how a scanner stores data: an ISMRMRD acquisition images an object
**centred in the FOV**. Without `shifted_image_dims` every reconstruction from raw data comes out
rolled by half the FOV along each spatial axis. With it, no manual `fftshift` is needed anywhere,
and `estimate_sensitivities(acq)` returns maps on the same (centred) image grid — see
[Coil Sensitivity Estimation](@ref). Maps estimated by hand from a bare k-space array are in MRT's
*default* convention and must be `fftshift`ed before being attached to such an acquisition.

Non-Cartesian raw data (`raw.params["trajectory"] != "cartesian"`) builds a
`NonCartesianAcquisitionInfo` from `MRIBase.trajectory`/`MRIBase.rawdata` for one `slice`/
`contrast` at a time (keywords, both defaulting to `1`) — it does not collect multiple slices,
contrasts or repetitions into batch dimensions the way the Cartesian path does.

## Validation Rules

`AcquisitionInfo` performs comprehensive validation to ensure configuration consistency.

### Dimension Name Validation

K-space `NamedDimsArray` must have specific dimension names:

```@example acqinfo
try
    # ❌ Wrong: using image dimension names
    bad_ksp = NamedDimsArray{(:x, :y, :coil)}(
        rand(ComplexF32, 64, 64, 8)
    )
    AcquisitionInfo(bad_ksp)
catch e
    println("Error: ", e.msg)
end
```

```@example acqinfo
# ✓ Correct: proper k-space dimension names
good_ksp = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 8)
)
info = AcquisitionInfo(good_ksp)
println("Success! Dimensions: ", dimnames(info.kspace_data))
```

### Dimension Order Validation

Dimensions must be in the correct order:

```@example acqinfo
try
    # ❌ Wrong: kx and ky swapped
    bad_order = NamedDimsArray{(:ky, :kx, :coil)}(
        rand(ComplexF32, 64, 64, 8)
    )
    AcquisitionInfo(bad_order)
catch e
    println("Error: ", e.msg)
end
```

```@example acqinfo
# ✓ Correct: kx first, then ky
good_order = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 8)
)
info = AcquisitionInfo(good_order)
println("Success!")
```

### Size Compatibility Validation

Sensitivity maps must match k-space spatial dimensions:

```@example acqinfo
try
    # ❌ Wrong: size mismatch
    ksp_wrong = rand(ComplexF32, 64, 64, 8)
    smaps_wrong = rand(ComplexF32, 128, 128, 8)  # Different size!
    AcquisitionInfo(ksp_wrong; is3D=false, sensitivity_maps=smaps_wrong)
catch e
    println("Error: ", e.msg)
end
```

```@example acqinfo
# ✓ Correct: matching sizes
ksp = rand(ComplexF32, 64, 64, 8)
smaps = rand(ComplexF32, 64, 64, 8)
info = AcquisitionInfo(ksp; is3D=false, sensitivity_maps=smaps)
println("Success! K-space: ", size(ksp)[1:2], ", Smaps: ", size(smaps)[1:2])
```

### Element Type Consistency

K-space and sensitivity maps must have matching element types:

```@example acqinfo
try
    # ❌ Wrong: different precision
    ksp_f32 = rand(ComplexF32, 64, 64, 8)
    smaps_f64 = rand(ComplexF64, 64, 64, 8)  # Different type!
    AcquisitionInfo(ksp_f32; is3D=false, sensitivity_maps=smaps_f64)
catch e
    println("Error: ", e.msg)
end
```

```@example acqinfo
# ✓ Correct: same element type
ksp = rand(ComplexF32, 64, 64, 8)
smaps = rand(ComplexF32, 64, 64, 8)
info = AcquisitionInfo(ksp; is3D=false, sensitivity_maps=smaps)
println("Success! Both are ", eltype(ksp))
```

### Coil Dimension Requirements

When using `NamedDimsArray` with sensitivity maps, `:coil` dimension is required:

```@example acqinfo
try
    # ❌ Wrong: missing :coil dimension
    ksp_no_coil = NamedDimsArray{(:kx, :ky)}(
        rand(ComplexF32, 64, 64)
    )
    smaps_wrong = NamedDimsArray{(:x, :y, :coil)}(
        rand(ComplexF32, 64, 64, 4)
    )
    AcquisitionInfo(ksp_no_coil; sensitivity_maps=smaps_wrong)
catch e
    println("Error: ", e.msg)
end
```

```@example acqinfo
# ✓ Correct: :coil dimension present
ksp_with_coil = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 4)
)
smaps = NamedDimsArray{(:x, :y, :coil)}(
    rand(ComplexF32, 64, 64, 4)
)
info = AcquisitionInfo(ksp_with_coil; sensitivity_maps=smaps)
println("Success! Coil dimension: ", size(info.kspace_data, 3), " coils")
```

### Image Size Inference and Validation

`image_size` is inferred when possible, but can be explicitly provided:

```@example acqinfo
# Inferred from fully sampled k-space
ksp = rand(ComplexF32, 64, 64)
info = AcquisitionInfo(ksp; is3D=false)
println("Inferred image size: ", info.image_size)
```

```@example acqinfo
# Required when subsampling without k-space data
mask = rand(Bool, 128, 128)
info = AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(128, 128),
    subsampling=mask
)
println("Explicit image size: ", info.image_size)
```

```@example acqinfo
try
    # ❌ Wrong: missing image_size with subsampling
    mask_wrong = rand(Bool, 64, 64)
    AcquisitionInfo(
        nothing;
        is3D=false,
        subsampling=mask_wrong
        # Missing image_size!
    )
catch e
    println("Error: ", e.msg)
end
```

### 3D vs 2D Validation

Correct dimensionality must be specified or inferred:

```@example acqinfo
# 3D requires :kz dimension or explicit is3D=true
ksp_3d = NamedDimsArray{(:kx, :ky, :kz, :coil)}(
    rand(ComplexF32, 32, 32, 16, 4)
)
AcquisitionInfo(ksp_3d)  # Automatically infers is3D=true
```

```@example acqinfo
# 2D must NOT have :kz dimension
ksp_2d = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 8)
)
AcquisitionInfo(ksp_2d)  # Automatically infers is3D=false
```

## Updating Existing Configurations

You can create new configurations based on existing ones:

```@example acqinfo
# Start with basic config
ksp = rand(ComplexF32, 64, 64, 8)
info1 = AcquisitionInfo(ksp; is3D=false)
println("Initial config: ", info1)

# Add sensitivity maps
smaps = rand(ComplexF32, 64, 64, 8)
info2 = AcquisitionInfo(info1; sensitivity_maps=smaps)
println("With sensitivity maps:", info2)
```

## Non-Cartesian Acquisitions

Passing a `trajectory` dispatches `AcquisitionInfo` to the non-Cartesian variant. No k-space
data is required to set up the acquisition — this is the normal way to prepare a trajectory for
[`simulate_acquisition`](@ref), with no placeholder array to invent:

```@example acqinfo
using NamedDims

traj = NamedDimsArray{(:coord, :sample, :spoke)}(rand(Float32, 2, 64, 32) .- 0.5f0)
acq_nc = AcquisitionInfo(; trajectory = traj, image_size = (64, 64))
println("k-space data: ", acq_nc.kspace_data)  # nothing until simulated or measured
```

Sensitivity maps can be attached the same way, still with no k-space data:

```@example acqinfo
smaps = coil_sensitivities(64, 64, 4)
acq_nc_smaps = AcquisitionInfo(; trajectory = traj, image_size = (64, 64), sensitivity_maps = smaps)
```

Passing measured k-space data works the same way as the Cartesian case — as the first
(positional) argument:

```@example acqinfo
measured_ksp = rand(ComplexF32, 64, 32)
acq_nc_data = AcquisitionInfo(measured_ksp; trajectory = traj, image_size = (64, 64))
```

`AcquisitionInfo` constructs a `NonCartesianAcquisitionInfo` under the hood. That concrete type
is `public` (documented, stable, dispatchable) but **not exported** — `AcquisitionInfo(...)` is
the advertised, non-expert-facing constructor. Code that needs the concrete type explicitly
(e.g. for a type annotation or `isa` check) imports it or qualifies it:

```julia
using MriReconstructionToolbox: NonCartesianAcquisitionInfo
# or: MriReconstructionToolbox.NonCartesianAcquisitionInfo
```

See `docs/src/high-level/simulation.md` for ready-made trajectory generators
(`radial_trajectory`, `stack_of_stars_trajectory`, `kooshball_trajectory`, `spiral_trajectory`).

## Density Compensation (Non-Cartesian)

Non-Cartesian acquisitions (such as radial, spiral, or arbitrary k-space trajectories) need density compensation factors (DCF) to turn the adjoint NFFT into a usable direct (gridding) reconstruction — the adjoint on its own is *not* an inverse for non-uniformly sampled data. `NonCartesianAcquisitionInfo` (`public`, not exported — see above) holds the trajectory and an optional `dcf` array.

**`acq.dcf` defaults to `nothing`, and `nothing` means no density compensation is applied**: the
encoding operator's adjoint stays the mathematically true adjoint of the forward NFFT. This
matters for anything that assumes `𝒜'` is the true adjoint of `𝒜` — operator-norm estimation,
CG/CGNR, and any algorithm built on that relationship. Reconstructing directly from a
`NonCartesianAcquisitionInfo` you have not run `density_compensation` on therefore does *not*
silently pull in a density-weighted adjoint; call `density_compensation` explicitly when you want
one (e.g. for a quick direct reconstruction), and be aware that once you do, the operator's
adjoint is a density-compensated approximate inverse, not the true adjoint. This is also why the
non-Cartesian trajectory generators and [`simulate_acquisition`](@ref) never apply density
compensation on their own — see `docs/src/high-level/simulation.md`.

You can compute the DCF directly using `density_compensation`:

```julia
# Compute iterative Pipe-Menon DCF (default)
acq_dcf = density_compensation(acq; method = PipeMenonDCF(maxit = 20))

# Or compute geometric Voronoi DCF for 2D trajectories
acq_vor = density_compensation(acq; method = VoronoiDCF())
```

```@docs
density_compensation
DensityCompensation
PipeMenonDCF
VoronoiDCF
```

## Integration with Other Functions

`AcquisitionInfo` is accepted by:

- **Reconstruction**: `reconstruct(acq_info, ...)`
- **Operators**: `get_encoding_operator(acq_info)`, `get_fourier_operator(acq_info)`, etc.
- **Simulation**: `simulate_acquisition(image, acq_info)`

This unified interface simplifies complex workflows and reduces parameter passing errors.


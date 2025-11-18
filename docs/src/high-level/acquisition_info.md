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

## Integration with Other Functions

`AcquisitionInfo` is accepted by:

- **Reconstruction**: `reconstruct(acq_info, ...)`
- **Operators**: `get_encoding_operator(acq_info)`, `get_fourier_operator(acq_info)`, etc.
- **Simulation**: `simulate_acquisition(image, acq_info)`

This unified interface simplifies complex workflows and reduces parameter passing errors.


# Named Dimensions Workflow

Named dimensions provide a type-safe, self-documenting way to work with MRI data. This guide explains how to use NamedDims.jl with MriReconstructionToolbox for clearer, less error-prone code.

```@setup imports
using MriReconstructionToolbox
```

## Why Named Dimensions?

### The Problem with Raw Arrays

Consider this common mistake:

```@example
# Raw arrays - easy to mix up dimensions
ksp = rand(ComplexF32, 256, 256, 8, 20)  # What do these dimensions mean?
                                         # Is it (kx, ky, coil, time)?
                                         # Or (kx, ky, time, coil)?
                                         # Or something else?

# Easy to make mistakes
result = sum(ksp, dims=3)  # Did we mean to average over coils?
                            # Or time? Hard to tell!
nothing # hide
```

### The Solution: Named Dimensions

```@example
using NamedDims

# Named dimensions - crystal clear
ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(
    rand(ComplexF32, 256, 256, 8, 20)
)

# Now it's obvious what we're doing
result = sum(ksp, dims=:coil)  # Clearly averaging over coils
slice = ksp[kx=128, ky=1:256, coil=1, time=:]  # Self-documenting indexing
nothing # hide
```

**Benefits:**
- ✅ **Prevents dimension mix-ups** - Compiler catches dimension errors
- ✅ **Self-documenting code** - Clear what each dimension represents
- ✅ **Safer refactoring** - Reorder dimensions without breaking code
- ✅ **Better IDE support** - Autocomplete shows dimension names

## Getting Started

### Installation

NamedDims.jl is re-exported by MriReconstructionToolbox:

```julia
using MriReconstructionToolbox
# NamedDims is now available
```

### Creating Named Arrays

```@example imports
# Start with regular array
ksp_data = rand(ComplexF32, 256, 256, 8)

# Add names
ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp_data)

# Or in one line
ksp = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 256, 256, 8)
)
nothing # hide
```

## Dimension Naming Conventions

### K-Space Dimensions

Standard names for k-space data: `:kx`, `:ky`, `:kz` or `:z`, `:coil`

```julia
# 2D single-coil
ksp = NamedDimsArray{(:kx, :ky)}(...)

# 2D multi-coil
ksp = NamedDimsArray{(:kx, :ky, :coil)}(...)

# 3D single-coil
ksp = NamedDimsArray{(:kx, :ky, :kz)}(...)

# 3D multi-coil
ksp = NamedDimsArray{(:kx, :ky, :kz, :coil)}(...)
```

!!! note
    The presence of the `:kz` dimension indicates 3D data. If `:kz` is absent, the data is treated as 2D.
    Multi-slice data should include `:z` as the first batch dimension, not `:kz` or `:slice`.

### Image Dimensions

Standard names for image space: `:x`, `:y`, `:z`, `:coil`

```julia
# 2D image
img = NamedDimsArray{(:x, :y)}(...)

# 2D image with coils (before combination)
img = NamedDimsArray{(:x, :y, :coil)}(...)

# 3D image
img = NamedDimsArray{(:x, :y, :z)}(...)
```

### Batch Dimensions

Additional dimensions for multi-dimensional data can be freely named:

```julia
# Dynamic 2D (time series)
ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(...)

# Multi-slice 2D
ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(...)

# Multi-contrast
ksp = NamedDimsArray{(:kx, :ky, :coil, :contrast)}(...)

# Multiple batch dimensions
ksp = NamedDimsArray{(:kx, :ky, :coil, :z, :contrast)}(...)

# 3D dynamic
ksp = NamedDimsArray{(:kx, :ky, :kz, :coil, :time)}(...)
```

## Working with Named Arrays

### Indexing

```@example imports
ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(
    rand(ComplexF32, 256, 256, 8, 30)
)

# Named indexing - clear and safe
first_coil = ksp[coil=1]  # All data from first coil
center_kspace = ksp[kx=128, ky=128, coil=:, time=:]
center_kspace = ksp[kx=128, ky=128]  # same as above
time_series = ksp[kx=128, ky=128, coil=1, time=:]

# Mix names and regular indices
middle_time = ksp[:, :, :, 15]  # Still works
# Or better:
middle_time = ksp[time=15]  # Clearer!

# Ranges
subregion = ksp[kx=100:150, ky=100:150, coil=:, time=1:10]
nothing # hide
```

### Reductions

```julia
# Named dimensions make reductions clear
mean_over_time = mean(ksp, dims=:time)
sum_over_coils = sum(ksp, dims=:coil)
std_spatial = std(ksp, dims=(:kx, :ky))

# Result preserves other dimension names
# If ksp is (:kx, :ky, :coil, :time)
# mean(ksp, dims=:time) is (:kx, :ky, :coil, :time) with size(..., ..., ..., 1)
```

### Dimension Queries

```julia
# Get dimension names
dimnames(ksp)  # Returns (:kx, :ky, :coil, :time)

# Check if dimension exists
:time in dimnames(ksp)  # true
:z in dimnames(ksp)     # false
```

## Using with MriReconstructionToolbox

### Automatic Operator Creation

The package automatically handles named dimensions:

```@repl imports
ksp = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 256, 256, 8)
);

smaps = NamedDimsArray{(:x, :y, :coil)}(
    coil_sensitivities(256, 256, 8)
);

data = AcquisitionInfo(ksp; sensitivity_maps=smaps)

E = get_encoding_operator(ksp; sensitivity_maps=smaps)
E = get_encoding_operator(data)  # Same result

img = E' * ksp;
dimnames(img)
```

### Reconstruction with Named Dimensions

```julia
# High-level reconstruct function
img = reconstruct(AcquisitionInfo(ksp; sensitivity_maps=smaps))

# Result has named dimensions
dimnames(img)  # (:x, :y)

# Can index result by name
center_pixel = img[x=128, y=128]
```

### Dimension Inference

The package infers 2D vs 3D from dimension names:

```julia
# 2D - no :kz dimension
ksp_2d = NamedDimsArray{(:kx, :ky, :coil)}(...)
E_2d = get_encoding_operator(ksp_2d)  # Creates 2D operator

# 3D - has :kz dimension
ksp_3d = NamedDimsArray{(:kx, :ky, :kz, :coil)}(...)
E_3d = get_encoding_operator(ksp_3d)  # Creates 3D operator

# No need to specify is3D=true/false!
```

## Complete Workflow Examples

### Example 1: Basic 2D Reconstruction

```@example
using MriReconstructionToolbox

# 1. Create phantom with names
img_true = NamedDimsArray{(:x, :y)}(shepp_logan(256, 256))

# 2. Create sensitivity maps with names
smaps = NamedDimsArray{(:x, :y, :coil)}(
    coil_sensitivities(256, 256, 8)
)

# 3. Create sampling pattern (regular array is fine)
pdf = VariableDensitySampling(GaussianDistribution(3), 3.0)
mask = create_sampling_pattern(pdf, (256, 256))

# 4. Simulate acquisition
acq = AcquisitionInfo(image_size=(256, 256),
                      sensitivity_maps=smaps,
                      subsampling=mask)
acq = simulate_acquisition(img_true, acq)

# acq.kspace_data is now NamedDimsArray{(:kx, :ky, :coil)}

# 5. Reconstruct
img_recon = reconstruct(acq, L1Wavelet2D(5e-3), verbose=false)

# 6. Compare
dimnames(img_recon)  # (:x, :y) - preserved from input
nothing # hide
```

### Example 2: Dynamic Cardiac Imaging

```@example imports
# Create 2D+time data
nt = 30
img_data = rand(ComplexF32, 256, 256, nt)
img_true = NamedDimsArray{(:x, :y, :cardiac_phase)}(img_data)

# Sensitivity maps (same for all phases)
smaps = NamedDimsArray{(:x, :y, :coil)}(
    coil_sensitivities(256, 256, 8)
)

# Create and simulate acquisition
pdf = VariableDensitySampling(GaussianDistribution(3), 3.0)
mask = create_sampling_pattern(pdf, (256, 256))
acq = AcquisitionInfo(image_size=(256, 256), 
                      sensitivity_maps=smaps,
                      subsampling=mask)
acq = simulate_acquisition(img_true, acq)

# K-space is now (:kx, :ky, :coil, :cardiac_phase)
dimnames(acq.kspace_data)

# Reconstruct with temporal regularization
img_recon = reconstruct(acq, verbose=false)

# Result preserves batch dimension
dimnames(img_recon)  # (:x, :y, :cardiac_phase)

# Easy to extract specific phases
systolic_phase = img_recon[cardiac_phase=5]
diastolic_phase = img_recon[cardiac_phase=20]
nothing # hide
```

### Example 3: Multi-Slice 3D

```julia
# Multi-slice 2D acquisition
nslices = 40
ksp_data = rand(ComplexF32, 256, 256, 8, nslices)
ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(ksp_data)

# 3D sensitivity maps
smaps_data = coil_sensitivities(256, 256, 8)
smaps = NamedDimsArray{(:x, :y, :coil)}(smaps_data)

# Create acquisition
acq = AcquisitionInfo(ksp; sensitivity_maps=smaps)

# Reconstruct
img = reconstruct(acq, verbose=false)

# Result is (:x, :y, :z)
dimnames(img)

# Easy to access individual slices
slice_10 = img[z=10]
middle_slices = img[z=15:25]
nothing # hide
```

## Advanced Techniques

### Dimension Transformations

```julia
# Rename dimensions
ksp = NamedDimsArray{(:kx, :ky, :coil, :time)}(data)
# Convert :time to :cardiac_phase
img = reconstruct(acq)  # Result has same batch dim names
# Can't directly rename in NamedDims, but can rewrap
img_renamed = NamedDimsArray{(:x, :y, :cardiac_phase)}(parent(img))
```

### Combining with Regular Arrays

```julia
# Named array
ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(...)

# Extract regular array when needed
ksp_regular = parent(ksp_named)
ksp_regular = unname(ksp_named)  # Alternative way that works even if input is not NamedDimsArray

# Some functions may require regular arrays
result = some_function(parent(ksp_named))

# Re-wrap result
result_named = NamedDimsArray{(:x, :y)}(result)
```

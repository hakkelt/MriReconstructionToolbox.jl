# Encoding Operators

The main interface for creating MRI encoding operators.

```@docs
get_encoding_operator
```

## Implementation Notes

The `get_encoding_operator` function has two main methods:

1. **Regular arrays**: For when you want to explicitly specify whether data is 2D or 3D
2. **Named dimension arrays**: Automatically infers dimensionality from dimension names

### Method 1: Regular Arrays

```julia
get_encoding_operator(ksp, is3D::Bool; kwargs...)
```

Use this method when working with regular Julia arrays and you want explicit control over dimensionality.

### Method 2: Named Dimensions  

```julia
get_encoding_operator(ksp::NamedDimsArray; kwargs...)
```

This method automatically determines 2D vs 3D based on the presence of the `:kz` dimension and provides better type safety.

## Keyword Arguments

All methods support the following keyword arguments:

- `smaps=nothing`: Sensitivity maps for parallel imaging
- `img_size=nothing`: Image size for subsampled reconstruction  
- `subsampling=nothing`: Subsampling pattern
- `threaded::Bool=true`: Enable multi-threading
- `fast_planning::Bool=false`: Use fast FFTW planning

## Examples

### Basic Usage

```julia
# Single-coil, fully sampled
ksp = rand(ComplexF32, 64, 64)
E = get_encoding_operator(ksp, false)

# Multi-coil parallel imaging
ksp = rand(ComplexF32, 64, 64, 8)
smaps = rand(ComplexF32, 64, 64, 8)
E = get_encoding_operator(ksp, false; smaps=smaps)
```

### With Named Dimensions

```julia
using NamedDims

ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8))
E = get_encoding_operator(ksp; smaps=smaps)
```

### Undersampled Reconstruction

```julia
# Create undersampling mask
mask = rand(Bool, 64, 64)
ksp_sub = ksp[mask, :]

# Encoding operator for undersampled data
E = get_encoding_operator(ksp_sub, false; 
                         smaps=smaps,
                         img_size=(64, 64), 
                         subsampling=mask)
```

# MRI Encoding Operators Documentation

This directory contains the modular implementation of MRI encoding operators for the MriReconstructionToolbox.jl package. The operators model the complete MRI data acquisition process from image space to measured k-space data.

## File Structure

The encoding operators have been organized into the following modules:

### Core Files

- **`types.jl`** - Type definitions and constants for subsampling patterns
- **`fourier_operators.jl`** - Fourier transform operators (image ↔ k-space)
- **`sensitivity_map_operators.jl`** - Parallel imaging sensitivity map operators
- **`subsampling_operators.jl`** - Subsampling operators for accelerated imaging
- **`encoding_operators.jl`** - Main encoding operator interface

### Supporting Files

- **`named_dims_op.jl`** - Named dimensions wrapper for operators
- **`utils.jl`** - Utility functions
- **`regularization.jl`** - Regularization operators for reconstruction

## Overview

### The MRI Forward Model

The MRI encoding operator models the complete data acquisition chain:

```
Image → [Sensitivity Maps] → [Fourier Transform] → [Subsampling] → Observed k-space
```

Mathematically: `y = Γ F S x`

Where:
- `x`: Image to be reconstructed
- `S`: Sensitivity map operator (parallel imaging)
- `F`: Fourier transform operator
- `Γ`: Subsampling operator (undersampling)
- `y`: Observed k-space data

### Key Functions

#### `get_encoding_operator`
The main interface function that creates the complete encoding operator. Two versions:

1. **Regular arrays**: `get_encoding_operator(ksp, is3D; kwargs...)`
2. **Named dimensions**: `get_encoding_operator(ksp::NamedDimsArray; kwargs...)`

#### Supporting Operators

- `get_fourier_operator` - Creates Fourier transform operators
- `get_sensitivity_map_operator` - Creates sensitivity map operators  
- `get_subsampled_fourier_op` - Creates combined Fourier + subsampling operators

## Usage Examples

### Simple 2D Reconstruction
```julia
using MriReconstructionToolbox

# Fully sampled single-coil data
ksp = rand(ComplexF32, 64, 64)
E = get_encoding_operator(ksp, false)
img_recon = E' * ksp  # Adjoint gives IFFT
```

### Parallel Imaging
```julia
# Multi-coil data with sensitivity maps
ksp = rand(ComplexF32, 64, 64, 8)  # 8 coils
smaps = rand(ComplexF32, 64, 64, 8)
E = get_encoding_operator(ksp, false; smaps=smaps)
img_recon = E' * ksp  # Coil combination + IFFT
```

### Accelerated Reconstruction
```julia
# Undersampled parallel imaging
mask = rand(Bool, 64, 64)  # Random sampling mask
ksp_sub = ksp[mask, :]      # Subsampled data
E = get_encoding_operator(ksp_sub, false; 
                         smaps=smaps, 
                         img_size=(64, 64), 
                         subsampling=mask)
img_recon = E' * ksp_sub   # Zero-filled reconstruction
```

### Named Dimensions Interface
```julia
# Type-safe interface with named dimensions
ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8))

E = get_encoding_operator(ksp; smaps=smaps)
img = E' * ksp  # Returns NamedDimsArray{(:x, :y)}
```

## Subsampling Patterns

The toolbox supports various subsampling patterns for accelerated imaging:

### 2D Patterns
- **Boolean masks**: `AbstractArray{Bool,2}` - Arbitrary sampling patterns
- **Linear indices**: `AbstractVector{Int}` - Specific k-space locations  
- **Cartesian indices**: `AbstractVector{CartesianIndex{2}}` - 2D coordinates
- **Separable patterns**: `Tuple{pattern_x, pattern_y}` - Independent sampling per dimension

### 3D Patterns
- **Boolean masks**: `AbstractArray{Bool,3}` - 3D sampling patterns
- **Mixed patterns**: `Tuple{pattern_x, pattern_yz}` - Hybrid sampling
- **Separable patterns**: `Tuple{pattern_x, pattern_y, pattern_z}` - Per-dimension sampling

### 1D Building Blocks
- **Full sampling**: `:` (Colon) - Select all indices
- **Regular undersampling**: `1:R:N` - Every R-th sample
- **Boolean masks**: `AbstractArray{Bool,1}` - Arbitrary 1D patterns
- **Index vectors**: `AbstractVector{Int}` - Specific indices

## Dimension Naming Conventions

### K-space Dimensions
- `:kx`, `:ky`, `:kz` - Spatial frequency dimensions
- `:coil` - Receiver coil dimension
- `:batch`, `:time`, `:contrast` - Additional batch dimensions

### Image Dimensions  
- `:x`, `:y`, `:z` - Spatial dimensions
- Batch dimensions preserved from k-space

## Performance Considerations

### Threading
- Set `threaded=true` (default) for multi-threaded operations
- Automatically scales thread usage based on problem size
- Disable threading with `threaded=false` for small problems

### FFTW Planning
- `fast_planning=false` (default): Uses FFTW.MEASURE for optimal performance
- `fast_planning=true`: Uses FFTW.ESTIMATE for faster setup, potentially slower execution

### Memory Layout
- Operators work in-place when possible
- Batch operations are optimized for memory efficiency
- Named dimensions add minimal overhead

## Error Handling

The operators perform extensive validation:

- **Dimension compatibility** between k-space data and sensitivity maps
- **Element type consistency** across all inputs
- **Required dimension presence** for named dimension arrays
- **Subsampling pattern validity** with respect to image sizes

All validation uses `@argcheck` macros that provide clear error messages.

## Integration with Reconstruction Algorithms

These operators are designed to work seamlessly with optimization packages:

```julia
using ProximalAlgorithms, ProximalOperators

# Set up reconstruction problem
E = get_encoding_operator(ksp_sub; smaps=smaps, img_size=(64, 64), subsampling=mask)
f = LeastSquares(E, ksp_sub)  # Data fidelity term
g = NormL1(1e-3)              # Sparsity regularization

# Solve with ISTA
x0 = E' * ksp_sub             # Initial estimate  
result = ista(f, g, x0)       # Iterative reconstruction
```

This modular design enables flexible composition of MRI reconstruction algorithms while maintaining computational efficiency and type safety.

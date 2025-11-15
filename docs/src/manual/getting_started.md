# Getting Started

This guide will help you get started with MriReconstructionToolbox.jl for MRI reconstruction.

## Installation

Install the package using Julia's package manager:

```julia
using Pkg
Pkg.add("MriReconstructionToolbox")
```

Then load the package:

```julia
using MriReconstructionToolbox
```

## Basic Concepts

### K-space and Image Space

In MRI, data is acquired in the spatial frequency domain (k-space) and must be transformed to image space for visualization. The relationship is governed by the Fourier transform:

- **Forward**: Image → k-space via FFT
- **Inverse**: k-space → Image via IFFT

### Encoding Operators

The encoding operator `E` models the complete MRI acquisition process:

```julia
observed_kspace = E * image
reconstructed_image = E' * observed_kspace  # Adjoint operation
```

## Your First Reconstruction

### Fully Sampled Single-Coil Data

```julia
# Create some synthetic k-space data
ksp = rand(ComplexF32, 64, 64)

# Create encoding operator (just Fourier transform)
E = get_encoding_operator(ksp, false)  # false = 2D (not 3D)

# Reconstruct image using adjoint
img_recon = E' * ksp

# The result is equivalent to:
# img_recon = ifft(ksp)
```

### Multi-Coil Parallel Imaging

```julia
# Multi-coil k-space data (8 coils)
ksp_multi = rand(ComplexF32, 64, 64, 8)

# Sensitivity maps for each coil
smaps = rand(ComplexF32, 64, 64, 8)

# Create encoding operator with sensitivity maps
E = get_encoding_operator(ksp_multi, false; smaps=smaps)

# Reconstruct with coil combination
img_recon = E' * ksp_multi
```

### Undersampled Reconstruction

```julia
# Create undersampling mask
mask = rand(Bool, 64, 64)
mask[32-5:32+5, 32-5:32+5] .= true  # Ensure center is sampled

# Subsample the k-space data
ksp_sub = ksp_multi[mask, :]

# Create encoding operator for undersampled data
E = get_encoding_operator(ksp_sub, false; 
                         smaps=smaps, 
                         img_size=(64, 64), 
                         subsampling=mask)

# Zero-filled reconstruction
img_recon = E' * ksp_sub
```

## Named Dimensions Interface

For better type safety and clearer code, use the named dimensions interface:

```julia
using NamedDims

# Create k-space data with named dimensions
ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8))

# Create encoding operator (automatically detects 2D from dimension names)
E = get_encoding_operator(ksp; smaps=smaps)

# Reconstruct - result has named dimensions too
img = E' * ksp  # Returns NamedDimsArray{(:x, :y)}
```

## Working with Different Data Types

The package supports various numeric types:

```julia
# Complex single precision (most common)
ksp_f32 = rand(ComplexF32, 64, 64)

# Complex double precision
ksp_f64 = rand(ComplexF64, 64, 64)

# The encoding operator preserves the element type
E_f32 = get_encoding_operator(ksp_f32, false)
E_f64 = get_encoding_operator(ksp_f64, false)
```

## Next Steps

- Learn about the [MRI Forward Model](@ref) for deeper understanding
- Explore [Usage Examples](@ref) for more complex scenarios
- Check [Subsampling Patterns](@ref) for different undersampling strategies
- Review [Performance](@ref) tips for large-scale reconstructions

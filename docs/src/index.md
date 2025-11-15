# MriReconstructionToolbox.jl

*A Julia package for MRI reconstruction with encoding operators*

MriReconstructionToolbox.jl provides a comprehensive set of tools for modeling and reconstructing magnetic resonance imaging (MRI) data. The package implements encoding operators that model the complete MRI data acquisition process, from image space to measured k-space data.

## Features

- **Complete MRI Forward Model**: Models the entire acquisition chain including Fourier transforms, sensitivity maps, and subsampling
- **Parallel Imaging Support**: Full support for multi-coil parallel imaging with sensitivity map operators
- **Flexible Subsampling**: Various undersampling patterns for compressed sensing and accelerated imaging
- **Named Dimensions**: Type-safe interface using NamedDims.jl for clear dimension semantics
- **High Performance**: Optimized operators with multi-threading and efficient memory usage
- **Modular Design**: Clean separation of concerns with focused operator modules

## Quick Start

```julia
using MriReconstructionToolbox

# Simple 2D reconstruction
ksp = rand(ComplexF32, 64, 64)
E = get_encoding_operator(ksp, false)
img_recon = E' * ksp  # Adjoint gives IFFT

# Parallel imaging with sensitivity maps
ksp_multi = rand(ComplexF32, 64, 64, 8)  # 8 coils
smaps = rand(ComplexF32, 64, 64, 8)
E = get_encoding_operator(ksp_multi, false; smaps=smaps)
img_recon = E' * ksp_multi  # Coil combination + IFFT
```

## The MRI Forward Model

The encoding operator models the complete MRI acquisition process:

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

## Installation

```julia
using Pkg
Pkg.add("MriReconstructionToolbox")
```

## Package Architecture

The package is organized into focused modules:

- **Types**: Subsampling pattern type definitions
- **Fourier Operators**: Image ↔ k-space transforms
- **Sensitivity Map Operators**: Parallel imaging coil sensitivity modeling
- **Subsampling Operators**: Undersampling pattern handling
- **Encoding Operators**: Main interface combining all components

## Integration

The operators are designed to work seamlessly with optimization packages like ProximalAlgorithms.jl:

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

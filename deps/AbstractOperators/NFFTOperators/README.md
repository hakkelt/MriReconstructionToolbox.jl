# NFFTOperators.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/stable/operators/#NFFT)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/latest/operators/#NFFT)

Non-Uniform Fast Fourier Transform operators for the AbstractOperators.jl framework.

## Overview

NFFTOperators.jl is a specialized extension package for [AbstractOperators.jl](../README.md) that provides linear operators for Non-Uniform Fast Fourier Transforms (NFFT). It wraps the high-performance NFFT.jl library to offer efficient transforms between spatial and non-uniform k-space domains, which is essential for magnetic resonance imaging (MRI), radar, and other applications with non-uniform sampling patterns.

## Relationship to AbstractOperators.jl

NFFTOperators.jl is a **subpackage** of the AbstractOperators.jl ecosystem. While AbstractOperators.jl provides the core abstract operator framework, NFFTOperators.jl extends it with specialized functionality for non-uniform Fourier transforms. This modular design allows users to access advanced NFFT capabilities optimized for applications like MRI reconstruction only when needed.

## Installation

```julia
pkg> add NFFTOperators
```

## GPU Support

NFFTOperators.jl supports GPU execution only with CUDA.jl. The GPU implementation relies on CUDA-specific FFT planning, so the package does not currently support AMDGPU.jl, oneAPI.jl, or OpenCL.jl backends.

## Usage Example

```julia
using NFFTOperators, NFFT

# Define MRI parameters
image_size = (256, 256)

# Create k-space trajectory (normalized to [-0.5, 0.5))
trajectory = rand(2, 100, 50) .- 0.5  # 2D, 100 points per readout, 50 readouts

# Create NFFT operator
nfft_op = NFFTOp(image_size, trajectory)

# Transform image to k-space
x = randn(image_size...)
k_space = nfft_op * x

# Adjoint operation: k-space to image
x_recon = nfft_op' * k_space
```

## Main Features

### 1. **NFFTOp** - Non-Uniform Fast Fourier Transform Operator
Transforms data between uniform spatial domain and non-uniform k-space domain.

- **Flexible sampling patterns**: Supports arbitrary non-uniform sampling trajectories
- **Density compensation**: Built-in support for density compensation functions (DCF)
- **Efficient computation**: Uses the NFFT algorithm for fast transforms
- **Multi-dimensional support**: Works with 1D, 2D, and 3D data
- **Threaded operations**: Optional multi-threaded computation for large datasets

```julia
using NFFTOperators, NFFT

# Create a 2D radial k-space trajectory (normalized to [-0.5, 0.5))
image_size = (128, 128)
n_angles = 50
n_points = 100
trajectory = zeros(2, n_points, n_angles)
for i in 1:n_angles
    angle = (i-1) * π / n_angles
    trajectory[1, :, i] = cos(angle) .* range(-0.5, 0.5, length=n_points)
    trajectory[2, :, i] = sin(angle) .* range(-0.5, 0.5, length=n_points)
end

# Create operator with automatic DCF estimation
nfft_op = NFFTOp(image_size, trajectory)

# Single-threaded for use in parallel regions
nfft_op_st = NFFTOp(image_size, trajectory, threaded=false)
```

### 2. **NormalNFFTOp** - Normal Equations Operator
Provides efficient implementation of the normal equations (A'A) for NFFT operators.

- **Optimized computation**: Faster than applying NFFT' * NFFT sequentially
- **Regularization support**: Easily add regularization terms
- **Iterative algorithms**: Natural fit for iterative reconstruction algorithms
- **Memory efficient**: Reduces memory footprint for large problems

```julia
using NFFTOperators, NFFT

# Standard NFFT
image_size = (128, 128)
trajectory = rand(2, 100, 50) .- 0.5  # Normalized trajectory: 2D, 100 points, 50 readouts
nfft_op = NFFTOp(image_size, trajectory)

# Normal operator (implicit A'A)
normal_op = nfft_op' * nfft_op
```

## License

See LICENSE.md for details.

# WaveletOperators.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/stable/operators/#Wavelet)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/latest/operators/#Wavelet)

Wavelet transform operators for the AbstractOperators.jl framework.

## Overview

WaveletOperators.jl is a specialized extension package for [AbstractOperators.jl](../README.md) that provides linear operators for wavelet transforms. It wraps the high-performance Wavelets.jl library to offer Discrete Wavelet Transforms (DWT) and their inverses (IDWT) as seamlessly integrated `LinearOperator` instances.

## Relationship to AbstractOperators.jl

WaveletOperators.jl is a **subpackage** of the AbstractOperators.jl ecosystem. While AbstractOperators.jl provides the core abstract operator framework, WaveletOperators.jl extends it with domain-specific functionality for wavelet-based signal and image analysis. This modular design allows users to access advanced wavelet capabilities only when needed.

## Installation

```julia
pkg> add WaveletOperators
```

## GPU Support

WaveletOperators.jl is currently CPU-only. Unlike most of the AbstractOperators.jl ecosystem, its operators do not yet support CUDA.jl, AMDGPU.jl, oneAPI.jl, or OpenCL.jl arrays.

## Usage Example

```julia
using WaveletOperators

# Create input signal
x = randn(256)

# Create a wavelet operator with Daubechies-4 wavelet
W = WaveletOp(wavelet(WT.db4), 256)

# Compute the wavelet transform
x_wt = W * x

# Inverse wavelet transform
x_recon = W' * x_wt

# Use in optimization or iterative algorithms
# The operator integrates with AbstractOperators.jl algorithms
```

## Main Features

### 1. **WaveletOp** - Discrete Wavelet Transform Operator
Computes multilevel Discrete Wavelet Transforms using various wavelet bases.

- **Multiple wavelets**: Supports Daubechies, Symlet, Coiflet, Biorthogonal, and many other wavelet families
- **Multi-level transforms**: Applies wavelet decomposition over multiple scales
- **Efficient computation**: Uses optimized wavelet algorithms from Wavelets.jl
- **Invertible operation**: Forward transform (DWT) and adjoint (IDWT) for analysis and synthesis
- **Multi-dimensional support**: Works with 1D, 2D, and multi-dimensional arrays
- **Automatic levels**: Automatically determines optimal number of decomposition levels

```julia
using WaveletOperators, Wavelets

# Create a wavelet operator with Daubechies-4 wavelet
W = WaveletOp(wavelet(WT.db4), 128)

# Apply the forward transform
x = randn(128)
x_wt = W * x

# Inverse (via adjoint)
x_recon = W' * x_wt

# Specify decomposition levels (positional argument)
W = WaveletOp(wavelet(WT.db8), 256, 4)

# Work with 2D arrays
W2D = WaveletOp(wavelet(WT.sym5), (256, 256))
```

## Supported Wavelets

WaveletOperators.jl supports all wavelet families from Wavelets.jl, including:

### Orthogonal Wavelets
- **Daubechies** (`db1`, `db2`, ..., `db20`): Compact support, orthogonal
- **Symlet** (`sym2`, `sym3`, ..., `sym10`): Nearly symmetric, orthogonal
- **Coiflet** (`coif1`, `coif2`, ..., `coif5`): More symmetric, orthogonal

### Biorthogonal Wavelets
- **BiOrthogonal** (`bior1.1`, `bior1.3`, ..., `bior6.8`): Non-orthogonal, better symmetry
- **Reverse BiOrthogonal** (`rbio1.1`, ..., `rbio6.8`): Reverse of biorthogonal

### Discrete Meyer Wavelets
- **dmey**: Discrete Meyer wavelet

```julia
using WaveletOperators, Wavelets

# Daubechies-8
W_db8 = WaveletOp(wavelet(WT.db8), 256)

# Symlet-5
W_sym5 = WaveletOp(wavelet(WT.sym5), 256)

# Biorthogonal 3.5
W_bior = WaveletOp(wavelet(WT.bior3.5), 256)
```

## License

See LICENSE.md for details.

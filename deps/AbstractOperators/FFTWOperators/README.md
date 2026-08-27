# FFTWOperators.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/stable/operators/#FFTW)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/latest/operators/#FFTW)

Fast Fourier Transform operators for the AbstractOperators.jl framework.

## Overview

FFTWOperators.jl is a specialized extension package for [AbstractOperators.jl](../README.md) that provides linear operators for efficient Fourier and frequency-domain transforms. It wraps the high-performance FFTW library to offer Discrete Fourier Transforms (DFT), Real FFTs (RDFT), Inverse Real FFTs (IRDFT), and Discrete Cosine Transforms (DCT) as seamlessly integrated `LinearOperator` instances.

## Relationship to AbstractOperators.jl

FFTWOperators.jl is a **subpackage** of the AbstractOperators.jl ecosystem. While AbstractOperators.jl provides the core abstract operator framework, FFTWOperators.jl extends it with domain-specific functionality for Fourier analysis and frequency-domain operations. This modular design allows users to access high-performance FFT capabilities only when needed.

## Installation

```julia
pkg> add FFTWOperators
```

## GPU Support

FFTWOperators.jl has mixed GPU support:

- All operators except `DCT` and `IDCT` work with GPU arrays on CUDA.jl, AMDGPU.jl, oneAPI.jl, and OpenCL.jl.
- `DCT` and `IDCT` use FFTW CPU plans by default, but loading `AcceleratedDCTs` activates GPU support for those operators. If `AcceleratedDCTs` is not imported, these operators will not work with GPU arrays and should be wrapped with `CpuOperatorWrapper` for use in GPU pipelines.

## Usage Example

```julia
using FFTWOperators

# Create input data
x = randn(64, 64)

# Create a DFT operator
dft_op = DFT(x)

# Compute the Fourier transform
X_fft = dft_op * x

# Use in optimization or iterative algorithms
# The operator integrates with AbstractOperators.jl algorithms
```

## Main Features

### 1. **DFT** - Discrete Fourier Transform
Computes the N-dimensional Discrete Fourier Transform using FFTW's optimized algorithms.

- **Multi-dimensional support**: Works on arbitrary-dimensional arrays
- **Selective dimensions**: Transform over specified dimensions of a multi-dimensional array
- **Multiple normalizations**: UNNORMALIZED, ORTHO, FORWARD, and BACKWARD schemes
- **Customizable planning**: Control FFTW planning flags and time limits
- **Orthogonal properties**: Supports orthogonal DFT operations for optimization algorithms

```julia
using FFTWOperators

# Complex-valued DFT
dft_op = DFT(Complex{Float64}, (10, 10))

# Real-valued input (transforms to complex output)
x = randn(64, 64)
dft_op = DFT(x)
X = dft_op * x  # Fourier transform

# Transform along specific dimensions
dft_op = DFT(x, 1)  # Only along first dimension
```

### 2. **RDFT** - Real FFT
Specialized Fast Fourier Transform for real-valued inputs, exploiting Hermitian symmetry for efficiency.

- **Real input efficiency**: Optimized for real-valued signals, outputs complex values
- **Dimension-selective**: Transform along specific dimensions
- **Conjugate symmetry**: Automatically exploits Hermitian symmetry properties
- **Reduced computation**: Approximately 50% faster than complex FFT for real inputs

```julia
using FFTWOperators

# Real FFT of a real-valued array
rdft_op = RDFT(Float64, (10, 10))

# Apply along a specific dimension
x = randn(100, 10, 10)
rdft_op = RDFT(x, 2)  # Transform along dimension 2
X = rdft_op * x
```

### 3. **IRDFT** - Inverse Real FFT
Transforms complex-valued k-space data back to real-valued spatial domain, inverse of RDFT.

- **Hermitian reconstruction**: Properly reconstructs real values from complex k-space
- **Adjoint operation**: Acts as the adjoint of RDFT for linear algebra operations
- **Energy preserving**: Maintains signal energy in the inverse transform

```julia
using FFTWOperators

# Inverse real FFT - transforms complex-valued input to real-valued output
# Takes the k-space dimension, the desired output spatial dimension, and optionally the transform dimension
irdft_op = IRDFT((51,), 100, 1)  # 51 complex input -> 100 real output along dimension 1
```

### 4. **DCT** - Discrete Cosine Transform
Computes the Discrete Cosine Transform, useful for image compression and spectral analysis.

- **Real-valued I/O**: Works with real-valued signals for many applications
- **Complex support**: Also supports complex-valued transforms
- **Standard DCT**: Implements standard DCT-II (the most common variant)
- **Orthogonal transform**: Invertible with fast IDCT operation

```julia
using FFTWOperators

# Real-valued DCT
dct_op = DCT(Float64, (10, 10))
x = randn(10, 10)
y = dct_op * x  # Cosine transform

# Complex DCT
dct_complex = DCT(Complex{Float64}, (8, 8))

# Inverse DCT
idct_op = IDCT(Float64, (10, 10))
```

#### GPU Support for DCT/IDCT

To use `DCT` and `IDCT` on GPU arrays, load `AcceleratedDCTs` before constructing the operator:

```julia
using FFTWOperators, CUDA
import AcceleratedDCTs # explicitly import to activate GPU support for DCT/IDCT

x_gpu = CUDA.randn(Float32, 10, 10)
dct_gpu = DCT(x_gpu)
y_gpu = dct_gpu * x_gpu

idct_gpu = IDCT(x_gpu)
x_rec_gpu = idct_gpu * y_gpu
```

If `AcceleratedDCTs` is not imported, `DCT` and `IDCT` will use CPU FFTW plans and will not work with GPU arrays. In that case, wrap them with `CpuOperatorWrapper` for use in GPU pipelines:

```julia
using FFTWOperators, CUDA

x_gpu = CUDA.randn(Float32, 10, 10)
dct_gpu = CpuOperatorWrapper(DCT(Float32, (10, 10)); array_type = CuArray{Float32})  # CPU operator wrapped for GPU use
y_gpu = dct_gpu * x_gpu  # GPU in → CPU DCT → GPU out
```

### 5. **Shift** - Frequency Shift Operator
Applies frequency shifts and zero-padding to signals.

- **Frequency shifting**: Shift zero-frequency component to the center
- **Zero-padding**: Efficiently apply padding for zero-padded transforms
- **Composable**: Combines seamlessly with other operators

## License

See LICENSE.md for details.

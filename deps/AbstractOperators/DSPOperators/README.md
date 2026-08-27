# DSPOperators.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/stable/operators/#Convolution)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/latest/operators/#Convolution)

Digital Signal Processing operators for the AbstractOperators.jl framework.

## Overview

DSPOperators.jl is a specialized extension package for [AbstractOperators.jl](../README.md) that provides linear operators for common digital signal processing (DSP) operations. It allows DSP operations such as filtering, convolution, and cross-correlation to be expressed using the unified `LinearOperator` interface, making them seamlessly compatible with optimization algorithms and iterative solvers built on AbstractOperators.

## Relationship to AbstractOperators.jl

DSPOperators.jl is a **subpackage** of the AbstractOperators.jl ecosystem. While AbstractOperators.jl provides the core abstract operator framework and basic linear algebra operators, DSPOperators.jl extends this framework with domain-specific functionality for signal processing. This modular design keeps the base package lightweight while allowing users to opt-in to DSP-specific features only when needed.

## Installation

```julia
pkg> add DSPOperators
```

## GPU Support

DSPOperators.jl follows the generic GPU-array support provided by AbstractOperators.jl. In practice, its operators are intended to work with CUDA.jl, AMDGPU.jl, oneAPI.jl, and OpenCL.jl when the underlying array operations are GPU-compatible.

## Usage Example

```julia
using DSPOperators

# Create signals
signal = randn(100)
filter_kernel = randn(5)

# Create a convolution operator
conv_op = Conv(signal, filter_kernel)

# Apply convolution via operator multiplication
result = conv_op * signal

# Use in optimization or iterative algorithms
# The operator integrates with AbstractOperators.jl algorithms
```

## Main Features

### 1. **Conv** - FFT-based Convolution Operator
Implements 1D convolution using efficient FFT algorithms. Creates a `LinearOperator` that computes the convolution between an input signal and a filter kernel.

- **Efficient computation**: Uses FFTW for fast convolution via the FFT algorithm
- **Type flexibility**: Supports both real and complex-valued signals
- **Automatic padding**: Handles dimension expansion automatically

```julia
using DSPOperators
op = Conv((10,), randn(5))  # Convolve length-10 signal with length-5 kernel
```

### 2. **Filt** - IIR/FIR Filtering Operator
Implements infinite impulse response (IIR) and finite impulse response (FIR) filters for single-input-single-output (SISO) systems.

- **IIR filters**: Specify both numerator (`b`) and denominator (`a`) coefficients
- **FIR filters**: Specify only numerator coefficients (`b`)
- **Coefficient normalization**: Automatically normalizes filter coefficients
- **Stateful operation**: Maintains filter state for sequential filtering

```julia
using DSPOperators
# IIR filter
filt_op = Filt(Float64, (10,), [1.0, 2.0, 1.0], [1.0, -1.0])

# FIR filter
fir_op = Filt(Float64, (10,), [1.0, 0.5, 0.2])
```

### 3. **MIMOFilt** - Multiple-Input-Multiple-Output Filtering
Extends filtering capabilities to multi-channel systems with multiple input and output signals.

- **MIMO systems**: Handles coupled filtering between multiple input/output channels
- **Flexible specifications**: IIR or FIR filters per channel pair
- **Matrix operations**: Works directly with signal matrices where columns represent channels

```julia
using DSPOperators
m, n = 10, 3  # 10 time samples, 3 inputs
# 3×2 MIMO system: 3 inputs, 2 outputs (6 filter vectors total)
B = [[1.; 0.; 1.], [1.; 0.; 1.], [1.; 0.; 1.],
     [1.; 0.; 1.], [1.; 0.; 1.], [1.; 0.; 1.]]
op = MIMOFilt((m, n), B)  # FIR filters
```

### 4. **Xcorr** - Cross-Correlation Operator
Computes cross-correlation between input signals and a reference signal using DSP.jl's optimized routines.

- **Efficient computation**: Leverages DSP.jl's cross-correlation implementation
- **Symmetric operation**: Supports both forward and adjoint operations
- **Flexible mode**: Uses the longest padding mode by default

```julia
using DSPOperators
xcorr_op = Xcorr(Float64, (10,), [1.0, 0.5, 0.2])
```

## License

See LICENSE.md for details.

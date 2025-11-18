# [Image Reconstruction](@id reconstruction)

The `reconstruct` function is the primary high-level interface for MRI image reconstruction from k-space data. It handles both direct (adjoint-based) and iterative reconstruction with regularization, with automatic problem decomposition and performance optimization.

## API Reference

```@docs
reconstruct
Config
```

## Basic Usage

The simplest reconstruction requires only k-space data:

```@setup recon
using MriReconstructionToolbox
using Random
Random.seed!(123)
```

```@example recon
using MriReconstructionToolbox

# Create k-space data
ksp = rand(ComplexF32, 128, 128, 8)
acq = AcquisitionInfo(ksp; is3D=false)

# Direct reconstruction (adjoint of encoding operator)
x_direct = reconstruct(acq)
println("Reconstructed image size: ", size(x_direct))
```

## Reconstruction Workflow

### 1. Direct Reconstruction (No Regularization)

When no regularization is specified, `reconstruct` performs a direct reconstruction using the adjoint of the encoding operator:

```@example recon
# Fully sampled data
ksp_full = rand(ComplexF32, 64, 64, 4)
smaps = rand(ComplexF32, 64, 64, 4)
acq_full = AcquisitionInfo(ksp_full; is3D=false, sensitivity_maps=smaps)

# Direct reconstruction: x = 𝒜' * y
x_direct = reconstruct(acq_full)
println("Direct reconstruction completed")
println("Output type: ", typeof(x_direct))
```

This is equivalent to:
```math
\hat{x} = \mathcal{A}^H y
```
where $\mathcal{A}$ is the encoding operator and $y$ is the k-space data.

### 2. Iterative Reconstruction with Regularization

For undersampled data, add regularization to solve:
```math
\min_x \frac{1}{2}\|\mathcal{A}x - y\|_2^2 + \sum_i \lambda_i R_i(x)
```

```@example recon
# Undersampled acquisition
mask = rand(Bool, 64, 64)
mask[25:40, 25:40] .= true  # Fully sample center
acq_under = AcquisitionInfo(
    nothing;
    is3D=false,
    image_size=(64, 64),
    subsampling=mask,
    sensitivity_maps=smaps
)

# Simulate undersampled data
phantom = rand(ComplexF32, 64, 64)
data = simulate_acquisition(phantom, acq_under)

# Reconstruct with L2 regularization
x_tikhonov = reconstruct(data, Tikhonov(0.01); maxit=20, verbose=false)
println("Tikhonov reconstruction completed")
```

## Configuration Control

The `Config` struct centralizes reconstruction parameters. You can pass configuration either as a `Config` object or as keyword arguments. It can make easy to pass common options together, while still allowing overrides via keywords.

```@example recon
# Method 1: Keyword arguments
x1 = reconstruct(data; maxit=50, tol=1e-5, verbose=false)
nothing # hide
```

```@example recon
# Method 2: Config object
config = Config(maxit=50, tol=1e-5, verbose=false)
x2 = reconstruct(data; config=config)
nothing # hide
```

```@example recon
# Method 3: Override config fields with keywords
config_base = Config(maxit=100, verbose=false)
x3 = reconstruct(data; config=config_base, maxit=25, verbose=false)  # Uses maxit=25
nothing # hide
```

### Important Configuration Options

#### Iteration Control

```@example recon
# Basic iteration parameters
Config(
    maxit=100,     # Maximum iterations
    tol=1e-4,      # Stopping tolerance
    freq=10,       # Print progress every 10 iterations
    verbose=true   # Enable logging
)
```

#### Performance Options

```@example recon
# Threading and performance
Config(
    threaded=true,                          # Enable multi-threading
    exact_opnorm=false,                     # Fast operator norm estimation
    disable_problem_decomposition=false,    # Enable automatic decomposition
    disable_operator_normalization=false    # Enable operator normalization
)
```

#### Normalization

```@docs
NoScaling
BartScaling
MeasurementBasedScaling
```

```@example recon
# Data scaling strategies
config_bart = Config(normalization=BartScaling())
config_none = Config(normalization=NoScaling())
nothing # hide
```

## Algorithm Selection

Choose optimization algorithms based on your problem:

```@example recon
# Single algorithm
x_cg = reconstruct(data, Tikhonov(0.01), FISTA(); maxit=30, verbose=false)
nothing # hide
```

```@example recon
# Tuple of algorithms (tries in order until convergence)
x_auto = reconstruct(
    data,
    Tikhonov(0.01),
    (CG(), FISTA(), ADMM());
    maxit=50,
    verbose=false
)
nothing # hide
```

Common algorithms:
- **CG**: Conjugate Gradient - best for quadratic problems (Tikhonov regularization)
- **FISTA**: Fast Iterative Shrinkage-Thresholding - for L1 regularization
- **ADMM**: Alternating Direction Method of Multipliers - for composite regularization

See [Optimization Algorithms](algorithms.md) for detailed information.

## Multiple Regularization Terms

Combine multiple regularization terms for advanced reconstruction:

```@example recon
# Wavelet sparsity + Total Variation
x_composite = reconstruct(
    data,
    (L1Wavelet2D(0.005), TotalVariation2D(0.002));
    maxit=50,
    verbose=false
)
nothing # hide
```

See [Regularization](regularization.md) for available regularization methods.

## Initial Guess

Provide a custom initial estimate:

```@example recon
# Use direct reconstruction as initial guess
x_init = reconstruct(data; verbose=false)

# Refine with regularization
x_refined = reconstruct(
    data,
    L1Wavelet2D(0.005);
    x₀=x_init,
    maxit=30,
    verbose=false
)
nothing # hide
```

## Advanced Features

### Operator Normalization

By default, the encoding operator is normalized for better convergence:

```@example recon
# Standard (normalized operator)
x_norm = reconstruct(data, Tikhonov(0.01); maxit=20, verbose=false)

# Disable normalization
x_unnorm = reconstruct(
    data,
    Tikhonov(0.01);
    disable_operator_normalization=true,
    maxit=20,
    verbose=false
)
println("Both reconstructions completed")
```

Operator normalization typically improves convergence by ensuring the encoding operator has unit norm, which helps algorithms choose better stepsizes.

### Normal Operator Optimization

For least-squares problems, `reconstruct` can exploit efficient normal operator implementations:

```@example recon
# Standard optimization (enabled by default)
config_opt = Config(disable_normalop_optimization=false)

# Disable for debugging
config_noopt = Config(disable_normalop_optimization=true)
nothing # hide
```

When enabled, instead of computing $\|\mathcal{A}x - y\|_2^2$ directly, it computes $\|\mathcal{A}^H\mathcal{A}x - \mathcal{A}^Hy\|_2^2$, which can be more efficient when $\mathcal{A}^H\mathcal{A}$ has an optimized implementation.

### Output Scaling

Control whether the output is scaled back to the original data range:

```@example recon
# Standard (output is inverse-scaled)
x_scaled = reconstruct(data, Tikhonov(0.01); maxit=20, verbose=false)

# Keep scaled output
x_unscaled = reconstruct(
    data,
    Tikhonov(0.01);
    disable_inverse_scale_output=true,
    maxit=20,
    verbose=false
)

println("Scaled max: ", maximum(abs, x_scaled))
println("Unscaled max: ", maximum(abs, x_unscaled))
```

## Problem Decomposition

For multi-dimensional data (e.g., 2D+time, multi-slice), `reconstruct` automatically decomposes the problem over independent dimensions:

```@example recon
# Multi-slice 2D data
nx, ny, nslices, nc = 32, 32, 5, 4
ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
smaps_ms = rand(ComplexF32, nx, ny, nc, nslices)

acq_ms = AcquisitionInfo(ksp_ms; is3D=false, sensitivity_maps=smaps_ms)

# Automatically decomposes over slice dimension
x_slices = reconstruct(acq_ms; maxit=10, verbose=false)
println("Reconstructed slices: ", size(x_slices))
```

The decomposition:
- Identifies batch dimensions not affected by Fourier transforms or regularization
- Reconstructs each batch element independently
- Utilizes multiple CPU cores for parallel execution
- Combines results into a single output array

To disable decomposition (e.g., for debugging):

```@example recon
x_no_decomp = reconstruct(
    acq_ms;
    disable_problem_decomposition=true,
    maxit=10,
    verbose=false
)
println("Sequential reconstruction completed")
```

See [Problem Decomposition](decomposition.md) for details.

## Custom Progress Logging

Replace the default logging function:

```@example recon
# Custom print function
messages = String[]
custom_print(args...) = push!(messages, string(args...))

config_custom = Config(
    printfunc=custom_print,
    verbose=true
)

x_custom = reconstruct(data, Tikhonov(0.01); config=config_custom, maxit=5)
println("Captured ", length(messages), " log messages")
println("First message: ", messages[1])
```

## Named Dimensions Support

`reconstruct` preserves `NamedDimsArray` metadata:

```@example recon
using NamedDims

# Create named k-space data
ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(
    rand(ComplexF32, 64, 64, 4)
)
smaps_named = NamedDimsArray{(:x, :y, :coil)}(
    rand(ComplexF32, 64, 64, 4)
)

acq_named = AcquisitionInfo(ksp_named; sensitivity_maps=smaps_named)
x_named = reconstruct(acq_named; verbose=false)

println("Output dimensions: ", dimnames(x_named))
```

See [Named Dimensions](nameddims.md) for more information.

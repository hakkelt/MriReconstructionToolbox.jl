# [Image Reconstruction](@id reconstruction)

The `reconstruct` function is the primary high-level interface for MRI image reconstruction from k-space data. It accepts an `AcquisitionInfo` object and an `AbstractReconstructionMethod` (defaulting to `DirectReconstruction()`), with automatic problem decomposition and performance optimization.

## API Reference

```@docs
reconstruct
DirectReconstruction
IterativeReconstruction
Config
```

## Basic Usage

The simplest reconstruction performs direct adjoint reconstruction ($\mathcal{A}^* y$):

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

When no method is explicitly specified, `reconstruct` defaults to `DirectReconstruction()` using the adjoint of the encoding operator:

```@example recon
# Fully sampled data
ksp_full = rand(ComplexF32, 64, 64, 4)
smaps = rand(ComplexF32, 64, 64, 4)
acq_full = AcquisitionInfo(ksp_full; is3D=false, sensitivity_maps=smaps)

# Direct reconstruction: x = 𝒜' * y
x_direct = reconstruct(acq_full, DirectReconstruction())
println("Direct reconstruction completed")
println("Output type: ", typeof(x_direct))
```

This is equivalent to:
```math
\hat{x} = \mathcal{A}^H y
```
where $\mathcal{A}$ is the encoding operator and $y$ is the k-space data.

### 2. Iterative Reconstruction with Regularization

For undersampled data, configure an `IterativeReconstruction` to solve:
```math
\min_x \frac{1}{2}\|\mathcal{A}x - y\|_2^2 + \sum_i \lambda_i R_i(x)
```

```@example recon
# Undersampled acquisition
mask = rand(Bool, 64, 64)
mask[25:40, 25:40] .= true  # Fully sample center
smaps = rand(ComplexF32, 64, 64, 4)
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
x_tikhonov = reconstruct(data, IterativeReconstruction(Tikhonov(0.01)); maxit=20, verbose=false)
println("Tikhonov reconstruction completed")
```

## Configuration Control

The `Config` struct centralizes execution parameters such as iterations, tolerances, normalization, and threading. You can pass configuration either as a `Config` object or as keyword arguments.

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

Specify optimization algorithms via `IterativeReconstruction`:

```@example recon
# Single algorithm
x_fista = reconstruct(data, IterativeReconstruction(Tikhonov(0.01); algorithm=FISTA()); maxit=30, verbose=false)
nothing # hide
```

```@example recon
# Tuple of algorithms (tries in order based on problem structure)
x_auto = reconstruct(
    data,
    IterativeReconstruction(
        Tikhonov(0.01);
        algorithm=(CG(), FISTA(), ADMM())
    );
    maxit=50,
    verbose=false
)
nothing # hide
```

Common algorithms:
- **CG / CGNR**: Conjugate Gradient - best for quadratic problems (Tikhonov / least squares)
- **FISTA**: Fast Iterative Shrinkage-Thresholding - for L1 / sparsity regularization
- **ADMM**: Alternating Direction Method of Multipliers - for composite / multi-term regularization

See [Optimization Algorithms](algorithms.md) and [Reconstruction Methods](methods.md) for detailed information.

## Multiple Regularization Terms

Combine multiple regularization terms for composite regularization:

```@example recon
# Wavelet sparsity + Total Variation
x_composite = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(0.005), TotalVariation2D(0.002));
    maxit=50,
    verbose=false
)
nothing # hide
```

See [Regularization](regularization.md) for available regularization terms.

For additive multi-component models (e.g. low-rank background plus sparse foreground), pass `Component`s into `IterativeReconstruction`; see [Image Decomposition](image_decomposition.md).

## Initial Guess

Provide a custom initial estimate via `x₀`:

```@example recon
# Use direct reconstruction as initial guess
x_init = reconstruct(data; verbose=false)

# Refine with regularization
x_refined = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(0.005));
    x₀=x_init,
    maxit=30,
    verbose=false
)
nothing # hide
```

## Advanced Method Options

### Operator Normalization

By default, the encoding operator is normalized to unit norm for stable step size selection. This can be configured on `IterativeReconstruction`:

```@example recon
# Disable operator normalization
x_unnorm = reconstruct(
    data,
    IterativeReconstruction(Tikhonov(0.01); disable_operator_normalization=true);
    maxit=20,
    verbose=false
)
println("Unnormalized reconstruction completed")
```

### Normal Operator Optimization

For least-squares problems, `IterativeReconstruction` can exploit efficient normal operator implementations $\mathcal{A}^*\mathcal{A}$:

```@example recon
# Disable normal operator optimization for debugging
method_noopt = IterativeReconstruction(
    Tikhonov(0.01);
    disable_normalop_optimization=true
)
nothing # hide
```

### Output Scaling

Control whether the output is scaled back to the original data range:

```@example recon
# Standard (output is inverse-scaled)
x_scaled = reconstruct(data, IterativeReconstruction(Tikhonov(0.01)); maxit=20, verbose=false)

# Keep scaled output
x_unscaled = reconstruct(
    data,
    IterativeReconstruction(Tikhonov(0.01));
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

x_custom = reconstruct(data, IterativeReconstruction(Tikhonov(0.01)); config=config_custom, maxit=5)
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

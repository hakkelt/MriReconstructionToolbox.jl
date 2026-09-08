# [Image Reconstruction](@id reconstruction)

The `reconstruct` function is the primary high-level interface for MRI image reconstruction from k-space data. It accepts an `AcquisitionInfo` object and an `ReconstructionMethod` (defaulting to `DirectReconstruction()`), with automatic task splitting and performance optimization.

## API Reference

```@docs
reconstruct
DirectReconstruction
IterativeReconstruction
ReconstructionConfig
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
x_tikhonov = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 20); verbosity = Silent())
println("L2Image reconstruction completed")
```

## Configuration Control

There are two separate places a parameter can live, and which one it belongs to is decided by
one question: *does it mean anything without knowing the method?*

- **Method parameters** — `maxit`, `tol`, `algorithm`, and everything else only a particular
  method can act on — go to that method's constructor. They are keyword-only there.
- **Run settings** — scaling, output, threading, task splitting — go to `ReconstructionConfig`, or
  straight to `reconstruct` as keywords.

Passing `maxit`, `tol` or `algorithm` to `reconstruct` throws rather than being silently
ignored, which is what happened before this split.

```@example recon
# Method 1: keyword arguments for the run, constructor arguments for the method
x1 = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 50, tol = 1e-5); verbosity = Silent())
nothing # hide
```

```@example recon
# Method 2: a ReconstructionConfig object, reusable across methods
config = ReconstructionConfig(; verbosity = Silent(), scaling = BartScaling())
x2 = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 50, tol = 1e-5); config = config)
nothing # hide
```

```@example recon
# Method 3: override config fields with keywords
config_base = ReconstructionConfig(; verbosity = Verbose())
x3 = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 25); config = config_base, verbosity = Silent())
nothing # hide
```

### Important Configuration Options

#### Iteration Control

```@example recon
# Iteration parameters belong to the method
IterativeReconstruction(
    L2Image(0.01);
    maxit = 100,          # Maximum iterations
    tol = 1e-4,           # Relative stopping tolerance (`nothing` defers to the algorithm)
    algorithm = FISTA(),  # Solver
)
```

`tol` is relative: the absolute threshold handed to the solver is
`max(10*eps, tol * maximum(abs, x₀))`. Setting `maxit = nothing` or `tol = nothing` leaves the
corresponding parameter to the `algorithm` itself, so
`IterativeReconstruction(reg; algorithm = FISTA(maxit = 500), maxit = nothing)` really runs 500
iterations.

#### Output

```@example recon
# Three mutually exclusive output modes
reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Silent())      # nothing at all
reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = ProgressBar()) # one progress bar
reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Verbose(; freq = 1))  # textual log
nothing # hide
```

`verbosity` also accepts `true`/`false` and the symbols `:verbose`, `:progress`, `:silent`.

```@docs
Verbosity
Silent
ProgressBar
Verbose
```

#### Scaling

```@docs
NoScaling
BartScaling
MeasurementBasedScaling
```

```@example recon
# Data scaling strategies
config_bart = ReconstructionConfig(scaling=BartScaling())
config_none = ReconstructionConfig(scaling=NoScaling())
nothing # hide
```

## Algorithm Selection

Specify optimization algorithms via `IterativeReconstruction`:

```@example recon
# Single algorithm
x_fista = reconstruct(data, IterativeReconstruction(L2Image(0.01); algorithm=FISTA(), maxit = 30); verbosity = Silent())
nothing # hide
```

```@example recon
# Tuple of algorithms (tries in order based on problem structure)
x_auto = reconstruct(
    data,
    IterativeReconstruction(
        L2Image(0.01);
        algorithm=(CG(), FISTA(), ADMM()), maxit = 50); verbosity = Silent())
nothing # hide
```

Common algorithms:
- **CG / CGNR**: Conjugate Gradient - best for quadratic problems (L2Image / least squares)
- **FISTA**: Fast Iterative Shrinkage-Thresholding - for L1 / sparsity regularization
- **ADMM**: Alternating Direction Method of Multipliers - for composite / multi-term regularization

See [Optimization Algorithms](algorithms.md) and [Reconstruction Methods](methods.md) for detailed information.

## Multiple Regularization Terms

Combine multiple regularization terms for composite regularization:

```@example recon
# Wavelet sparsity + Total Variation
x_composite = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(0.005), TotalVariation2D(0.002); maxit = 50); verbosity = Silent())
nothing # hide
```

See [Regularization](regularization.md) for available regularization terms.

For additive multi-component models (e.g. low-rank background plus sparse foreground), pass `Component`s into `IterativeReconstruction`; see [Image Decomposition](image_decomposition.md).

## Initial Guess

Provide a custom initial estimate via `x₀`:

```@example recon
# Use direct reconstruction as initial guess
x_init = reconstruct(data; verbosity = Silent())

# Refine with regularization
x_refined = reconstruct(
    data,
    IterativeReconstruction(L1Wavelet2D(0.005); maxit = 30); x₀=x_init, verbosity = Silent())
nothing # hide
```

## Advanced Method Options

### Operator Scaling

By default, the encoding operator is normalized to unit norm for stable step size selection. This can be configured on `IterativeReconstruction`:

```@example recon
# Disable operator normalization
x_unnorm = reconstruct(
    data,
    IterativeReconstruction(L2Image(0.01); disable_operator_normalization=true, maxit = 20); verbosity = Silent())
println("Unnormalized reconstruction completed")
```

### Normal Operator Optimization

For least-squares problems, `IterativeReconstruction` can exploit efficient normal operator implementations $\mathcal{A}^*\mathcal{A}$:

```@example recon
# Disable normal operator optimization for debugging
method_noopt = IterativeReconstruction(
    L2Image(0.01);
    disable_normalop_optimization=true
)
nothing # hide
```

### Output Scaling

Control whether the output is scaled back to the original data range:

```@example recon
# Standard (output is inverse-scaled)
x_scaled = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 20); verbosity = Silent())

# Keep scaled output
x_unscaled = reconstruct(
    data,
    IterativeReconstruction(L2Image(0.01); maxit = 20); disable_inverse_scale_output=true, verbosity = Silent())

println("Scaled max: ", maximum(abs, x_scaled))
println("Unscaled max: ", maximum(abs, x_unscaled))
```

## Task Splitting

For multi-dimensional data (e.g., 2D+time, multi-slice), `reconstruct` automatically splits the task over independent dimensions:

```@example recon
# Multi-slice 2D data
nx, ny, nslices, nc = 32, 32, 5, 4
ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
smaps_ms = rand(ComplexF32, nx, ny, nc, nslices)

acq_ms = AcquisitionInfo(ksp_ms; is3D=false, sensitivity_maps=smaps_ms)

# Automatically splits over slice dimension
x_slices = reconstruct(acq_ms; verbosity = Silent())
println("Reconstructed slices: ", size(x_slices))
```

Task splitting:
- Identifies batch dimensions not affected by Fourier transforms or regularization
- Reconstructs each batch element independently
- Utilizes multiple CPU cores for parallel execution
- Combines results into a single output array

To disable task splitting (e.g., for debugging):

```@example recon
x_no_split = reconstruct(
    acq_ms; disable_task_splitting=true, verbosity = Silent())
println("Sequential reconstruction completed")
```

See [Task Splitting](task_splitting.md) for details.

## Custom Progress Logging

Replace the default logging function:

```@example recon
# Custom print function
messages = String[]
custom_print(args...) = push!(messages, string(args...))

config_custom = ReconstructionConfig(; verbosity = Verbose(; printfunc = custom_print))

x_custom = reconstruct(data, IterativeReconstruction(L2Image(0.01); maxit = 5); config=config_custom)
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
x_named = reconstruct(acq_named; verbosity = Silent())

println("Output dimensions: ", dimnames(x_named))
```

See [Named Dimensions](nameddims.md) for more information.

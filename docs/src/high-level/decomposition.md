# Problem Decomposition

Problem decomposition is an advanced feature that automatically parallelizes reconstruction across independent data dimensions, significantly speeding up processing when you have multiple CPU cores.

## What is Problem Decomposition?

When you have batch dimensions in your data (like multiple slices, time points, or contrasts), and these dimensions don't interact through the Fourier transform or regularization, the reconstruction problem can be **decomposed** into independent sub-problems that can be solved in parallel.

### Simple Example

Imagine you have 20 slices to reconstruct:

**Without decomposition:**
- Reconstruct all 20 slices as one large problem
- Uses 1 CPU core
- Takes 20 time units

**With decomposition:**
- Automatically splits into 20 independent problems
- Each reconstructed on a separate core
- With 8 cores: takes ~2.5 time units
- **~8× speedup!***

*Actual speedup is expected to be lower because even without decomposition, some operations (like FFTs) are multi-threaded.

## When Does Decomposition Happen?

The `reconstruct` function automatically determines if decomposition is beneficial:

```julia
# This will automatically use decomposition if:
# 1. There are batch dimensions (e.g., time, slice)
# 2. The batch dimensions aren't affected by regularization
# 3. decomposition is not disabled
img = reconstruct(acq, regularization)
```

### Conditions for Decomposition

✅ **Decomposition is used when:**
- Data has batch dimensions beyond spatial dimensions and coils
- Regularization doesn't couple the batch dimensions
- You haven't disabled it with `disable_problem_decomposition=true`

❌ **Decomposition is NOT used when:**
- No batch dimensions exist
- Regularization couples batch dimensions (e.g., temporal regularization)
- Explicitly disabled
- No regularization specified (direct reconstruction doesn't benefit)

## Understanding Batch Dimensions

### What are Batch Dimensions?

Batch dimensions are additional dimensions beyond the standard spatial and coil dimensions:

**2D Imaging:**
- Spatial: x, y, and slice if sensitivity maps are 3D
- Coil: coil
- **Batch examples:** [slice,] time, contrast, phase, etc.

**3D Imaging:**
- Spatial: x, y, z
- Coil: coil
- **Batch examples:** time, contrast, phase, etc.

### Examples of Batch Dimensions

**Multi-slice 2D:**
```julia
ksp = rand(ComplexF32, 256, 256, 8, 20)  # (kx, ky, coil, slice)
# Slice is a batch dimension - each slice independent
```

**Dynamic 2D:**
```julia
ksp = rand(ComplexF32, 256, 256, 8, 30)  # (kx, ky, coil, time)
# Time is a batch dimension (if no temporal regularization)
```

**Multi-contrast 3D:**
```julia
ksp = rand(ComplexF32, 128, 128, 64, 8, 4)  # (kx, ky, kz, coil, contrast)
# Contrast is a batch dimension - each contrast independent
```

**Named dimensions version:**
```julia
ksp = NamedDimsArray{(:kx, :ky, :coil, :slice)}(...)
# Automatically identifies :slice as batch dimension
```

## How Regularization Affects Decomposition

### Regularization that ALLOWS Decomposition

These regularizations only affect spatial dimensions:

```julia
# Each time point reconstructed independently
img = reconstruct(acq_dynamic, L1Wavelet2D(5e-3))  # ✅ Decomposed over time

# Each slice reconstructed independently  
img = reconstruct(acq_multislice, TotalVariation2D(1e-3))  # ✅ Decomposed over slices

# Multiple spatial regularizers still allow decomposition
img = reconstruct(acq, (L1Wavelet2D(5e-3), TotalVariation2D(1e-3)))  # ✅ Decomposed
```

### Regularization that PREVENTS Decomposition

These regularizations couple batch dimensions:

```julia
# 3D wavelets couple slice dimension
img = reconstruct(acq_multislice, L1Wavelet3D(5e-3))  # ❌ No decomposition

# Temporal regularization couples time points
img = reconstruct(acq_dynamic, TemporalFourier(1e-2))  # ❌ No decomposition

# Low-rank couples time points
img = reconstruct(acq_dynamic, LowRank(1e-1))  # ❌ No decomposition
```

### Mixed Cases

```julia
# Spatial + Temporal regularization
# Cannot decompose over time (coupled by TemporalFourier)
# But could decompose over other batch dimensions, like slice
reg = (L1Wavelet2D(5e-3), TemporalFourier(1e-2))
img = reconstruct(acq, reg)  # Partial decomposition possible
```

## Controlling Decomposition

### Automatic (Recommended)

Just call reconstruct normally:

```julia
# Automatic decomposition when beneficial
img = reconstruct(acq, L1Wavelet2D(5e-3))
```

The function will:
1. Analyze your data dimensions
2. Check if regularization allows decomposition
3. Automatically parallelize if beneficial
4. Use all available CPU cores

### Manual Control

Disable decomposition if needed:

```julia
# Force sequential reconstruction
img = reconstruct(acq, regularization; 
                 disable_problem_decomposition=true)
```

**When to disable:**
- Debugging (easier to trace single-threaded execution)
- Limited memory (decomposition uses more memory)
- Benchmarking specific aspects

### Choosing Execution Strategy

```julia
# Default: Multi-threading (recommended)
img = reconstruct(acq, reg)

# Can explicitly specify executor
using MriReconstructionToolbox: MultiThreadingExecutor, SequentialExecutor

# Force multi-threading
img = reconstruct(acq, reg; 
                 decomposition_executor=MultiThreadingExecutor())

# Force sequential
img = reconstruct(acq, reg; 
                 decomposition_executor=SequentialExecutor())
```

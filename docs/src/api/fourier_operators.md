# Fourier Operators

Fourier transform operators for converting between image space and k-space.

```@docs
get_fourier_operator
```

## Overview

The Fourier operators implement the discrete Fourier transform (DFT) and its inverse, which form the mathematical foundation for converting between image space and k-space in MRI.

## Mathematical Background

The discrete Fourier transform for MRI is defined as:

**Forward (image → k-space):**
```math
k(u,v) = \sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} I(x,y) e^{-2\pi i (ux/N_x + vy/N_y)}
```

**Inverse (k-space → image):**
```math
I(x,y) = \frac{1}{N_x N_y} \sum_{u=0}^{N_x-1} \sum_{v=0}^{N_y-1} k(u,v) e^{2\pi i (ux/N_x + vy/N_y)}
```

## Implementation Details

### FFTW Integration

The operators use FFTW.jl for high-performance FFT computation:

- **FFTW.MEASURE** (default): Optimal performance with longer planning time
- **FFTW.ESTIMATE**: Faster planning with potentially slower execution

### Multi-threading

- Automatically uses all available threads when `threaded=true`
- Thread count determined by `Threads.nthreads()`
- Can be disabled for small problems to avoid overhead

### Dimension Handling

**2D Data**: FFT applied along dimensions `(1, 2)`
**3D Data**: FFT applied along dimensions `(1, 2, 3)`
**Batch Data**: Additional dimensions preserved (e.g., coils, time points)

## Usage Examples

### Basic 2D Fourier Transform

```julia
# Create some image data
img = rand(ComplexF32, 64, 64)

# Method 1: Specify dimensionality explicitly  
F = get_fourier_operator(img, false)  # false = 2D
ksp = F * img     # Forward FFT
img_recon = F' * ksp  # Inverse FFT

# Method 2: Named dimensions (automatic detection)
img_named = NamedDimsArray{(:x, :y)}(img)
F = get_fourier_operator(img_named)
ksp_named = F * img_named  # Returns NamedDimsArray{(:kx, :ky)}
```

### 3D Fourier Transform

```julia
# 3D image data
img_3d = rand(ComplexF32, 64, 64, 32)

# Explicit 3D
F = get_fourier_operator(img_3d, true)  # true = 3D

# Named dimensions  
img_3d_named = NamedDimsArray{(:x, :y, :z)}(img_3d)
F = get_fourier_operator(img_3d_named)  # Auto-detects 3D from :z dimension
```

### Multi-coil Data

```julia
# Multi-coil k-space data (FFT applied to each coil)
ksp_multi = rand(ComplexF32, 64, 64, 8)  # 8 coils
F = get_fourier_operator(ksp_multi, false)

# Transform all coils simultaneously
img_multi = F' * ksp_multi  # Shape: (64, 64, 8)
```

### Performance Tuning

```julia
# Fast planning for prototyping
F_fast = get_fourier_operator(ksp, false; fast_planning=true)

# Disable threading for small problems  
F_single = get_fourier_operator(ksp, false; threaded=false)

# Optimal performance for production (default)
F_optimal = get_fourier_operator(ksp, false)
```

## Dimension Name Mappings

When using named dimensions, the operators automatically map dimension names:

| Image Space | K-space     |
|-------------|-------------|
| `:x`        | `:kx`       |
| `:y`        | `:ky`       |
| `:z`        | `:kz`       |
| `:coil`     | `:coil`     |
| `:time`     | `:time`     |
| `:slice`    | `:slice`    |

## Error Conditions

The function validates inputs and throws descriptive errors for:

- Missing required dimensions (`:kx`, `:ky` for k-space data)
- Incorrect dimension ordering
- Dimension name conflicts

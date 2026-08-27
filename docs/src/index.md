# MriReconstructionToolbox.jl

*A comprehensive Julia package for MRI reconstruction*

MriReconstructionToolbox.jl provides everything you need to reconstruct images from MRI k-space data, from simple FFT-based reconstruction to advanced compressed sensing with sophisticated regularization.

## Installation

**Note:** This package is not yet registered in the Julia General registry because it needs enhancements to upstream packages. These changes are currently under pull requests, and hopefully will be merged soon. Installation requires adding dependencies from GitHub repositories:

```julia
using Pkg

# Add the package from GitHub
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="FFTWOperators")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="DSPOperators")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="NFFTOperators")
Pkg.add(url="https://github.com/hakkelt/AbstractOperators.jl", subdir="WaveletOperators")
Pkg.add(url="https://github.com/hakkelt/ProximalCore.jl")
Pkg.add(url="https://github.com/hakkelt/ProximalOperators.jl")
Pkg.add(url="https://github.com/hakkelt/ProximalAlgorithms.jl")
Pkg.add(url="https://github.com/hakkelt/StructuredOptimization.jl")
Pkg.add(url="https://github.com/hakkelt/MriReconstructionToolbox.jl")
```

## What This Package Does

MriReconstructionToolbox.jl solves the MRI reconstruction inverse problem:

```
Given: k-space measurements (undersampled, multi-coil)
Find: Image that best explains the measurements
```

The package provides:
- **Complete MRI Forward Model**: Models the entire acquisition chain
- **Flexible Regularization**: Multiple methods for different image properties
- **Efficient Algorithms**: Many iterative solvers from [ProximalAlgorithms.jl](https://github.com/hakkelt/ProximalAlgorithms.jl)
- **High-Level Interface**: Simple `reconstruct()` function for common tasks
- **Low-Level Control**: Direct operator access for custom algorithms

## Features

- ✅ **Complete MRI Forward Model** - Fourier transform + sensitivity maps + subsampling
- ✅ **Parallel Imaging** - Multi-coil reconstruction with sensitivity maps
- ✅ **Compressed Sensing** - Advanced undersampling and regularization
- ✅ **Multiple Regularizers** - Sparsity, wavelets, total variation, low-rank
- ✅ **Image Decomposition** - Additive multi-component reconstruction (e.g. low-rank + sparse)
- ✅ **Fast Algorithms** - FISTA, ADMM, Conjugate Gradient
- ✅ **Auto-Parallelization** - Automatic decomposition over batch dimensions
- ✅ **Named Dimensions** - Type-safe interface prevents dimension errors
- ✅ **Simulation Tools** - Built-in phantoms and sampling patterns
- ✅ **High Performance** - Multi-threaded FFTs and optimized operators

## Quick Start

### Simulation Example

#### Shepp-Logan Phantom and Noisy Observation

```@setup imports
using MriReconstructionToolbox
using GeometricMedicalPhantoms
using MIRTjim: jim
using Plots
```

```@example imports
using MriReconstructionToolbox
using MIRTjim: jim

nx, ny, nc = 256, 256, 8
xᵍᵗ = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
noise_level = 0.03f0
x = xᵍᵗ + noise_level * randn(ComplexF32, nx, ny)
p1 = jim(xᵍᵗ; title = "Ground truth")
p2 = jim(x; title = "Noisy image")
jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("shepp_logan_noisy.png"); nothing # hide
```

![shepp_logan_noisy.png](shepp_logan_noisy.png)

#### Coil Sensitivity Maps

```@example imports
smaps = coil_sensitivities(nx, ny, nc)
jim(smaps; title = "Coil sensitivity maps", nrow=1, size = (1400, 200))
savefig("coil_sensitivity_maps.png"); nothing # hide
```

![coil_sensitivity_maps.png](coil_sensitivity_maps.png)

#### k-space Undersampling Pattern

```@example imports
using Plots

pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
W = MriReconstructionToolbox.construct_weights(pdf, (nx,))
p1 = plot(W; title = "1D Sampling weights", legend = false)

pattern = create_sampling_pattern(pdf, (nx, ny))
p2 = jim(to_displayable_mask(pattern, (nx, ny)); title = "Sampling pattern")

jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("sampling_pattern.png"); nothing # hide
```

![sampling_pattern.png](sampling_pattern.png)

#### Simulation of Acquisition

```@example imports
# Create acquisition info that contains every knowledge about the acquisition, but no actual data yet
acq_info = AcquisitionInfo(
   is3D=false,
   image_size=(nx, ny),
   subsampling=pattern,
   sensitivity_maps=smaps)

# Simulate k-space acquisition
data = simulate_acquisition(x, acq_info)
```

### Reconstruction Examples

#### Direct Reconstruction via Adjoint

```@example imports
reconstruct(data, verbose=false); # hide
x̂_direct = reconstruct(data)
p1 = jim(x̂_direct; title = "Direct reconstruction")
p2 = jim(abs.(x̂_direct - xᵍᵗ); title = "Error map")
jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("direct_reconstruction.png"); nothing # hide
```

![direct_reconstruction.png](direct_reconstruction.png)

#### Compressed Sensing Reconstruction with Wavelet Regularization

```@example imports
reg = L1Wavelet2D(0.01f0)
reconstruct(data, reg; maxit=3, verbose=false) # hide
x̂_cs = reconstruct(data, reg; maxit=50)
p1 = jim(x̂_cs; title = "CS Reconstruction")
p2 = jim(abs.(x̂_cs - xᵍᵗ); title = "Error map")
jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("cs_reconstruction.png"); nothing # hide
```

![cs_reconstruction.png](cs_reconstruction.png)

#### Custom Reconstruction With Low-Level Interface

```@example imports
# Prepare encoding operator
ℳ = get_subsampling_operator(data)
ℱ = get_fourier_operator(data)
𝒮 = get_sensitivity_map_operator(data)
𝒜 = ℳ * ℱ * 𝒮
```

```@example imports
# Get k-space data and direct reconstruction
b = data.kspace_data
x̂ = 𝒜' * b # direct reconstruction as adjoint operation
p1 = jim(x̂; title = "Direct reconstruction")
p2 = jim(abs.(x̂ - xᵍᵗ); title = "Error map")
jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("direct_reconstruction_lowlevel.png"); nothing # hide
```

![direct_reconstruction_lowlevel.png](direct_reconstruction_lowlevel.png)

```@example imports
# Set up and solve custom optimization problem with StructuredOptimization.jl
v = Variable(x̂); # use direct reconstruction as initial guess
𝒲 = WaveletOp(ComplexF32, wavelet(WT.db4), (nx, ny))
alg = FISTA(maxit=50, verbose=true, freq=5)
x̂_custom, it = @minimize ls(𝒜 * v - b) + 0.01 * norm(𝒲 * v, 1) with alg

# Visualize results
println("Reconstruction completed in $it iterations.")
p1 = jim(~x̂_custom; title = "CS Reconstruction")
p2 = jim(abs.(~x̂_custom - xᵍᵗ); title = "Error map")
jim(p1, p2; layout = (1, 2), size = (700, 300))
savefig("custom_cs_reconstruction.png"); nothing # hide
```

![custom_cs_reconstruction.png](custom_cs_reconstruction.png)

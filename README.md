# MriReconstructionToolbox.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://hakkelt.github.io/MriReconstructionToolbox.jl/)

_A comprehensive Julia package for MRI reconstruction_

## What is this package?

MriReconstructionToolbox.jl provides everything you need to reconstruct images from MRI data. Whether you're working with simple single-coil acquisitions or complex parallel imaging with advanced regularization, this toolbox has you covered.

**Perfect for:**
- Researchers developing new MRI reconstruction methods
- Students learning about MRI physics and reconstruction
- Engineers prototyping imaging pipelines
- Anyone working with MRI k-space data

## Key Features

✨ **Complete MRI Forward Model** - Models the entire acquisition process from image to k-space  
🔧 **High-Level Interface** - Simple `reconstruct()` function gets you started in seconds  
🎯 **Low-Level Control** - Fine-grained operator access for custom algorithms  
🚀 **Parallel Imaging** - Full support for multi-coil data with sensitivity maps  
⚡ **High Performance** - Multi-threaded operations and optimized FFTs  
🎨 **Multiple Regularization** - Sparsity, wavelets, total variation, low-rank, and more  
📐 **Named Dimensions** - Type-safe interface prevents dimension mix-ups  
🔬 **Simulation Tools** - Built-in phantoms and sampling pattern generators

## Quick Start

### Installation

**Note:** This package is not yet registered in the Julia General registry because it needs enhancements to upstream packages. These changes are currently under pull requests, and hopefully will be merged soon. Installation requires adding dependencies from GitHub repositories:

```julia
using Pkg

# Add the package from GitHub
Pkg.add(url="https://github.com/hakkelt/OperatorCore.jl")
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

### Your First Reconstruction

```julia
using MriReconstructionToolbox

# Load your k-space data (or create synthetic data)
ksp = rand(ComplexF32, 128, 128)  # Single-coil k-space data
acq = AcquisitionInfo(ksp, is3D=false)

# Reconstruct with one function call
img = reconstruct(acq)

# That's it! You have your image.
```

### Example with Parallel Imaging

```julia
# Multi-coil k-space data
ksp = rand(ComplexF32, 128, 128, 8)  # 8 receiver coils

# Coil sensitivity maps
smaps = rand(ComplexF32, 128, 128, 8)

# Reconstruct with automatic coil combination
acq = AcquisitionInfo(ksp; is3D=false, sensitivity_maps=smaps)
img = reconstruct(acq)
```

### Advanced: Compressed Sensing Reconstruction

```julia
# Multi-coil k-space data
ksp = rand(ComplexF32, 128, 128, 8)

# Undersampled k-space data
pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
pattern = create_sampling_pattern(pdf, (128, 128))
ksp_sub = ksp[pattern..., :]

# Set up acquisition info with subsampling
acq = AcquisitionInfo(ksp_sub;
                      image_size=(128, 128),
                      sensitivity_maps=smaps,
                      subsampling=pattern)

# Reconstruct with wavelet sparsity regularization
img = reconstruct(acq, L1Wavelet2D(5e-3))
```

## Documentation

📚 **[Full Documentation](https://hakkelt.github.io/MriReconstructionToolbox.jl/)** - Comprehensive guides and API reference

**Quick Links:**
- [Getting Started Guide](https://hakkelt.github.io/MriReconstructionToolbox.jl/manual/getting_started/) - First steps with the package
- [MRI Forward Model](https://hakkelt.github.io/MriReconstructionToolbox.jl/manual/forward_model/) - Understanding the physics
- [Regularization Options](https://hakkelt.github.io/MriReconstructionToolbox.jl/manual/regularization/) - Available reconstruction methods
- [Simulation Tools](https://hakkelt.github.io/MriReconstructionToolbox.jl/manual/simulation/) - Creating synthetic data

## Design Philosophy

**Beginner-Friendly, Expert-Powerful**

The package provides two levels of interface:

1. **High-Level API** - The `reconstruct()` function handles all the complexity for you. Just provide your data and optional regularization.

2. **Low-Level API** - Direct access to encoding operators, optimization primitives, and algorithmic building blocks for maximum flexibility.

You can start simple and progressively unlock more control as your needs grow.

## Core Concepts

### The MRI Forward Model

MRI reconstruction solves an inverse problem. The forward model describes how images become k-space data:

```
Image → [Coil Sensitivities] → [Fourier Transform] → [Subsampling] → K-space Data
```

This package provides operators for each step, which can be combined or used individually.

### Named Dimensions

Avoid dimension confusion with named dimensions:

```julia
# Dimensions are labeled for clarity
ksp = NamedDimsArray{(:kx, :ky, :coil)}(ksp_data)

# The package knows which dimensions to FFT
E = get_encoding_operator(ksp)  # Automatically detects structure
```

## Project Status

This package is under active development. The core functionality is stable, but the API may evolve. Several dependencies are currently available only via GitHub (not yet in the Julia General registry).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{MriReconstructionToolbox,
  author = {Hakkel, Tamás},
  title = {MriReconstructionToolbox.jl: A Julia Package for MRI Reconstruction},
  year = {2024},
  url = {https://github.com/hakkelt/MriReconstructionToolbox.jl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

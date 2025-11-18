# Regularization

Regularization is essential for reconstructing high-quality images from undersampled k-space data. This page explains the available regularization methods and how to use them.

## Why Regularization?

When k-space is undersampled (as in compressed sensing or parallel imaging), the reconstruction problem becomes **ill-posed** - there are many possible images that could have produced the observed data. Regularization adds prior knowledge about what "good" images look like to guide the reconstruction toward a unique, high-quality solution.

## Understanding the Math

For those interested in the mathematical details, the reconstruction solves:

```
minimize  (1/2)‖E·x - y‖₂² + ∑ᵢ λᵢ·Rᵢ(x)
```

Where:
- `E` is the encoding operator (Fourier + sensitivity + subsampling)
- `x` is the image to reconstruct
- `y` is the observed k-space data
- `‖E·x - y‖₂²` is the data fidelity term
- `Rᵢ(x)` are the regularization terms
- `λᵢ` are the regularization parameters

The first term ensures the reconstruction is consistent with observed data. The regularization terms encode prior knowledge about image properties.

## Available Regularization Methods

```@setup imports
using MriReconstructionToolbox
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)
```

The code snippets in the following sections assume that `MriReconstructionToolbox` and `MIRTjim` are already imported. `MIRTjim` is a convenience wrapper around `Plots.jl` for displaying multidimensional images. Also, assume you have an `AcquisitionInfo` object `acq` representing your k-space data and acquisition settings for simulated Shepp-Logan phantom:

```@example imports
using MriReconstructionToolbox
using MIRTjim: jim

# Simulate 2D acquisition
x = shepp_logan(128, 128)
x_noisy = x + 0.02f0 * randn(ComplexF32, 128, 128)
smaps = coil_sensitivities(128, 128, 8)
pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05)
pattern = create_sampling_pattern(pdf, (128, 128))
acq_full = AcquisitionInfo(
    is3D=false, 
    image_size=(128, 128), 
    subsampling=pattern, 
    sensitivity_maps=smaps
)
data = simulate_acquisition(x_noisy, acq_full)

# Simulate 3D acquisition
x3d = shepp_logan(64, 64, 32)
smaps3d = coil_sensitivities(64, 64, 32, 8)
subsampling3d = create_sampling_pattern(
    VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), 
    (64, 64, 32)
)
acq3d = AcquisitionInfo(
    image_size=(64, 64, 32), 
    sensitivity_maps=smaps3d,
    subsampling=subsampling3d,
)
data3d = simulate_acquisition(x3d, acq3d)

jim(x3d; title="Shepp-Logan Phantom (3D)", size=(800,400))
savefig("shepp_logan_phantom_3d.png"); nothing # hide
```

![shepp_logan_phantom_3d.png](shepp_logan_phantom_3d.png)

### Image Domain Regularization

#### Tikhonov (L2) Regularization

The simplest form of regularization, penalizing large pixel values:

```@docs
Tikhonov
```

**When to use:**
- Noise reduction without strong assumptions about image structure
- As a baseline for comparison with other methods, especially for parallel imaging
- Fast and simple regularization (it is computationally cheap and can be optimized with Conjugate Gradient)

**Example:**
```@example imports
img₁ = reconstruct(data, Tikhonov(1e-1), verbose=false)
img₂ = reconstruct(data, Tikhonov(1e-6), verbose=false)
p1 = jim(img₁; title="Tikhonov λ=1e-1")
p2 = jim(img₂; title="Tikhonov λ=1e-6")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("tikhonov_regularization.png"); nothing # hide
```

![tikhonov_regularization.png](tikhonov_regularization.png)

#### L1 Regularization

Promotes sparse images (many pixels close to zero):

```@docs
L1Image
```

**When to use:**
- Images that are naturally sparse (e.g., angiography) and you want to suppress small values

**Example:**
```@example imports
img₁ = reconstruct(data, L1Image(1e-2), verbose=false)
img₂ = reconstruct(data, L1Image(1e-5), verbose=false)
p1 = jim(img₁; title="L1Image λ=1e-2")
p2 = jim(img₂; title="L1Image λ=1e-5")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("l1image_regularization.png"); nothing # hide
```

![l1image_regularization.png](l1image_regularization.png)

### Wavelet Domain Regularization

#### 2D Wavelet Sparsity

Promotes sparsity in the wavelet domain:

```@docs
L1Wavelet2D
```

**When to use:**
- Natural images with structure at multiple scales
- Most MRI applications (anatomy has multi-scale features)
- Standard compressed sensing reconstruction

**Parameters:**
- `λ`: Regularization strength (try 1e-3 to 1e-2)
- `wavelet`: Wavelet type (default: Daubechies, also try Haar, etc.)
- `levels`: Number of decomposition levels (default: 4)

**Example:**
```@example imports
reg = L1Wavelet2D(1e-3)
example_img = rand(ComplexF32, 128, 128)
op = get_operator(reg, example_img)
transformed = op * x_noisy
p1 = jim(transformed; title="Wavelet Coefficients")
img = reconstruct(data, reg, verbose=false)
p2 = jim(img; title="L1Wavelet2D Reconstruction")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("l1wavelet2d_regularization.png"); nothing # hide
```

![l1wavelet2d_regularization.png](l1wavelet2d_regularization.png)

**Options for `L1Wavelet2D`:**
- `wavelet`: Specify wavelet type (e.g., `WT.haar`, `WT.db4`)
- `levels`: Number of decomposition levels (default: 4)

```@example imports
reg_haar = L1Wavelet2D(1e-2; wavelet=WT.haar)
op_haar = get_operator(reg_haar, example_img)
transformed_haar = op_haar * x_noisy
img_haar = reconstruct(data, reg_haar, verbose=false)
p1 = jim(transformed_haar; title="Haar Coefficients")
p2 = jim(img_haar; title="Haar Reconstruction")

reg_level8 = L1Wavelet2D(1e-3; levels=8)
op_level8 = get_operator(reg_level8, example_img)
transformed_level8 = op_level8 * x_noisy
img_level8 = reconstruct(data, reg_level8, verbose=false)
p3 = jim(transformed_level8; title="Level 8 Coefficients")
p4 = jim(img_level8; title="Level 8 Reconstruction")
jim(p1, p2, p3, p4; layout=(2,2), size=(800,700))
savefig("l1wavelet2d_options.png"); nothing # hide
```

![l1wavelet2d_options.png](l1wavelet2d_options.png)

#### 3D Wavelet Sparsity

For volumetric / multislice data, promotes sparsity in 3D wavelet domain:

```@docs
L1Wavelet3D
```

**When to use:**
- 3D acquisitions or multi-slice 2D data
- When you want to exploit 3D structure

**Example:**
```@example imports
reg3d = L1Wavelet3D(1e-3)

op = get_operator(reg3d, rand(ComplexF32, 64, 64, 32))
transformed = op * x3d
jim(abs.(transformed); title="3D Wavelet Coefficients", size=(800,400))
savefig("l1wavelet3d_coefficients.png"); nothing # hide
```

![l1wavelet3d_coefficients.png](l1wavelet3d_coefficients.png)

### Total Variation

#### 2D Total Variation

Promotes piecewise-constant images by penalizing rapid changes:

```@docs
TotalVariation2D
```

**When to use:**
- Images with sharp edges and flat regions
- Brain imaging with gray/white matter boundaries
- When you want strong edge preservation

**Example:**
```@example imports
reg = TotalVariation2D(1e-3)
op = get_operator(reg, example_img)
transformed = op * x_noisy
img = reconstruct(data, reg, verbose=false)
p1 = jim(transformed[:,:,1]; title="Δx Coefficients")
p2 = jim(transformed[:,:,2]; title="Δy Coefficients")
p3 = jim(img; title="TotalVariation2D Reconstruction")
jim(p1, p2, p3; layout=(1, 3), size=(900,250))
savefig("totalvariation2d_coefficients.png"); nothing # hide
```

![totalvariation2d_coefficients.png](totalvariation2d_coefficients.png)

**Practical tip:** TV can create a "cartoon-like" appearance. Use lower λ values (1e-4 to 5e-3) to preserve texture.

#### 3D Total Variation

For volumetric / multislice data, promotes piecewise-constant structure in 3D:

```@docs
TotalVariation3D
```

### Temporal Regularization

#### Temporal Fourier Sparsity

For dynamic imaging, promotes sparsity in the temporal Fourier domain:

```@docs
TemporalFourier
```

**When to use:**
- Dynamic or cine imaging
- Cardiac MRI
- DCE-MRI (dynamic contrast enhanced)
- When motion is periodic or smoothly varying

**Example:**
```julia
img = reconstruct(acq, TemporalFourier(1e-2, time_dim=4))
```

**Practical tip:** This works best when temporal changes are smooth or periodic. For irregular motion, consider low-rank methods instead.

### Low-Rank Regularization

#### Nuclear Norm

Promotes low-rank structure in dynamic data:

```@docs
LowRank
RankLimit
```

**When to use:**
- Dynamic imaging with temporal correlations
- Background suppression in DCE-MRI
- Data with strong spatiotemporal correlations
- When images share common features across time

**Example:**
```julia
# Dynamic series with low-rank structure
img = reconstruct(acq_dynamic, LowRank(1e-1))
```

**Practical tip:** Low-rank methods can be computationally expensive. Use for datasets where temporal correlations are strong.

## Combining Multiple Regularizers

You can combine multiple regularization terms to exploit different image properties simultaneously:

```julia
# Comprehensive regularization for dynamic imaging
reg = (
    L1Wavelet2D(5e-3),      # Spatial sparsity
    TotalVariation2D(1e-3),  # Edge preservation
    TemporalFourier(2e-2)    # Temporal smoothness
)
img = reconstruct(acq_dynamic, reg)
```

**When to combine:**
- Wavelet + TV: Exploit both multi-scale structure and edge preservation
- Spatial + Temporal: Regularize both space and time dimensions
- Multiple spatial regularizers: When images have complex structure

## Choosing Regularization Parameters

The regularization parameter λ controls the trade-off between data fidelity and regularization:

- **Too small (λ → 0)**: Noisy, artifacts remain
- **Too large (λ → ∞)**: Over-smoothed, loss of detail
- **Just right**: Balance between noise/artifact suppression and detail preservation

### Practical Guidelines

**Starting values by regularization type:**
- Tikhonov: `1e-5` to `1e-3`
- L1Image: `1e-4` to `1e-2`
- L1Wavelet: `1e-3` to `1e-2`
- TotalVariation: `1e-4` to `5e-3`
- TemporalFourier: `1e-2` to `1e-1`
- LowRank: `1e-2` to `1`

**Adjustment strategy:**
1. Start with the suggested value
2. If too noisy/aliased → increase λ
3. If too smooth/blurry → decrease λ
4. Typical range: adjust by factors of 2-5

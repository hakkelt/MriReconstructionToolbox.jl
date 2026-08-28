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
using GeometricMedicalPhantoms
using MIRTjim: jim

# Simulate 2D acquisition
x = create_shepp_logan_phantom(128, 128, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
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
x3d = create_shepp_logan_phantom(64, 64, 32; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
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

**Practical tip:** This works best when temporal changes are smooth or periodic. For irregular motion, consider temporal total variation or low-rank methods instead.

#### Temporal Total Variation

For dynamic imaging with irregular or non-periodic motion, penalizes the frame-to-frame differences:

```@docs
TemporalTotalVariation
```

**When to use:**
- Free-breathing and real-time acquisitions, where the temporal Fourier assumption of periodicity fails
- Contrast dynamics that are piecewise smooth in time (DCE-MRI, first-pass perfusion)
- As the sparse part of an L+S model (see [Image Decomposition](image_decomposition.md))

**Example:**
```julia
img = reconstruct(acq_dynamic, TemporalTotalVariation(2e-2; time_dim = 3))
```

**Practical tip:** Like spatial TV, this term uses a non-tight operator, so reconstruction falls back to ADMM. It is the temporal counterpart of [`TotalVariation2D`](@ref) and is often combined with it (`(TotalVariation2D(1e-3), TemporalTotalVariation(2e-2))`) — the "spatiotemporal TV" of the golden-angle radial sparse parallel (GRASP) literature.

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

#### Locally Low Rank

Instead of one Casorati matrix for the whole image, penalizes the nuclear norm of every spatial block separately:

```@docs
LocallyLowRank
```

**When to use:**
- Dynamic series where the temporal dynamics differ across the field of view (cardiac motion vs. static background, focal contrast uptake) — a global low-rank model needs a high rank to represent all of them at once, a local one does not
- Quantitative parameter mapping (T1/T2 relaxometry, MR fingerprinting), where each voxel neighbourhood follows a low-dimensional signal model

**Example:**
```julia
img = reconstruct(acq_dynamic, LocallyLowRank(5e-2; block_size = 8, time_dim = 3))
```

**Practical tip:** `block_size` trades locality against cost and stability: 4-8 voxels for strongly varying dynamics, 12-16 when the temporal signal is smooth over larger regions. Each iteration performs one SVD of a `(∏ block_size) × n_frames` matrix per block. The block grid is fixed, so residual block boundaries can remain visible at large λ; using a smaller λ with more iterations usually removes them.

### Joint Sparsity

For multi-contrast, multi-echo or multi-directional data, forces the components to share a common support:

```@docs
JointSparsity
```

**When to use:**
- Multi-echo, multi-contrast (T1w/T2w/FLAIR) or diffusion data of the same anatomy: the edges are in the same place in every image, only their intensities differ
- Velocity- or phase-encoded series
- Preferable to independent `L1Image`/`L1Wavelet2D` on each contrast, because the joint norm couples them

**Example:**
```julia
# echoes stored along dimension 3, sharing the same support
img = reconstruct(acq_multiecho, JointSparsity(1e-2; dim = 3))
```

**Practical tip:** Joint sparsity is most effective on a sparsifying transform of the images. Combining `JointSparsity` with a wavelet regularizer per contrast (`(JointSparsity(1e-2; dim = 3), L1Wavelet2D(1e-3))`) is a common compromise.

### Reference-Image Prior

Promotes sparsity of the *difference* to a known image instead of the image itself:

```@docs
ReferencePrior
```

**When to use:**
- Dynamic series where a high-quality temporal average or a previous time frame is available
- Follow-up or multi-contrast exams where an earlier high-SNR scan of the same anatomy exists
- Interventional / real-time imaging with a fully sampled baseline

**Example:**
```julia
x_ref = reconstruct(acq_reference, L1Wavelet2D(1e-3))
img = reconstruct(acq, (ReferencePrior(1e-2, x_ref), L1Wavelet2D(1e-3)))
```

**Practical tip:** The reference must be in the same units as the reconstruction; when data scaling is enabled the reference is rescaled automatically. A wrong reference biases the result toward it, so combine it with an ordinary sparsity term (as in the PICCS convex combination) rather than using it alone.

### Constraints

Constraints are enforced exactly by projection instead of being traded off against data consistency, so they carry no `λ`:

```@docs
NonNegative
BoxConstraint
```

**When to use:**
- Quantitative maps with a physically meaningful range (proton density, relaxation rates, diffusion coefficients)
- Magnitude-only or phase-resolved real-valued reconstructions

Both are defined for real-valued images only; applying them to complex data throws an `ArgumentError`.

**Example:**
```julia
img = reconstruct(acq_real, (TotalVariation2D(1e-3), NonNegative()))
```

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
- TemporalTotalVariation: `1e-2` to `1e-1`
- LowRank: `1e-2` to `1`
- LocallyLowRank: `1e-2` to `5e-1`
- JointSparsity: `1e-3` to `1e-2`
- ReferencePrior: `1e-3` to `1e-1`
- NonNegative / BoxConstraint: no parameter

**Adjustment strategy:**
1. Start with the suggested value
2. If too noisy/aliased → increase λ
3. If too smooth/blurry → decrease λ
4. Typical range: adjust by factors of 2-5

## Choosing a Regularizer

| Data | First choice | Common combinations |
|---|---|---|
| Static 2D/3D anatomy | [`L1Wavelet2D`](@ref) / [`L1Wavelet3D`](@ref) | + [`TotalVariation2D`](@ref) |
| Piecewise-constant anatomy, strong edges | [`TotalVariation2D`](@ref) / [`TotalVariation3D`](@ref) | + [`L1Wavelet2D`](@ref) |
| Periodic dynamics (cine, cardiac) | [`TemporalFourier`](@ref) | + [`TotalVariation2D`](@ref) |
| Irregular dynamics (free-breathing, real-time) | [`TemporalTotalVariation`](@ref) | + [`TotalVariation2D`](@ref) |
| Strong global spatiotemporal correlation (DCE, perfusion) | [`LowRank`](@ref) | L+S: [`LowRank`](@ref) + [`TemporalTotalVariation`](@ref), see [Image Decomposition](image_decomposition.md) |
| Spatially varying dynamics, parameter mapping | [`LocallyLowRank`](@ref) | + [`TotalVariation2D`](@ref) |
| Multi-contrast / multi-echo / diffusion | [`JointSparsity`](@ref) | + [`L1Wavelet2D`](@ref) |
| A high-quality prior image exists | [`ReferencePrior`](@ref) | + [`L1Wavelet2D`](@ref) |
| Real-valued images, physical range known | [`NonNegative`](@ref) / [`BoxConstraint`](@ref) | + any penalty |
| Parallel imaging without sparsity assumptions | [`Tikhonov`](@ref) | — |

## References

Sparsity and total variation:
- Lustig, M., Donoho, D., & Pauly, J. M. (2007). *Sparse MRI: The application of compressed sensing for rapid MR imaging.* Magnetic Resonance in Medicine, 58(6), 1182-1195. — the original CS-MRI formulation with ℓ₁-wavelet and total variation.
- Block, K. T., Uecker, M., & Frahm, J. (2007). *Undersampled radial MRI with multiple coils: Iterative image reconstruction using a total variation constraint.* Magnetic Resonance in Medicine, 57(6), 1086-1098.
- Fessler, J. A. (2010). *Model-based image reconstruction for MRI.* IEEE Signal Processing Magazine, 27(4), 81-89. — quadratic and edge-preserving penalties, non-negativity.

Dynamic imaging:
- Lustig, M., Santos, J. M., Donoho, D. L., & Pauly, J. M. (2006). *k-t SPARSE: High frame rate dynamic MRI exploiting spatio-temporal sparsity.* Proc. ISMRM. — sparsity in the temporal Fourier domain ([`TemporalFourier`](@ref)).
- Feng, L., Grimm, R., Block, K. T., et al. (2014). *Golden-angle radial sparse parallel MRI: Combination of compressed sensing, parallel imaging, and golden-angle radial sampling for fast and flexible dynamic volumetric MRI.* Magnetic Resonance in Medicine, 72(3), 707-717. — temporal total variation ([`TemporalTotalVariation`](@ref)).
- Otazo, R., Candès, E., & Sodickson, D. K. (2015). *Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components.* Magnetic Resonance in Medicine, 73(3), 1125-1136. — the L+S model, see [Image Decomposition](image_decomposition.md).

Low-rank models:
- Liang, Z.-P. (2007). *Spatiotemporal imaging with partially separable functions.* Proc. IEEE ISBI, 988-991. — the partially separable / globally low-rank model behind [`LowRank`](@ref) and [`RankLimit`](@ref).
- Trzasko, J. D., & Manduca, A. (2011). *Local versus global low-rank promotion in dynamic MRI series reconstruction.* Proc. ISMRM, 4371. — [`LocallyLowRank`](@ref).
- Zhang, T., Pauly, J. M., & Levesque, I. R. (2015). *Accelerating parameter mapping with a locally low rank constraint.* Magnetic Resonance in Medicine, 73(2), 655-661.

Joint sparsity and prior images:
- Majumdar, A., & Ward, R. K. (2011). *Joint reconstruction of multiecho MR images using correlated sparsity.* Magnetic Resonance Imaging, 29(7), 899-906. — [`JointSparsity`](@ref).
- Huang, J., Chen, C., & Axel, L. (2014). *Fast multi-contrast MRI reconstruction.* Magnetic Resonance Imaging, 32(10), 1344-1352.
- Chen, G.-H., Tang, J., & Leng, S. (2008). *Prior image constrained compressed sensing (PICCS).* Medical Physics, 35(2), 660-663. — [`ReferencePrior`](@ref).

Algorithms:
- Beck, A., & Teboulle, M. (2009). *A fast iterative shrinkage-thresholding algorithm for linear inverse problems.* SIAM Journal on Imaging Sciences, 2(1), 183-202. — FISTA.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). *Distributed optimization and statistical learning via the alternating direction method of multipliers.* Foundations and Trends in Machine Learning, 3(1), 1-122. — ADMM.

## Regularizers Not Currently Available

The following terms appear in the literature and in other reconstruction packages but are not implemented here, because they need building blocks the package does not yet have:

- **Total generalized variation (TGV)** and **infimal-convolution TV**: need a second-order (symmetrized gradient) operator, and TGV additionally needs an auxiliary variable that is coupled to the image but absent from the data term — which is *not* the same as the additive components of [Image Decomposition](image_decomposition.md).
- **Structured low-rank k-space methods** (SAKE, LORAKS, ALOHA): need a block-Hankel lifting operator with an adjoint.
- **Plug-and-play denoiser priors**: need a denoiser to be plugged in as a proximal operator; the machinery (a custom proximable function on the identity operator, as used by [`LocallyLowRank`](@ref)) is in place, only the denoisers are missing.
- **Shift-invariant (randomly shifted) LLR**: the block grid of [`LocallyLowRank`](@ref) is fixed between iterations.

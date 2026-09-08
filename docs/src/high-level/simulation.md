# Simulation Tools

MriReconstructionToolbox provides comprehensive tools for simulating MRI acquisitions. These are essential for testing reconstruction algorithms, teaching MRI concepts, and prototyping new acquisition strategies.

## Why Simulate?

Simulation is useful for:

- **Testing reconstruction algorithms** without needing real MRI data
- **Understanding MRI physics** through hands-on experimentation
- **Prototyping new acquisition strategies** before scanner implementation
- **Teaching and learning** MRI concepts interactively
- **Benchmarking** reconstruction methods with known ground truth

## Overview: Complete Simulation Pipeline

```@setup imports
using MriReconstructionToolbox
using GeometricMedicalPhantoms
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)
```

A typical simulation workflow:

```@example imports
using MriReconstructionToolbox

# 1. Create a phantom (ground truth image)
img = create_shepp_logan_phantom(256, 256, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

# 2. Generate coil sensitivity maps
smaps = coil_sensitivities(256, 256, 8)

# 3. Create a subsampling pattern
pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
pattern = create_sampling_pattern(pdf, (256, 256))

# 4. Simulate the acquisition
acq = AcquisitionInfo(is3D=false,
                      sensitivity_maps=smaps,
                      subsampling=pattern)
acq_with_data = simulate_acquisition(img, acq)

# 5. Reconstruct and compare
img_recon = reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(5e-3)); verbosity = Silent())
nothing # hide
```

## Phantoms

Phantoms are synthetic images that serve as ground truth for testing.

### Shepp-Logan Phantom

The classic test phantom for MRI reconstruction. MriReconstructionToolbox uses phantoms from the [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl) package.

```@example imports
using MIRTjim: jim

# 2D Shepp-Logan
img = create_shepp_logan_phantom(256, 256, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

# 3D Shepp-Logan
img_3d = create_shepp_logan_phantom(128, 128, 64; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
jim(img_3d; title="Shepp-Logan Phantom", nrow=4, size=(1200,300))
savefig("3D_shepp_logan_phantom.png"); nothing # hide
```

![3D_shepp_logan_phantom.png](3D_shepp_logan_phantom.png)

## Coil Sensitivity Maps

Sensitivity maps model the spatial response of receiver coils in parallel imaging.

```@docs
coil_sensitivities
```

```@example imports
# Generate sensitivity maps for 8 coils
smaps = coil_sensitivities(256, 256, 8)

# 3D sensitivity maps
smaps_3d = coil_sensitivities(128, 128, 64, 8)

# Visualize 2D coil sensitivities
jim(smaps; title="Coil Sensitivity Maps", nrow=1, size=(1400, 200))
savefig("coil_sensitivity_maps.png"); nothing # hide
```

![coil_sensitivity_maps.png](coil_sensitivity_maps.png)

**What you get:**
- Smooth, realistic sensitivity patterns
- Each coil has higher sensitivity near its location
- Proper phase variations
- Returns ComplexF32 array with shape (nx, ny, [nz,] ncoils)

## Subsampling Patterns

Subsampling patterns determine which k-space locations are measured.

### Uniform Cartesian Sampling

```@docs
UniformRandomSampling
```

#### Example: 2D Uniform Random Sampling
```@example imports
using Plots

# Center fraction = 0.1 (10% fully sampled center, default)
pdf = UniformRandomSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256))
p1 = jim(to_displayable_mask(pattern, (256, 256)); title = "cf=0.1")

# Center fraction = 0.3 (30% fully sampled center)
pdf = UniformRandomSampling(3.0, 0.3)
pattern = create_sampling_pattern(pdf, (256, 256))
p2 = jim(to_displayable_mask(pattern, (256, 256)); title = "cf=0.3")

# 2D subsampling: freq encoding also subsampled (not realistic, for demo)
pdf = UniformRandomSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256), subsample_freq_encoding=true)
p3 = jim(to_displayable_mask(pattern, (256, 256)); title = "Freq encoding subsampled")

jim(p1, p2, p3; layout = (1, 3), size = (1000, 300))
savefig("2D_uniform_random_sampling_weights.png"); nothing # hide
```

![2D_uniform_random_sampling_weights.png](2D_uniform_random_sampling_weights.png)

#### Example: 3D Uniform Random Sampling
```@example imports
pdf = UniformRandomSampling(4.0, 0.1)
pattern_3d = create_sampling_pattern(pdf, (128, 128, 64))
mask = zeros(Bool, 128, 128, 64)
mask[pattern_3d...] .= true
p1 = jim(mask[:, :, 32]; title="x-y plane")
p2 = jim(mask[:, 64, :]; title="x-z plane")
p3 = jim(mask[64, :, :]; title="y-z plane")
jim(p1, p2, p3; layout=(1,3), size=(1000,300))
savefig("3D_uniform_random_sampling_weights.png"); nothing # hide
```

![3D_uniform_random_sampling_weights.png](3D_uniform_random_sampling_weights.png)

### Variable Density Sampling

The most common pattern for compressed sensing. There are many options for generating variable density patterns.

```@docs
GaussianDistribution
PolynomialDistribution
VariableDensitySampling
```

#### Example: Different Variable Density Patterns in 2D

```@example imports
gaussian_pdf₁ = VariableDensitySampling(GaussianDistribution(1/3), 3.0)
gaussian_pattern₁ = create_sampling_pattern(gaussian_pdf₁, (256, 256))
W = MriReconstructionToolbox.construct_weights(gaussian_pdf₁, (256,))

p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(gaussian_pattern₁, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Gaussian std=1/3", size = (700, 300))
savefig("gaussian_sampling_pattern_1.png"); nothing # hide
```

![gaussian_sampling_pattern_1.png](gaussian_sampling_pattern_1.png)

```@example imports
gaussian_pdf₂ = VariableDensitySampling(GaussianDistribution(1/5), 3.0)
gaussian_pattern₂ = create_sampling_pattern(gaussian_pdf₂, (256, 256))
W = MriReconstructionToolbox.construct_weights(gaussian_pdf₂, (256,))

p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(gaussian_pattern₂, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Gaussian std=1/5", size = (700, 300))
savefig("gaussian_sampling_pattern_2.png"); nothing # hide
```

![gaussian_sampling_pattern_2.png](gaussian_sampling_pattern_2.png)

```@example imports
poly_pdf₁ = VariableDensitySampling(PolynomialDistribution(2), 3.0)
poly_pattern₁ = create_sampling_pattern(poly_pdf₁, (256, 256))

W = MriReconstructionToolbox.construct_weights(poly_pdf₁, (256,))
p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(poly_pattern₁, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Polynomial p=2", size = (700, 300))
savefig("polynomial_sampling_pattern_1.png"); nothing # hide
```

![polynomial_sampling_pattern_1.png](polynomial_sampling_pattern_1.png)

```@example imports
poly_pdf₂ = VariableDensitySampling(PolynomialDistribution(4), 3.0)
poly_pattern₂ = create_sampling_pattern(poly_pdf₂, (256, 256))
W = MriReconstructionToolbox.construct_weights(poly_pdf₂, (256,))
p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(poly_pattern₂, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Polynomial p=4", size = (700, 300))
savefig("polynomial_sampling_pattern_2.png"); nothing # hide
```

![polynomial_sampling_pattern_2.png](polynomial_sampling_pattern_2.png)

#### Example: Variable Density in 3D

```@example imports
poly_pdf = VariableDensitySampling(PolynomialDistribution(2), 3.0)
poly_pattern = create_sampling_pattern(poly_pdf, (256, 256, 256))
mask = zeros(Bool, 256, 256, 256)
mask[poly_pattern...] .= true
p1 = jim(mask[:, :, 128]; title="x-y plane")
p2 = jim(mask[:, 128, :]; title="x-z plane")
p3 = jim(mask[128, :, :]; title="y-z plane")
jim(p1, p2, p3; layout=(1,3), size=(900,200))
savefig("3D_variable_density_sampling_weights.png"); nothing # hide
```

![3D_variable_density_sampling_weights.png](3D_variable_density_sampling_weights.png)

### Poisson Disk Sampling

Spatially uniform but avoiding clustering.

```@docs
PoissonDiskSampling
```

```@example imports
pdf = PoissonDiskSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256), subsample_freq_encoding=true)
jim(to_displayable_mask(pattern, (256, 256)); title="Poisson Disk Sampling", size = (300, 300))
savefig("poisson_disk_sampling_pattern.png"); nothing # hide
```

![poisson_disk_sampling_pattern.png](poisson_disk_sampling_pattern.png)

**Properties:**
- Maintains minimum distance between samples
- More uniform coverage than random
- Good incoherence properties

## Non-Cartesian Trajectories

Ready-made generators for the common non-Cartesian sampling patterns, returned as
`NamedDimsArray`s in exactly the layout `AcquisitionInfo`/[`simulate_acquisition`](@ref) expect
(coordinate axis first, normalized to `[-0.5, 0.5)`, NFFT.jl convention):

```@docs
radial_trajectory
stack_of_stars_trajectory
kooshball_trajectory
spiral_trajectory
```

```@example imports
using NamedDims

traj = radial_trajectory(128, 96; ordering = :golden_angle)
acq_radial = AcquisitionInfo(; trajectory = traj, image_size = (256, 256))
data_radial = simulate_acquisition(img, acq_radial)
size(data_radial.kspace_data)
```

## Simulate Acquisition

```@docs
simulate_acquisition
```

## Advanced Simulation

### Adding Noise

`add_noise` adds complex Gaussian noise to k-space data, either as a target SNR (in dB, relative
to the RMS of the data) or as an absolute standard deviation. It accepts a plain array, a
`NamedDimsArray`, or an `AcquisitionInfo` directly (in which case a *copy* with noisy
`kspace_data` is returned, via the same copy-constructor pattern as `AcquisitionInfo(acq; ...)`):

```@docs
add_noise
```

```@example imports
# Target SNR in dB
noisy_acq = add_noise(acq_with_data; snr_db = 20)

# Or an absolute noise standard deviation
noisy_acq2 = add_noise(acq_with_data; noise_std = 0.02)

# Reconstruct noisy data
img_recon = reconstruct(noisy_acq, IterativeReconstruction(L1Wavelet2D(5e-3)); verbosity = Silent())
nothing # hide
```

### Subsampling: Indexing Expressions vs. Boolean Masks

`subsampling` accepts two forms (see [AcquisitionInfo](@ref) for the full set, including plain
boolean masks):

1. **A tuple of indexing expressions** (`Colon`, ranges, or integer vectors) — the **default**
   idiom for anything with regular structure. It reads directly as "keep these indices along
   each dimension", is cheaper to construct and store than a full mask, and composes naturally
   with `Base`'s indexing:

   ```@example imports
   ny = 256

   # Partial Fourier: keep the first 65% of phase-encoding lines
   pf_pattern = (:, 1:round(Int, 0.65 * ny))

   # Uniform GRAPPA-style undersampling (every 4th line) with a fully sampled ACS block —
   # a range/step expression reads directly as the acceleration factor and ACS width, where a
   # mask only shows the result of them.
   R, acs_half_width = 4, 12
   center = div(ny, 2)
   grappa_lines = sort(union(1:R:ny, (center - acs_half_width):(center + acs_half_width)))
   grappa_pattern = (:, grappa_lines)
   ```

2. **A boolean mask** (or `(:, mask)` for a fully sampled frequency-encoding dimension) — the
   special case for genuinely irregular or random patterns, where there is no indexing
   expression to write down. `create_sampling_pattern` (above) returns this form.

Both forms are passed the same way:

```@example imports
acq_pf = AcquisitionInfo(nothing; is3D=false, image_size=(ny, ny), subsampling=pf_pattern)
acq_grappa = AcquisitionInfo(nothing; is3D=false, image_size=(ny, ny), subsampling=grappa_pattern)
nothing # hide
```

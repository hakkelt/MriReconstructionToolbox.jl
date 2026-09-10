# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,src//jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Julia 1.12.7
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # 3 — Simulation tools
#
# MRT ships everything needed to fabricate a realistic acquisition: phantoms, coil sensitivity
# maps, sampling-pattern generators and a forward simulator. This notebook is a tour of all of
# them.
#
# **Contents**
# 1. Phantoms (2D and 3D)
# 2. Coil sensitivity maps
# 3. Sampling patterns — uniform random, variable density, Poisson disk
# 4. Sampling patterns in 3D
# 5. Hand-written patterns
# 6. `simulate_acquisition`
# 7. Noise, SNR and dynamic series

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: parts
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities,
    create_torso_phantom, generate_respiratory_signal
using MIRTjim: jim
using Plots
using LinearAlgebra: norm
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Phantoms
#
# Phantoms come from [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl).
# `MRISheppLoganIntensities()` gives the ellipse intensities MRI papers use (rather than the CT
# values of the original Shepp–Logan), and the phantom can be produced directly as `ComplexF32`.

# %%
x2d = create_shepp_logan_phantom(
    256, 256, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32
)
jim(x2d; title = "2D Shepp–Logan (axial)", size = (400, 350))

# %%
# Other orientations of the same 3D model.
jim(
    jim(create_shepp_logan_phantom(128, 128, :axial; ti = MRISheppLoganIntensities()); title = "axial"),
    jim(create_shepp_logan_phantom(128, 128, :coronal; ti = MRISheppLoganIntensities()); title = "coronal"),
    jim(create_shepp_logan_phantom(128, 128, :sagittal; ti = MRISheppLoganIntensities()); title = "sagittal");
    layout = (1, 3), size = (1000, 300)
)

# %%
# A 3D volume: (nx, ny, nz).
x3d = create_shepp_logan_phantom(128, 128, 32; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
println(size(x3d), " ", eltype(x3d))
jim(x3d[:, :, 5:4:32]; title = "3D Shepp–Logan, every 4th slice", nrow = 2, size = (900, 450))

# %% [markdown]
# ## 2. Coil sensitivity maps
#
# `coil_sensitivities(nx, ny[, nz], ncoils)` returns smooth complex profiles arranged around the
# field of view — enough structure for parallel imaging to be non-trivial.

# %%
smaps = coil_sensitivities(128, 128, 8)
println(size(smaps), " ", eltype(smaps))
jim(smaps; title = "Coil sensitivity maps (magnitude)", nrow = 2, size = (800, 400))

# %%
# The phase is what makes coil combination non-trivial.
jim(angle.(smaps); title = "Coil sensitivity phase", nrow = 2, size = (800, 400), clim = (-π, π))

# %%
# 3D maps: (nx, ny, nz, ncoils).
smaps3d = coil_sensitivities(64, 64, 16, 8)
println(size(smaps3d))
jim(smaps3d[:, :, 8, :]; title = "3D maps, central slice", nrow = 2, size = (800, 400))

# %% [markdown]
# ## 3. Sampling patterns
#
# A `Subsampling` object describes *how* to sample; `create_sampling_pattern` draws one
# realisation for a given image size. By default the frequency-encoding direction is fully
# sampled (that is free on a Cartesian scanner), so the returned pattern is `(:, mask)`.

# %%
pdf = UniformRandomSampling(3.0)                    # R = 3, default 10% central band
pattern = create_sampling_pattern(pdf, (256, 256))
println(typeof(pattern))
println("acquired phase encodes: ", sum(pattern[2]), " of 256")

# %%
# Uniform random sampling with different fully-sampled centre fractions.
p1 = jim(to_displayable_mask(create_sampling_pattern(UniformRandomSampling(3.0), (256, 256)), (256, 256)); title = "cf = 0.1")
p2 = jim(to_displayable_mask(create_sampling_pattern(UniformRandomSampling(3.0, 0.3), (256, 256)), (256, 256)); title = "cf = 0.3")
p3 = jim(
    to_displayable_mask(
        create_sampling_pattern(UniformRandomSampling(3.0), (256, 256); subsample_freq_encoding = true),
        (256, 256)
    ); title = "kx also undersampled"
)
jim(p1, p2, p3; layout = (1, 3), size = (1000, 320))

# %% [markdown]
# ### Variable density
#
# The compressed-sensing workhorse: sample the centre of k-space densely and the periphery
# sparsely. The density profile is either Gaussian or polynomial.

# %%
using MriReconstructionToolbox: construct_weights

function show_density(pdf, label)
    W = construct_weights(pdf, (256,))
    pat = create_sampling_pattern(pdf, (256, 256))
    p1 = plot(W; legend = false, title = "$label — density")
    p2 = jim(to_displayable_mask(pat, (256, 256)); title = "$label — pattern")
    return jim(p1, p2; layout = (1, 2), size = (750, 300))
end

show_density(VariableDensitySampling(GaussianDistribution(1 / 3), 3.0), "Gaussian σ=1/3")

# %%
show_density(VariableDensitySampling(GaussianDistribution(1 / 5), 3.0), "Gaussian σ=1/5")

# %%
show_density(VariableDensitySampling(PolynomialDistribution(2), 3.0), "Polynomial p=2")

# %%
show_density(VariableDensitySampling(PolynomialDistribution(4), 3.0), "Polynomial p=4")

# %% [markdown]
# ### Poisson disk
#
# Keeps a minimum distance between samples, so the coverage is uniform without the clumping of
# purely random sampling — the incoherent-but-even pattern favoured for 2D-undersampled 3D
# acquisitions.

# %%
pat_pd = create_sampling_pattern(PoissonDiskSampling(3.0), (256, 256); subsample_freq_encoding = true)
jim(to_displayable_mask(pat_pd, (256, 256)); title = "Poisson disk, R = 3", size = (400, 350))

# %% [markdown]
# ## 4. Patterns in 3D
#
# In 3D both phase-encoding directions can be undersampled, and the pattern is a 3-tuple.

# %%
pat3d = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(2), 4.0), (128, 128, 64))
mask3d = zeros(Bool, 128, 128, 64)
mask3d[pat3d...] .= true
println("acceleration: ", round(length(mask3d) / sum(mask3d), digits = 2), "×")

jim(
    jim(mask3d[:, :, 32]; title = "kx–ky"),
    jim(mask3d[:, 64, :]; title = "kx–kz"),
    jim(mask3d[64, :, :]; title = "ky–kz");
    layout = (1, 3), size = (1000, 300)
)

# %% [markdown]
# ## 5. Hand-written patterns
#
# `subsampling` does not have to come from a generator. The **default idiom** for a
# scanner-specific scheme is a tuple of per-dimension indexing expressions — a range, a
# `step`, a `Vector` of indices, `:` for "everything acquired" — because it states the
# pattern directly (a stride, a start, a count) instead of making the reader recover those
# numbers by scanning a boolean array. A boolean mask is still accepted, and stays the right
# tool for a pattern that has no closed form — variable-density and Poisson-disc random
# sampling in section 3 are exactly that case, which is why `create_sampling_pattern` returns
# one there.

# %%
# Partial Fourier: the first 65% of phase encodes, as a plain range.
ny = 256
subsampling_pf = (:, 1:round(Int, 0.65 * ny))
acq_pf = AcquisitionInfo(
    nothing; is3D = false, image_size = (256, 256), subsampling = subsampling_pf
)
println("phase encodes: ", length(subsampling_pf[2]), " of ", ny)

# %%
# Regular R = 4 with a 21-line autocalibration (ACS) band — the GRAPPA-style pattern. A
# strided range unioned with the ACS range states the acceleration and the calibration extent
# directly; recovering either number from a boolean mask would mean scanning it (or plotting
# it, as below) instead of just reading the expression.
acs_half = 10
subsampling_grappa = (:, sort(union(1:4:ny, (ny ÷ 2 - acs_half):(ny ÷ 2 + acs_half))))
acq_grappa_like = AcquisitionInfo(
    nothing; is3D = false, image_size = (256, 256), subsampling = subsampling_grappa
)
println("net acceleration: ", round(ny / length(subsampling_grappa[2]), digits = 2), "×")

# %%
# The two patterns above, rendered as masks purely for display.
mask_grappa = falses(ny)
mask_grappa[subsampling_grappa[2]] .= true
mask_pf = falses(ny)
mask_pf[subsampling_pf[2]] .= true
jim(
    jim(repeat(reshape(mask_grappa, 1, :), 256, 1); title = "regular R=4 + ACS"),
    jim(repeat(reshape(mask_pf, 1, :), 256, 1); title = "partial Fourier 65%");
    layout = (1, 2), size = (800, 320)
)

# %% [markdown]
# ## 6. `simulate_acquisition`
#
# `simulate_acquisition(image, acq)` applies the encoding operator described by `acq` and
# returns a *new* `AcquisitionInfo` carrying the simulated k-space. Whatever the configuration
# describes — coils, undersampling, shifts, 3D, batch dimensions — is what gets simulated.

# %%
nx, ny, nc = 128, 128, 8
x = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps = coil_sensitivities(nx, ny, nc)
pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (nx, ny))

acq = AcquisitionInfo(;
    is3D = false, image_size = (nx, ny), sensitivity_maps = smaps, subsampling = pattern
)
data = simulate_acquisition(x, acq)
println("simulated k-space: ", size(data.kspace_data))

# %%
# Fully sampled single-coil, for comparison: the k-space is the whole grid.
acq_full = AcquisitionInfo(; is3D = false, image_size = (nx, ny))
data_full = simulate_acquisition(x, acq_full)
println("fully sampled k-space: ", size(data_full.kspace_data))
jim(log.(abs.(data_full.kspace_data) .+ 1.0f-6); title = "log |k-space|", size = (400, 350))

# %%
# 3D acquisition. (The variable-density weights need a reasonably long kz axis; a very short
# one — 16 partitions, say — makes the polynomial density go negative and throws.)
x3 = create_shepp_logan_phantom(64, 64, 32; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
acq3 = AcquisitionInfo(;
    image_size = (64, 64, 32),
    sensitivity_maps = coil_sensitivities(64, 64, 32, 4),
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (64, 64, 32)
    ),
)
data3 = simulate_acquisition(x3, acq3)
println("3D k-space: ", size(data3.kspace_data))

# %% [markdown]
# ## 7. Noise, SNR and dynamic series
#
# `add_noise` is part of the package: `snr_db` targets a signal-to-noise ratio (RMS-relative,
# `σ = rms(data) * 10^(-snr_db / 20)`), `noise_std` an absolute complex standard deviation.
# Passing an `AcquisitionInfo` adds noise to its k-space and returns a new configuration via
# the copy constructor.

# %%
recs = map((40, 20, 10)) do snr
    rec = reconstruct(add_noise(data_full; snr_db = snr); verbosity = Silent())
    jim(rec; title = "SNR $(snr) dB")
end
jim(recs...; layout = (1, 3), size = (1000, 320))

# %% [markdown]
# ### A dynamic series
#
# A batch dimension (here `:time`) is simulated exactly like anything else. For a realistic
# dynamic series, `create_torso_phantom` from GeometricMedicalPhantoms.jl (already a dependency
# of these notebooks) takes a `respiratory_signal` (in litres, from `generate_respiratory_signal`)
# and returns a 4D `(nx, ny, nz, nt)` phantom breathing along with it; a single coronal slice
# (through the diaphragm, where respiratory motion is largest) gives a 2D dynamic series driven
# by real respiratory motion instead of a hand-rolled bolus.

# %%
using NamedDims

nt = 16
t_resp, resp_liters = generate_respiratory_signal(nt * 1.0, 1.0, 15.0)   # nt samples, 15 breaths/min
vol_dyn = create_torso_phantom(64, 64, 64; respiratory_signal = resp_liters[1:nt], eltype = ComplexF32)
series = NamedDimsArray{(:x, :y, :time)}(vol_dyn[:, 32, :, :])           # one coronal slice, all frames

jim(series[:, :, 1:5:16]; title = "dynamic frames 1, 6, 11, 16", nrow = 1, size = (1000, 280))

# %% [markdown]
# #### The same pattern for every frame
#
# The `(:, mask)`/index-expression idiom from section 5 applies unchanged to a series with a
# `:time` batch dimension: one `subsampling` acquires the same phase encodes at every frame.
# `AcquisitionInfo` also accepts a `Vector` of subsampling specs, one per frame, which
# `simulate_acquisition` acquires all at once through a single `AcquisitionInfo` — this is
# already the right idiom for "a different pattern per frame" when every frame selects the same
# *number* of samples (see below).

# %%
smaps_dyn = coil_sensitivities(64, 64, 4)
subsampling_same = (:, 1:2:64)
acq_dyn_same = AcquisitionInfo(
    nothing; is3D = false, image_size = (64, 64),
    sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps_dyn), subsampling = subsampling_same,
)
data_dyn_same = simulate_acquisition(series, acq_dyn_same)
println(dimnames(data_dyn_same.kspace_data), " ", size(data_dyn_same.kspace_data))

# %% [markdown]
# #### A different pattern per frame
#
# The interesting case for temporal regularizers: incoherent aliasing across time, so that a
# temporal-Fourier or low-rank penalty has something to exploit. One `AcquisitionInfo` still
# does it — `subsampling` is simply a `Vector` of per-frame specs, each acquiring a *different*
# but *equal-count* set of phase encodes (an equal-count requirement, since the simulated
# k-space is one dense array).

# %%
ny_dyn = 64
# Alternate between two R=2 patterns (odd/even phase encodes) — incoherent frame to frame,
# same number of samples every frame.
subsampling_per_frame = [(:, isodd(t) ? (1:2:ny_dyn) : (2:2:ny_dyn)) for t in 1:nt]
println("acquired phase encodes per frame: ", length.(getindex.(subsampling_per_frame, 2)))

acq_dyn_varying = AcquisitionInfo(
    nothing; is3D = false, image_size = (64, 64),
    sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps_dyn),
    subsampling = subsampling_per_frame,
)
data_dyn_varying = simulate_acquisition(series, acq_dyn_varying)
println(dimnames(data_dyn_varying.kspace_data), " ", size(data_dyn_varying.kspace_data))

# %% [markdown]
# #### Unequal sample counts per frame
#
# When the per-frame specs select *different numbers* of samples — a more aggressive schedule
# for later frames, say — the result can no longer be one dense array. `simulate_acquisition`
# detects this automatically and returns a `PartitionedKSpace` instead, wrapping one dense
# k-space array per frame. `reconstruct` still returns a plain `Array`: the partitioning is an
# internal storage detail of the measurement, not something that propagates to the image.

# %%
subsampling_unequal = [(:, 1:(t + 1):ny_dyn) for t in 1:nt]      # acceleration increases with t
println("acquired phase encodes per frame: ", length.(getindex.(subsampling_unequal, 2)))

acq_dyn_unequal = AcquisitionInfo(
    nothing; is3D = false, image_size = (64, 64),
    sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(smaps_dyn),
    subsampling = subsampling_unequal,
)
data_dyn_unequal = simulate_acquisition(series, acq_dyn_unequal)
println(typeof(data_dyn_unequal.kspace_data))
println("per-frame k-space sizes: ", size.(parts(data_dyn_unequal.kspace_data)))

rec_unequal = reconstruct(data_dyn_unequal; verbosity = Silent())
println(typeof(rec_unequal), " ", size(rec_unequal))

# %% [markdown]
# ## Environment

# %%
print_versions()

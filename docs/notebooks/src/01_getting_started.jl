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
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # 1 — Getting started with MriReconstructionToolbox
#
# This notebook walks through a complete MRI reconstruction in about twenty lines of code:
# build a phantom, simulate an undersampled multi-coil acquisition, and reconstruct it
# first directly and then with compressed sensing.
#
# Every notebook in this folder uses the environment in `docs/notebooks/Project.toml`
# (see `README.md` for the one-time setup).
#
# **Contents**
# 1. The pieces of an MRI acquisition
# 2. Simulating k-space
# 3. Direct (adjoint) reconstruction
# 4. Compressed-sensing reconstruction
# 5. Where to go next

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)
jim(:colorbar, true);

# %% [markdown]
# ## 1. The pieces of an MRI acquisition
#
# The forward model MRT solves is
#
# $$ y = \mathcal{A}x = \mathcal{P}\,\mathcal{F}\,\mathcal{S}\,x $$
#
# with $\mathcal{S}$ the coil sensitivities, $\mathcal{F}$ the Fourier transform and
# $\mathcal{P}$ the k-space sampling pattern. We need a ground-truth image $x$, sensitivity
# maps, and a sampling pattern.

# %%
nx, ny, nc = 128, 128, 8

x_true = create_shepp_logan_phantom(
    nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32
)
jim(x_true; title = "Shepp–Logan phantom (ground truth)", size = (400, 350))

# %%
# Smooth, complex-valued receive profiles for an 8-element array.
smaps = coil_sensitivities(nx, ny, nc)
jim(smaps; title = "Coil sensitivity maps", nrow = 2, size = (800, 400))

# %%
# Variable-density random sampling: 4× acceleration, fully sampled 5% centre.
pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05)
pattern = create_sampling_pattern(pdf, (nx, ny))

mask = to_displayable_mask(pattern, (nx, ny))
println("acceleration: ", round(nx * ny / sum(mask), digits = 2), "×")
jim(mask; title = "Sampling pattern (white = acquired)", size = (400, 350))

# %% [markdown]
# ## 2. Simulating k-space
#
# `AcquisitionInfo` is the container that holds everything known about the acquisition.
# Constructed without k-space data it describes an acquisition that has not happened yet,
# which is exactly what `simulate_acquisition` needs.

# %%
acq = AcquisitionInfo(;
    is3D = false,
    image_size = (nx, ny),
    subsampling = pattern,
    sensitivity_maps = smaps,
)

# %%
# A little measurement noise, then the simulated acquisition.
x_noisy = x_true + 0.01f0 * randn(ComplexF32, nx, ny)
data = simulate_acquisition(x_noisy, acq)

println("k-space data: ", size(data.kspace_data), " ", eltype(data.kspace_data))

# %% [markdown]
# ## 3. Direct (adjoint) reconstruction
#
# With no method argument, `reconstruct` applies $\mathcal{A}^H$ — zero-filling the missing
# k-space and combining the coils with the sensitivity maps. It is instantaneous, and on
# 4×-undersampled data it is visibly aliased.

# %%
x_direct = reconstruct(data; verbosity = Silent())

println("direct NRMSE: ", round(nrmse(x_direct, x_true), digits = 4))

jim(jim(x_direct; title = "Direct (adjoint)"), difference_image(x_direct, x_true); layout = (1, 2), size = (800, 350))

# %% [markdown]
# ## 4. Compressed-sensing reconstruction
#
# The undersampling is random, so the aliasing is incoherent and an $\ell_1$ penalty on the
# wavelet coefficients can remove it. `IterativeReconstruction` takes the regularizers as
# positional arguments; everything that tunes the solve (`maxit`, `tol`, `algorithm`) is a
# keyword of the method, not of `reconstruct`.

# %%
method = IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 60)
x_cs = reconstruct(data, method; verbosity = ProgressBar())

println("CS NRMSE:     ", round(nrmse(x_cs, x_true), digits = 4))

jim(jim(x_cs; title = "L1-wavelet CS"), difference_image(x_cs, x_true); layout = (1, 2), size = (800, 350))

# %%
# Side by side with the ground truth.
side_by_side(x_true, x_direct, x_cs; titles = ("Ground truth", "Direct", "CS"), size = (1100, 330))

# %% [markdown]
# ## 5. Where to go next
#
# | Notebook | Topic |
# |---|---|
# | `02_acquisition_info.ipynb` | `AcquisitionInfo`, named dimensions, validation |
# | `03_simulation.ipynb` | phantoms, coil maps, sampling patterns, noise |
# | `04_regularization.ipynb` | every regularizer in the package |
# | `05_reconstruction_methods.ipynb` | GRAPPA, SPIRiT, partial Fourier, signal models |
# | `06_algorithms_and_configuration.ipynb` | solvers, tolerances, scaling, verbosity, task splitting |
# | `07_dynamic_and_decomposition.ipynb` | dynamic imaging, L+S, image decomposition |
# | `08_non_cartesian.ipynb` | radial/spiral trajectories, NFFT, density compensation |
# | `09_low_level_interface.ipynb` | operators, `StructuredOptimization`, custom terms |
# | `10_real_data_cartesian.ipynb` | real 0.3 T brain data end to end |
# | `11_real_data_dynamic.ipynb` | real 1.5 T cardiac cine, low-rank + sparse |

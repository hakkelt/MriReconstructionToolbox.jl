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
# # 2 — `AcquisitionInfo`: describing the acquisition
#
# `AcquisitionInfo` is the validated container that every other part of MRT consumes: it holds
# the k-space data, the sensitivity maps, the image size, the sampling pattern and the FFT-shift
# conventions, and it checks them against each other at construction time.
#
# **Contents**
# 1. Constructing it (plain arrays, named dimensions, no data at all)
# 2. Sensitivity maps and the dimension order rules
# 3. Subsampling patterns
# 4. FFT-shift conventions
# 5. What the validation catches
# 6. Copy constructors
# 7. Cartesian vs. non-Cartesian
# 8. Getting the operators back out

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using NamedDims
using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator,
    get_sensitivity_map_operator, get_subsampling_operator
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Constructing it
#
# The first (positional) argument is the k-space data. It can be a plain array, a
# `NamedDimsArray`, or `nothing` when the acquisition has not happened yet.
#
# With a plain array, `is3D` has to be stated: MRT cannot tell a 3D volume from a multi-slice
# 2D stack by shape alone.

# %%
ksp_plain = rand(ComplexF32, 64, 64, 8)
AcquisitionInfo(ksp_plain; is3D = false)

# %%
# With named dimensions, `is3D` is inferred from the presence of `:kz`.
ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
AcquisitionInfo(ksp_named)

# %%
ksp_3d = NamedDimsArray{(:kx, :ky, :kz, :coil)}(rand(ComplexF32, 32, 32, 16, 4))
AcquisitionInfo(ksp_3d)

# %%
# No data yet — the configuration a simulation starts from.
AcquisitionInfo(nothing; is3D = false, image_size = (128, 128))

# %% [markdown]
# ## 2. Sensitivity maps
#
# Sensitivity maps must match the spatial dimensions and the element type of the k-space:
#
# | encoding | k-space | maps |
# |---|---|---|
# | 2D | `(kx, ky, coil[, slice])` | `(x, y, coil[, slice])` |
# | 3D | `(kx, ky, kz, coil)` | `(x, y, z, coil)` |

# %%
smaps = coil_sensitivities(64, 64, 8)
acq = AcquisitionInfo(rand(ComplexF32, 64, 64, 8); is3D = false, sensitivity_maps = smaps)

# %%
# 2D multi-slice: the slice dimension comes *after* the coil dimension.
ksp_ms = rand(ComplexF32, 64, 64, 4, 10)     # 4 coils, 10 slices
smaps_ms = rand(ComplexF32, 64, 64, 4, 10)
AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

# %%
# 3D volume.
ksp_vol = rand(ComplexF32, 32, 32, 16, 4)
smaps_vol = rand(ComplexF32, 32, 32, 16, 4)
AcquisitionInfo(ksp_vol; is3D = true, sensitivity_maps = smaps_vol)

# %% [markdown]
# ## 3. Subsampling patterns
#
# `subsampling` accepts a boolean mask, a tuple of per-dimension patterns, index vectors, or
# the tuple `create_sampling_pattern` returns. When k-space is compacted to the acquired
# samples, `image_size` has to be given because it can no longer be inferred.

# %%
mask = rand(Bool, 64, 64)
mask[25:40, 25:40] .= true                    # fully sampled centre
AcquisitionInfo(nothing; is3D = false, image_size = (64, 64), subsampling = mask)

# %%
# The realistic Cartesian case: every readout is acquired, phase encodes are undersampled.
mask_ky = rand(Bool, 64)
mask_ky[28:36] .= true
AcquisitionInfo(nothing; is3D = false, image_size = (64, 64), subsampling = (:, mask_ky))

# %%
# `create_sampling_pattern` returns exactly such a tuple.
pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 3.0), (64, 64))
println(typeof(pattern))
AcquisitionInfo(nothing; is3D = false, image_size = (64, 64), subsampling = pattern)

# %%
# 3D mask.
mask_3d = rand(Bool, 32, 32, 16)
mask_3d[13:20, 13:20, 5:12] .= true
AcquisitionInfo(nothing; is3D = true, image_size = (32, 32, 16), subsampling = mask_3d)

# %% [markdown]
# ## 4. FFT-shift conventions
#
# MRT assumes DC sits at the centre of the array. Data that comes off a scanner unshifted
# (DC at index 1), or that needs an image-space shift, is declared rather than pre-processed:
# the shift is folded into the Fourier operator instead of costing a copy.

# %%
ksp = rand(ComplexF32, 64, 64)

# DC already at the first index in both encoded dimensions
AcquisitionInfo(ksp; is3D = false, shifted_kspace_dims = (1, 2))

# %%
# Image-space shift (equivalent to sign alternation in k-space)
AcquisitionInfo(ksp; is3D = false, shifted_image_dims = (1,))

# %%
# With named dimensions the shifts are named too.
ksp_n = NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 64, 64))
AcquisitionInfo(ksp_n; shifted_kspace_dims = (:kx, :ky))

# %% [markdown]
# ## 5. What the validation catches
#
# Everything below throws at construction time rather than producing a wrong image later.

# %%
# Image-space names on k-space data
try
    AcquisitionInfo(NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 64, 64, 8)))
catch e
    println("Error: ", e.msg)
end

# %%
# Dimensions in the wrong order
try
    AcquisitionInfo(NamedDimsArray{(:ky, :kx, :coil)}(rand(ComplexF32, 64, 64, 8)))
catch e
    println("Error: ", e.msg)
end

# %%
# Sensitivity maps of a different size
try
    AcquisitionInfo(
        rand(ComplexF32, 64, 64, 8);
        is3D = false, sensitivity_maps = rand(ComplexF32, 128, 128, 8)
    )
catch e
    println("Error: ", e.msg)
end

# %%
# Mixed precision between k-space and maps
try
    AcquisitionInfo(
        rand(ComplexF32, 64, 64, 8);
        is3D = false, sensitivity_maps = rand(ComplexF64, 64, 64, 8)
    )
catch e
    println("Error: ", e.msg)
end

# %%
# Subsampling without an image size
try
    AcquisitionInfo(nothing; is3D = false, subsampling = rand(Bool, 64, 64))
catch e
    println("Error: ", e.msg)
end

# %% [markdown]
# ## 6. Copy constructors
#
# `AcquisitionInfo(info; kwargs...)` returns a new, re-validated configuration with the named
# fields replaced — the way to add sensitivity maps, swap in noisy k-space, or change the
# sampling pattern without rebuilding everything.

# %%
info = AcquisitionInfo(rand(ComplexF32, 64, 64, 8); is3D = false)
info_with_maps = AcquisitionInfo(info; sensitivity_maps = coil_sensitivities(64, 64, 8))

# %%
noisy = info_with_maps.kspace_data .+ 0.01f0 .* randn(ComplexF32, size(info_with_maps.kspace_data))
AcquisitionInfo(info_with_maps; kspace_data = noisy)

# %% [markdown]
# ## 7. Cartesian vs. non-Cartesian
#
# `AcquisitionInfo` is an abstract type and also a constructor that dispatches on its keywords:
# pass a `trajectory` and you get a `NonCartesianAcquisitionInfo`, otherwise a
# `CartesianAcquisitionInfo`. Only the Cartesian concrete type is exported and can be named
# directly; `NonCartesianAcquisitionInfo` is `public` but not exported, so non-Cartesian
# acquisitions are always built through the `AcquisitionInfo(; trajectory, ...)` dispatch.

# %%
nsamp, nspokes = 64, 32
traj = zeros(Float32, 2, nsamp, nspokes)         # first dimension = coordinate axes
for s in 1:nspokes, k in 1:nsamp
    θ = Float32((s - 1) * π / nspokes)
    r = Float32((k - 1 - nsamp / 2) / nsamp * 0.99)
    traj[1, k, s] = r * cos(θ)
    traj[2, k, s] = r * sin(θ)
end

acq_radial = AcquisitionInfo(;
    trajectory = traj, image_size = (64, 64)
)

# %%
println(typeof(AcquisitionInfo(rand(ComplexF32, 64, 64); is3D = false)))
println(typeof(acq_radial))
println("is3D: ", acq_radial.is3D, "  image_size: ", acq_radial.image_size)

# %% [markdown]
# ## 8. Getting the operators back out
#
# Every operator MRT would build internally is available from the configuration. These names
# are `public` but not exported, so they have to be imported explicitly.

# %%
mask_acq = rand(Bool, 64, 64)
mask_acq[28:36, 28:36] .= true
acq_ops = AcquisitionInfo(
    rand(ComplexF32, sum(mask_acq), 8);
    is3D = false,
    image_size = (64, 64),
    subsampling = (mask_acq,),
    sensitivity_maps = coil_sensitivities(64, 64, 8),
)

𝒜 = get_encoding_operator(acq_ops)
ℱ = get_fourier_operator(acq_ops)
𝒮 = get_sensitivity_map_operator(acq_ops)
𝒫 = get_subsampling_operator(acq_ops)

# size(op) is (codomain, domain) — matrix convention — so domain → codomain reads naturally.
println("𝒜 : ", size(𝒜)[2], " → ", size(𝒜)[1])
println("ℱ : ", size(ℱ)[2], " → ", size(ℱ)[1])
println("𝒮 : ", size(𝒮)[2], " → ", size(𝒮)[1])
println("𝒫 : ", size(𝒫)[2], " → ", size(𝒫)[1])

# %%
# The forward model, applied by hand.
img = rand(ComplexF32, 64, 64)
y = 𝒜 * img
x̂ = 𝒜' * y
println("𝒜  (forward): ", size(img), " → ", size(y))
println("𝒜' (adjoint): ", size(y), " → ", size(x̂))

# %% [markdown]
# ## Environment

# %%
print_versions()

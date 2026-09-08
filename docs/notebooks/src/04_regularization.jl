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
# # 4 — Regularization
#
# Undersampled reconstruction is ill-posed: many images explain the measured samples. MRT solves
#
# $$ \min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \sum_i \lambda_i R_i(x) $$
#
# and this notebook walks through every regularizer $R_i$ the package offers on one synthetic
# 2D problem, plus the 3D variants. Temporal and low-rank terms have a notebook of their own
# (`07_dynamic_and_decomposition.ipynb`) because they need a dynamic series.
#
# **Contents**
# 1. The common test problem
# 2. Image-domain terms — `L2Image`, `L1Image`
# 3. Transform sparsity — `L1Wavelet2D/3D`, `L1Contourlet`
# 4. Total variation — `TotalVariation2D/3D`, second order, TGV, Huber
# 5. Non-convex sparsity — `HardThreshold`, `SparsityLimit`
# 6. Plug-and-play priors
# 7. Joint sparsity and reference priors
# 8. Constraints
# 9. Combining terms, and choosing λ

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: get_operator
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using LinearAlgebra: norm
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. The common test problem
#
# 128², eight coils, 4× variable-density undersampling, a little noise.

# %%
nx, ny, nc = 128, 128, 8

x_true = create_shepp_logan_phantom(
    nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32
)
x_noisy = x_true + 0.02f0 * randn(ComplexF32, nx, ny)
smaps = coil_sensitivities(nx, ny, nc)
pattern = create_sampling_pattern(
    VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (nx, ny)
)

acq = AcquisitionInfo(;
    is3D = false, image_size = (nx, ny), subsampling = pattern, sensitivity_maps = smaps
)
data = simulate_acquisition(x_noisy, acq)

nrmse(x̂) = norm(abs.(x̂) - abs.(x_true)) / norm(abs.(x_true))
x_direct = reconstruct(data; verbosity = Silent())
println("direct (adjoint) NRMSE: ", round(nrmse(x_direct), digits = 4))
jim(x_direct; title = "starting point: direct reconstruction", size = (400, 350))

# %%
# A helper that reconstructs and reports, used throughout the notebook.
function show_recon(method, title; kwargs...)
    x̂ = reconstruct(data, method; verbosity = Silent(), kwargs...)
    println(title, " — NRMSE ", round(nrmse(x̂), digits = 4))
    return x̂
end

# %% [markdown]
# ## 2. Image-domain terms
#
# ### `L2Image` (Tikhonov)
#
# The quadratic penalty $\lambda\|x\|_2^2$. Cheap, smooth, solvable with conjugate gradient —
# the standard regularized-SENSE baseline. `Tikhonov` is an exported alias for the same type.

# %%
x_l2_strong = show_recon(IterativeReconstruction(L2Image(1.0f-1); maxit = 40), "L2Image λ=1e-1")
x_l2_weak = show_recon(IterativeReconstruction(Tikhonov(1.0f-4); maxit = 40), "Tikhonov λ=1e-4")

jim(
    jim(x_l2_strong; title = "λ = 1e-1 (over-smoothed)"),
    jim(x_l2_weak; title = "λ = 1e-4 (residual aliasing)");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ### `L1Image`
#
# $\lambda\|x\|_1$ — sparsity of the image itself. Right for genuinely sparse objects
# (angiography), too aggressive for anatomy.

# %%
x_l1 = show_recon(IterativeReconstruction(L1Image(5.0f-3); maxit = 40), "L1Image λ=5e-3")
jim(x_l1; title = "L1Image", size = (400, 350))

# %% [markdown]
# ## 3. Transform sparsity
#
# ### `L1Wavelet2D`
#
# The compressed-sensing default: anatomy is sparse in a wavelet basis. `get_operator` gives the
# transform itself, which is worth looking at.

# %%
reg_w = L1Wavelet2D(2.0f-3)
𝒲 = get_operator(reg_w, x_true)
coeffs = 𝒲 * x_noisy

x_wav = show_recon(IterativeReconstruction(reg_w; maxit = 60), "L1Wavelet2D λ=2e-3")
jim(
    jim(log.(abs.(coeffs) .+ 1.0f-4); title = "wavelet coefficients (log)"),
    jim(x_wav; title = "L1Wavelet2D reconstruction");
    layout = (1, 2), size = (800, 350)
)

# %%
# Options: the wavelet family and the number of decomposition levels.
x_haar = show_recon(IterativeReconstruction(L1Wavelet2D(2.0f-3; wavelet = WT.haar); maxit = 60), "Haar")
x_db8 = show_recon(IterativeReconstruction(L1Wavelet2D(2.0f-3; wavelet = WT.db8, levels = 3); maxit = 60), "db8, 3 levels")

jim(
    jim(x_haar; title = "Haar"),
    jim(x_db8; title = "Daubechies-8, 3 levels");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ### `L1Contourlet`
#
# The nonsubsampled contourlet transform is directional: elongated, oriented structures (vessels,
# fibres) need fewer coefficients than in a wavelet basis. It is more expensive per iteration.

# %%
reg_c = L1Contourlet(2.0f-3)
𝒞 = get_operator(reg_c, x_true)
bands = 𝒞 * x_noisy
println("contourlet stack: ", size(bands))

x_cont = show_recon(IterativeReconstruction(reg_c; maxit = 30), "L1Contourlet λ=2e-3")
jim(
    jim(bands[:, :, 1]; title = "coarse band"),
    jim(x_cont; title = "L1Contourlet reconstruction");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ### 3D: `L1Wavelet3D`
#
# For volumes and multi-slice stacks — the transform couples the slice direction, which also
# means the problem no longer decomposes over slices.

# %%
x3d = create_shepp_logan_phantom(64, 64, 32; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps3d = coil_sensitivities(64, 64, 32, 4)
pattern3d = create_sampling_pattern(
    VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (64, 64, 32)
)
acq3d = AcquisitionInfo(;
    image_size = (64, 64, 32), sensitivity_maps = smaps3d, subsampling = pattern3d
)
data3d = simulate_acquisition(x3d, acq3d)

x3d_wav = reconstruct(data3d, IterativeReconstruction(L1Wavelet3D(2.0f-3); maxit = 30); verbosity = Silent())
println("3D NRMSE: ", round(norm(abs.(x3d_wav) - abs.(x3d)) / norm(abs.(x3d)), digits = 4))
jim(x3d_wav[:, :, 9:4:29]; title = "L1Wavelet3D, four slices", nrow = 1, size = (1000, 280))

# %% [markdown]
# ## 4. Total variation
#
# ### `TotalVariation2D`
#
# Penalizes the gradient magnitude, favouring piecewise-constant images — strong edge
# preservation, at the price of a cartoon-like ("staircasing") appearance if λ is too large.

# %%
reg_tv = TotalVariation2D(1.0f-3)
∇ = get_operator(reg_tv, x_true)
grad = ∇ * x_noisy

x_tv = show_recon(IterativeReconstruction(reg_tv; maxit = 60), "TotalVariation2D λ=1e-3")
jim(
    jim(grad[:, :, 1]; title = "∂x"),
    jim(grad[:, :, 2]; title = "∂y"),
    jim(x_tv; title = "TV reconstruction");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ### Second-order TV and TGV
#
# First-order TV charges a smooth intensity ramp; second-order TV does not, but blurs jumps.
# `TotalGeneralizedVariation2D` makes the trade-off adaptively per voxel through an auxiliary
# vector field — it needs ADMM, and roughly doubles the unknowns.

# %%
x_tv2 = show_recon(
    IterativeReconstruction(TotalVariation2D(1.0f-3), SecondOrderTotalVariation2D(2.0f-3);
        algorithm = ADMM(), maxit = 60),
    "TV + second-order TV"
)

x_tgv = show_recon(
    IterativeReconstruction(TotalGeneralizedVariation2D(1.0f-3); algorithm = ADMM(), maxit = 60),
    "TotalGeneralizedVariation2D λ=1e-3"
)

jim(
    jim(x_tv; title = "TV"),
    jim(x_tv2; title = "TV + TV²"),
    jim(x_tgv; title = "TGV");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ### `EdgePreservingRoughness2D` (Huber)
#
# A smooth interpolation between a quadratic roughness penalty and TV: differences below `δ`
# are treated as noise and smoothed quadratically, those above are preserved. Being
# differentiable everywhere, it needs no proximal step. `δ` is an absolute intensity — a good
# recipe is a low percentile of the finite differences of a preliminary reconstruction.

# %%
using Statistics: quantile

diffs = abs.(diff(abs.(x_direct); dims = 1))
δ = Float32(quantile(vec(diffs), 0.15))
println("δ from the 15th percentile of |∇x_direct|: ", round(δ, digits = 5))

x_huber = show_recon(
    IterativeReconstruction(EdgePreservingRoughness2D(1.0f-3; δ = δ); maxit = 60),
    "EdgePreservingRoughness2D"
)
jim(x_huber; title = "Huber roughness penalty", size = (400, 350))

# %% [markdown]
# ### 3D total variation

# %%
x3d_tv = reconstruct(data3d, IterativeReconstruction(TotalVariation3D(1.0f-3); maxit = 30); verbosity = Silent())
println("3D TV NRMSE: ", round(norm(abs.(x3d_tv) - abs.(x3d)) / norm(abs.(x3d)), digits = 4))
jim(x3d_tv[:, :, 9:4:29]; title = "TotalVariation3D, four slices", nrow = 1, size = (1000, 280))

# %% [markdown]
# ## 5. Non-convex sparsity
#
# $\ell_1$ shrinks the coefficients it keeps, so intensities are systematically underestimated.
# Hard thresholding does not shrink — at the price of non-convexity, which makes the result
# depend on the starting image. Warm-starting from the $\ell_1$ solution is the standard recipe;
# note the threshold is `sqrt(2γλ)` rather than `γλ`, so an ℓ₁ λ carried over gives a completely
# different sparsity level and has to be retuned.
#
# On this phantom the ℓ₁ solution stays ahead on NRMSE — the argument for the ℓ₀ terms is
# unbiased amplitudes (lesion or vessel intensities that are not systematically shrunk), not a
# better global error.

# %%
x_hard = show_recon(
    IterativeReconstruction(HardThreshold(2.0f-4; domain = :wavelet2d); maxit = 40),
    "HardThreshold (warm start)"; x₀ = x_wav
)

# `SparsityLimit` constrains the *number* of non-zero coefficients instead of penalizing them.
x_sparsity = show_recon(
    IterativeReconstruction(SparsityLimit(2000; domain = :wavelet2d); maxit = 40),
    "SparsityLimit (2000 coefficients)"; x₀ = x_wav
)

jim(
    jim(x_wav; title = "ℓ₁ wavelet"),
    jim(x_hard; title = "hard threshold"),
    jim(x_sparsity; title = "sparsity limit");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ## 6. Plug-and-play priors
#
# `PlugAndPlay` uses any callable `denoiser(image, σ)` as the proximal operator, i.e. as an
# implicit image prior. No denoiser ships with MRT — BM3D or a trained network are the usual
# choices. To show the wiring (and to check it), a soft-thresholding "denoiser" reproduces the
# proximal operator of `L1Image` exactly; the two reconstructions then agree to a couple of
# percent, the remaining difference coming from the adaptive step size (the plug-and-play term
# has no objective value to backtrack on).
#
# `complex_handling` decides what the denoiser sees: `:split` (default) denoises the real and
# imaginary parts separately, `:magnitude` denoises the magnitude and keeps the phase — the
# latter is what matches complex soft thresholding.

# %%
# The denoiser is called as `denoiser(image, σ)` with `σ = strength * sqrt(γ)`, and the prox of
# `L1Image(λ)` corresponds to thresholding at `σ²` — hence `strength = sqrt(λ)` below.
soft(image, σ) = sign.(image) .* max.(abs.(image) .- σ^2, 0)

λ_pnp = 5.0f-3
x_pnp = show_recon(
    IterativeReconstruction(
        PlugAndPlay(soft; strength = sqrt(λ_pnp), complex_handling = :magnitude);
        algorithm = ISTA(), maxit = 60
    ),
    "PlugAndPlay(soft threshold)"
)
x_l1_ista = reconstruct(
    data, IterativeReconstruction(L1Image(λ_pnp); algorithm = ISTA(), maxit = 60); verbosity = Silent()
)
println("‖PnP − L1Image‖/‖L1Image‖ = ", round(norm(x_pnp - x_l1_ista) / norm(x_l1_ista), digits = 6))

# %% [markdown]
# The implicit prior has no value function, so the reported objective is `NaN` and
# objective-based convergence checks are meaningless: use `ISTA`, `FISTA` or `ADMM` with a fixed
# iteration budget, never a line-search algorithm. (With this MRT version, the `FISTA` path
# rejects the term at problem-parsing time — `ISTA` and `ADMM` both work.)

# %% [markdown]
# ## 7. Joint sparsity and reference priors
#
# ### `JointSparsity`
#
# Multi-echo / multi-contrast images of the same anatomy share their edge locations. The joint
# $\ell_{2,1}$ norm couples them so that a coefficient is either non-zero in every contrast or
# in none.

# %%
# Three "echoes" of the same anatomy with different contrast.
n = 64
base = create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
echoes = cat((base .* w for w in (1.0f0, 0.7f0, 0.45f0))...; dims = 3)

acq_me = AcquisitionInfo(;
    is3D = false,
    image_size = (n, n),
    sensitivity_maps = coil_sensitivities(n, n, 4),
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.05), (n, n)
    ),
)
data_me = simulate_acquisition(echoes, acq_me)

x_joint = reconstruct(
    data_me, IterativeReconstruction(JointSparsity(5.0f-3; dim = 3); maxit = 40); verbosity = Silent()
)
println("joint NRMSE: ", round(norm(abs.(x_joint) - abs.(echoes)) / norm(abs.(echoes)), digits = 4))
jim(x_joint; title = "JointSparsity — three echoes", nrow = 1, size = (900, 300))

# %% [markdown]
# ### `ReferencePrior`
#
# Penalizes the difference to a known image (a temporal average, a previous exam) instead of the
# image itself — the PICCS idea. Combine it with an ordinary sparsity term so a wrong reference
# cannot dominate.

# %%
x_ref = x_wav                                   # pretend this is a prior high-quality scan
x_piccs = show_recon(
    IterativeReconstruction(ReferencePrior(1.0f-2, x_ref), L1Wavelet2D(1.0f-3);
        algorithm = ADMM(), maxit = 40),
    "ReferencePrior + L1Wavelet2D"
)
jim(x_piccs; title = "reference-constrained reconstruction", size = (400, 350))

# %% [markdown]
# ## 8. Constraints
#
# `NonNegative` and `BoxConstraint` are enforced by projection, so they carry no λ. They are
# defined for **real-valued** images only — applying them to a complex image (which is what a
# standard MRI reconstruction produces) throws. They belong on quantitative maps and
# magnitude-only models.

# %%
println(NonNegative())
println(BoxConstraint(0.0, 1.0))

try
    reconstruct(data, IterativeReconstruction(TotalVariation2D(1.0f-3), NonNegative()); verbosity = Silent())
catch e
    println("\nOn complex data: ", sprint(showerror, e))
end

# %% [markdown]
# ## 9. Combining terms, and choosing λ
#
# Terms are simply listed; MRT picks a solver that can handle the combination (ADMM, in
# practice, as soon as there is more than one non-smooth term or a non-tight operator).

# %%
x_combo = show_recon(
    IterativeReconstruction(L1Wavelet2D(1.5f-3), TotalVariation2D(5.0f-4);
        algorithm = ADMM(), maxit = 60),
    "L1Wavelet2D + TotalVariation2D"
)
jim(
    jim(x_wav; title = "wavelet only"),
    jim(x_combo; title = "wavelet + TV");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ### Starting values for λ
#
# Because the problem solved is $\tfrac12\|\mathcal{A}x-y\|^2 + \lambda R(x)$ with no operator
# rescaling, λ is in the data's own units and the reconstructed image comes back in those units.
#
# | Regularizer | Typical λ |
# |---|---|
# | `L2Image` | 1e-5 … 1e-3 |
# | `L1Image` | 1e-4 … 1e-2 |
# | `L1Wavelet2D/3D`, `L1Contourlet` | 1e-3 … 1e-2 |
# | `TotalVariation2D/3D` | 1e-4 … 5e-3 |
# | `SecondOrderTotalVariation2D` | ≈ 2× the first-order λ |
# | `TotalGeneralizedVariation2D` | as `TotalVariation2D`; leave `ratio = 2.0` |
# | `EdgePreservingRoughness2D` | 1e-4 … 5e-3, `δ` from the data |
# | `HardThreshold` | 1e-4 … 1e-2 (threshold is `sqrt(2γλ)` — retune, don't reuse an ℓ₁ λ) |
# | `JointSparsity` | 1e-3 … 1e-2 |
# | `ReferencePrior` | 1e-3 … 1e-1 |
# | `L1TemporalFourier`, `TemporalTotalVariation` | 1e-2 … 1e-1 |
# | `LowRank`, `LocallyLowRank`, `MultiScaleLowRank` | 1e-2 … 5e-1 |
#
# Too noisy or aliased → increase λ; too smooth → decrease it; move in factors of 2–5.
#
# A quick sweep, all warm-started from the previous solution:

# %%
λs = Float32[8.0e-4, 2.0e-3, 5.0e-3, 1.2e-2]
sweep = map(λs) do λ
    x̂ = reconstruct(data, IterativeReconstruction(L1Wavelet2D(λ); maxit = 40); verbosity = Silent())
    jim(x̂; title = "λ = $λ\nNRMSE $(round(nrmse(x̂), digits = 3))")
end
jim(sweep...; layout = (1, 4), size = (1300, 320))

# %% [markdown]
# ### Not covered here
#
# The temporal and low-rank terms — `L1TemporalFourier`, `TemporalTotalVariation`, `LowRank`,
# `RankLimit`, `LocallyLowRank`, `MultiScaleLowRank` — need a dynamic series; they are covered in
# `07_dynamic_and_decomposition.ipynb`, together with the additive `Component` models (L+S).

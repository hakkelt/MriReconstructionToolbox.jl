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
# # 5 — Regularization
#
# Undersampled reconstruction is ill-posed: many images explain the measured samples. MRT solves
#
# $$ \min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \sum_i \lambda_i R_i(x) $$
#
# and this notebook walks through every regularizer $R_i$ the package offers on one synthetic
# 2D problem, plus the 3D variants. Every regularizer below is presented the same way:
#
# - **Problem** — the optimization problem this term actually poses, written out in full.
# - **Description** — what it does, and what it is good and bad at.
# - **References** — where it comes from.
# - **Availability in other toolboxes** — how the same idea is reached in BART, SigPy and
#   MRIReco.jl (whose regularizers come from RegularizedLeastSquares.jl).
#
# See `docs/src/high-level/regularization.md` for the full reference list and a "Choosing a
# regularizer" table. Temporal and low-rank terms have a notebook of their own
# (`07_dynamic_and_decomposition.ipynb`) because they need a dynamic series.
#
# **Contents**
# 1. The common test problem
# 2. Image-domain terms — `L2Image`, `L1Image`
# 3. Transform sparsity — `L1Wavelet2D/3D`, `L1Contourlet`
# 4. Total variation — `TotalVariation2D/3D`, second order, TGV, Huber
# 5. Non-convex sparsity — `L0Image`, `L0Wavelet2D`
# 6. Plug-and-play priors
# 7. Joint sparsity and reference priors
# 8. Constraints
# 9. Combining terms, and choosing λ

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: get_operator
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities,
    create_tubes_phantom, TubesIntensities
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

nrmse1(x̂) = nrmse(x̂, x_true)
x_direct = reconstruct(data; verbosity = Silent())
println("direct (adjoint) NRMSE: ", round(nrmse1(x_direct), digits = 4))
jim(x_direct; title = "starting point: direct reconstruction", size = (400, 350))

# %%
# A helper that reconstructs and reports, used throughout the notebook.
function show_recon(method, title; kwargs...)
    x̂ = reconstruct(data, method; verbosity = Silent(), kwargs...)
    println(title, " — NRMSE ", round(nrmse1(x̂), digits = 4))
    return x̂
end

# %% [markdown]
# ## 2. Image-domain terms
#
# ### `L2Image` (Tikhonov)
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|x\|_2^2$
#
# **Description:** Quadratic, smooth, solvable with conjugate gradient — the standard
# regularized-SENSE baseline, and the only term here that needs no proximal step at all.
# `Tikhonov` is an exported alias for the same type. Its strength is speed and predictability: the
# problem stays a linear system, so there is nothing to tune but λ. Its weakness is that it
# penalizes edges exactly as hard as noise, so noise suppression and resolution are traded one for
# one — too small barely regularizes (the NRMSE floor is the aliasing/noise level of the direct
# reconstruction), too large smooths away the anatomy along with the noise. The two λ below are
# chosen so the difference is visible at a glance, not just in the NRMSE number.
#
# **References:** Fessler, *Model-based image reconstruction for MRI*, IEEE Signal Processing
# Magazine 27(4), 81–89 (2010), covers the quadratic penalty alongside the edge-preserving one
# below.
#
# **Availability in other toolboxes:** BART — `pics -R Q:λ` (ℓ₂ in image domain), or `-l2 -r λ`.
# SigPy — `sigpy.prox.L2Reg`, or `sigpy.mri.app.SenseRecon(..., lamda=λ)`. MRIReco.jl —
# `L2Regularization(λ)`, the default `reg` of its CG-SENSE solver.

# %%
x_l2_good = show_recon(IterativeReconstruction(L2Image(1.0f-4); maxit = 40), "L2Image λ=1e-4 (well chosen)")
x_l2_over = show_recon(IterativeReconstruction(L2Image(1.0f0); maxit = 40), "L2Image λ=1e0 (over-regularized)")

# The over-regularized solution is heavily shrunk in magnitude; rescale it to the well-chosen
# image's peak before display, so the comparison is about lost structure, not lost brightness.
x_l2_over_scaled = x_l2_over .* (maximum(abs, x_l2_good) / maximum(abs, x_l2_over))
side_by_side(
    x_l2_good, x_l2_over_scaled;
    titles = (
        "λ = 1e-4 (well chosen)\nNRMSE $(round(nrmse1(x_l2_good), digits = 3))",
        "λ = 1e0 (over-regularized)\nNRMSE $(round(nrmse1(x_l2_over), digits = 3))",
    ),
)

# %% [markdown]
# ### `L1Image`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|x\|_1$
#
# **Description:** Sparsity of the image itself, enforced by soft thresholding. Right for
# genuinely sparse objects — angiography, where most of the FOV is background — and too aggressive
# for anatomy, where it eats low-contrast tissue along with the noise. It also biases the
# amplitudes it keeps downward by λγ, which §5's ℓ₀ terms exist to avoid.
#
# **References:** Lustig, Donoho & Pauly, *Sparse MRI: The application of compressed sensing for
# rapid MR imaging*, Magnetic Resonance in Medicine 58(6), 1182–1195 (2007), is the reference for
# ℓ₁ sparsity penalties in MRI generally.
#
# **Availability in other toolboxes:** BART — `pics -R I:λ`. SigPy — `sigpy.prox.L1Reg`.
# MRIReco.jl — `L1Regularization(λ)`.

# %%
x_l1 = show_recon(IterativeReconstruction(L1Image(5.0f-3); maxit = 40), "L1Image λ=5e-3")
jim(x_l1; title = "L1Image", size = (400, 350))

# %% [markdown]
# ## 3. Transform sparsity
#
# ### `L1Wavelet2D`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|\mathcal{W}x\|_1$
#
# **Description:** The compressed-sensing default: anatomy is compressible in a wavelet basis, so
# an ℓ₁ penalty on the coefficients removes incoherent aliasing while keeping edges. Cheap
# (an orthogonal transform, so the prox is exact soft thresholding of the coefficients) and robust
# across anatomies; its weakness is the blocky, texture-suppressing look at high λ, and a
# dependence on the wavelet family and level count shown below. `get_operator` gives the transform
# itself, which is worth looking at.
#
# **References:** Lustig, Donoho & Pauly (2007), cited above, is the paper that popularized this
# combination for MRI.
#
# **Availability in other toolboxes:** BART — `pics -R W:7:0:λ` (the first number selects the
# transformed dimensions). SigPy — `sigpy.mri.app.L1WaveletRecon`. MRIReco.jl — no dedicated
# wavelet type; the same term is `L1Regularization` with a wavelet `regTrafo`.

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
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|\mathcal{C}x\|_1$
#
# **Description:** The nonsubsampled contourlet transform is directional: elongated, oriented
# structures — vessels, fibres, sharp curved boundaries — need far fewer coefficients than in a
# wavelet basis, so at the same λ they survive better. The costs are real: the transform is
# redundant (a stack of directional subbands rather than a basis), so it is several times more
# expensive per iteration and uses more memory. The bands shown below are what the penalty acts
# on — coarse approximation first, then increasingly fine directional detail.

# **References:** da Cunha, Zhou & Do, *The nonsubsampled contourlet transform: theory, design,
# and applications*, IEEE Transactions on Image Processing 15(10), 3089–3101 (2006).
#
# **Availability in other toolboxes:** none of BART, SigPy or MRIReco.jl ships a contourlet
# regularizer — no direct equivalent anywhere else.

# %%
reg_c = L1Contourlet(2.0f-3)
𝒞 = get_operator(reg_c, x_true)
bands = 𝒞 * x_noisy
println("contourlet stack: ", size(bands))

x_cont = show_recon(IterativeReconstruction(reg_c; maxit = 30), "L1Contourlet λ=2e-3")
nbands = size(bands, 3)
band_idx = unique(round.(Int, range(1, nbands; length = min(4, nbands))))
rows_c, cols_c = grid_layout(length(band_idx) + 1)
jim(
    (jim(bands[:, :, b]; title = "band $b/$nbands") for b in band_idx)...,
    jim(x_cont; title = "L1Contourlet reconstruction");
    layout = (rows_c, cols_c), size = (330 * cols_c, 330 * rows_c)
)

# %% [markdown]
# ### 3D: `L1Wavelet3D`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|\mathcal{W}_{3D}x\|_1$
#
# **Description:** The same penalty as `L1Wavelet2D` with a three-dimensional transform, for
# volumes and multi-slice stacks. It exploits correlation between neighbouring slices, so at the
# same λ it is stronger than a per-slice 2D transform on a genuine volume — and worse than one on
# a stack of unrelated slices.
#
# **References:** as `L1Wavelet2D` above.
#
# **Availability in other toolboxes:** BART — the same `-R W` with the 3D dimension flags set
# (`-R W:7:0:λ`). SigPy — `L1WaveletRecon` over a 3D image shape. MRIReco.jl — `L1Regularization`
# with a 3D wavelet `regTrafo`.

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
println("3D NRMSE: ", round(nrmse(x3d_wav, x3d), digits = 4))
jim(x3d_wav[:, :, 9:4:29]; title = "L1Wavelet3D, four slices", nrow = 1, size = (1000, 280))

# %% [markdown]
# ## 4. Total variation
#
# ### `TotalVariation2D`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 +
# \lambda\sum_{\text{pixels}} \|\nabla x\|_2$
#
# **Description:** The isotropic gradient magnitude, summed over pixels. It favours
# piecewise-constant images, which is why it preserves edges better than any of the terms above —
# and why, at too large a λ, it renders smooth intensity variation as a flight of steps
# (*staircasing*), the artefact the second-order terms below exist to remove. The finite-difference
# operator is not tight, so this term cannot be composed into a single proximal map: MRT solves it
# with ADMM rather than FISTA.
#
# **References:** Block, Uecker & Frahm, *Undersampled radial MRI with multiple coils: Iterative
# image reconstruction using a total variation constraint*, Magnetic Resonance in Medicine 57(6),
# 1086–1098 (2007).
#
# **Availability in other toolboxes:** BART — `pics -R T:7:0:λ`. SigPy —
# `sigpy.mri.app.TotalVariationRecon`. MRIReco.jl — `TVRegularization(λ)`, though its own docs
# recommend `L1Regularization` with a `GradientOp` transform instead, because `TVRegularization`
# routes through an inexact nested dual solve.

# %%
reg_tv = TotalVariation2D(1.0f-3)
∇ = get_operator(reg_tv, x_true)
grad = ∇ * x_noisy

x_tv = show_recon(IterativeReconstruction(reg_tv; maxit = 60), "TotalVariation2D λ=1e-3")
jim(
    jim(grad[:, :, 1]; title = "dx"),
    jim(grad[:, :, 2]; title = "dy"),
    jim(x_tv; title = "TV reconstruction");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ### Second-order TV and TGV
#
# **Problem:** second-order TV adds the term as written,
#
# $$ \min_x \ \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda_1\|\nabla x\|_1
#    + \lambda_2\|\nabla^2 x\|_1, $$
#
# while TGV introduces an auxiliary vector field $w$ and minimizes over both variables:
#
# $$ \min_{x, w} \ \tfrac12\|\mathcal{A}x - y\|_2^2
#    + \alpha_1\|\nabla x - w\|_1 + \alpha_0\|\mathcal{E}w\|_1 $$
#
# with $\mathcal{E}$ the symmetrized gradient.
#
# **Description:** First-order TV charges a smooth intensity ramp; second-order TV does not, but
# blurs jumps. TGV makes that trade-off adaptively per voxel: where $w \approx \nabla x$ the
# penalty falls on $\mathcal{E}w$ and the region is allowed to be smooth, where $w \approx 0$ it
# falls on $\nabla x$ and the region is allowed a jump. The cost is roughly twice the unknowns and
# a coupled problem: $w$ is tied to $x$ through $\nabla x - w$, which proximal-gradient algorithms
# cannot separate, so both this and TV+TV² pin `algorithm = ADMM()` below.
#
# **References:** Bredies, Kunisch & Pock, *Total generalized variation*, SIAM Journal on Imaging
# Sciences 3(3), 492–526 (2010), introduces TGV; Knoll, Bredies, Pock & Stollberger, *Second order
# total generalized variation (TGV) for MRI*, Magnetic Resonance in Medicine 65(2), 480–491 (2011),
# applies it to reconstruction.
#
# **Availability in other toolboxes:** BART — `pics -R G:7:0:λ` for TGV (and `-R C` / `-R V` for
# the infimal-convolution variants of notebook 7 §5). Neither SigPy nor MRIReco.jl ships a TGV
# term — no direct equivalent there.

# %%
x_tv2 = show_recon(
    IterativeReconstruction(
        TotalVariation2D(1.0f-3), SecondOrderTotalVariation2D(2.0f-3);
        algorithm = ADMM(), maxit = 60
    ),
    "TV + second-order TV"
)

x_tgv = show_recon(
    IterativeReconstruction(TotalGeneralizedVariation2D(1.0f-3); algorithm = ADMM(), maxit = 60),
    "TotalGeneralizedVariation2D λ=1e-3"
)

jim(
    jim(x_tv; title = "TV"),
    jim(x_tv2; title = "TV + TV^2"),
    jim(x_tgv; title = "TGV");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ### `EdgePreservingRoughness2D` (Huber)
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 +
# \lambda \sum_{\text{pixels}} \phi_\delta(\nabla x)$, with
#
# $$ \phi_\delta(t) = \begin{cases} t^2 & |t| \le \delta \\ 2\delta|t| - \delta^2 & |t| > \delta \end{cases} $$
#
# **Description:** A smooth interpolation between a quadratic roughness penalty and TV:
# differences below `δ` are treated as noise and smoothed quadratically, those above are preserved
# like TV. Being differentiable everywhere it needs no proximal step at all, so a plain gradient
# method solves it and there is no staircasing — the reason to prefer it over TV when the object
# has genuine smooth gradients. The price is a second parameter: `δ` is an absolute intensity, and
# a good recipe is a low percentile of the finite differences of a preliminary reconstruction, as
# below.
#
# **References:** Charbonnier, Blanc-Féraud, Aubert & Barlaud, *Deterministic edge-preserving
# regularization in computed imaging*, IEEE Transactions on Image Processing 6(2), 298–311 (1997),
# is the Huber-type potential this implements; Fessler (2010), cited above, is the
# model-based-MRI application.
#
# **Availability in other toolboxes:** none of BART, SigPy or MRIReco.jl ships this potential — no
# direct equivalent.

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
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 +
# \lambda\sum_{\text{voxels}} \|\nabla_{3D} x\|_2$
#
# **Description:** the same penalty as `TotalVariation2D` with the gradient taken over all three
# spatial dimensions, so through-plane structure is regularized too.
#
# **References and availability in other toolboxes:** as `TotalVariation2D` above (BART's
# `-R T` flags carry the dimension set, so the 3D form is the same option).

# %%
x3d_tv = reconstruct(data3d, IterativeReconstruction(TotalVariation3D(1.0f-3); maxit = 30); verbosity = Silent())
println("3D TV NRMSE: ", round(nrmse(x3d_tv, x3d), digits = 4))
jim(x3d_tv[:, :, 9:4:29]; title = "TotalVariation3D, four slices", nrow = 1, size = (1000, 280))

# %% [markdown]
# ## 5. Non-convex sparsity
#
# **Problem:** in the penalty form,
#
# $$ \min_x \ \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|\mathcal{W}x\|_0, $$
#
# and in the constraint form, which is how the `count` variant is stated,
#
# $$ \min_x \ \tfrac12\|\mathcal{A}x - y\|_2^2
#    \quad \text{subject to} \quad \|\mathcal{W}x\|_0 \le n. $$
#
# **Description:** $\ell_1$ shrinks the coefficients it keeps, so intensities come out
# systematically underestimated. Hard thresholding keeps or kills a coefficient and never shrinks
# it — unbiased amplitudes, which for a lesion or a vessel is the quantity of interest. The price
# is non-convexity: the result depends on the starting image, so warm-starting from the $\ell_1$
# solution is the standard recipe (and is what the cells below do). Note the threshold is
# `sqrt(2γλ)` rather than `γλ`, so an ℓ₁ λ carried over gives a completely different sparsity
# level and has to be retuned. On this phantom the ℓ₁ solution stays ahead on NRMSE — the argument
# for the ℓ₀ terms is the unbiased amplitude, not a better global error.
#
# **References:** Blumensath & Davies, *Iterative hard thresholding for compressed sensing*,
# Applied and Computational Harmonic Analysis 27(3), 265–274 (2009), is the algorithm both the
# penalty form (`threshold`) and the constraint form (`count`) come from.
#
# **Availability in other toolboxes:** BART — `pics -R N:7:0:λ` is NIHT in the image domain
# (`L0Image`) and `-R H:7:0:λ` is NIHT on wavelet coefficients (`L0Wavelet2D`). Neither SigPy nor
# MRIReco.jl ships a hard-thresholding regularizer. The `count` constraint form has no equivalent
# in any of the three; it is the sparsity analogue of `RankLimit` — see
# `docs/src/high-level/regularization.md`.

# %%
x_hard = show_recon(
    IterativeReconstruction(L0Wavelet2D(threshold = 2.0f-4); maxit = 40),
    "L0Wavelet2D threshold (warm start)"; x₀ = x_wav
)

# The `count` form constrains the *number* of non-zero coefficients instead of penalizing them.
x_sparsity = show_recon(
    IterativeReconstruction(L0Wavelet2D(count = 2000); maxit = 40),
    "L0Wavelet2D count=2000 (warm start)"; x₀ = x_wav
)

jim(
    jim(x_wav; title = "L1 wavelet"),
    jim(x_hard; title = "L0 (threshold) wavelet"),
    jim(x_sparsity; title = "L0 (count) wavelet");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ## 6. Plug-and-play priors
#
# **Problem:** there is no explicit penalty to write down. The problem solved is
#
# $$ \min_x \ \tfrac12\|\mathcal{A}x - y\|_2^2 + R(x) $$
#
# where $R$ is known *only* through its proximal operator,
# $\mathrm{prox}_{\gamma R}(x) = \mathrm{denoiser}(x, \sigma)$ — the algorithm never needs
# $R$ itself, only that step.
#
# **Description:** `PlugAndPlay` uses any callable `denoiser(image, σ)` as that proximal operator,
# i.e. as an implicit image prior. Its strength is that state-of-the-art denoisers (BM3D, trained
# networks) are far better image models than any penalty anyone can write down. Its weaknesses
# follow from the same fact: there is no objective value, so convergence cannot be checked against
# one and line-search algorithms cannot be used; convergence is only guaranteed for denoisers with
# properties most real ones are not proven to have; and the result depends on a denoiser that is
# not part of the reconstruction's own mathematics.
#
# **References:** Venkatakrishnan, Bouman & Wohlberg, *Plug-and-play priors for model based
# reconstruction*, Proc. IEEE GlobalSIP, 945–948 (2013), introduces the idea; Ahmad, Bouman,
# Buzzard et al., *Plug-and-play methods for magnetic resonance imaging*, IEEE Signal Processing
# Magazine 37(1), 105–116 (2020), surveys it for MRI.
#
# **Availability in other toolboxes:** MRIReco.jl — `PlugAndPlayRegularization`, which takes a
# Julia callable, the closest match to MRT's interface. BART — `pics -R TF:{graph}:λ`, which does
# have a plug-and-play prior but restricted to a denoiser exported as a TensorFlow model, a
# narrower interface than "any callable". SigPy — no plug-and-play regularizer.
#
# No denoiser ships with MRT — BM3D or a trained network are the usual choices. To show the
# wiring (and to check it), a soft-thresholding "denoiser" reproduces the proximal operator of
# `L1Image` exactly; the two reconstructions then agree to a couple of percent, the remaining
# difference coming from the adaptive step size (the plug-and-play term has no objective value to
# backtrack on). Because the implicit prior has no value function, the reported objective is
# `NaN` and objective-based convergence checks are meaningless — this is the one place in the
# notebook where the algorithm has to be pinned explicitly: `ISTA`, `FISTA` or `ADMM` with a
# fixed iteration budget, never a line-search algorithm.
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
# (With this MRT version, the `FISTA` path rejects the term at problem-parsing time — `ISTA` and
# `ADMM` both work.)

# %% [markdown]
# ## 7. Joint sparsity and reference priors
#
# ### `JointSparsity`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|x\|_{2,1}$, with
# $\|x\|_{2,1} = \sum_{\text{pixels}} \big(\sum_{\text{contrasts}} |x|^2\big)^{1/2}$
#
# **Description:** Multi-echo / multi-contrast images of the same anatomy share their edge
# locations. The joint $\ell_{2,1}$ norm couples them so that a coefficient is either non-zero in
# every contrast or in none, which recovers a weak contrast from the support the strong ones
# agree on. It fails exactly when the assumption does: a structure genuinely present in one
# contrast only is penalized as if it were noise.
#
# **References:** Majumdar & Ward, *Joint reconstruction of multiecho MR images using correlated
# sparsity*, Magnetic Resonance Imaging 29(7), 899–906 (2011).
#
# **Availability in other toolboxes:** MRIReco.jl — `L21Regularization(λ; slices = n)`. BART and
# SigPy — no dedicated joint-sparsity option; the ℓ₂,₁ prox would have to be supplied by hand.

# %%
# Three "echoes" of the same anatomy with different contrast: six tubes at fixed relative
# fillings, scaled together per echo the way multi-echo signal decays — a more realistic
# multi-contrast test than a single phantom uniformly dimmed.
n = 64
base_fillings = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
echo_intensities = [TubesIntensities(tube_fillings = base_fillings .* w) for w in (1.0, 0.7, 0.45)]
echoes = create_tubes_phantom(n, n, :axial; ti = echo_intensities, eltype = ComplexF32)

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
println("joint NRMSE: ", round(nrmse(x_joint, echoes), digits = 4))
jim(x_joint; title = "JointSparsity — three echoes", nrow = 1, size = (900, 300))

# %% [markdown]
# ### `ReferencePrior`
#
# **Problem:** $\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \lambda\|x - x_{\text{ref}}\|_1
# + \mu\|\mathcal{W}x\|_1$ — the second term is the ordinary sparsity penalty the reference
# prior should always be paired with.
#
# **Description:** Penalizes the difference to a known image — a temporal average, a previous exam
# — instead of the image itself, which is the PICCS idea. Where the reference is right this is by
# far the strongest prior available, because it constrains the *value* rather than the smoothness.
# Where it is wrong it hallucinates the reference into the result, which is why it is combined
# with an ordinary sparsity term so a wrong reference cannot dominate.
#
# **References:** Chen, Tang & Leng, *Prior image constrained compressed sensing (PICCS)*,
# Medical Physics 35(2), 660–663 (2008).
#
# **Availability in other toolboxes:** none of BART, SigPy or MRIReco.jl ships a reference-prior
# term; the closest building block elsewhere is a plain ℓ₁ penalty applied to a manually formed
# difference image.
#
# MRT picks the solver automatically — two non-smooth terms together mean ADMM, without needing
# `algorithm = ADMM()` spelled out.

# %%
x_ref = x_wav                                   # pretend this is a prior high-quality scan
x_piccs = show_recon(
    IterativeReconstruction(ReferencePrior(1.0f-2, x_ref), L1Wavelet2D(1.0f-3); maxit = 40),
    "ReferencePrior + L1Wavelet2D"
)
jim(x_piccs; title = "reference-constrained reconstruction", size = (400, 350))

# %% [markdown]
# ## 8. Constraints
#
# **Problem:** a constraint, not a penalty — which is why these carry no λ:
#
# $$ \min_x \ \tfrac12\|\mathcal{A}x - y\|_2^2 \quad \text{subject to} \quad x \in C, $$
#
# with $C = \{x \ge 0\}$ for `NonNegative` and $C = [a, b]$ for `BoxConstraint`. Internally this
# is the indicator function $\iota_C$, whose proximal operator is the projection onto $C$.
#
# **Description:** Constraints encode what an image *cannot* be rather than what it should look
# like, so unlike every penalty above they cost no accuracy where they are true: a proton density
# or a $T_2$ map is non-negative as a matter of physics. Their natural home is quantitative maps
# and magnitude-only models. The catch is that a standard MRI reconstruction produces a *complex*
# image, for which "non-negative" is not defined — MRT therefore throws by default and offers
# `complex_handling = :real` (below) to project onto the real non-negative orthant instead.
#
# **References:** the projection is elementary; Fessler (2010), cited above, discusses
# non-negativity in model-based MRI reconstruction.
#
# **Availability in other toolboxes:** BART — `pics -R S:0:0:0` is the non-negative constraint,
# and `pics -c` separately constrains the image to be real-valued (there is no `-R POS`). SigPy —
# `sigpy.prox.BoxConstraint(shape, lower, upper)`, with `lower = 0` for non-negativity. MRIReco.jl
# — `PositiveRegularization()` for non-negativity and `RealRegularization()` for the real-valued
# constraint, both of which take the real part the way MRT's `:real` handling does.

# %%
println(NonNegative())
println(BoxConstraint(0.0, 1.0))

try
    reconstruct(data, IterativeReconstruction(TotalVariation2D(1.0f-3), NonNegative()); verbosity = Silent())
catch e
    println("\nOn complex data, default (:error): ", sprint(showerror, e))
end

# %% [markdown]
# `complex_handling = :real` projects onto the real, non-negative orthant instead of throwing:
# the imaginary part is discarded and the real part clamped at 0. On the real, non-negative
# Shepp–Logan phantom used throughout this notebook, adding that constraint to TV should only
# help — it rules out images the true one could never be.

# %%
x_tv_only = show_recon(IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 60), "TV alone")
x_tv_pos = show_recon(
    IterativeReconstruction(TotalVariation2D(1.0f-3), NonNegative(; complex_handling = :real); maxit = 60),
    "TV + NonNegative(:real)"
)
println("NonNegative(:real) improves on TV alone: ", nrmse1(x_tv_pos) < nrmse1(x_tv_only))

side_by_side(x_tv_only, x_tv_pos; titles = ("TV alone", "TV + NonNegative(:real)"))

# %% [markdown]
# ## 9. Combining terms, and choosing λ
#
# Terms are simply listed; MRT picks a solver that can handle the combination (ADMM, in
# practice, as soon as there is more than one non-smooth term or a non-tight operator) — no
# `algorithm = ...` keyword is needed here either.

# %%
x_combo = show_recon(
    IterativeReconstruction(L1Wavelet2D(1.5f-3), TotalVariation2D(5.0f-4); maxit = 60),
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
# | `L0Image`, `L0Wavelet2D`, `L0Wavelet3D` (`threshold` form) | 1e-4 … 1e-2 (threshold is `sqrt(2γλ)` — retune, don't reuse an ℓ₁ λ) |
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
    jim(x̂; title = "λ = $λ\nNRMSE $(round(nrmse1(x̂), digits = 3))")
end
jim(sweep...; layout = grid_layout(length(sweep)), size = (1000, 660))

# %% [markdown]
# ### Not covered here
#
# The temporal and low-rank terms — `L1TemporalFourier`, `TemporalTotalVariation`, `LowRank`,
# `RankLimit`, `LocallyLowRank`, `MultiScaleLowRank` — need a dynamic series; they are covered in
# `07_dynamic_and_decomposition.ipynb`, together with the additive `Component` models (L+S).

# %% [markdown]
# ## Environment

# %%
print_versions()

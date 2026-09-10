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
# 2D problem, plus the 3D variants. Each section states the penalty, cites the paper it comes
# from, and says how the same idea is reached in another toolbox (BART's `-R` flag, SigPy, or
# RegularizedLeastSquares.jl) — see `docs/src/high-level/regularization.md` for the full
# reference list and a "Choosing a regularizer" table. Temporal and low-rank terms have a
# notebook of their own (`07_dynamic_and_decomposition.ipynb`) because they need a dynamic series.
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
# $$ \lambda\|x\|_2^2 $$
#
# Quadratic, smooth, solvable with conjugate gradient — the standard regularized-SENSE baseline.
# `Tikhonov` is an exported alias for the same type. Fessler, *Model-based image reconstruction
# for MRI*, IEEE Signal Processing Magazine 27(4), 81–89 (2010), covers the quadratic penalty
# alongside the edge-preserving one below. It is BART's `-R Q`, SigPy's `L2Reg`, and
# RegularizedLeastSquares.jl's `L2Regularization`.
#
# λ trades noise suppression for detail directly: too small barely regularizes at all (the NRMSE
# floor here is essentially the aliasing/noise level of the direct reconstruction), too large
# smooths away the anatomy along with the noise. The two λ below are chosen so the difference is
# visible at a glance, not just in the NRMSE number.

# %%
x_l2_good = show_recon(IterativeReconstruction(L2Image(1.0f-4); maxit = 40), "L2Image λ=1e-4 (well chosen)")
x_l2_over = show_recon(IterativeReconstruction(L2Image(1.0f0); maxit = 40), "L2Image λ=1e0 (over-regularized)")

side_by_side(
    x_l2_good, x_l2_over;
    titles = ("λ = 1e-4 (well chosen)\nNRMSE $(round(nrmse1(x_l2_good), digits = 3))",
        "λ = 1e0 (over-regularized)\nNRMSE $(round(nrmse1(x_l2_over), digits = 3))"),
)

# %% [markdown]
# ### `L1Image`
#
# $$ \lambda\|x\|_1 $$
#
# Sparsity of the image itself. Right for genuinely sparse objects (angiography), too aggressive
# for anatomy. Lustig, Donoho & Pauly, *Sparse MRI: The application of compressed sensing for
# rapid MR imaging*, Magnetic Resonance in Medicine 58(6), 1182–1195 (2007), is the reference for
# ℓ₁ sparsity penalties in MRI generally. It is BART's `-R I`, SigPy's `L1Reg`, and
# RegularizedLeastSquares.jl's `L1Regularization`.

# %%
x_l1 = show_recon(IterativeReconstruction(L1Image(5.0f-3); maxit = 40), "L1Image λ=5e-3")
jim(x_l1; title = "L1Image", size = (400, 350))

# %% [markdown]
# ## 3. Transform sparsity
#
# ### `L1Wavelet2D`
#
# $$ \lambda\|\mathcal{W}x\|_1 $$
#
# The compressed-sensing default: anatomy is sparse in a wavelet basis. Lustig, Donoho & Pauly
# (2007), cited above, is the paper that popularized this combination for MRI. It is BART's
# `-R W` and SigPy's `L1WaveletRecon`; RegularizedLeastSquares.jl has no dedicated wavelet type —
# the same term is `L1Regularization` composed with a wavelet `regTrafo`. `get_operator` gives
# the transform itself, which is worth looking at.

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
# $$ \lambda\|\mathcal{C}x\|_1 $$
#
# The nonsubsampled contourlet transform is directional: elongated, oriented structures (vessels,
# fibres) need fewer coefficients than in a wavelet basis. It is more expensive per iteration.
# da Cunha, Zhou & Do, *The nonsubsampled contourlet transform: theory, design, and
# applications*, IEEE Transactions on Image Processing 15(10), 3089–3101 (2006), is the transform
# this term sparsifies in. None of BART, SigPy or RegularizedLeastSquares.jl ships a contourlet
# regularizer — no direct equivalent.

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
# Same penalty $\lambda\|\mathcal{W}_{3D}x\|_1$ as `L1Wavelet2D`, over volumes and multi-slice
# stacks — the transform couples the slice direction, which also means the problem no longer
# decomposes over slices. Same references and toolbox equivalents as `L1Wavelet2D` above.

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
# $$ \lambda\sum_{\text{pixels}} \|\nabla x\|_2 $$
#
# The isotropic gradient magnitude, summed over pixels — favouring piecewise-constant images,
# strong edge preservation, at the price of a cartoon-like ("staircasing") appearance if λ is too
# large. Block, Uecker & Frahm, *Undersampled radial MRI with multiple coils: Iterative image
# reconstruction using a total variation constraint*, Magnetic Resonance in Medicine 57(6),
# 1086–1098 (2007), is the MRI-specific reference. It is BART's `-R T`, SigPy's
# `TotalVariationRecon`; RegularizedLeastSquares.jl has no dedicated `TVRegularization` path for
# this — its own docs recommend `L1Regularization` with a `GradientOp` transform instead, because
# the alternative (`TVRegularization` as `reg`) routes through an inexact nested dual solve.

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
# First-order TV, $\lambda\|\nabla x\|$, charges a smooth intensity ramp; second-order TV,
# $\lambda\|\nabla^2 x\|$, does not, but blurs jumps. `TotalGeneralizedVariation2D` makes the
# trade-off adaptively per voxel through an auxiliary vector field $w$,
#
# $$ \min_w \ \alpha_1\|\nabla x - w\|_1 + \alpha_0\|\mathcal{E}w\|_1 $$
#
# with $\mathcal{E}$ the symmetrized gradient — it needs ADMM (the auxiliary field is coupled to
# $x$ through $\nabla x - w$, which the proximal-gradient algorithms cannot separate), and
# roughly doubles the unknowns. This section pins `algorithm = ADMM()` for exactly that reason —
# neither TV+TV² nor TGV admits a plain ISTA/FISTA step.
#
# Bredies, Kunisch & Pock, *Total generalized variation*, SIAM Journal on Imaging Sciences 3(3),
# 492–526 (2010) introduces TGV; Knoll, Bredies, Pock & Stollberger, *Second order total
# generalized variation (TGV) for MRI*, Magnetic Resonance in Medicine 65(2), 480–491 (2011)
# applies it to reconstruction. It is BART's `-R G`; neither SigPy nor RegularizedLeastSquares.jl
# ships a TGV term — no direct equivalent there.

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
    jim(x_tv2; title = "TV + TV^2"),
    jim(x_tgv; title = "TGV");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ### `EdgePreservingRoughness2D` (Huber)
#
# $$ \lambda \sum_{\text{pixels}} \phi_\delta(\nabla x), \qquad
#    \phi_\delta(t) = \begin{cases} t^2 & |t| \le \delta \\ 2\delta|t| - \delta^2 & |t| > \delta \end{cases} $$
#
# A smooth interpolation between a quadratic roughness penalty and TV: differences below `δ` are
# treated as noise and smoothed quadratically, those above are preserved. Being differentiable
# everywhere, it needs no proximal step. `δ` is an absolute intensity — a good recipe is a low
# percentile of the finite differences of a preliminary reconstruction. Charbonnier, Blanc-Féraud,
# Aubert & Barlaud, *Deterministic edge-preserving regularization in computed imaging*, IEEE
# Transactions on Image Processing 6(2), 298–311 (1997), is the Huber-type potential this
# implements; Fessler (2010), cited above, is the model-based-MRI application. None of BART,
# SigPy or RegularizedLeastSquares.jl ships this potential — no direct equivalent.

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
# Same penalty as `TotalVariation2D`, over all three spatial dimensions; same references and
# toolbox equivalents.

# %%
x3d_tv = reconstruct(data3d, IterativeReconstruction(TotalVariation3D(1.0f-3); maxit = 30); verbosity = Silent())
println("3D TV NRMSE: ", round(nrmse(x3d_tv, x3d), digits = 4))
jim(x3d_tv[:, :, 9:4:29]; title = "TotalVariation3D, four slices", nrow = 1, size = (1000, 280))

# %% [markdown]
# ## 5. Non-convex sparsity
#
# $$ \lambda\|\mathcal{W}x\|_0 \qquad \text{or, as a constraint,} \qquad \|\mathcal{W}x\|_0 \le n $$
#
# $\ell_1$ shrinks the coefficients it keeps, so intensities are systematically underestimated.
# Hard thresholding does not shrink — at the price of non-convexity, which makes the result
# depend on the starting image. Warm-starting from the $\ell_1$ solution is the standard recipe;
# note the threshold is `sqrt(2γλ)` rather than `γλ`, so an ℓ₁ λ carried over gives a completely
# different sparsity level and has to be retuned. Blumensath & Davies, *Iterative hard
# thresholding for compressed sensing*, Applied and Computational Harmonic Analysis 27(3),
# 265–274 (2009), is the algorithm both the penalty form (`threshold`) and constraint form
# (`count`) come from. `L0Image`'s threshold form is BART's `-R H`; `L0Wavelet2D`'s threshold
# form is BART's `-R N` (NIHT on wavelet coefficients). Neither SigPy nor
# RegularizedLeastSquares.jl ships a hard-thresholding regularizer — no direct equivalent there.
# The `count` constraint form has no BART/SigPy/RegularizedLeastSquares.jl equivalent either; it
# is the sparsity analogue of [`RankLimit`](@ref) — see `docs/src/high-level/regularization.md`.
#
# On this phantom the ℓ₁ solution stays ahead on NRMSE — the argument for the ℓ₀ terms is
# unbiased amplitudes (lesion or vessel intensities that are not systematically shrunk), not a
# better global error.

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
    jim(x_hard; title = "L0 threshold"),
    jim(x_sparsity; title = "L0 count");
    layout = (1, 3), size = (1050, 300)
)

# %% [markdown]
# ## 6. Plug-and-play priors
#
# `PlugAndPlay` uses any callable `denoiser(image, σ)` as the proximal operator, i.e. as an
# implicit image prior — there is no explicit penalty $R(x)$ to write down, only its proximal
# operator, $\mathrm{prox}_{\gamma R}(x) = \mathrm{denoiser}(x, \sigma)$. Venkatakrishnan, Bouman
# & Wohlberg, *Plug-and-play priors for model based reconstruction*, Proc. IEEE GlobalSIP,
# 945–948 (2013), introduces the idea; Ahmad, Bouman, Buzzard et al., *Plug-and-play methods for
# magnetic resonance imaging*, IEEE Signal Processing Magazine 37(1), 105–116 (2020), surveys it
# for MRI specifically. RegularizedLeastSquares.jl has a matching `PlugAndPlayRegularization`;
# neither BART nor SigPy ships one.
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
# $$ \lambda\|x\|_{2,1} = \lambda \sum_{\text{pixels}} \Big(\sum_{\text{contrasts}} |x|^2\Big)^{1/2} $$
#
# Multi-echo / multi-contrast images of the same anatomy share their edge locations. The joint
# $\ell_{2,1}$ norm couples them so that a coefficient is either non-zero in every contrast or in
# none. Majumdar & Ward, *Joint reconstruction of multiecho MR images using correlated
# sparsity*, Magnetic Resonance Imaging 29(7), 899–906 (2011), is the reference. It matches
# RegularizedLeastSquares.jl's `L21Regularization`; BART and SigPy have no dedicated joint-
# sparsity flag.

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
println("joint NRMSE: ", round(nrmse(x_joint, echoes), digits = 4))
jim(x_joint; title = "JointSparsity — three echoes", nrow = 1, size = (900, 300))

# %% [markdown]
# ### `ReferencePrior`
#
# $$ \lambda\|x - x_{\text{ref}}\|_1 $$
#
# Penalizes the difference to a known image (a temporal average, a previous exam) instead of the
# image itself — the PICCS idea. Combine it with an ordinary sparsity term so a wrong reference
# cannot dominate. Chen, Tang & Leng, *Prior image constrained compressed sensing (PICCS)*,
# Medical Physics 35(2), 660–663 (2008), is the reference. None of BART, SigPy or
# RegularizedLeastSquares.jl ships a reference-prior term — no direct equivalent; the closest
# available building block elsewhere is a plain ℓ₁ penalty applied to a manually-formed
# difference image.
#
# `MRT` picks the solver automatically — two non-smooth terms together mean ADMM, without needing
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
# `NonNegative` and `BoxConstraint` are indicator functions, enforced by projection, so they
# carry no λ:
#
# $$ R(x) = \iota_C(x) = \begin{cases} 0 & x \in C \\ +\infty & x \notin C \end{cases} $$
#
# with $C = \{x \ge 0\}$ or $C = [a, b]$. They are defined for **real-valued** images only —
# applying them to a complex image (which is what a standard MRI reconstruction produces) throws.
# They belong on quantitative maps and magnitude-only models. They are BART's `-R POS`, SigPy's
# `BoxConstraint`, and RegularizedLeastSquares.jl's `PositiveRegularization`.

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
jim(sweep...; layout = (1, 4), size = (1300, 320))

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

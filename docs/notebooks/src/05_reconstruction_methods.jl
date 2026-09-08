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
# # 6 — Reconstruction methods
#
# Everything MRT can do is expressed as a *method* object handed to `reconstruct`. This notebook
# covers the ones that are not "iterative SENSE with a regularizer": direct reconstruction and
# coil combination, partial Fourier, autocalibrated parallel imaging (GRAPPA, SPIRiT), the data
# fidelity choices, and the signal models.
#
# ```
# ReconstructionMethod
# ├── DirectMethod     → DirectReconstruction, GRAPPA, Homodyne, PhaseConstrained, …
# └── IterativeMethod  → IterativeReconstruction, POCS, SPIRiT(iterative = true), …
# ```
#
# **Contents**
# 1. Direct reconstruction and coil combination
# 2. Partial Fourier — Homodyne, phase-constrained, POCS
# 3. GRAPPA and SPIRiT
# 4. Data fidelity — `L2Loss`, `HardConsistency`, `NoFidelity`
# 5. Signal models — `TemporalBasis`, `KSpaceToImage`
# 6. Checking applicability

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using NamedDims
using FFTW
using LinearAlgebra
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Direct reconstruction and coil combination
#
# `DirectReconstruction` is $\mathcal{A}^H y$: inverse FFT, then combine the coils. Which
# combination is used follows from the acquisition — if it carries sensitivity maps, the adjoint
# of the sensitivity operator does the combining ($\sum_c \bar s_c x_c$, the SNR-optimal one);
# if it does not, the reconstruction comes back per coil.
#
# The `coil_combination` keyword names the three strategies:
#
# - `AdjointSensitivity()` — $\sum_c \bar s_c x_c$
# - `RootSumSquares()` — $\sqrt{\sum_c |x_c|^2}$, needs no maps but loses the phase
# - `NoCoilCombination()` — keep the coil channels separate
#
# !!! note
#     In this version of MRT the keyword is only acted on by the methods that synthesize
#     k-space — `GRAPPA`, `SPIRiT` and the `KSpaceToImage` signal model (all three default to
#     `RootSumSquares()`). `DirectReconstruction` always applies $\mathcal{A}^H$, so its own
#     `coil_combination` argument currently changes nothing; the root-sum-of-squares of an
#     uncombined reconstruction is one line of Julia, shown below.

# %%
nx, ny, nc = 128, 128, 8
x_true = NamedDimsArray{(:x, :y)}(
    create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
)
smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))

acq_full = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil)}(zeros(ComplexF32, nx, ny, nc));
    is3D = false, sensitivity_maps = smaps
)
data_full = simulate_acquisition(x_true, acq_full)

# With sensitivity maps: one combined image.
x_adj = reconstruct(data_full, DirectReconstruction(); verbosity = Silent())
println("with maps:    ", size(x_adj), " ", dimnames(x_adj))

# The same k-space, described without maps: one image per coil.
acq_nomaps = AcquisitionInfo(data_full.kspace_data; is3D = false)
x_coils = reconstruct(acq_nomaps, DirectReconstruction(); verbosity = Silent())
println("without maps: ", size(x_coils), " ", dimnames(x_coils))

# Root sum of squares over the coil dimension.
x_rss = sqrt.(sum(abs2, unname(x_coils); dims = 3)[:, :, 1])

jim(
    jim(x_adj; title = "adjoint sensitivity combination"),
    jim(x_rss; title = "root sum of squares");
    layout = (1, 2), size = (800, 350)
)

# %%
jim(x_coils; title = "uncombined coil images", nrow = 2, size = (800, 400))

# %% [markdown]
# ## 2. Partial Fourier
#
# A partial-Fourier acquisition measures somewhat more than half of k-space and relies on
# conjugate symmetry for the rest. MRT detects the asymmetric band from the sampling pattern.

# %%
Nx, Ny = 128, 128

# A phantom with smooth phase — partial Fourier lives or dies on the phase estimate.
mag = abs.(create_shepp_logan_phantom(Nx, Ny, :axial; ti = MRISheppLoganIntensities()))
X = [(x - Nx / 2) / Nx for x in 1:Nx, y in 1:Ny]
Y = [(y - Ny / 2) / Ny for x in 1:Nx, y in 1:Ny]
img_pf = ComplexF32.(mag .* cis.(0.8f0 .* (X .+ Y)))

# 65% of the phase encodes, on one side
frac = 0.65
mask_y = falses(Ny)
mask_y[1:round(Int, frac * Ny)] .= true

acq_pf = simulate_acquisition(
    img_pf,
    CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_y)
    )
)
println("acquired k-space: ", size(acq_pf.kspace_data), " of ", (Nx, Ny))

band = partial_fourier_band(acq_pf)
println("partial-Fourier band: dimension ", band.dim, ", lines ", first(band.acquired_range), ":", last(band.acquired_range))

# %%
# The partial-Fourier methods in this version return an image scaled by `sqrt(Nx*Ny)` relative
# to the adjoint path, so the errors below are amplitude-aligned (the least-squares scale factor
# is divided out before comparing) — which is what one does anyway when comparing magnitude
# images from different pipelines.
function rel(x̂)
    a = abs.(unname(x̂))
    α = sum(a .* mag) / sum(abs2, a)
    return norm(α .* a - mag) / norm(mag)
end

x_zf = reconstruct(acq_pf; verbosity = Silent())
x_hom_lin = reconstruct(acq_pf, Homodyne(filter = LinearRamp()); verbosity = Silent())
x_hom_step = reconstruct(acq_pf, Homodyne(filter = StepRamp()); verbosity = Silent())
x_pc = reconstruct(acq_pf, PhaseConstrained(); verbosity = Silent())
x_pocs = reconstruct(acq_pf, POCS(maxit = 30); verbosity = Silent())

for (label, x̂) in (
        ("zero-filled", x_zf), ("Homodyne / LinearRamp", x_hom_lin), ("Homodyne / StepRamp", x_hom_step),
        ("PhaseConstrained", x_pc), ("POCS", x_pocs),
    )
    println(rpad(label, 24), " relative magnitude error ", round(rel(x̂), digits = 4))
end

jim(
    jim(abs.(unname(x_zf)); title = "zero-filled"),
    jim(abs.(unname(x_hom_lin)); title = "Homodyne"),
    jim(abs.(unname(x_pc)); title = "PhaseConstrained"),
    jim(abs.(unname(x_pocs)); title = "POCS");
    layout = (2, 2), size = (800, 700)
)

# %% [markdown]
# The two filters shape the transition band of the homodyne weighting; `POCS` iterates between
# enforcing the measured samples and the estimated phase, and `PhaseConstrained` solves a
# least-squares problem with the phase fixed. All three use only the acquired band — no
# sensitivity maps needed.

# %% [markdown]
# ## 3. GRAPPA and SPIRiT
#
# Autocalibrated parallel imaging fills in the missing k-space lines from a kernel fitted on a
# fully-sampled autocalibration (ACS) region — no explicit sensitivity maps are used for the
# interpolation itself.

# %%
Nc = 8
img_pi = ComplexF32.(abs.(create_shepp_logan_phantom(Nx, Ny, :axial; ti = MRISheppLoganIntensities())))
sens = coil_sensitivities(Nx, Ny, Nc)

# R = 2 with a 24-line ACS block in the centre
R = 2
acs = (Ny ÷ 2 - 11):(Ny ÷ 2 + 12)
mask_pi = falses(Ny)
mask_pi[1:R:Ny] .= true
mask_pi[acs] .= true
println("net acceleration: ", round(Ny / sum(mask_pi), digits = 2), "×")

acq_pi = simulate_acquisition(
    img_pi,
    CartesianAcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_pi), sensitivity_maps = sens
    )
)

# %%
# Amplitude-aligned again: GRAPPA and SPIRiT synthesize k-space and combine coils by
# root-sum-of-squares, which carries its own scale.
function relpi(x̂)
    a = abs.(unname(x̂))
    α = sum(a .* abs.(img_pi)) / sum(abs2, a)
    return norm(α .* a - abs.(img_pi)) / norm(abs.(img_pi))
end

x_grappa = reconstruct(acq_pi, GRAPPA(kernel_size = (3, 2), calib_size = (Nx, 24)); verbosity = Silent())
x_spirit = reconstruct(acq_pi, SPIRiT(kernel_size = (5, 5), calib_size = (Nx, 24), maxit = 30); verbosity = Silent())
x_sense = reconstruct(acq_pi, IterativeReconstruction(L2Image(1.0f-5); maxit = 30); verbosity = Silent())

println("GRAPPA         ", round(relpi(x_grappa), digits = 4))
println("SPIRiT         ", round(relpi(x_spirit), digits = 4))
println("CG-SENSE (L2)  ", round(relpi(x_sense), digits = 4))

jim(
    jim(abs.(unname(x_grappa)); title = "GRAPPA"),
    jim(abs.(unname(x_spirit)); title = "SPIRiT"),
    jim(abs.(unname(x_sense)); title = "CG-SENSE");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# On this phantom GRAPPA and CG-SENSE recover the image almost exactly (the data is noiseless
# and R = 2 with a generous ACS block), while SPIRiT lags behind by an order of magnitude and
# does not improve with more iterations, a larger kernel or a different `λ`. Treat the SPIRiT
# numbers here as a demonstration of the interface rather than of achievable SPIRiT quality.

# %%
# SPIRiT can also be run as an iterative k-space problem: the SPIRiT kernel becomes a
# consistency term on the full multi-channel k-space (`KSpaceToImage` signal model).
x_spirit_it = reconstruct(
    acq_pi, SPIRiT(kernel_size = (5, 5), calib_size = (Nx, 24), maxit = 30, iterative = true);
    verbosity = Silent()
)
println("SPIRiT (iterative) ", round(relpi(x_spirit_it), digits = 4))

# %% [markdown]
# ## 4. Data fidelity
#
# `IterativeReconstruction(...; fidelity = ...)` chooses how the measurements enter the problem.
#
# - `L2Loss()` (default) — the penalty $\tfrac12\|\mathcal{A}x-y\|^2$.
# - `HardConsistency()` — the constraint $\{x : \mathcal{A}x = y\}$, enforced by projection.
#   Closed-form when $\mathcal{A}\mathcal{A}^*$ is diagonal, otherwise an inner CG.
# - `NoFidelity()` — no data term at all; the "reconstruction" is then pure denoising of the
#   initial estimate, which is occasionally what you want (or a building block for a custom model).

# %%
# Undersampled, noisy, so the three behave differently.
acq_us = AcquisitionInfo(;
    is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens,
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (Nx, Ny)
    ),
)
data_us = simulate_acquisition(img_pi + 0.02f0 * randn(ComplexF32, Nx, Ny), acq_us)

x_l2 = reconstruct(data_us, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40); verbosity = Silent())
x_hc = reconstruct(
    data_us,
    IterativeReconstruction(
        L1Wavelet2D(2.0f-3); fidelity = HardConsistency(maxit = 20),
        algorithm = DouglasRachford(), maxit = 40
    );
    verbosity = Silent()
)
x_nf = reconstruct(
    data_us,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); fidelity = NoFidelity(), algorithm = FISTA(), maxit = 20);
    verbosity = Silent()
)

println("L2Loss          ", round(relpi(x_l2), digits = 4))
println("HardConsistency ", round(relpi(x_hc), digits = 4))
println("NoFidelity      ", round(relpi(x_nf), digits = 4), "   (denoising of the initial estimate)")

jim(
    jim(x_l2; title = "L2Loss"),
    jim(x_hc; title = "HardConsistency"),
    jim(x_nf; title = "NoFidelity");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# ## 5. Signal models
#
# A signal model changes what the optimization variable *is*.
#
# ### `TemporalBasis` — subspace reconstruction
#
# The variable holds $K$ coefficient maps, expanded to $N_t$ frames by a basis $\Phi$. This is
# the standard model for relaxometry and MR fingerprinting, and for cine data whose temporal
# behaviour is known to be low-dimensional.

# %%
n, nt, K = 64, 24, 4

# A series that really is low-dimensional: three tissue classes with different time courses.
base = abs.(create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities()))
roi1 = base .> 0.9
roi2 = (base .> 0.3) .& (base .≤ 0.9)
roi3 = (base .> 0.05) .& (base .≤ 0.3)

t = range(0, 1; length = nt)
curves = [exp.(-2 .* t), 1 .- exp.(-3 .* t), 0.5 .+ 0.4 .* sin.(2π .* t)]

series = zeros(ComplexF32, n, n, nt)
for (roi, curve) in zip((roi1, roi2, roi3), curves), k in 1:nt
    series[:, :, k] .+= ComplexF32(curve[k]) .* roi
end
series = NamedDimsArray{(:x, :y, :time)}(series)

# The temporal basis: the leading left singular vectors of the Casorati matrix.
casorati = reshape(unname(series), n * n, nt)
Φ = Matrix{ComplexF32}(svd(casorati').U[:, 1:K])
println("basis Φ: ", size(Φ))

# %%
acq_dyn = CartesianAcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :time)}(zeros(ComplexF32, n, n, nt)); is3D = false
)
data_dyn = simulate_acquisition(series, acq_dyn)

x_sub = reconstruct(
    data_dyn,
    IterativeReconstruction(; signal_model = TemporalBasis(Φ; time_dim = :time), algorithm = CGNR(), maxit = 20);
    verbosity = Silent()
)
println("subspace reconstruction: ", size(x_sub), " ", dimnames(x_sub))
println("relative error: ", round(norm(unname(x_sub) - unname(series)) / norm(unname(series)), digits = 4))

jim(
    jim(abs.(unname(series)[:, :, 1]); title = "frame 1 — truth"),
    jim(abs.(unname(x_sub)[:, :, 1]); title = "frame 1 — subspace"),
    jim(abs.(unname(series)[:, :, nt]); title = "frame $nt — truth"),
    jim(abs.(unname(x_sub)[:, :, nt]); title = "frame $nt — subspace");
    layout = (2, 2), size = (800, 700)
)

# %% [markdown]
# ### `KSpaceToImage`
#
# The optimization variable is the full multi-channel k-space rather than the image; data
# consistency is enforced through the subsampling operator alone, and the result is transformed
# to an image afterwards. This is the model `SPIRiT(; iterative = true)` runs on.

# %%
x_ksp = reconstruct(
    acq_pi,
    IterativeReconstruction(;
        signal_model = KSpaceToImage(AdjointSensitivity()), algorithm = CGNR(), maxit = 10
    );
    verbosity = Silent()
)
println("k-space-domain solve: ", size(x_ksp), "  error ", round(relpi(x_ksp), digits = 4))

# %% [markdown]
# ## 6. Checking applicability
#
# `check_applicable(method, acq)` is the hook that decides whether a method can run on given
# data; `reconstruct` calls it for you, and a custom method overrides it. It is what produces the
# error below, rather than a confusing failure deeper in the pipeline.

# %%
using MriReconstructionToolbox: check_applicable

# GRAPPA needs a Cartesian acquisition with an autocalibration region.
try
    check_applicable(GRAPPA(), data_us)          # random variable-density pattern, no ACS block
catch e
    println(sprint(showerror, e))
end

# %%
check_applicable(GRAPPA(calib_size = (Nx, 24)), acq_pi)   # returns without complaint
println("GRAPPA is applicable to the R = 2 + ACS acquisition")

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
# # 4 — Reconstruction methods
#
# Everything MRT can do is expressed as a *method* object handed to `reconstruct`. A method says
# **what kind of reconstruction this is** — how the measurements become an image. This notebook is
# a tour of the methods that are not "iterative SENSE with a regularizer": direct reconstruction
# and coil combination, partial Fourier, and autocalibrated parallel imaging (GRAPPA, SPIRiT).
# Data-fidelity choices and signal models — including calibrationless structured low-rank k-space
# filling — are the subject of
# [`11_advanced_reconstruction`](11_advanced_reconstruction.ipynb).
#
# ```
# ReconstructionMethod
# ├── DirectMethod     → DirectReconstruction, GRAPPA, Homodyne, PhaseConstrained, …
# └── IterativeMethod  → IterativeReconstruction, POCS, SPIRiT(iterative = true), …
# ```
#
# Several of these methods do iterate internally, and a few of the comparisons below run
# `IterativeReconstruction` as a reference. That is deliberately kept in the background here: how
# the iteration is *driven* — which solver, how many iterations, what stopping tolerance, how much
# the run prints — is the subject of the next notebook,
# [`06_algorithms_and_configuration`](06_algorithms_and_configuration.ipynb). Where this notebook
# passes `maxit` or `algorithm`, treat the values as "enough to converge on this small phantom"
# and look there for how to choose them.
#
# **Contents**
# 1. Direct reconstruction and coil combination
# 2. Partial Fourier — Homodyne, phase-constrained, POCS
# 3. GRAPPA and SPIRiT
# 4. Checking applicability

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms:
    create_shepp_logan_phantom, create_torso_phantom, MRISheppLoganIntensities, TissueMask
using MIRTjim: jim
using Plots
using NamedDims
using FFTW
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Direct reconstruction and coil combination
#
# `DirectReconstruction` is $\mathcal{A}^H y$: inverse FFT, then combine the coils. The
# `coil_combination` keyword names the three strategies:
#
# - `AdjointSensitivity()` (the default) — $\sum_c \bar s_c x_c$, the SNR-optimal combination, and
#   the only one that keeps the image's phase. It needs sensitivity maps.
# - `RootSumSquares()` — $\sqrt{\sum_c |x_c|^2}$, needs no maps but discards the phase.
# - `NoCoilCombination()` — keep the coil channels separate.
#
# All three are honored by `DirectReconstruction` on Cartesian data, and by the methods that
# synthesize k-space (`GRAPPA`, `SPIRiT`, and the `KSpaceToImage` signal model — those three
# default to `RootSumSquares()`, since they do not need maps for anything else). On non-Cartesian
# data `DirectReconstruction` supports `AdjointSensitivity()` only; the other two raise an error
# rather than silently ignoring the request.
#
# When the acquisition carries no sensitivity maps and the combination is the default
# `AdjointSensitivity()`, there is nothing to combine with, so the reconstruction comes back per
# coil.

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

# With sensitivity maps and the default combination: one combined image.
x_adj = reconstruct(data_full, DirectReconstruction(); verbosity = Silent())
println("AdjointSensitivity: ", size(x_adj), " ", dimnames(x_adj))

# Same data, root sum of squares: also one image, but no phase.
x_rss = reconstruct(data_full, DirectReconstruction(RootSumSquares()); verbosity = Silent())
println("RootSumSquares:     ", size(x_rss), " ", dimnames(x_rss))

# Same data, coils kept apart.
x_coils = reconstruct(data_full, DirectReconstruction(NoCoilCombination()); verbosity = Silent())
println("NoCoilCombination:  ", size(x_coils), " ", dimnames(x_coils))

# The same k-space described *without* maps also comes back per coil under the default.
acq_nomaps = AcquisitionInfo(data_full.kspace_data; is3D = false)
x_nomaps = reconstruct(acq_nomaps, DirectReconstruction(); verbosity = Silent())
println("no maps, default:   ", size(x_nomaps), " ", dimnames(x_nomaps))

# %%
jim(x_coils; title = "uncombined coil images", nrow = 2, size = (800, 400))

# %% [markdown]
# ### Where the two combinations actually differ
#
# On noiseless data with normalized maps ($\sum_c |s_c|^2 \equiv 1$, which is what
# `coil_sensitivities` produces) the two magnitude images are nearly indistinguishable — which is
# why a side-by-side of them teaches nothing. The difference is a **noise** effect, and it shows up
# in two places:
#
# - Root sum of squares is a *biased* magnitude estimator. Squaring and adding the coil channels
#   rectifies the noise, so signal-free regions acquire a positive floor that grows with the coil
#   count; the sensitivity-weighted sum keeps noise zero-mean and complex.
# - The sensitivity-weighted sum is the matched filter for the coil array, so it is SNR-optimal;
#   root sum of squares is not, and loses the most where a single coil dominates.
#
# So the comparison below is run on noisy data, and the figure carries a difference panel on its
# own color scale next to the two magnitude images.

# %%
data_noisy = add_noise(data_full; snr_db = 12)

xn_adj = reconstruct(data_noisy, DirectReconstruction(); verbosity = Silent())
xn_rss = reconstruct(data_noisy, DirectReconstruction(RootSumSquares()); verbosity = Silent())

println("NRMSE vs. truth")
println("  AdjointSensitivity ", round(nrmse(xn_adj, x_true), digits = 4))
println("  RootSumSquares     ", round(nrmse(xn_rss, x_true), digits = 4))

background = abs.(unname(x_true)) .< 1.0e-6
println("mean magnitude in the signal-free background (the RSS noise floor)")
println("  AdjointSensitivity ", round(mean(abs.(unname(xn_adj))[background]), digits = 4))
println("  RootSumSquares     ", round(mean(abs.(unname(xn_rss))[background]), digits = 4))

# %%
side_by_side(
    unname(xn_adj), unname(xn_rss);
    titles = ("adjoint sensitivity", "root sum of squares"), size = (1100, 360)
)

# %%
difference_image(
    unname(xn_rss), unname(xn_adj);
    title = "|RSS| - |adjoint sensitivity|", size = (450, 380)
)

# %% [markdown]
# The difference image is not noise-shaped scatter: it is a picture of the object, brightest where
# the phantom is dark, because that is where the rectification bias is largest relative to the
# signal. Only `AdjointSensitivity` keeps the phase, so it is the one to use whenever the phase
# matters (partial Fourier, off-resonance correction, phase-contrast flow).

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
# Every method here returns an image in the data's own units — the zero-filled adjoint, the three
# partial-Fourier methods and the phantom are all on one scale, so a plain NRMSE against the
# magnitude phantom is meaningful with no amplitude alignment. (The least-squares scale factor
# that would align them is printed below to make that concrete: it is 1 to within a percent.)
x_zf = reconstruct(acq_pf; verbosity = Silent())
x_hom_lin = reconstruct(acq_pf, Homodyne(filter = LinearRamp()); verbosity = Silent())
x_hom_step = reconstruct(acq_pf, Homodyne(filter = StepRamp()); verbosity = Silent())
x_pc = reconstruct(acq_pf, PhaseConstrained(); verbosity = Silent())
x_pocs = reconstruct(acq_pf, POCS(maxit = 30); verbosity = Silent())

for (label, x̂) in (
        ("zero-filled", x_zf), ("Homodyne / LinearRamp", x_hom_lin), ("Homodyne / StepRamp", x_hom_step),
        ("PhaseConstrained", x_pc), ("POCS", x_pocs),
    )
    a = abs.(unname(x̂))
    α = sum(a .* mag) / sum(abs2, a)
    println(rpad(label, 24), " NRMSE ", rpad(round(nrmse(x̂, mag), digits = 4), 8), " (scale factor ", round(α, digits = 3), ")")
end

# %%
side_by_side(
    unname(x_zf), unname(x_hom_lin), unname(x_pc), unname(x_pocs);
    titles = ("zero-filled", "Homodyne", "PhaseConstrained", "POCS"), size = (1300, 340)
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
#
# The two differ in what the kernel is fitted to do:
#
# - **GRAPPA** (Griswold 2002) fits, for each missing-line offset, the weights that predict one
#   target sample from a neighbourhood of *acquired* lines. It therefore needs a regular
#   undersampling pattern, and it fills each hole once.
# - **SPIRiT** (Lustig & Pauly 2010) fits a kernel that predicts *every* sample from all of its
#   neighbours, acquired or not — a self-consistency relation $k = G k$ on the whole multi-channel
#   k-space. That relation is then iterated to a fixed point while the acquired samples are held.
#   It is not restricted to a regular pattern.

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

# Noise is what makes this a comparison rather than a formality: on noiseless data at R = 2 every
# method below recovers the phantom to within a fraction of a percent.
acq_pi = add_noise(
    simulate_acquisition(
        img_pi,
        CartesianAcquisitionInfo(;
            is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_pi), sensitivity_maps = sens
        )
    );
    snr_db = 30
)

# %%
x_grappa = reconstruct(acq_pi, GRAPPA(kernel_size = (3, 2), calib_size = (Nx, 24)); verbosity = Silent())
x_spirit = reconstruct(acq_pi, SPIRiT(kernel_size = (5, 5), calib_size = (Nx, 24), maxit = 30); verbosity = Silent())
x_sense = reconstruct(acq_pi, IterativeReconstruction(L2Image(1.0f-3); maxit = 30); verbosity = Silent())

println("GRAPPA         ", round(nrmse(x_grappa, img_pi), digits = 4))
println("SPIRiT         ", round(nrmse(x_spirit, img_pi), digits = 4))
println("CG-SENSE (L2)  ", round(nrmse(x_sense, img_pi), digits = 4))

side_by_side(
    unname(x_grappa), unname(x_spirit), unname(x_sense);
    titles = ("GRAPPA", "SPIRiT", "CG-SENSE"), size = (1200, 350)
)

# %% [markdown]
# CG-SENSE wins here because it is the only one of the three given the *true* sensitivity maps;
# GRAPPA and SPIRiT calibrate everything they know from the 24-line ACS block. Between the two
# autocalibrated methods SPIRiT is the more accurate, which is what its extra work buys: a 5×5
# kernel over all coils, applied to every k-space location rather than only to the holes.
#
# !!! note "A tuning knob worth knowing about"
#     `SPIRiT(; calib_λ = 1e-4)` is a relative Tikhonov penalty on the *calibration* solve (not on
#     the reconstruction). Neighbouring ACS samples are highly correlated, so the fit is close to
#     rank-deficient and the unregularized kernel amplifies noise. The default is small; raise it
#     on low-SNR data, set it to `0` for the plain least-squares fit.

# %%
# SPIRiT can also be run as an iterative k-space problem: the SPIRiT kernel becomes a
# consistency term on the full multi-channel k-space (`KSpaceToImage` signal model, notebook 11
# §2.2).
x_spirit_it = reconstruct(
    acq_pi, SPIRiT(kernel_size = (5, 5), calib_size = (Nx, 24), maxit = 30, iterative = true);
    verbosity = Silent()
)
println("SPIRiT (fixed point) ", round(nrmse(x_spirit, img_pi), digits = 4))
println("SPIRiT (iterative)   ", round(nrmse(x_spirit_it, img_pi), digits = 4))

# %% [markdown]
# ## 4. Checking applicability
#
# `check_applicable(method, acq)` decides whether a method can run on given data. `reconstruct`
# calls it for you, before any work is done, so an unsupported combination fails with a sentence
# that names the problem instead of producing a plausible-looking wrong image.
#
# `GRAPPA` is the method with the most to check. Its kernel is fitted once per missing-line offset
# $t = 1 \ldots R-1$ and then applied everywhere, which presupposes:
#
# 1. a Cartesian acquisition with fully sampled readout lines,
# 2. acquired phase-encoding lines on a **regular lattice** of stride $R$, and
# 3. a contiguous fully sampled ACS block, long enough for the kernel.
#
# Requirement 2 is the one that surprises people: **GRAPPA cannot reconstruct randomly
# undersampled data at all.** There is no "GRAPPA kernel" for an irregular pattern — the weights
# are defined by a fixed geometric relationship between a hole and its neighbours, and a random
# mask does not have one. (A variable-density mask often *does* contain a fully sampled centre,
# so the presence of an ACS region is not what disqualifies it.)

# %%
using MriReconstructionToolbox: check_applicable

# The variable-density pattern from §4: it even has a fully sampled centre, but its acquired lines
# are not on any lattice.
try
    check_applicable(GRAPPA(), data_us)
catch e
    println(sprint(showerror, e))
end

# %%
# Regular stride, but no ACS block at all: nothing to calibrate the kernel on. `check_applicable`
# inspects the sampling pattern of *acquired data*, so this needs simulated k-space, not just the
# empty `AcquisitionInfo` description.
mask_no_acs = falses(Ny)
mask_no_acs[1:2:Ny] .= true
acq_no_acs = simulate_acquisition(
    img_pi,
    AcquisitionInfo(;
        is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens, subsampling = (:, mask_no_acs)
    )
)
try
    check_applicable(GRAPPA(), acq_no_acs)
catch e
    println(sprint(showerror, e))
end

# %%
# The R = 2 + ACS acquisition from §3 passes.
check_applicable(GRAPPA(calib_size = (Nx, 24)), acq_pi)
println("GRAPPA is applicable to the R = 2 + ACS acquisition")

# %% [markdown]
# For the patterns GRAPPA rejects, the alternatives are the ones this notebook has already shown:
# `SPIRiT`, whose self-consistency relation holds at every k-space location and so does not care
# about the lattice (it still needs a calibration region), or an `IterativeReconstruction` with a
# sparsity prior — which is what a variable-density mask was designed for in the first place.
#
# A custom method plugs into the same hook: subtype `DirectMethod` or `IterativeMethod` and
# override `check_applicable` to state your own preconditions (see notebook 12).

# %% [markdown]
# ## References
#
# - Griswold M. A. *et al.*, *Generalized autocalibrating partially parallel acquisitions
#   (GRAPPA)*, Magn. Reson. Med. 47:1202–1210 (2002).
# - Lustig M., Pauly J. M., *SPIRiT: Iterative self-consistent parallel imaging reconstruction from
#   arbitrary k-space*, Magn. Reson. Med. 64:457–471 (2010).
# - Noll D. C., Nishimura D. G., Macovski A., *Homodyne detection in magnetic resonance imaging*,
#   IEEE Trans. Med. Imaging 10:154–163 (1991).

# %% [markdown]
# ## Environment

# %%
print_versions()

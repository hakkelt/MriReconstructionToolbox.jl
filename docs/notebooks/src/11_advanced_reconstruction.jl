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
# # 11 — Advanced reconstruction methods
#
# Two mechanisms go past the "operator + regularizer" pattern the rest of the notebooks use:
# **data fidelity** terms that change how the mismatch to the measured k-space is scored, and
# **signal models** that change what the optimization variable itself is. Structured low-rank
# k-space filling — including ALOHA's transform-domain weighting — is presented here too, as an
# application of the `KSpaceToImage` signal model.
#
# **Contents**
# 1. Data fidelity — `L2Loss`, `HardConsistency`, `NoFidelity`
# 2. Signal models — `TemporalBasis`, `KSpaceToImage`, and calibrationless structured low-rank

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms: create_shepp_logan_phantom, create_torso_phantom, MRISheppLoganIntensities, TissueMask
using MIRTjim: jim
using Plots
using AbstractOperators: Hankel
using NamedDims
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

# %% [markdown]
# `img_pi`, `sens` and `acq_pi` below are the same R = 2, 8-channel, 24-line-ACS phantom setup
# used throughout notebook 4 — reproduced here so this notebook runs standalone.

# %%
Nx, Ny, Nc = 128, 128, 8
img_pi = ComplexF32.(abs.(create_shepp_logan_phantom(Nx, Ny, :axial; ti = MRISheppLoganIntensities())))
sens = coil_sensitivities(Nx, Ny, Nc)

R = 2
acs = (Ny ÷ 2 - 11):(Ny ÷ 2 + 12)
mask_pi = falses(Ny)
mask_pi[1:R:Ny] .= true
mask_pi[acs] .= true

acq_pi = add_noise(
    simulate_acquisition(
        img_pi,
        CartesianAcquisitionInfo(;
            is3D = false, image_size = (Nx, Ny), subsampling = (:, mask_pi), sensitivity_maps = sens
        )
    );
    snr_db = 30
)

# %% [markdown]
# ## 1. Data fidelity
#
# `IterativeReconstruction(...; fidelity = ...)` chooses how the measurements enter the problem.
#
# - `L2Loss()` (default) — the penalty $\tfrac12\|\mathcal{A}x-y\|^2$.
# - `HardConsistency()` — the constraint $\{x : \mathcal{A}x = y\}$, enforced by projection.
#   Closed-form when $\mathcal{A}\mathcal{A}^*$ is diagonal, otherwise an inner CG.
# - `NoFidelity()` — no data term at all; the "reconstruction" is then pure denoising of the
#   initial estimate, which is occasionally what you want (or a building block for a custom model).
#
# The choice is not a matter of taste: it is a statement about how much you trust `y`. An equality
# constraint says the measurements are *exact*. That is why the comparison below is run on
# noiseless data first — and why the second cell then shows what noise does to it.
#
# The algorithms named below (`DouglasRachford`, `FISTA`) are picked because they accept the
# corresponding term; notebook 6 covers which solver goes with which problem.

# %%
# A 4x variable-density mask, first with exact (noiseless) measurements.
acq_us = AcquisitionInfo(;
    is3D = false, image_size = (Nx, Ny), sensitivity_maps = sens,
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), (Nx, Ny)
    ),
)
data_clean = simulate_acquisition(img_pi, acq_us)
data_us = add_noise(data_clean; snr_db = 30)

x_l2 = reconstruct(data_clean, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40); verbosity = Silent())
x_hc = reconstruct(
    data_clean,
    IterativeReconstruction(
        L1Wavelet2D(2.0f-3); fidelity = HardConsistency(maxit = 20),
        algorithm = DouglasRachford(), maxit = 40
    );
    verbosity = Silent()
)
x_nf = reconstruct(
    data_clean,
    IterativeReconstruction(L1Wavelet2D(2.0f-3); fidelity = NoFidelity(), algorithm = FISTA(), maxit = 20);
    verbosity = Silent()
)

println("zero-filled     ", round(nrmse(reconstruct(data_clean; verbosity = Silent()), img_pi), digits = 4))
println("L2Loss          ", round(nrmse(x_l2, img_pi), digits = 4))
println("HardConsistency ", round(nrmse(x_hc, img_pi), digits = 4))
println("NoFidelity      ", round(nrmse(x_nf, img_pi), digits = 4), "   (denoising of the initial estimate)")

side_by_side(
    unname(x_l2), unname(x_hc), unname(x_nf);
    titles = ("L2Loss", "HardConsistency", "NoFidelity"), size = (1200, 350)
)

# %% [markdown]
# `NoFidelity` is the outlier, and it should be: with no data term the solver never looks at `y`
# at all, so it can only denoise the zero-filled adjoint it started from. The aliasing the other two
# *undo* is merely smoothed, which is why the result lands slightly behind the zero-filled image it
# began with. It is a building block, not a reconstruction.

# %% [markdown]
# ### Why `HardConsistency` is a statement about the noise
#
# $\{x : \mathcal{A}x = y\}$ asks the solution to reproduce every measured sample exactly — noise
# included. With a well-conditioned encoding that is harmless. With sensitivity maps and heavy
# undersampling, $\mathcal{A}\mathcal{A}^*$ has very small eigenvalues, and satisfying the
# constraint along those directions means multiplying the noise by their inverse. The projection is
# doing exactly what was asked, so the failure mode is *silent*: the error grows the harder the
# solver works.

# %%
for label in ("noiseless", "SNR 30 dB")
    data = label == "noiseless" ? data_clean : data_us
    errs = map((10, 40, 100)) do maxit
        x̂ = reconstruct(
            data,
            IterativeReconstruction(
                L1Wavelet2D(2.0f-3); fidelity = HardConsistency(maxit = 20),
                algorithm = DouglasRachford(), maxit = maxit
            );
            verbosity = Silent()
        )
        round(nrmse(x̂, img_pi), digits = 4)
    end
    println(rpad(label, 12), " HardConsistency NRMSE at maxit = 10 / 40 / 100: ", join(errs, "  "))
end

x_l2_noisy = reconstruct(data_us, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40); verbosity = Silent())
println("SNR 30 dB    L2Loss NRMSE at maxit = 40:                ", round(nrmse(x_l2_noisy, img_pi), digits = 4))

# %% [markdown]
# On exact data more iterations help, as they should. On noisy data the same run gets *worse* with
# every iteration, and ends far behind the `L2Loss` reconstruction of the same data. The rule that
# follows:
#
# - **`L2Loss` is the default for measured data**, because $\tfrac12\|\mathcal{A}x-y\|^2$ tolerates
#   noise by construction and the regularizer sets how much.
# - **`HardConsistency` belongs where the constraint is well posed**: essentially noiseless data, or
#   — much more usefully — a problem in which $\mathcal{A}\mathcal{A}^*$ is *diagonal*, where the
#   projection is closed-form and needs no inner CG at all. That is exactly the case for the
#   `KSpaceToImage` signal model of §2.2, and it is why `SPIRiT(; iterative = true)` uses hard
#   consistency: there $\mathcal{A}$ is the subsampling operator, so "keep the measured samples"
#   really is just "keep the measured samples".
# - **`NoFidelity`** is for denoising an estimate you already have, or as a component of a custom
#   model.


# %% [markdown]
# ## 2. Signal models
#
# A regularizer says what an image *should look like*. A signal model goes further: it changes
# what the optimization variable **is**, so that the unknown is smaller than the image series and
# the model is imposed exactly rather than penalized.
#
# ```
# variable  ──ℳ──▶  image series  ──𝒜──▶  k-space
#    c                  x(r, t)                y
# ```
#
# MRT ships two: `TemporalBasis`, whose variable is a set of subspace coefficient maps, and
# `KSpaceToImage`, whose variable is the multi-channel k-space itself.


# %% [markdown]
# ### 2.1 `TemporalBasis` — subspace (low-rank) modelling of the time dimension
#
# #### What a temporal basis is
#
# Stack a dynamic or multi-contrast series as a **Casorati matrix** $X \in \mathbb{C}^{N \times
# N_t}$: one row per voxel, one column per frame/echo/contrast. Nothing forces $X$ to have full
# rank. If the signal at every voxel is one of a small family of time courses — the same
# exponential decay at different rates, the same cardiac cycle at different amplitudes — then those
# $N$ rows all live in a $K$-dimensional subspace of $\mathbb{C}^{N_t}$ with $K \ll N_t$.
#
# Write an orthonormal basis of that subspace as the columns of $\Phi \in \mathbb{C}^{N_t \times
# K}$. Then
#
# $$X \approx C\,\Phi^{\mathsf T}, \qquad\text{i.e.}\qquad x(r, t) \;=\; \sum_{k=1}^{K} \Phi(t, k)\, c(r, k),$$
#
# and the unknown is the **coefficient array** $c \in \mathbb{C}^{N_x \times N_y \times K}$ rather
# than the $N_x \times N_y \times N_t$ series. `TemporalBasis(Φ; time_dim)` installs exactly this
# map, so the operator the solver sees is $\mathcal{A}\,\mathcal{M}_\Phi$ and the reconstruction
# solves for $c$; `reconstruct` expands the result back to the full series before returning it.
#
# Two things follow, and they are the whole reason to do it:
#
# - **The problem shrinks.** $K/N_t$ as many unknowns, so a given number of measurements goes
#   further — this is what makes high accelerations feasible.
# - **The model is a hard constraint, not a penalty.** Anything outside the subspace — including
#   most of the noise, and undersampling artifacts that do not resemble a plausible time course —
#   cannot be represented at all. A subspace reconstruction denoises for free.
#
# This is the "low-rank"/"partially separable" idea of Liang's *k-t* PCA line of work (Liang 2007;
# Pedersen 2009; Petzschner 2011), and it is what T2-shuffling (Tamir et al., MRM 2017) and MR
# fingerprinting reconstructions are built on.

# %% [markdown]
# #### A phantom that really is low-dimensional
#
# The torso phantom from `GeometricMedicalPhantoms` can be asked for one tissue at a time
# (`TissueMask(; heart = true)` and friends), which makes it easy to build a physically meaningful
# multi-echo series: give each tissue a $T_2$ and a proton density, and sample the decay
# $M_0 e^{-\mathrm{TE}/T_2}$ at 24 echo times. This is a spin-echo train, the acquisition
# T2-shuffling was designed for.

# %%
n, nt, ncoils = 96, 24, 4

tissue_T2 = (lung = 60.0, heart = 50.0, bones = 20.0, body = 90.0, lv_blood = 250.0, rv_blood = 250.0)
tissue_M0 = (lung = 0.35, heart = 0.85, bones = 0.25, body = 0.70, lv_blood = 1.0, rv_blood = 1.0)

TE = collect(range(10, 240; length = nt))     # ms

series = zeros(ComplexF32, n, n, nt)
tissue_maps = Dict{Symbol, BitMatrix}()
for tissue in keys(tissue_T2)
    mask = create_torso_phantom(
        n, n, :axial; fov = (40, 40), ti = TissueMask(; NamedTuple{(tissue,)}((true,))...)
    )[:, :, 1]
    tissue_maps[tissue] = mask
    curve = tissue_M0[tissue] .* exp.(-TE ./ tissue_T2[tissue])
    for k in 1:nt
        @views series[:, :, k] .+= ComplexF32(curve[k]) .* mask
    end
end
series = NamedDimsArray{(:x, :y, :time)}(series)

side_by_side(
    unname(series)[:, :, 1], unname(series)[:, :, 8], unname(series)[:, :, 24];
    titles = ("TE = $(round(Int, TE[1])) ms", "TE = $(round(Int, TE[8])) ms", "TE = $(round(Int, TE[end])) ms"),
    size = (1100, 340)
)

# %%
plot(
    TE, [tissue_M0[t] .* exp.(-TE ./ tissue_T2[t]) for t in keys(tissue_T2)];
    label = reshape(["$t (T2 = $(round(Int, tissue_T2[t])) ms)" for t in keys(tissue_T2)], 1, :),
    lw = 2, xlabel = "TE (ms)", ylabel = "signal", title = "Tissue signal evolutions",
    size = (700, 350)
)

# %% [markdown]
# #### How low-dimensional? The Casorati spectrum
#
# The singular values of the Casorati matrix say how many basis functions the series actually
# needs. A handful of exponentials at different rates is a textbook low-rank family.

# %%
casorati = reshape(unname(series), n * n, nt)
F = svd(casorati)
σ = F.S ./ F.S[1]

plot(
    1:nt, max.(σ, 1.0e-8);
    yscale = :log10, lw = 2, marker = :circle, label = "",
    xlabel = "index", ylabel = "singular value / largest",
    title = "Casorati spectrum of the echo series", size = (650, 330)
)

# %% [markdown]
# #### Where the basis comes from in practice
#
# The SVD above uses the ground-truth series, which you do not have at reconstruction time. In
# practice $\Phi$ comes from a **dictionary of plausible signal evolutions**, simulated from the
# sequence:
#
# 1. Sweep the tissue parameters over the physiological range ($T_2$ here; $T_1$/$T_2$/$B_1$ for
#    fingerprinting).
# 2. Simulate the signal each parameter combination would produce under the actual pulse sequence
#    — an analytic expression for a simple decay, an extended phase graph or a full Bloch
#    simulation for anything realistic.
# 3. Take the leading $K$ left singular vectors of that dictionary. They span the signal manifold
#    without ever having seen the patient.
#
# This is exactly the recipe in Tamir et al. 2017 and in the fingerprinting literature; the *k-t*
# PCA variants instead build the dictionary from low-resolution training data acquired in the same
# scan.

# %%
# Step 1-2: a "Bloch-simulated" dictionary — here the analytic spin-echo decay over a log-spaced
# T2 range, which is what the extended-phase-graph simulation reduces to for this sequence.
T2_dict = exp.(range(log(15), log(400); length = 256))
dictionary = Float32[exp(-te / t2) for te in TE, t2 in T2_dict]

# Step 3: the temporal basis.
Φ_full = Matrix{ComplexF32}(svd(dictionary).U)
println("dictionary: ", size(dictionary), "   basis: ", size(Φ_full))

plot(
    TE, real.(Φ_full[:, 1:5]);
    lw = 2, label = ["Φ₁" "Φ₂" "Φ₃" "Φ₄" "Φ₅"],
    xlabel = "TE (ms)", ylabel = "amplitude", title = "Leading dictionary basis functions",
    size = (700, 350)
)

# %% [markdown]
# #### How many basis functions? The projection error
#
# Before running any reconstruction, the basis can be scored directly: project the true series onto
# the first $K$ dictionary components and measure what is lost. That is the *model error floor* —
# no reconstruction using this basis can do better.

# %%
proj_err = Float64[]
for K in 1:12
    Φ = Φ_full[:, 1:K]
    projected = casorati * conj(Φ) * transpose(Φ)
    push!(proj_err, norm(projected - casorati) / norm(casorati))
end

data_svd_err = [sqrt(sum(abs2, F.S[(K + 1):end]) / sum(abs2, F.S)) for K in 1:12]

plot(
    1:12, [proj_err data_svd_err];
    yscale = :log10, lw = 2, marker = :circle,
    label = ["dictionary basis" "data SVD (unattainable)"],
    xlabel = "number of basis functions K", ylabel = "relative projection error",
    title = "Model error floor vs. K", size = (700, 350)
)

# %% [markdown]
# The dictionary basis tracks the (unattainable) data SVD closely and the error falls off a cliff
# by $K \approx 4$–$6$: six exponentials at six rates need six components, and the dictionary found
# them without being told the tissue parameters.

# %% [markdown]
# #### Reconstruction at K = 2, 4, 8
#
# Now undersample. All echoes share one phase-encoding pattern here — a real subspace acquisition
# would vary it per echo, which helps considerably more — and the data is noisy, so both effects a
# subspace model is good at are in play.

# %%
smaps_dyn = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(n, n, ncoils))

mask_dyn = falses(n)
mask_dyn[1:3:n] .= true
mask_dyn[(n ÷ 2 - 5):(n ÷ 2 + 5)] .= true
println("acceleration: ", round(n / sum(mask_dyn), digits = 2), "×")

acq_dyn = AcquisitionInfo(;
    is3D = false, image_size = (n, n), sensitivity_maps = smaps_dyn, subsampling = (:, mask_dyn)
)
data_dyn = add_noise(simulate_acquisition(series, acq_dyn); snr_db = 25)
println("k-space: ", size(data_dyn.kspace_data), " ", dimnames(data_dyn.kspace_data))

# %%
x_zf_dyn = reconstruct(data_dyn; verbosity = Silent())
x_cg_dyn = reconstruct(data_dyn, IterativeReconstruction(; algorithm = CGNR(), maxit = 40); verbosity = Silent())

println("zero-filled           ", round(nrmse(x_zf_dyn, series), digits = 4))
println("CG, no signal model   ", round(nrmse(x_cg_dyn, series), digits = 4))

subspace_recons = Dict{Int, Any}()
for K in (1, 2, 4, 8)
    x̂ = reconstruct(
        data_dyn,
        IterativeReconstruction(;
            signal_model = TemporalBasis(Φ_full[:, 1:K]; time_dim = :time),
            algorithm = CGNR(), maxit = 40
        );
        verbosity = Silent()
    )
    subspace_recons[K] = x̂
    println("TemporalBasis, K = ", rpad(K, 2), "   ", round(nrmse(x̂, series), digits = 4))
end

# %% [markdown]
# The pattern is the one to remember: $K = 1$ **underfits** — a single decay cannot describe six
# tissues — while $K = 8$ starts spending its extra components on noise. The best $K$ sits just
# past the knee of the projection-error curve, and the subspace reconstruction beats the
# unconstrained CG reconstruction by a wide margin even though it is solving for a third as many
# unknowns.

# %%
side_by_side(
    unname(series)[:, :, 12], unname(x_cg_dyn)[:, :, 12],
    unname(subspace_recons[2])[:, :, 12], unname(subspace_recons[4])[:, :, 12];
    titles = ("truth, echo 12", "CG, no model", "K = 2", "K = 4"), size = (1300, 340)
)

# %%
difference_image(
    unname(subspace_recons[4])[:, :, 12], unname(series)[:, :, 12];
    title = "K = 4 error, echo 12", size = (450, 380)
)

# %% [markdown]
# Because the coefficient maps are the variable, the *fitted decay curve* is available everywhere,
# not just the images — which is the point of the whole exercise for parameter mapping.

# %%
roi = tissue_maps[:heart]
plot(
    TE, [
        [mean(abs.(unname(series))[roi, k]) for k in 1:nt],
        [mean(abs.(unname(x_cg_dyn))[roi, k]) for k in 1:nt],
        [mean(abs.(unname(subspace_recons[4]))[roi, k]) for k in 1:nt],
    ];
    lw = 2, label = ["truth" "CG, no model" "K = 4 subspace"],
    xlabel = "TE (ms)", ylabel = "mean |x| in the myocardium ROI",
    title = "Recovered signal evolution", size = (700, 350)
)

# %% [markdown]
# #### Other places this model is the right one
#
# - **Quantitative / parameter mapping.** Multi-echo $T_2$ (above), inversion-recovery $T_1$,
#   multi-echo $T_2^{*}$ and $B_0$ mapping, diffusion with many $b$-values. The dictionary is a
#   forward simulation of the sequence, and the parameter map is fitted afterwards from the
#   reconstructed evolutions.
# - **MR fingerprinting.** The same construction with a much larger dictionary over
#   $(T_1, T_2, B_1, \ldots)$; the subspace reconstruction is the standard way to make the
#   highly-undersampled fingerprinting time series tractable.
# - **Dynamic contrast enhancement.** The dictionary is a family of plausible enhancement curves
#   (arterial input convolved with tissue responses) rather than a Bloch simulation.
# - **Cardiac cine and real-time imaging.** The *k-t* PCA family, with the basis learned from
#   training data acquired in the same scan. Beware: cine dynamics are driven by *motion*, and a
#   moving edge is much less low-rank than a decaying exponential — expect to need more components,
#   or a locally low-rank model instead (see notebook 7).
#
# When the low-dimensional structure is real but you cannot write down a basis in advance, use a
# low-rank *regularizer* (`LowRank`, `LocallyLowRank`, notebook 7) instead: same intuition, learned
# during the solve, at the cost of a penalty rather than a hard constraint.


# %% [markdown]
# ### 2.2 `KSpaceToImage` — solving in the k-space domain
#
# The other signal model turns the problem inside out. The optimization variable is the full
# multi-channel k-space $k \in \mathbb{C}^{N_x \times N_y \times N_c}$; the encoding operator
# during the solve is then just the subsampling operator $\mathcal{P}$, so data consistency is
# $\mathcal{P}k = y$ and needs no Fourier transform and no sensitivity maps at all. The result is
# mapped to an image afterwards by an inverse FFT and the model's own `coil_combination`.
#
# $$\hat k = \arg\min_k\; \tfrac12\|\mathcal{P}k - y\|^2 + \mathcal{R}(k), \qquad \hat x = \text{combine}(\mathcal{F}^{-1}\hat k)$$
#
# This is the natural home for any regularizer that is a statement about k-space rather than about
# the image — `SPIRiTConsistency` being the example the package ships, and structured low-rank
# methods being the other family. `SPIRiT(; iterative = true)` lowers to precisely this: a
# `KSpaceToImage` variable, a `SPIRiTConsistency` term built from the calibrated kernel, and hard
# data consistency.
#
# Because $\mathcal{P}\mathcal{P}^*$ is diagonal, `HardConsistency()` is closed-form here, which is
# why the lowered SPIRiT can use it without an inner CG.

# %%
# Plain CG in the k-space domain, no k-space regularizer: this just interpolates nothing and
# combines the coils, so it is the k-space-domain spelling of a zero-filled reconstruction.
x_ksp = reconstruct(
    acq_pi,
    IterativeReconstruction(;
        signal_model = KSpaceToImage(AdjointSensitivity()), algorithm = CGNR(), maxit = 10
    );
    verbosity = Silent()
)
println("KSpaceToImage, no k-space prior  ", round(nrmse(x_ksp, img_pi), digits = 4))

# Adding the SPIRiT self-consistency term is what makes the k-space variable pay off — this is the
# hand-built version of `SPIRiT(; iterative = true)`.
kernel = MriReconstructionToolbox._calibrate_spirit_kernel(
    acq_pi, SPIRiT(kernel_size = (5, 5), calib_size = (Nx, 24))
)
x_ksp_spirit = reconstruct(
    acq_pi,
    IterativeReconstruction(
        SPIRiTConsistency(kernel; λ = 1.0);
        signal_model = KSpaceToImage(RootSumSquares()),
        fidelity = HardConsistency(), algorithm = FISTA(adaptive = true), maxit = 30
    );
    verbosity = Silent()
)
println("KSpaceToImage + SPIRiTConsistency ", round(nrmse(x_ksp_spirit, img_pi), digits = 4))

side_by_side(
    unname(x_ksp), unname(x_ksp_spirit);
    titles = ("k-space CG, no prior", "+ SPIRiT consistency"), size = (900, 360)
)


# %% [markdown]
# ### 2.3 No calibration region at all: structured low-rank k-space
#
# Both methods above need an ACS block. Take it away — an irregular sampling pattern with no
# fully-sampled centre — and GRAPPA has nothing to fit its kernel to and SPIRiT has nothing to
# calibrate $G$ from. This happens in practice more often than it sounds: prospectively
# undersampled scans that never acquired a calibration region, patterns where motion corrupted
# the centre, and acquisitions where the ACS lines would cost too much time.
#
# The way out is to notice that the *same* relation GRAPPA and SPIRiT calibrate — every k-space
# sample is a linear combination of its neighbours across coils — can be read off the undersampled
# data itself, without ever writing the kernel down. Stack every sliding window of multi-coil
# k-space as a row of one big matrix (a **block-Hankel** matrix, with the coils stacked as extra
# columns) and that matrix is low rank exactly when such linear relations exist. So: fill in the
# missing samples by asking for the matrix to be low rank. No sensitivity maps, no ACS —
# *calibrationless* parallel imaging.
#
# `StructuredLowRank` is that regularizer, in two forms:
#
# - `StructuredLowRank(; λ, window = ...)` — the nuclear norm of the lifted matrix, i.e. its
#   convex relaxation. This is LORAKS' C-matrix penalty (Haldar 2014).
# - `StructuredLowRank(; max_rank, window = ...)` — a hard cap on the rank, imposed by truncating
#   the SVD of the lifted matrix each iteration. This is SAKE (Shin et al. 2014), and it is the
#   Cadzow alternating-projection idea applied to MRI.
#
# The penalty lives on k-space, not on the image, so the reconstruction is set up with
# `signal_model = KSpaceToImage(...)` — the same trick `SPIRiT(; iterative = true)` uses above.

# %%
# Calibrationless data: irregular ky sampling at R = 2, and no dense centre (the second argument
# of `UniformRandomSampling` is the fraction of fully-sampled central lines — here, none).
mask_cl = create_sampling_pattern(UniformRandomSampling(2.0, 0.0), (Nx, Ny))
println("sampled ky lines: ", sum(mask_cl[2]), " / ", Ny, "  (no ACS block)")

acq_cl_maps = add_noise(
    simulate_acquisition(
        img_pi,
        CartesianAcquisitionInfo(;
            is3D = false, image_size = (Nx, Ny), subsampling = mask_cl, sensitivity_maps = sens
        )
    );
    snr_db = 30
)

# The reconstruction is handed the coil data and nothing else -- rebuilding the acquisition
# without `sensitivity_maps` is what makes this calibrationless.
acq_cl = CartesianAcquisitionInfo(
    acq_cl_maps.kspace_data; is3D = false, image_size = (Nx, Ny), subsampling = mask_cl
)

# Without maps, `DirectReconstruction` returns the individual coil images, so combine them here.
rss(x) = sqrt.(dropdims(sum(abs2, unname(x); dims = 3); dims = 3))
x_zf = rss(reconstruct(acq_cl, DirectReconstruction(); verbosity = Silent()))
println("zero-filled RSS  ", round(nrmse(x_zf, img_pi), digits = 4))

# %% [markdown]
# Before reconstructing, it is worth looking at the object the whole method rests on. Lift the
# zero-filled multi-coil k-space into its block-Hankel matrix and look at the singular values: if
# the low-rank story is true, they should fall off a cliff. The index at which they do is the
# `max_rank` to ask for.

# %%
ksp_grid = zeros(ComplexF32, Nx, Ny, Nc)
ksp_grid[:, mask_cl[2], :] .= unname(acq_cl.kspace_data)

H_cl = Hankel(ComplexF32, (Nx, Ny), (5, 5); nchannels = Nc, channels = true)
σ_cl = svdvals(H_cl * ksp_grid)
println("lifted matrix: ", size(H_cl)[1][1], " x ", size(H_cl)[1][2])

plot(
    1:length(σ_cl), σ_cl ./ σ_cl[1];
    yscale = :log10, lw = 2, label = "",
    xlabel = "index", ylabel = "singular value / largest",
    title = "Block-Hankel spectrum, 5x5 window, 8 coils", size = (650, 330)
)
vline!([25]; ls = :dash, lw = 2, label = "max_rank = 25")

# %%
slr(reg) = reconstruct(
    acq_cl,
    IterativeReconstruction(
        reg; signal_model = KSpaceToImage(RootSumSquares()), algorithm = ADMM(), maxit = 40
    );
    verbosity = Silent()
)

x_loraks = slr(StructuredLowRank(; λ = 1.0f-2, window = (5, 5)))     # convex, LORAKS-C
x_sake = slr(StructuredLowRank(; max_rank = 25, window = (5, 5)))    # non-convex, SAKE

println("zero-filled RSS      ", round(nrmse(x_zf, img_pi), digits = 4))
println("LORAKS-C (nuclear)   ", round(nrmse(x_loraks, img_pi), digits = 4))
println("SAKE (rank 25)       ", round(nrmse(x_sake, img_pi), digits = 4))

side_by_side(
    x_zf, unname(x_loraks), unname(x_sake), abs.(unname(img_pi));
    titles = ("zero-filled RSS", "LORAKS-C", "SAKE", "ground truth"), size = (1400, 350)
)

# %% [markdown]
# Both forms turn an unusable zero-filled image into a usable one from data that GRAPPA and
# SPIRiT cannot touch, and the hard-rank form is the more accurate of the two here — which is the
# usual finding, and the reason SAKE is stated as a rank constraint in the first place. The
# nuclear norm shrinks *every* singular value, including the ones carrying signal, so it pays a
# bias for its convexity.
#
# !!! warning "`max_rank` gives up convexity"
#     A rank cap is a projection onto a non-convex set, and it is applied to the lifted matrix
#     rather than to k-space itself, so a splitting algorithm using it is a heuristic: there is no
#     convergence guarantee, and the answer depends on where the iteration starts. The `λ` form is
#     convex and will not surprise you. Treat a good SAKE result as "this initialization worked",
#     not as "this is the global optimum".
#
# Two practical notes:
#
# - Cost is one economy SVD of the lifted matrix per iteration — here a
#   $(N_x - 4)(N_y - 4) \times 25 N_c$ matrix — so the `window` is the knob that decides whether
#   this is affordable. `(5, 5)` or `(6, 6)` in 2D, `(4, 4, 4)` in 3D.
# - Only the plain block-Hankel structure is implemented (`structure = :c`). LORAKS' S- and
#   G-matrices, which additionally impose conjugate symmetry and phase constraints, are not
#   available.


# %% [markdown]
# ## References
#
# - Shin P. J. *et al.*, *Calibrationless parallel imaging reconstruction based on structured
#   low-rank matrix completion*, Magn. Reson. Med. 72:959–970 (2014). — SAKE.
# - Haldar J. P., *Low-rank modeling of local k-space neighborhoods (LORAKS) for constrained MRI*,
#   IEEE Trans. Med. Imaging 33:668–681 (2014). — LORAKS.
# - Liang Z.-P., *Spatiotemporal imaging with partially separable functions*, ISBI 2007, 988–991.
# - Pedersen H. *et al.*, *k-t PCA: temporally constrained k-t BLAST reconstruction using principal
#   component analysis*, Magn. Reson. Med. 62:706–716 (2009).
# - Petzschner F. H. *et al.*, *Fast MR parameter mapping using k-t principal component analysis*,
#   Magn. Reson. Med. 66:706–716 (2011).
# - Tamir J. I. *et al.*, *T2 shuffling: sharp, multicontrast, volumetric fast spin-echo imaging*,
#   Magn. Reson. Med. 77:180–195 (2017).

# %% [markdown]
# ## Environment

# %%
print_versions()

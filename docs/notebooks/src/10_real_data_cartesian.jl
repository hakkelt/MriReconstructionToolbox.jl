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
# # 10 — Real scanner data, end to end
#
# Everything so far ran on simulated data. This notebook takes a real, fully sampled Cartesian
# brain acquisition and walks the whole pipeline: raw ISMRMRD profiles → k-space → preprocessing
# → sensitivity maps → reference reconstruction → retrospective undersampling → compressed
# sensing and parallel imaging → noise analysis.
#
# The data comes from [M4Raw](https://github.com/mylyu/M4Raw) (0.3 T low-field brain, 4 channels,
# CC-BY), downloaded through
# [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl). The first run downloads ~12 MB;
# afterwards it is cached.
#
# > **Data terms.** The datasets have their own licenses and citation requirements, separate from
# > the packages that download them. M4Raw is CC-BY: cite Lyu et al., *M4Raw: A multi-contrast,
# > multi-repetition, multi-channel MRI k-space dataset for low-field MRI research*,
# > Scientific Data 10, 264 (2023).
#
# **Contents**
# 1. Loading the raw data
# 2. Assembling one slice of k-space
# 3. Preprocessing — prewhitening, coil compression, sensitivity maps
# 4. The fully-sampled reference
# 5. Retrospective undersampling and compressed sensing
# 6. Parallel imaging on the same data
# 7. Noise analysis with pseudo-replicas

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: get_encoding_operator
using MRITestData
using MIRTjim: jim
using Plots
using NamedDims
using FFTW
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Loading the raw data
#
# `MRITestData` needs to be told once where downloads go; `:cache` uses the package's own scratch
# space. `load_raw` returns an `MRIBase.RawAcquisitionData`: a list of profiles (one readout
# each), plus the ISMRMRD header.

# %%
if MRITestData.get_download_path() === nothing
    MRITestData.set_download_path!(:cache)
end

entry = MRITestData.dataset(MRITestData.M4RAW, "multicoil_train/2022062402_T203")
raw = MRITestData.load_raw(entry)

println("profiles:      ", length(raw.profiles))
println("per profile:   ", size(raw.profiles[1].data), "  (readout samples × channels)")
println("slices:        ", length(unique(Int(p.head.idx.slice) for p in raw.profiles)))
println("phase encodes: ", maximum(Int(p.head.idx.kspace_encode_step_1) for p in raw.profiles) + 1)

# %% [markdown]
# ## 2. Assembling one slice of k-space
#
# Each profile carries its own encoding counters; a Cartesian slice is assembled by placing every
# profile at the phase-encode line its header names. This is the step that turns a scanner file
# into the `(kx, ky, coil)` array MRT works with.

# %%
function assemble_slice(raw, slice)
    profiles = [
        p for p in raw.profiles if
            Int(p.head.idx.slice) == slice && Int(p.head.idx.contrast) == 0 &&
            Int(p.head.idx.repetition) == 0 && Int(p.head.idx.average) == 0
    ]
    nsamples, ncoils = size(profiles[1].data)
    pre, post = Int(profiles[1].head.discard_pre), Int(profiles[1].head.discard_post)
    nkx = nsamples - pre - post
    nky = maximum(Int(p.head.idx.kspace_encode_step_1) for p in profiles) + 1

    ksp = zeros(ComplexF32, nkx, nky, ncoils)
    for p in profiles
        ksp[:, Int(p.head.idx.kspace_encode_step_1) + 1, :] .= ComplexF32.(p.data[(pre + 1):(pre + nkx), :])
    end
    return ksp
end

slices = sort(unique(Int(p.head.idx.slice) for p in raw.profiles))
ksp_raw = assemble_slice(raw, slices[cld(length(slices), 2)])       # middle slice
nkx, nky, ncoil = size(ksp_raw)
println("assembled k-space: ", size(ksp_raw), " ", eltype(ksp_raw))

jim(log.(abs.(ksp_raw) .+ 1.0f-8); title = "log |k-space|, per channel", nrow = 1, size = (1100, 300))

# %% [markdown]
# Note the black bands: this scan measured only a central block of phase encodes (a partial
# `ky` coverage), so the array it was assembled into is wider than the data. Everything below
# treats that measured block as the fully sampled grid — the honest thing to do, since the
# unmeasured lines carry no information that any reconstruction could be scored against.

# %%
acquired = [any(!iszero, ksp_raw[:, j, :]) for j in 1:nky]
println("phase-encode lines measured: ", sum(acquired), " of ", nky,
    "  (lines ", findfirst(acquired), ":", findlast(acquired), ")")

ksp = ksp_raw[:, acquired, :]
nkx, nky, ncoil = size(ksp)
println("working k-space: ", size(ksp))

# %%
# The coil images and their root-sum-of-squares — the coil-independent reference every comparison
# below is scored against. Reconstructing through MRT (rather than calling `ifft` by hand) keeps
# one FFT-shift convention throughout the notebook.
acq_coils = AcquisitionInfo(NamedDimsArray{(:kx, :ky, :coil)}(ksp); is3D = false)
coil_images = reconstruct(acq_coils; verbosity = Silent())        # no maps ⇒ one image per coil
reference = sqrt.(sum(abs2, unname(coil_images); dims = 3)[:, :, 1])

jim(
    jim(abs.(unname(coil_images)); title = "coil images", nrow = 1),
    jim(reference; title = "root sum of squares");
    layout = (2, 1), size = (900, 600)
)

# %% [markdown]
# ## 3. Preprocessing
#
# ### Noise prewhitening
#
# Receiver channels see correlated noise. `estimate_noise_covariance` takes noise-only samples —
# ideally a dedicated noise scan, here the far corners of k-space, which contain no signal — and
# `prewhiten` decorrelates both the k-space and the sensitivity maps.

# %%
# Corners of the *measured* region: high frequency in both directions, so almost pure noise.
# (A dedicated noise scan is better when the sequence provides one; this dataset has none.)
corner = 20
noise_patch = cat(
    ksp[1:corner, 1:corner, :],
    ksp[(end - corner + 1):end, 1:corner, :],
    ksp[1:corner, (end - corner + 1):end, :],
    ksp[(end - corner + 1):end, (end - corner + 1):end, :];
    dims = 2
)
Ψ = estimate_noise_covariance(NamedDimsArray{(:kx, :ky, :coil)}(noise_patch))

println("noise covariance (magnitude):")
display(round.(abs.(Ψ); digits = 4))
println("\ncorrelation between channels 1 and 2: ",
    round(abs(Ψ[1, 2]) / sqrt(abs(Ψ[1, 1]) * abs(Ψ[2, 2])), digits = 3))

# %%
acq_white = prewhiten(acq_coils, Ψ)

Ψ_after = estimate_noise_covariance(
    NamedDimsArray{(:kx, :ky, :coil)}(unname(acq_white.kspace_data)[1:corner, 1:corner, :])
)
println("off-diagonal magnitude before: ", round(maximum(abs, Ψ - Diagonal(diag(Ψ))), digits = 5))
println("off-diagonal magnitude after:  ", round(maximum(abs, Ψ_after - Diagonal(diag(Ψ_after))), digits = 5))

# %% [markdown]
# ### Sensitivity maps
#
# Three estimators, all working from the fully sampled centre of k-space. ESPIRiT's maps have
# compact support (they are zero where there is no signal), which is what a SENSE-type
# reconstruction wants.

# %%
maps_selfcal = estimate_sensitivities(acq_white; method = SelfCalibrating(calib_size = 24)).sensitivity_maps
maps_adaptive = estimate_sensitivities(acq_white; method = AdaptiveCombine(kernel_size = 5)).sensitivity_maps
acq_espirit = estimate_sensitivities(acq_white; method = ESPIRiT(calib_size = 24, kernel_size = 6))
maps_espirit = acq_espirit.sensitivity_maps

jim(
    jim(abs.(unname(maps_selfcal)); title = "SelfCalibrating", nrow = 1),
    jim(abs.(unname(maps_adaptive)); title = "AdaptiveCombine", nrow = 1),
    jim(abs.(unname(maps_espirit)); title = "ESPIRiT", nrow = 1);
    layout = (3, 1), size = (1000, 800)
)

# %% [markdown]
# ### Coil compression
#
# With four channels there is little to gain, but the mechanics are the same as on a 32-channel
# array: `compress_coils` returns the compressed acquisition and the compression matrix.

# %%
acq_compressed, C = compress_coils(acq_espirit, 2; method = SVDCompression())
println("compression matrix: ", size(C), "  (virtual × physical)")
println("channels: ", size(acq_espirit.kspace_data, :coil), " → ", size(acq_compressed.kspace_data, :coil))

rec_full_4ch = reconstruct(acq_espirit; verbosity = Silent())
rec_full_2ch = reconstruct(acq_compressed; verbosity = Silent())

jim(
    jim(abs.(unname(rec_full_4ch)); title = "4 channels"),
    jim(abs.(unname(rec_full_2ch)); title = "2 virtual channels"),
    jim(abs.(abs.(unname(rec_full_4ch)) - abs.(unname(rec_full_2ch))); title = "difference");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# ## 4. The fully-sampled reference
#
# With sensitivity maps in hand, the adjoint reconstruction is the SNR-optimal coil combination.
# It is the target the accelerated reconstructions below are measured against.

# %%
x_ref = reconstruct(acq_espirit; verbosity = Silent())

# Everything below is scored against the root-sum-of-squares of the fully sampled data, on
# magnitude, with the amplitude aligned (different reconstructions carry different scalings) and
# restricted to the object — background pixels are noise and would dominate an unmasked norm.
support = reference .> 0.1maximum(reference)

function rel_err(x̂)
    a = abs.(unname(x̂))[support]
    b = reference[support]
    α = sum(a .* b) / sum(abs2, a)
    return norm(α .* a - b) / norm(b)
end

println("sensitivity-weighted combination vs. RSS: ", round(rel_err(x_ref), digits = 4))

jim(
    jim(reference; title = "root sum of squares"),
    jim(abs.(unname(x_ref)); title = "ESPIRiT + adjoint (𝒜ᴴy)");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ## 5. Retrospective undersampling and compressed sensing
#
# The data is fully sampled, so any sampling pattern can be applied after the fact — the standard
# way of evaluating an accelerated reconstruction against a real reference.

# %%
ksp_white = unname(acq_white.kspace_data)

pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.08)
mask_us = create_sampling_pattern(pdf, (nkx, nky))[2]
println("retained phase encodes: ", sum(mask_us), " of ", nky,
    "  (", round(nky / sum(mask_us), digits = 2), "× acceleration)")

acq_us = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil)}(ksp_white[:, mask_us, :]);
    is3D = false,
    image_size = (nkx, nky),
    subsampling = (:, mask_us),
    sensitivity_maps = maps_espirit,
)

x_zf = reconstruct(acq_us; verbosity = Silent())
println("zero-filled: ", round(rel_err(x_zf), digits = 4))
jim(abs.(unname(x_zf)); title = "zero-filled, 4× undersampled", size = (400, 350))

# %%
# λ is larger here than on the phantom of notebook 4: this is 0.3 T data with four channels, so
# the SNR is low and the noise, not the aliasing, is what limits the result.
methods = (
    "L2Image (CG-SENSE)" => IterativeReconstruction(L2Image(1.0f-2); maxit = 30),
    "L1Wavelet2D" => IterativeReconstruction(L1Wavelet2D(2.0f-2); maxit = 60),
    "TotalVariation2D" => IterativeReconstruction(TotalVariation2D(1.0f-2); maxit = 60),
    "wavelet + TV" => IterativeReconstruction(L1Wavelet2D(1.0f-2), TotalVariation2D(5.0f-3); algorithm = ADMM(), maxit = 60),
    "TGV" => IterativeReconstruction(TotalGeneralizedVariation2D(1.0f-2); algorithm = ADMM(), maxit = 60),
)

recons = map(methods) do (label, method)
    x̂ = reconstruct(acq_us, method; verbosity = Silent())
    println(rpad(label, 22), " ", round(rel_err(x̂), digits = 4))
    label => x̂
end

# %% [markdown]
# Two things are worth reading off those numbers. The unregularized parallel-imaging solve
# (`L2Image` with a small λ is CG-SENSE) is *worse* than the zero-filled adjoint here: with four
# low-field channels at 3× the inverse problem is badly conditioned, and CG happily amplifies
# noise into the answer — section 7 measures a g-factor around 3 for exactly this setup. The
# regularized reconstructions are what make the acceleration usable, and the edge-preserving ones
# (TV, TGV) do best on this low-SNR data.

# %%
jim(
    jim(abs.(unname(x_ref)); title = "reference (fully sampled)"),
    jim(abs.(unname(x_zf)); title = "zero-filled"),
    (jim(abs.(unname(x̂)); title = label) for (label, x̂) in recons)...;
    layout = (2, 4), size = (1400, 700)
)

# %%
# A λ sweep on the real data — the same exercise as on the phantom, with a real noise floor.
λs = Float32[2.0e-3, 8.0e-3, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1]
errs = map(λs) do λ
    rel_err(reconstruct(acq_us, IterativeReconstruction(L1Wavelet2D(λ); maxit = 60); verbosity = Silent()))
end
plot(
    λs, errs; xscale = :log10, marker = :circle, lw = 2, legend = false,
    xlabel = "λ", ylabel = "relative error vs. reference", title = "ℓ₁-wavelet λ sweep, real data",
    size = (600, 350)
)

# %% [markdown]
# ## 6. Parallel imaging on the same data
#
# A GRAPPA-style pattern — uniform R = 2 plus a fully sampled autocalibration block — lets the
# autocalibrated methods run on the same slice.

# %%
R = 2
acs = (nky ÷ 2 - 11):(nky ÷ 2 + 12)
mask_pi = falses(nky)
mask_pi[1:R:nky] .= true
mask_pi[acs] .= true
println("net acceleration: ", round(nky / sum(mask_pi), digits = 2), "×")

acq_pi = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil)}(ksp_white[:, mask_pi, :]);
    is3D = false, image_size = (nkx, nky), subsampling = (:, mask_pi),
    sensitivity_maps = maps_espirit,
)

x_grappa = reconstruct(acq_pi, GRAPPA(kernel_size = (3, 2), calib_size = (nkx, 24)); verbosity = Silent())
x_sense = reconstruct(acq_pi, IterativeReconstruction(L2Image(1.0f-2); maxit = 30); verbosity = Silent())
x_sense_cs = reconstruct(acq_pi, IterativeReconstruction(L1Wavelet2D(1.0f-2); maxit = 60); verbosity = Silent())

println("GRAPPA              ", round(rel_err(x_grappa), digits = 4))
println("CG-SENSE            ", round(rel_err(x_sense), digits = 4))
println("CS-SENSE (wavelet)  ", round(rel_err(x_sense_cs), digits = 4))

jim(
    jim(abs.(unname(x_grappa)); title = "GRAPPA"),
    jim(abs.(unname(x_sense)); title = "CG-SENSE"),
    jim(abs.(unname(x_sense_cs)); title = "CS-SENSE");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# ## 7. Noise analysis with pseudo-replicas
#
# The Monte Carlo pseudo-replica method (Robson et al. 2008) measures how a reconstruction — any
# reconstruction, including a non-linear regularized one — propagates noise: add synthetic noise
# to the measured k-space many times over, reconstruct each replica, and look at the pixel-wise
# standard deviation. The ratio to the fully-sampled case is the geometry factor.
#
# Data-dependent scaling would rescale every replica by its own noise level, so `pseudo_replica`
# insists on `NoScaling()` or `FixedScaling()`.

# %%
noise_level = 0.02 * sqrt(mean(abs2, ksp_white))

res_full = pseudo_replica(
    acq_espirit, IterativeReconstruction(L2Image(1.0f-2); maxit = 20);
    replicas = 16, noise_std = noise_level, scaling = NoScaling()
)
res_us = pseudo_replica(
    acq_us, IterativeReconstruction(L2Image(1.0f-2); maxit = 20);
    replicas = 16, noise_std = noise_level, scaling = NoScaling()
)

println("fields: ", keys(res_us))
println("mean g-factor over the object: ",
    round(mean(res_us.g_factor[reference .> 0.15maximum(reference)]), digits = 3))

# %%
jim(
    jim(res_full.std; title = "σ, fully sampled"),
    jim(res_us.std; title = "σ, 4× undersampled"),
    jim(res_us.g_factor .* support; title = "g-factor (masked)");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# The same machinery works on the parallel-imaging pattern, and on a regularized reconstruction —
# which is the point of the Monte Carlo approach: for a non-linear reconstruction there is no
# closed-form g-factor to compute.

# %%
res_cs = pseudo_replica(
    acq_us, IterativeReconstruction(L1Wavelet2D(2.0f-2); maxit = 30);
    replicas = 16, noise_std = noise_level, scaling = NoScaling()
)

println("mean σ over the object, CG-SENSE: ", round(mean(res_us.std[support]), digits = 5))
println("mean σ over the object, ℓ₁-wavelet: ", round(mean(res_cs.std[support]), digits = 5))

jim(
    jim(res_us.std .* support; title = "σ — CG-SENSE"),
    jim(res_cs.std .* support; title = "σ — ℓ₁-wavelet");
    layout = (1, 2), size = (800, 350)
)

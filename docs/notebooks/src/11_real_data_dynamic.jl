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
# # 11 — Real dynamic data: cardiac cine
#
# The dynamic machinery of notebook 7 on real scanner data: a fully sampled 1.5 T cardiac cine
# from [OCMR](https://ocmr.info), retrospectively undersampled and reconstructed with temporal,
# low-rank and low-rank-plus-sparse models.
#
# > **Data terms.** OCMR has its own data-use agreement and asks that you cite Chen et al.,
# > *OCMR (v1.0) — Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance
# > Imaging*, arXiv:2008.03410 (2020). The first run downloads ~200 MB.
#
# **Contents**
# 1. Loading and assembling the cine
# 2. Sensitivity maps and the fully-sampled reference
# 3. Retrospective undersampling
# 4. Frame-by-frame vs. temporal reconstruction
# 5. Low-rank and L+S on real data
# 6. Looking at the temporal profile

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
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
# ## 1. Loading and assembling the cine
#
# Each profile carries a cardiac phase index (`idx.phase`) in addition to the phase-encode
# counter, so the assembly is the 2D one of notebook 10 with one more axis.

# %%
if MRITestData.get_download_path() === nothing
    MRITestData.set_download_path!(:cache)
end

entry = MRITestData.dataset(MRITestData.OCMR_SOURCE, "fs_0001_1_5T")
raw = MRITestData.load_raw(entry)

println("profiles: ", length(raw.profiles))
println("per profile: ", size(raw.profiles[1].data))
println("cardiac phases: ", maximum(Int(p.head.idx.phase) for p in raw.profiles) + 1)

# %%
function assemble_cine(raw)
    slices = sort(unique(Int(p.head.idx.slice) for p in raw.profiles))
    slice = slices[cld(length(slices), 2)]
    profiles = [
        p for p in raw.profiles if
            Int(p.head.idx.slice) == slice && Int(p.head.idx.contrast) == 0 &&
            Int(p.head.idx.repetition) == 0 && Int(p.head.idx.average) == 0
    ]
    pre, post = Int(profiles[1].head.discard_pre), Int(profiles[1].head.discard_post)
    rows = (pre + 1):(size(profiles[1].data, 1) - post)
    nkx, ncoil = length(rows), size(profiles[1].data, 2)
    nky = maximum(Int(p.head.idx.kspace_encode_step_1) for p in profiles) + 1
    nframes = maximum(Int(p.head.idx.phase) for p in profiles) + 1

    ksp = zeros(ComplexF32, nkx, nky, ncoil, nframes)
    for p in profiles
        ksp[:, Int(p.head.idx.kspace_encode_step_1) + 1, :, Int(p.head.idx.phase) + 1] .=
            ComplexF32.(p.data[rows, :])
    end
    return ksp
end

# Crop the (2× oversampled, often asymmetric-echo) readout around the true DC — the energy peak,
# not the array midpoint — for a smaller and correctly centred problem.
function crop_readout(ksp, target)
    n = size(ksp, 1)
    profile = sum(abs2, reshape(ksp, n, :); dims = 2)[:, 1]
    dc = argmax(profile)
    lo, hi = dc - target ÷ 2 + 1, dc + target ÷ 2
    out = zeros(eltype(ksp), target, size(ksp)[2:end]...)
    src = max(1, lo):min(n, hi)
    out[(first(src) - lo + 1) .+ (0:(length(src) - 1)), :, :, :] .= ksp[src, :, :, :]
    return out
end

ksp_cine = crop_readout(assemble_cine(raw), 128)
println("cine k-space: ", size(ksp_cine), "  (kx, ky, coil, frame)")

# %%
# Fifteen channels is more than this problem needs; compressing to six virtual coils cuts the
# cost of every iterative reconstruction below by more than half at no visible cost in quality.
acq_full = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil, :time)}(ksp_cine); is3D = false
)
acq_full, _ = compress_coils(acq_full, 6; method = SVDCompression())
ksp_cine = unname(acq_full.kspace_data)
nkx, nky, ncoil, nframes = size(ksp_cine)
println("after coil compression: ", size(ksp_cine))

# %%
# The fully-sampled reference: per-frame root sum of squares of the coil images. Reconstructing
# through MRT (rather than calling `ifft` by hand) keeps one FFT-shift convention throughout.
coil_frames = reconstruct(acq_full; verbosity = Silent())      # no maps ⇒ per-coil images
reference = sqrt.(sum(abs2, unname(coil_frames); dims = 3)[:, :, 1, :])
println("reference: ", size(reference))

jim(reference[:, :, 1:4:min(nframes, 16)]; title = "reference, every 4th frame", nrow = 1, size = (1100, 300))

# %% [markdown]
# ## 2. Sensitivity maps
#
# Calibrate from a **single frame**, not the time average: cardiac motion smears a temporally
# averaged calibration region and corrupts the ESPIRiT maps.

# %%
smaps = estimate_sensitivities(
    NamedDimsArray{(:kx, :ky, :coil)}(ksp_cine[:, :, :, 1]);
    method = ESPIRiT(calib_size = 24, kernel_size = 6)
)
println("maps: ", size(smaps), " ", dimnames(smaps))
jim(abs.(unname(smaps)); title = "ESPIRiT maps (frame 1)", nrow = 2, size = (1000, 500))

# %% [markdown]
# ## 3. Retrospective undersampling
#
# Uniform R = 3 with a small fully-sampled centre block — the same pattern for every frame, which
# is the pessimistic case (an interleaved, frame-varying pattern gives temporal models much more
# to work with).

# %%
R, acs = 3, 8
mask = falses(nky)
mask[1:R:nky] .= true
mask[(nky ÷ 2 - acs):(nky ÷ 2 + acs)] .= true
println("acceleration: ", round(nky / sum(mask), digits = 2), "×")

acq_us = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil, :time)}(ksp_cine[:, mask, :, :]);
    is3D = false,
    image_size = (nkx, nky),
    subsampling = (:, mask),
    sensitivity_maps = smaps,
)

# Compare on magnitude, amplitude-aligned against the RSS reference, over the object only.
support = reference .> 0.1maximum(reference)

function rel_err(x̂)
    a = abs.(unname(x̂))[support]
    b = reference[support]
    α = sum(a .* b) / sum(abs2, a)
    return norm(α .* a - b) / norm(b)
end

x_zf = reconstruct(acq_us; verbosity = Silent())
println("zero-filled: ", round(rel_err(x_zf), digits = 4))

jim(
    jim(reference[:, :, 1]; title = "reference, frame 1"),
    jim(abs.(unname(x_zf))[:, :, 1]; title = "zero-filled, frame 1");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ## 4. Frame-by-frame vs. temporal reconstruction
#
# A purely spatial regularizer treats every frame as an independent problem (and the
# reconstruction decomposes over time, so it is also the fastest). The temporal terms couple the
# frames, which is where the real gain is.

# %%
methods = (
    "L2Image (per frame)" => IterativeReconstruction(L2Image(1.0f-3); maxit = 20),
    "L1Wavelet2D (per frame)" => IterativeReconstruction(L1Wavelet2D(5.0f-3); maxit = 30),
    "L1TemporalFourier" => IterativeReconstruction(L1TemporalFourier(2.0f-2; time_dim = :time); maxit = 30),
    "TemporalTotalVariation" => IterativeReconstruction(TemporalTotalVariation(2.0f-2; time_dim = :time); maxit = 20),
)

results = map(methods) do (label, method)
    t = @elapsed x̂ = reconstruct(acq_us, method; verbosity = Silent())
    println(rpad(label, 26), " error ", round(rel_err(x̂), digits = 4), "   ", round(t, digits = 1), " s")
    label => x̂
end

# %%
frame = 1 + nframes ÷ 3
jim(
    jim(reference[:, :, frame]; title = "reference"),
    jim(abs.(unname(x_zf))[:, :, frame]; title = "zero-filled"),
    (jim(abs.(unname(x̂))[:, :, frame]; title = label) for (label, x̂) in results)...;
    layout = (2, 3), size = (1200, 700)
)

# %% [markdown]
# ## 5. Low-rank and L+S
#
# A cine is nearly low rank: the frames differ mostly in one moving region. `LowRank` penalizes
# the nuclear norm of the whole space × time matrix; the L+S decomposition splits it into a
# static background and a dynamic foreground, each with its own prior.

# %%
x_lr = reconstruct(
    acq_us, IterativeReconstruction(LowRank(5.0f-2; time_dim = :time); maxit = 30); verbosity = Silent()
)
x_llr = reconstruct(
    acq_us,
    IterativeReconstruction(LocallyLowRank(5.0f-2; block_size = 8, time_dim = :time); maxit = 30);
    verbosity = Silent()
)
img_ls = reconstruct(
    acq_us,
    IterativeReconstruction(
        Component(:lowrank, LowRank(5.0f-2; time_dim = :time)),
        Component(:sparse, TemporalTotalVariation(2.0f-2; time_dim = :time));
        maxit = 30
    );
    verbosity = Silent()
)

println("LowRank          ", round(rel_err(x_lr), digits = 4))
println("LocallyLowRank   ", round(rel_err(x_llr), digits = 4))
println("L + S            ", round(rel_err(img_ls), digits = 4))

# %%
L = img_ls.components.lowrank
S = img_ls.components.sparse

jim(
    jim(abs.(unname(L))[:, :, frame]; title = "L — background"),
    jim(abs.(unname(S))[:, :, frame]; title = "S — dynamics"),
    jim(abs.(unname(img_ls))[:, :, frame]; title = "L + S"),
    jim(reference[:, :, frame]; title = "reference");
    layout = (2, 2), size = (800, 700)
)

# %%
# The singular-value spectrum shows how much of the cine is really low rank.
casorati(x) = reshape(abs.(unname(x)), nkx * nky, nframes)
plot(
    svdvals(casorati(reference))[1:12]; label = "reference", lw = 2, marker = :circle, yscale = :log10,
    xlabel = "index", ylabel = "singular value", size = (650, 350), title = "Casorati spectrum"
)
plot!(svdvals(casorati(x_zf))[1:12]; label = "zero-filled", lw = 2, marker = :circle)
plot!(svdvals(casorati(x_lr))[1:12]; label = "LowRank", lw = 2, marker = :circle)

# %% [markdown]
# ## 6. The temporal profile
#
# A y–t cut through the heart is the standard way to look at a cine reconstruction: temporal
# blurring and residual aliasing that are invisible in a single frame show up immediately.

# %%
# Column through the region that moves most.
motion = dropdims(std(reference; dims = 3); dims = 3)
col = argmax(vec(sum(motion; dims = 1)))
println("profiling column ", col)

profiles = (
    "reference" => reference[:, col, :],
    "zero-filled" => abs.(unname(x_zf))[:, col, :],
    "temporal TV" => abs.(unname(results[4][2]))[:, col, :],
    "L + S" => abs.(unname(img_ls))[:, col, :],
)
jim(
    (jim(p; title = label, aspect_ratio = :auto) for (label, p) in profiles)...;
    layout = (2, 2), size = (900, 700)
)

# %%
# And the intensity of one voxel through the cardiac cycle.
row = argmax(vec(sum(motion; dims = 2)))
plot(
    reference[row, col, :]; label = "reference", lw = 3, xlabel = "cardiac phase", ylabel = "|x|",
    size = (700, 380), title = "voxel ($row, $col) through the cycle"
)
for (label, x̂) in (("zero-filled", x_zf), ("temporal TV", results[4][2]), ("L+S", img_ls))
    a = abs.(unname(x̂))
    α = sum(a .* reference) / sum(abs2, a)
    plot!(α .* a[row, col, :]; label = label, lw = 2)
end
plot!()

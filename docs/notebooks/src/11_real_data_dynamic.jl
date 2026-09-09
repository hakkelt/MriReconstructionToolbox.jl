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
# This notebook is also the place where the comparison is done *honestly*. A table of methods
# each run at a λ someone once picked says nothing: half of what looks like "method A beats
# method B" is really "λ_A happened to suit this dataset". So every method here gets its own
# small λ sweep, is shown at its own best setting, and is scored twice — once globally, and once
# on the pixels that actually move, which is the only place the temporal models can differ from
# the frame-by-frame ones.
#
# > **Data terms.** OCMR has its own data-use agreement and asks that you cite Chen et al.,
# > *OCMR (v1.0) — Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance
# > Imaging*, arXiv:2008.03410 (2020). The first run downloads ~200 MB.
#
# **Contents**
# 1. From ISMRMRD file to `AcquisitionInfo` in one call
# 2. Readout oversampling, coil compression and the reference
# 3. Sensitivity maps
# 4. Two retrospective sampling patterns
# 5. Choosing λ per method
# 6. The comparison, at each method's best λ
# 7. Timing, measured properly
# 8. The temporal profile
# 9. What actually wins, and when

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
using Printf
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. From ISMRMRD file to `AcquisitionInfo` in one call
#
# `AcquisitionInfo(::MRIBase.RawAcquisitionData)` (the package extension that loading `MRIBase`
# activates — `MRITestData` already does) reads the ISMRMRD header and assembles the array. On
# this file it has three things to get right that hand-written assembly usually does not:
#
# * **A cardiac phase axis.** Every encoding counter that varies across the profiles becomes a
#   named batch dimension; `head.idx.phase` becomes `:time`. Counters that never vary contribute
#   nothing, which is why this single-slice cine comes back without a `:z` axis.
# * **An asymmetric echo.** Only a contiguous block of the 512-sample encoded readout was
#   acquired, and its k = 0 sits at `head.center_sample`, not at the middle of the block. The
#   constructor places each sample at `raw_index - center + N ÷ 2` and reports the gap as a
#   `subsampling` range rather than silently sliding the image along `x`.
# * **The image-domain convention.** A scanner images an object centred in the FOV, whereas MRT's
#   default is the plain-DFT one (image origin at index 1). The constructor sets
#   `shifted_image_dims` on both spatial axes, so **no `fftshift` appears anywhere in this
#   notebook** — without it every frame would come out rolled by half the FOV.

# %%
if MRITestData.get_download_path() === nothing
    MRITestData.set_download_path!(:cache)
end

entry = MRITestData.dataset(MRITestData.OCMR_SOURCE, "fs_0001_1_5T")
raw = MRITestData.load_raw(entry)

acq_enc = AcquisitionInfo(raw)

println(acq_enc)
println()
println("dimensions:        ", dimnames(acq_enc.kspace_data), " = ", size(acq_enc.kspace_data))
println("encoded matrix:    ", acq_enc.image_size)
println("reconstructed to:  ", Int.(raw.params["reconSize"]))
println("readout coverage:  ", acq_enc.subsampling[1], " of 1:", acq_enc.image_size[1])
println("cardiac phases:    ", size(acq_enc.kspace_data, :time))

# %% [markdown]
# ## 2. Readout oversampling, coil compression and the reference
#
# ### Removing the readout oversampling
#
# The encoded matrix is 512 wide but `reconSize` is 256, and `encodedFOV[1]` is twice
# `reconFOV[1]`: the readout is **2× oversampled**, which every Cartesian scanner does because
# oversampling along the readout is free (it costs sampling rate, not time) and it moves
# out-of-FOV anatomy out of the way instead of folding it in.
#
# Removing it is a crop in the *image* domain, and it is exact: nothing folds along a fully
# sampled axis, so the discarded half is simply anatomy outside the prescribed FOV. Doing it
# first halves the size of every reconstruction below.
#
# The round trip is worth reading closely, because it is the one place in these two notebooks
# where the FFT-shift convention has to be written out by hand. `reconstruct` hands back centred
# images (`shifted_image_dims`), so going back to centred k-space is `fftshift ∘ fft ∘ ifftshift`
# along both spatial axes — *not* a bare `fft`, which would treat array index 1 as the origin.

# %%
coil_enc = reconstruct(acq_enc; verbosity = Silent())    # no maps => one image per coil
nx_recon = Int(raw.params["reconSize"][1])
x_lo = (size(coil_enc, :x) - nx_recon) ÷ 2 + 1
coil_cropped = unname(coil_enc)[x_lo:(x_lo + nx_recon - 1), :, :, :]

ksp_cine = ComplexF32.(fftshift(fft(ifftshift(coil_cropped, (1, 2)), (1, 2)), (1, 2)))
acq_cine = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil, :time)}(ksp_cine);
    is3D = false, shifted_image_dims = (:x, :y),
)
println("encoded  ", size(acq_enc.kspace_data), " -> cine ", size(acq_cine.kspace_data))

# The crop is a projection, so re-reconstructing must return exactly what we cropped.
round_trip = unname(reconstruct(acq_cine; verbosity = Silent()))
println("round-trip error: ", norm(round_trip - coil_cropped) / norm(coil_cropped))

# %%
# Fifteen channels is more than this problem needs; compressing to six virtual coils cuts the
# cost of every iterative reconstruction below by more than half at no visible cost in quality.
acq_cine, _ = compress_coils(acq_cine, 6; method = SVDCompression())
ksp_cine = unname(acq_cine.kspace_data)
nkx, nky, ncoil, nframes = size(ksp_cine)
println("after coil compression: ", size(ksp_cine))

# The fully-sampled reference: per-frame root sum of squares of the coil images.
coil_frames = reconstruct(acq_cine; verbosity = Silent())
reference = sqrt.(sum(abs2, unname(coil_frames); dims = 3)[:, :, 1, :])
println("reference: ", size(reference))

# %% [markdown]
# ### Which frame to look at
#
# A cine has no "most representative" frame, but it does have a most *informative* one: the
# frame furthest from the temporal mean, i.e. the one where the myocardium has moved the most.
# That is where an over-smoothed temporal reconstruction gives itself away, so every single-frame
# picture below uses it rather than an arbitrary index. (This dataset is a single slice — the
# ISMRMRD `slice` counter never varies — so there is no slice to choose, only a frame.)

# %%
temporal_mean = mean(reference; dims = 3)
frame_deviation = [norm(reference[:, :, t] - temporal_mean[:, :, 1]) for t in 1:nframes]
frame = argmax(frame_deviation)
println("most dynamic frame: ", frame, " of ", nframes,
    "  (deviation from the temporal mean, normalized: ",
    join(round.(frame_deviation ./ maximum(frame_deviation); digits = 2), " "), ")")

jim(reference[:, :, 1:3:nframes]; title = "reference, every 3rd frame", nrow = 2, size = (1200, 620))

# %% [markdown]
# ## 3. Sensitivity maps
#
# Calibrate from a **single frame**, not the time average: cardiac motion smears a temporally
# averaged calibration region and corrupts the ESPIRiT maps.
#
# Take the frame out with the copy constructor rather than by slicing the bare array. Sensitivity
# maps live in the image domain, so they inherit whatever FFT-shift convention they were
# estimated under; calibrating from a raw `NamedDimsArray` would silently use MRT's *default*
# convention and hand back maps rolled by half the FOV relative to this acquisition. Going
# through the `AcquisitionInfo` carries `shifted_image_dims` along and cannot get that wrong.

# %%
acq_frame1 = AcquisitionInfo(acq_cine; kspace_data = acq_cine.kspace_data[time = 1])
smaps = estimate_sensitivities(
    acq_frame1; method = ESPIRiT(calib_size = 24, kernel_size = 6)
).sensitivity_maps
println("maps: ", size(smaps), " ", dimnames(smaps))
jim(abs.(unname(smaps)); title = "ESPIRiT maps (frame 1)", nrow = 2, size = (1000, 500))

# %% [markdown]
# ## 4. Two retrospective sampling patterns
#
# The sampling pattern decides in advance how much a temporal model can possibly gain, so this
# notebook uses two of them at the **same** net acceleration:
#
# * **Fixed** — uniform R = 3 plus a fully sampled centre block, the same ky lines at every
#   frame. The aliasing is then identical in every frame, i.e. perfectly *coherent* in time. A
#   temporal regularizer sees a time series whose artefact does not change with time, and cannot
#   separate the artefact from the anatomy by looking along time.
# * **Interleaved** — the same pattern cyclically shifted by one line per frame. Same number of
#   lines per frame, same net acceleration, but the aliasing now moves frame to frame, and the
#   *time average* of the acquired lines covers all of ky.
#
# MRT expresses a per-frame pattern as a `Vector` of subsampling specs, one per batch element —
# `[(:, mask_t) for t in 1:nframes]` — alongside a fixed-shape k-space array, which is why both
# masks are built with the same line count.

# %%
R, acs = 3, 8
mask_fixed = falses(nky)
mask_fixed[1:R:nky] .= true
mask_fixed[(nky ÷ 2 - acs):(nky ÷ 2 + acs)] .= true
masks_interleaved = [circshift(mask_fixed, t - 1) for t in 1:nframes]
nlines = sum(mask_fixed)

println("lines per frame: ", nlines, " of ", nky,
    "  (net acceleration ", round(nky / nlines, digits = 2), "x)")
println("lines per frame, interleaved: ", unique(sum.(masks_interleaved)),
    "   union over time: ", sum(reduce(.|, masks_interleaved)), " of ", nky)

# Both acquisitions are built with the copy constructor, `AcquisitionInfo(info; field = value)`,
# so `image_size`, `is3D` and — the one that bites — `shifted_image_dims` are inherited rather
# than retyped.
acq_fixed = AcquisitionInfo(
    acq_cine;
    kspace_data = acq_cine.kspace_data[ky = mask_fixed],
    subsampling = (:, mask_fixed), sensitivity_maps = smaps,
)

ksp_interleaved = similar(ksp_cine, nkx, nlines, ncoil, nframes)
for t in 1:nframes
    ksp_interleaved[:, :, :, t] .= ksp_cine[:, masks_interleaved[t], :, t]
end
acq_interleaved = AcquisitionInfo(
    acq_cine;
    kspace_data = NamedDimsArray{(:kx, :ky, :coil, :time)}(ksp_interleaved),
    subsampling = [(:, m) for m in masks_interleaved], sensitivity_maps = smaps,
)
println(acq_interleaved)

# %%
# What the two patterns look like as (ky, frame) maps, and what their aliasing does to a frame.
pattern_fixed = repeat(mask_fixed, 1, nframes)
pattern_interleaved = reduce(hcat, masks_interleaved)

x_zf_fixed = reconstruct(acq_fixed; verbosity = Silent())
x_zf_interleaved = reconstruct(acq_interleaved; verbosity = Silent())

jim(
    jim(pattern_fixed; title = "fixed pattern (ky vs frame)", aspect_ratio = :auto),
    jim(pattern_interleaved; title = "interleaved pattern (ky vs frame)", aspect_ratio = :auto),
    jim(abs.(unname(x_zf_fixed))[:, :, frame]; title = "zero-filled, fixed"),
    jim(abs.(unname(x_zf_interleaved))[:, :, frame]; title = "zero-filled, interleaved");
    layout = (2, 2), size = (1100, 900)
)

# %% [markdown]
# ### How the reconstructions are scored
#
# Two numbers, both against the fully sampled root-sum-of-squares reference, on magnitude, with
# the amplitude aligned (different reconstructions carry different scalings) :
#
# * **global** — over the whole object. This is the number everyone quotes, and it is dominated
#   by the static chest wall, which is most of the object and which every method reconstructs
#   well.
# * **dynamic** — over the pixels whose intensity varies most across the cardiac cycle (the top
#   of the temporal-standard-deviation map: heart, vessels, diaphragm). This is the only region
#   where a temporal or low-rank prior can differ from a frame-by-frame one at all, so it is the
#   number that actually discriminates between the methods.

# %%
support = reference .> 0.1maximum(reference)
motion = dropdims(std(reference; dims = 3); dims = 3)
dynamic = repeat(motion .> 0.25maximum(motion), 1, 1, nframes)
println("object pixels: ", sum(support), "   dynamic pixels: ", sum(dynamic))

function scores(x̂)
    a = abs.(unname(x̂))
    α = sum(a[support] .* reference[support]) / sum(abs2, a[support])
    a = α .* a
    return (
        global_err = norm(a[support] - reference[support]) / norm(reference[support]),
        dynamic_err = norm(a[dynamic] - reference[dynamic]) / norm(reference[dynamic]),
    )
end

for (name, x) in ("fixed" => x_zf_fixed, "interleaved" => x_zf_interleaved)
    s = scores(x)
    @printf("zero-filled %-12s global %.4f   dynamic %.4f\n", name, s.global_err, s.dynamic_err)
end

# %%
jim(
    jim(motion; title = "temporal standard deviation"),
    jim(dynamic[:, :, 1]; title = "dynamic-pixel mask"),
    jim(support[:, :, frame]; title = "object mask");
    layout = (1, 3), size = (1300, 420)
)

# %% [markdown]
# ## 5. Choosing λ per method
#
# Every method below is a different penalty on a different transform, so its λ lives on a
# different scale — nothing about "λ = 0.02" is comparable between `L1Wavelet2D` and `LowRank`.
# Comparing methods at a shared λ, or at whatever λ each one was first written with, mostly
# measures how lucky those choices were. So: a four-point log-spaced sweep per method, per
# pattern, and each method is reported at the λ that minimizes its **dynamic** error.
#
# Four points is coarse, and deliberately so — this has to run in a couple of minutes. The best
# λ is therefore accurate to a factor of ~3, which is enough to stop a method being shown at a
# grossly wrong setting but not enough to split hairs between two methods that land within a
# percent of each other.

# %%
sweeps = (
    "L2Image (per frame)" =>
        (Float32[1.0e-4, 1.0e-3, 5.0e-3, 2.0e-2], λ -> IterativeReconstruction(L2Image(λ); maxit = 20)),
    "L1Wavelet2D (per frame)" =>
        (Float32[1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2], λ -> IterativeReconstruction(L1Wavelet2D(λ); maxit = 30)),
    "L1TemporalFourier" =>
        (Float32[3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1], λ -> IterativeReconstruction(L1TemporalFourier(λ; time_dim = :time); maxit = 30)),
    "TemporalTotalVariation" =>
        (Float32[3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1], λ -> IterativeReconstruction(TemporalTotalVariation(λ; time_dim = :time); maxit = 30)),
    "LowRank" =>
        (Float32[1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1], λ -> IterativeReconstruction(LowRank(λ; time_dim = :time); maxit = 30)),
    "LocallyLowRank" =>
        (Float32[1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1], λ -> IterativeReconstruction(LocallyLowRank(λ; block_size = 8, time_dim = :time); maxit = 30)),
)

function sweep(acq)
    return map(sweeps) do (label, (λs, build))
        results = [(λ, reconstruct(acq, build(λ); verbosity = Silent())) for λ in λs]
        errs = [scores(x).dynamic_err for (_, x) in results]
        best = argmin(errs)
        @printf(
            "%-24s best λ = %-8g dynamic %.4f   (sweep: %s)\n",
            label, results[best][1], errs[best],
            join(map(e -> @sprintf("%.4f", e), errs), " ")
        )
        return label => (λ = results[best][1], x = results[best][2], errs = errs, λs = λs)
    end
end

println("--- fixed pattern ---")
best_fixed = sweep(acq_fixed)
println("\n--- interleaved pattern ---")
best_interleaved = sweep(acq_interleaved);

# %% [markdown]
# ## 6. The comparison, at each method's best λ

# %%
function summary_table(name, best, x_zf)
    @printf("%s\n%-24s %9s %9s\n", name, "", "global", "dynamic")
    s = scores(x_zf)
    @printf("%-24s %9.4f %9.4f\n", "zero-filled", s.global_err, s.dynamic_err)
    for (label, r) in best
        s = scores(r.x)
        @printf("%-24s %9.4f %9.4f   (λ = %g)\n", label, s.global_err, s.dynamic_err, r.λ)
    end
    return println()
end

summary_table("FIXED pattern (same lines every frame)", best_fixed, x_zf_fixed)
summary_table("INTERLEAVED pattern (lines shift with frame)", best_interleaved, x_zf_interleaved)

# %%
plot(
    plot(
        [r.λs for (_, r) in best_fixed], [r.errs for (_, r) in best_fixed];
        xscale = :log10, marker = :circle, lw = 2, xlabel = "lambda",
        ylabel = "dynamic-region error", title = "lambda sweep, fixed pattern",
        label = reshape([l for (l, _) in best_fixed], 1, :), legend = :outertopright
    ),
    plot(
        [r.λs for (_, r) in best_interleaved], [r.errs for (_, r) in best_interleaved];
        xscale = :log10, marker = :circle, lw = 2, xlabel = "lambda",
        ylabel = "dynamic-region error", title = "lambda sweep, interleaved pattern",
        label = reshape([l for (l, _) in best_interleaved], 1, :), legend = :outertopright
    );
    layout = (2, 1), size = (1000, 800)
)

# %%
jim(
    jim(reference[:, :, frame]; title = "reference"),
    jim(abs.(unname(x_zf_interleaved))[:, :, frame]; title = "zero-filled"),
    (
        jim(abs.(unname(r.x))[:, :, frame]; title = label)
            for (label, r) in best_interleaved
    )...;
    layout = (2, 4), size = (1700, 850)
)

# %% [markdown]
# ## 7. Timing, measured properly
#
# A bare `@elapsed` around the first call to a reconstruction measures Julia compiling it, which
# on this problem is comparable to the solve itself. Warm the method up on a two-iteration run
# first, then time it; and because this notebook may well be running on a shared machine, take
# the **best of three** rather than a single number — wall-clock timings on a loaded node swing
# by tens of percent, and the minimum is the least contaminated estimate of the work actually
# done.

# %%
function best_of(f, n = 3)
    f()                                        # warm-up: compile everything
    return minimum(@elapsed(f()) for _ in 1:n)
end

for (label, r) in best_interleaved
    (_, build) = sweeps[findfirst(((l, _),) -> l == label, sweeps)][2]
    t = best_of(() -> reconstruct(acq_interleaved, build(r.λ); verbosity = Silent()))
    @printf("%-24s %6.2f s  (best of 3, after warm-up)\n", label, t)
end

# %% [markdown]
# The per-frame methods are the fast ones for a structural reason, not an implementation one: a
# purely spatial regularizer leaves `:time` a batch dimension, so `reconstruct` splits the
# problem into `nframes` independent solves and runs them in parallel (notebook 7, section 5).
# A temporal or low-rank penalty couples the frames, so there is one large problem instead —
# see how the number changes with `JULIA_NUM_THREADS`.

# %% [markdown]
# ## 8. The temporal profile
#
# A y–t cut through the heart is the standard way to look at a cine reconstruction: temporal
# blurring and residual aliasing that are invisible in a single frame show up immediately as
# smearing or as banding along the time axis.

# %%
col = argmax(vec(sum(motion; dims = 1)))
println("profiling column ", col)

profile_methods = ("L1Wavelet2D (per frame)", "TemporalTotalVariation", "LowRank")
profiles = (
    "reference" => reference[:, col, :],
    "zero-filled" => abs.(unname(x_zf_interleaved))[:, col, :],
    (
        m => abs.(unname(best_interleaved[findfirst(((l, _),) -> l == m, best_interleaved)][2].x))[:, col, :]
            for m in profile_methods
    )...,
)
jim(
    (jim(p; title = label, aspect_ratio = :auto) for (label, p) in profiles)...;
    layout = (2, 3), size = (1400, 800)
)

# %%
# And the intensity of one voxel through the cardiac cycle.
row = argmax(vec(sum(motion; dims = 2)))
plot(
    reference[row, col, :]; label = "reference", lw = 3, xlabel = "cardiac phase", ylabel = "|x|",
    size = (850, 420), title = "voxel ($row, $col) through the cycle", legend = :outertopright
)
for (label, p) in profiles
    label == "reference" && continue
    a = p[row, :]
    α = sum(a .* reference[row, col, :]) / sum(abs2, a)
    plot!(α .* a; label = label, lw = 2)
end
plot!()

# %% [markdown]
# ## 9. What actually wins, and when
#
# (Filled in from the numbers above.)

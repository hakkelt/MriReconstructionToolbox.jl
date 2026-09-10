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
# # 9 — Real scanner data, end to end
#
# Everything so far ran on simulated data. This notebook takes a real Cartesian brain acquisition
# and walks the whole pipeline: raw ISMRMRD file → `AcquisitionInfo` → preprocessing →
# sensitivity maps → reference reconstruction → retrospective undersampling → compressed sensing
# and parallel imaging → noise analysis.
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
# 2. From `RawAcquisitionData` to `AcquisitionInfo` in one call
# 3. Preprocessing — prewhitening, sensitivity maps, coil compression
# 4. The fully-sampled reference
# 5. Retrospective undersampling and compressed sensing
# 6. Parallel imaging on the same data
# 7. Noise analysis with pseudo-replicas

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MRITestData
using MIRTjim: jim
using Plots
using NamedDims
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
println("per profile:   ", size(raw.profiles[1].data), "  (readout samples x channels)")
println("encoded size:  ", raw.params["encodedSize"])
println("slices:        ", length(unique(Int(p.head.idx.slice) for p in raw.profiles)))
println("field strength: ", raw.params["systemFieldStrength_T"], " T")

# %% [markdown]
# ## 2. From `RawAcquisitionData` to `AcquisitionInfo` in one call
#
# Turning a scanner file into the array a reconstruction can use is where most hand-written MRI
# code goes wrong, so MRT does it for you. Loading `MRIBase` (`MRITestData` already does)
# activates a package extension that adds
#
# `$julia
# AcquisitionInfo(raw::MRIBase.RawAcquisitionData; sensitivity_maps = nothing)
# $`
#
# It reads the encoding matrix from the header, drops noise-calibration profiles, decides
# Cartesian vs. non-Cartesian from `raw.params["trajectory"]`, turns every encoding counter that
# actually varies into a named batch dimension, and reduces the sampling pattern that is present
# in the profiles to a `subsampling` spec.
#
# Two conventions it gets right that hand-rolled assembly usually does not:
#
# * **k-space centring.** A profile's phase-encode index is an index into the *encoded* matrix
#   whose k = 0 line sits at `enc_lim_kspace_encoding_step_1.center`, and its readout has k = 0 at
#   `head.center_sample` — neither is necessarily the middle of the array. The constructor places
#   each sample at `raw_index - center + N ÷ 2`, so DC really does land at `N ÷ 2 + 1`.
# * **Image centring.** MRT's own default is the plain-DFT one, image origin at index 1 — fine for
#   k-space MRT itself simulated, wrong for a scanner, which images an object centred in the FOV.
#   The constructor therefore sets `shifted_image_dims` on every spatial axis.
#
# Together those mean **no `fftshift` anywhere in this notebook**. Get either wrong and the
# reconstruction comes out shifted — by half the FOV for the second one, which is the classic
# "my brain is in the four corners" bug.

# %%
acq_all = AcquisitionInfo(raw)

println(acq_all)
println()
println("dimensions:         ", dimnames(acq_all.kspace_data), " = ", size(acq_all.kspace_data))
println("image size:         ", acq_all.image_size)
println("shifted image dims: ", acq_all.shifted_image_dims)

# %% [markdown]
# The 18 slices came back as a `:z` batch dimension (a multi-slice 2D scan is *not* a 3D encoding,
# and the constructor keeps them apart). Every `AcquisitionInfo` also acts as a copy constructor,
# `AcquisitionInfo(info; field = new_value)`, which is how one slice is pulled out without
# rebuilding the conventions by hand.

# %%
z_mid = size(acq_all.kspace_data, :z) ÷ 2 + 1
acq_slice = AcquisitionInfo(acq_all; kspace_data = acq_all.kspace_data[z = z_mid])
println("slice $z_mid of $(size(acq_all.kspace_data, :z)): ",
    dimnames(acq_slice.kspace_data), " = ", size(acq_slice.kspace_data))

jim(
    log.(abs.(unname(acq_slice.kspace_data)) .+ 1.0f-8);
    title = "log |k-space|, per channel", nrow = 1, size = (1400, 380)
)

# %% [markdown]
# Note the black bands at the top and bottom of every channel. This protocol used a reduced
# **phase resolution**: only a centred block of the 256 encoded phase-encode lines was measured,
# and the file stores the rest as explicit zeros. The constructor cannot tell a measured zero
# from an unmeasured one, so it reports the scan as fully sampled — flagging the empty lines is
# on us.
#
# The honest way to handle them is *not* to crop k-space to the measured block and call it a
# smaller fully-sampled grid: that silently changes the pixel size along `y` and squashes every
# image in the notebook. It is to mark them as missing with a `subsampling` mask and keep the
# 256 × 256 image grid. MRT then zero-fills them, which is exactly what the scanner does.

# %%
ksp_slice = acq_slice.kspace_data
measured = [any(!iszero, view(unname(ksp_slice), :, j, :)) for j in axes(ksp_slice, :ky)]
println("phase-encode lines measured: ", sum(measured), " of ", length(measured),
    "  (lines ", findfirst(measured), ":", findlast(measured), ", ",
    round(100 * sum(measured) / length(measured), digits = 1), "% phase resolution)")

acq_coils = AcquisitionInfo(
    acq_slice; kspace_data = ksp_slice[ky = measured], subsampling = (:, measured)
)
println("stored k-space: ", size(acq_coils.kspace_data),
    "   reconstructed on: ", acq_coils.image_size)

# %%
# The coil images and their root-sum-of-squares — the coil-independent reference every comparison
# below is scored against.
coil_images = reconstruct(acq_coils; verbosity = Silent())        # no maps => one image per coil
reference = sqrt.(sum(abs2, unname(coil_images); dims = 3)[:, :, 1])

jim(
    jim(abs.(unname(coil_images)); title = "coil images", nrow = 1),
    jim(reference; title = "root sum of squares");
    layout = (2, 1), size = (1100, 780)
)

# %% [markdown]
# ## 3. Preprocessing
#
# ### Noise prewhitening
#
# Receiver channels do not see independent noise. Neighbouring elements couple to each other and
# to the same body noise, so the channel noise covariance $\Psi$ has off-diagonal entries, and
# the channels do not even have equal noise variance (different cable lengths, preamplifier
# gains, loading).
#
# That matters because every least-squares reconstruction here minimizes $\|\mathcal{A}x - y\|_2^2$,
# and that is the maximum-likelihood data term **only** when the noise is white with unit
# variance. Prewhitening makes it true: estimate $\Psi$ from noise-only samples, factor
# $\Psi = L L^H$, and replace $y$ by $L^{-1} y$ (and the sensitivity maps by $L^{-1} S$,
# which `prewhiten` does for you when they are attached).
#
# `estimate_noise_covariance` wants pure-noise samples. A dedicated noise scan is best; this
# dataset has none, so we use the four corners of the measured k-space block — high frequency in
# both directions, where the brain has no signal left.

# %%
corner = 20
ksp = unname(acq_coils.kspace_data)
noise_patch = cat(
    ksp[1:corner, 1:corner, :],
    ksp[(end - corner + 1):end, 1:corner, :],
    ksp[1:corner, (end - corner + 1):end, :],
    ksp[(end - corner + 1):end, (end - corner + 1):end, :];
    dims = 2
)
Ψ = estimate_noise_covariance(NamedDimsArray{(:kx, :ky, :coil)}(noise_patch))

acq_white = prewhiten(acq_coils, Ψ)

# The same corners, after whitening, to check the result.
ksp_w = unname(acq_white.kspace_data)
noise_patch_w = cat(
    ksp_w[1:corner, 1:corner, :],
    ksp_w[(end - corner + 1):end, 1:corner, :],
    ksp_w[1:corner, (end - corner + 1):end, :],
    ksp_w[(end - corner + 1):end, (end - corner + 1):end, :];
    dims = 2
)
Ψ_after = estimate_noise_covariance(NamedDimsArray{(:kx, :ky, :coil)}(noise_patch_w))

# Covariance is easier to read as a correlation matrix: unit diagonal by construction, so every
# visible off-diagonal value is genuine channel coupling.
correlation(Ψ) = abs.(Ψ) ./ sqrt.(real.(diag(Ψ)) * real.(diag(Ψ))')

println("noise std per channel, before: ", round.(sqrt.(real.(diag(Ψ))); sigdigits = 3))
println("noise std per channel, after:  ", round.(sqrt.(real.(diag(Ψ_after))); sigdigits = 3))
println("largest channel correlation, before: ",
    round(maximum(correlation(Ψ) - I), digits = 3))
println("largest channel correlation, after:  ",
    round(maximum(correlation(Ψ_after) - I), digits = 3))

# %% [markdown]
# **What to look for in the two heatmaps below.** The left one is the measured correlation
# matrix: a unit diagonal, and off-diagonal blobs wherever two elements of the array are coupled.
# The right one is the same estimate recomputed *after* whitening, and it must be the identity —
# black everywhere off the diagonal. That is not a soft convergence criterion but an algebraic
# identity: $L^{-1} \Psi L^{-H} = I$ exactly, so any visible off-diagonal structure on the
# right means the covariance was estimated from samples that were not pure noise (signal leaking
# into the corners, a too-small patch), not that whitening "did not work well enough".

# %%
chan = 1:size(Ψ, 1)
heatmap_args = (
    clim = (0, 1), c = :viridis, aspect_ratio = 1, xticks = chan, yticks = chan,
    xlabel = "channel", ylabel = "channel",
)
plot(
    heatmap(chan, chan, correlation(Ψ); title = "|correlation| before", heatmap_args...),
    heatmap(chan, chan, correlation(Ψ_after); title = "|correlation| after", heatmap_args...);
    layout = (1, 2), size = (950, 400)
)

# %% [markdown]
# The per-channel noise histograms make the second half of the story visible. The printed
# standard deviations above quantify what to look for: at this array's mild coupling (largest
# off-diagonal correlation around 0.3) the four channels differ by tens of percent, not by an
# order of magnitude, so do not expect the four curves before whitening to look dramatically
# different by eye — the important change is *after*, where all four collapse onto the same
# unit-variance Gaussian, which is the assumption the least-squares data term makes.

# %%
noise_before = reshape(noise_patch, :, size(noise_patch, 3))
noise_after = reshape(noise_patch_w, :, size(noise_patch_w, 3))
plot(
    plot(
        [real.(noise_before[:, c]) for c in chan]; seriestype = :stephist, bins = 60,
        label = reshape(["ch $c" for c in chan], 1, :), lw = 2,
        title = "noise, real part - before", xlabel = "value", ylabel = "count"
    ),
    plot(
        [real.(noise_after[:, c]) for c in chan]; seriestype = :stephist, bins = 60,
        label = reshape(["ch $c" for c in chan], 1, :), lw = 2,
        title = "noise, real part - after (whitened)", xlabel = "value", ylabel = "count"
    );
    layout = (1, 2), size = (1000, 380)
)

# %% [markdown]
# ### Does it change the picture?
#
# The two reconstructions below use the *same* sensitivity maps (estimated once, on the whitened
# data, then pushed back through `L` for the un-whitened path) so the only difference is whether
# the data term knows about $\Psi$. SNR is measured as the mean magnitude over a box inside the
# brain divided by the standard deviation over the image corners, which are pure background.
#
# Read the printed numbers, not the pictures: with four mildly correlated channels the gain is a
# **few percent**, and the difference image is dominated by noise texture rather than structure.
# That is the honest result at this array size. Prewhitening is cheap and always correct, and it
# is what makes the g-factor of section 7 mean anything; on a 32-channel array with correlations
# of 0.5 and up, the same step is worth much more.

# %%
# One set of maps, estimated on the measured data, carried into the whitened frame by the very
# same transform `prewhiten` applies to the k-space — so the two reconstructions differ in
# nothing but whether the data term knows about Ψ.
maps_raw = estimate_sensitivities(acq_coils; method = ESPIRiT(calib_size = 24, kernel_size = 6)).sensitivity_maps
maps_matched = prewhiten(maps_raw, Ψ)

x_raw = reconstruct(AcquisitionInfo(acq_coils; sensitivity_maps = maps_raw); verbosity = Silent())
x_white = reconstruct(AcquisitionInfo(acq_white; sensitivity_maps = maps_matched); verbosity = Silent())

box = 100:156
# A proper background region — a corner box outside the head, hundreds of pixels — not
# `reference .< threshold`: that thresholded set is dominated by a thin rim of near-zero pixels
# right at the object's edge and can come down to a literal handful of samples, whose standard
# deviation is itself noisy rather than a stable estimate of the background level.
bg_box = 1:40
bg_idx = CartesianIndices((bg_box, bg_box))
snr(x) = mean(abs.(unname(x))[box, box]) / std(abs.(unname(x))[bg_idx])

println("background pixels used: ", length(bg_idx))
println("SNR without prewhitening: ", round(snr(x_raw), digits = 2))
println("SNR with prewhitening:    ", round(snr(x_white), digits = 2))
println("relative change:          ",
    round(100 * (snr(x_white) / snr(x_raw) - 1), digits = 1), " %")

side_by_side(
    abs.(unname(x_raw)) ./ maximum(abs, unname(x_raw)),
    abs.(unname(x_white)) ./ maximum(abs, unname(x_white));
    titles = ("no prewhitening", "prewhitened"), size = (900, 420)
)

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
    layout = (3, 1), size = (1300, 1000)
)

# %% [markdown]
# ### Coil compression
#
# With four channels there is little to gain, but the mechanics are the same as on a 32-channel
# array: `compress_coils` returns the compressed acquisition and the compression matrix.

# %%
acq_compressed, C = compress_coils(acq_espirit, 2; method = SVDCompression())
println("compression matrix: ", size(C), "  (virtual x physical)")
println("channels: ", size(acq_espirit.kspace_data, :coil), " -> ", size(acq_compressed.kspace_data, :coil))

rec_full_4ch = reconstruct(acq_espirit; verbosity = Silent())
rec_full_2ch = reconstruct(acq_compressed; verbosity = Silent())

jim(
    jim(abs.(unname(rec_full_4ch)); title = "4 channels"),
    jim(abs.(unname(rec_full_2ch)); title = "2 virtual channels"),
    difference_image(abs.(unname(rec_full_4ch)), abs.(unname(rec_full_2ch)));
    layout = (1, 3), size = (1350, 430)
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

# Scale the adjoint panel by the same alpha rel_err aligns with, so the two panels are on a
# common scale rather than each other's own maximum.
α_ref = sum(abs.(unname(x_ref))[support] .* reference[support]) / sum(abs2, abs.(unname(x_ref))[support])
side_by_side(
    reference, α_ref .* abs.(unname(x_ref));
    titles = ("root sum of squares", "ESPIRiT + adjoint (A'y)"), size = (900, 420)
)

# %% [markdown]
# ## 5. Retrospective undersampling and compressed sensing
#
# Any sampling pattern can be applied after the fact — the standard way of evaluating an
# accelerated reconstruction against a real reference. The pattern has to stay inside the
# measured block, so the variable-density mask is intersected with `measured`; the acceleration
# below is quoted against the lines that were actually acquired, not against the encoded 256.

# %%
pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.08)
mask_us = create_sampling_pattern(pdf, acq_coils.image_size)[2] .& measured
println("retained phase encodes: ", sum(mask_us), " of ", sum(measured),
    "  (", round(sum(measured) / sum(mask_us), digits = 2), "x acceleration)")

acq_us = AcquisitionInfo(
    acq_white;
    kspace_data = acq_white.kspace_data[ky = mask_us[measured]],
    subsampling = (:, mask_us),
    sensitivity_maps = maps_espirit,
)

x_zf = reconstruct(acq_us; verbosity = Silent())
println("zero-filled: ", round(rel_err(x_zf), digits = 4))
jim(abs.(unname(x_zf)); title = "zero-filled, undersampled", size = (480, 420))

# %%
# λ is larger here than on the phantom of notebook 5: this is 0.3 T data with four channels, so
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
end;

# %% [markdown]
# Two things are worth reading off those numbers. The unregularized parallel-imaging solve
# (`L2Image` with a small λ is CG-SENSE) is *worse* than the zero-filled adjoint here: with four
# low-field channels the inverse problem is badly conditioned, and CG happily amplifies noise
# into the answer — section 7 measures the g-factor for exactly this setup. The regularized
# reconstructions are what make the acceleration usable.

# %%
jim(
    jim(reference; title = "reference (fully sampled)"),
    jim(abs.(unname(x_zf)); title = "zero-filled"),
    (jim(abs.(unname(x̂)); title = label) for (label, x̂) in recons)...;
    layout = (2, 4), size = (1800, 900)
)

# %%
# A λ sweep on the real data — the same exercise as on the phantom, with a real noise floor.
λs = Float32[2.0e-3, 8.0e-3, 2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1]
errs = map(λs) do λ
    rel_err(reconstruct(acq_us, IterativeReconstruction(L1Wavelet2D(λ); maxit = 60); verbosity = Silent()))
end
plot(
    λs, errs; xscale = :log10, marker = :circle, lw = 2, legend = false,
    xlabel = "lambda", ylabel = "relative error vs. reference",
    title = "L1-wavelet lambda sweep, real data", size = (700, 400)
)

# %% [markdown]
# ### Preconditioned CG-SENSE
#
# `CGNR(; P, P_is_inverse)` accepts a preconditioner. The natural choice for SENSE is the
# diagonal image-domain operator $P = 1/(\sum_c |S_c|^2 + \lambda)$: it approximates the inverse
# of $\mathcal{A}^H\mathcal{A}$'s diagonal, which is dominated by the coil sensitivity energy at
# each pixel. Built as a `DiagOp`, it costs one elementwise divide per application. The point of
# preconditioning is convergence *speed* at the same accuracy, not a different answer — so the
# comparison below is error against iteration count, not error against wall-clock time.

# %%
using AbstractOperators: DiagOp

coil_energy = dropdims(sum(abs2, unname(maps_espirit); dims = 3); dims = 3)
λ_precond = 1.0f-2
P = DiagOp(ComplexF32.(1 ./ (coil_energy .+ λ_precond)))

method_unprecond = IterativeReconstruction(L2Image(1.0f-2); algorithm = CGNR(), maxit = 30, tol = 0.0)
method_precond = IterativeReconstruction(
    L2Image(1.0f-2); algorithm = CGNR(P = P, P_is_inverse = true), maxit = 30, tol = 0.0
)

trace_unprecond = IterationTrace(rel_err)
trace_precond = IterationTrace(rel_err)
reconstruct(
    acq_us, IterativeReconstruction(L2Image(1.0f-2); algorithm = CGNR(), maxit = 30, tol = 0.0, on_iteration = trace_unprecond);
    verbosity = Silent()
)
reconstruct(
    acq_us,
    IterativeReconstruction(
        L2Image(1.0f-2); algorithm = CGNR(P = P, P_is_inverse = true), maxit = 30, tol = 0.0,
        on_iteration = trace_precond
    );
    verbosity = Silent()
)

println("relative error vs. reference, by iteration:")
println(rpad("iteration", 12), rpad("unpreconditioned", 20), "preconditioned")
for it in (1, 5, 10, 20, 30)
    println(
        rpad(it, 12), rpad(round(trace_unprecond.values[it], digits = 4), 20),
        round(trace_precond.values[it], digits = 4)
    )
end

# %% [markdown]
# The two columns above are identical on this dataset, which is worth explaining rather than
# leaving as a null result: `estimate_sensitivities` normalizes its output so
# $\sum_c|S_c|^2 \approx 1$ throughout the reconstructed support (ESPIRiT's own convention), so
# $P$ is nearly *constant* there — it only varies outside the object, where the coil energy
# drops to zero. `rel_err` is masked to `support`, so it cannot see the one region $P$ actually
# reweights. A preconditioner built from a genuinely non-uniform coil geometry (an array with
# strong near/far sensitivity falloff, or a support-masked error metric extended to the whole
# FOV) is where this preconditioner earns its keep; on unit-normalized maps evaluated only
# inside the object it is close to a no-op, which is itself useful to know before reaching for
# it as a default.
#
# The error growing with iteration count in both columns is the badly-conditioned inverse
# problem from the discussion above (§5): unregularized CG-SENSE on four low-field channels
# amplifies noise, so more iterations make it worse, not better — a preconditioner changes how
# fast that happens, not whether the underlying problem needs regularization.

plot(
    trace_unprecond.iterations, trace_unprecond.values; label = "unpreconditioned", lw = 2,
    xlabel = "iteration", ylabel = "relative error vs. reference", yscale = :log10, size = (650, 380)
)
plot!(trace_precond.iterations, trace_precond.values; label = "preconditioned", lw = 2)

# %% [markdown]
# ## 6. Parallel imaging on the same data
#
# A GRAPPA-style pattern — uniform R = 2 plus a fully sampled autocalibration block — lets the
# autocalibrated methods run on the same slice.

# %%
R = 2
nky = acq_coils.image_size[2]
ky_centre = nky ÷ 2 + 1
mask_pi = falses(nky)
mask_pi[1:R:nky] .= true
mask_pi[(ky_centre - 12):(ky_centre + 11)] .= true
mask_pi .&= measured
println("net acceleration: ", round(sum(measured) / sum(mask_pi), digits = 2), "x")

acq_pi = AcquisitionInfo(
    acq_white;
    kspace_data = acq_white.kspace_data[ky = mask_pi[measured]],
    subsampling = (:, mask_pi),
    sensitivity_maps = maps_espirit,
)

x_grappa = reconstruct(acq_pi, GRAPPA(kernel_size = (3, 2), calib_size = (size(ksp, 1), 24)); verbosity = Silent())
x_sense = reconstruct(acq_pi, IterativeReconstruction(L2Image(1.0f-2); maxit = 30); verbosity = Silent())
x_sense_cs = reconstruct(acq_pi, IterativeReconstruction(L1Wavelet2D(1.0f-2); maxit = 60); verbosity = Silent())

println("GRAPPA               ", round(rel_err(x_grappa), digits = 4))
println("SENSE (L2, CG)       ", round(rel_err(x_sense), digits = 4))
println("SENSE + wavelet CS   ", round(rel_err(x_sense_cs), digits = 4))

jim(
    jim(abs.(unname(x_grappa)); title = "GRAPPA"),
    jim(abs.(unname(x_sense)); title = "SENSE (L2, CG)"),
    jim(abs.(unname(x_sense_cs)); title = "SENSE + wavelet CS");
    layout = (1, 3), size = (1350, 430)
)

# %% [markdown]
# ## 7. Noise analysis with pseudo-replicas
#
# ### What question this answers
#
# Accelerating an acquisition costs SNR twice: once because fewer samples were collected
# (the $\sqrt{R}$ factor, which is unavoidable), and once because *unfolding* the aliased
# signal is an ill-conditioned inverse problem whose conditioning varies from pixel to pixel.
# The second factor is the **geometry factor** $g$: the local noise amplification caused by the
# geometry of the coil array relative to the sampling pattern. It is what makes accelerated
# images noisy in the middle of the FOV, where the coil sensitivities are most similar, while the
# edges stay clean.
#
# For plain SENSE there is a closed-form $g$ map, because the reconstruction is a linear
# operator you can write down. Every interesting reconstruction in this notebook is **not**
# linear: `L1Wavelet2D` thresholds, `TotalVariation2D` and TGV solve a non-smooth problem, GRAPPA
# fits kernels from the data. There is no matrix to invert, so there is no analytic $g$.
#
# ### How pseudo-replicas answer it anyway
#
# The Monte Carlo pseudo-replica method (Robson et al., *Comprehensive quantification of
# signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging
# reconstructions*, Magnetic Resonance in Medicine **60**:895–907, 2008) treats the
# reconstruction as a black box and measures what it does to noise:
#
# 1. Take the measured k-space and add a fresh draw of synthetic complex Gaussian noise to it.
# 2. Run the *whole* reconstruction — the same method, the same λ, the same number of iterations.
# 3. Repeat `replicas` times, then take the pixel-wise mean and standard deviation of the
#    resulting magnitude images.
#
# The standard-deviation map is the noise the reconstruction actually delivers, non-linearity and
# all. `pseudo_replica` also divides it by the corresponding fully-sampled map (with the
# $\sqrt{R}$ factor removed) to give a `g_factor` field.
#
# ### How to read the map
#
# $g = 1$ means the acceleration cost nothing beyond the $\sqrt{R}$ sample loss; $g = 3$
# means the noise in that pixel is three times worse than that. Expect a smooth map with its
# maximum near the centre of the object, growing with acceleration and shrinking as channels are
# added. Only the values inside the object mean anything — outside it both maps are noise divided
# by noise — so the map is shown masked.
#
# ### Limitations
#
# * It is Monte Carlo: the estimate has its own error, falling as $1/\sqrt{N_{\text{replicas}}}$.
#   Sixteen replicas, used here to keep the notebook fast, is enough for the pattern but not for
#   a number you would publish; Robson et al. use hundreds.
# * For a non-linear reconstruction the "g-factor" is not a property of the coil geometry alone —
#   it depends on λ, on the iteration count, and on the underlying image. A regularizer can push
#   $g$ below 1 by *biasing* the estimate: less noise, more smoothing. The σ map alone never
#   reveals that trade; compare it against the error maps in section 5.
# * The added noise must dominate nothing and change nothing else, so data-dependent scaling
#   would rescale every replica by its own noise level. `pseudo_replica` therefore insists on
#   `NoScaling()` or `FixedScaling()`.

# %%
noise_level = 0.02 * sqrt(mean(abs2, unname(acq_white.kspace_data)))

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
    jim(res_full.std; title = "sigma, fully sampled"),
    jim(res_us.std; title = "sigma, undersampled"),
    jim(res_us.g_factor .* support; title = "g-factor (masked)");
    layout = (1, 3), size = (1350, 430)
)

# %% [markdown]
# The same machinery works on a regularized reconstruction, which is the point of the Monte Carlo
# approach. Note the caveat above when reading the comparison: the ℓ₁-wavelet σ map is lower
# everywhere, but part of that is bias, not a better-conditioned inverse.

# %%
res_cs = pseudo_replica(
    acq_us, IterativeReconstruction(L1Wavelet2D(2.0f-2); maxit = 30);
    replicas = 16, noise_std = noise_level, scaling = NoScaling()
)

println("mean sigma over the object, CG-SENSE:   ", round(mean(res_us.std[support]), digits = 5))
println("mean sigma over the object, L1-wavelet: ", round(mean(res_cs.std[support]), digits = 5))

side_by_side(
    res_us.std .* support, res_cs.std .* support;
    titles = ("sigma - CG-SENSE", "sigma - L1-wavelet"), size = (900, 420)
)

# %% [markdown]
# ## Environment

# %%
print_versions()

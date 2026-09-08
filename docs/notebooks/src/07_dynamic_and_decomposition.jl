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
# # 7 — Dynamic imaging, image decomposition and problem decomposition
#
# Everything in this notebook needs more than one image: a time series, a slice stack, or a model
# that splits one image into additive parts.
#
# Two different things are called "decomposition" in MRT, and they compose:
#
# - **Image decomposition** — the image *is* a sum of components, each with its own regularizer
#   (low-rank + sparse, cartoon + ramp).
# - **Problem decomposition** — one independent reconstruction per batch element (slice,
#   contrast), run in parallel.
#
# **Contents**
# 1. A synthetic dynamic dataset
# 2. Temporal regularizers
# 3. Low-rank regularizers
# 4. Image decomposition (L+S)
# 5. Problem decomposition and threading

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using NamedDims
using LinearAlgebra
using Statistics: mean
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. A synthetic dynamic dataset
#
# 64², 16 frames, 4 coils: a static anatomy (the low-rank background), a bolus washing into one
# region (the sparse foreground) and a small pulsating structure. Phase encodes are undersampled
# 3× with a fully sampled centre — the same pattern for every frame.

# %%
n, nt, nc = 64, 16, 4

anatomy = create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
roi_bolus = falses(n, n); roi_bolus[24:36, 20:28] .= true
roi_beat = falses(n, n); roi_beat[30:38, 36:44] .= true

series = zeros(ComplexF32, n, n, nt)
for t in 1:nt
    frame = copy(anatomy)
    frame[roi_bolus] .+= 0.6f0 * (1 - exp(-3.0f0 * (t - 1) / nt))          # contrast uptake
    frame[roi_beat] .+= 0.3f0 * ComplexF32(sin(2π * (t - 1) / nt))          # pulsation
    series[:, :, t] = frame
end
series = NamedDimsArray{(:x, :y, :time)}(series)

jim(abs.(unname(series))[:, :, 1:5:16]; title = "frames 1, 6, 11, 16", nrow = 1, size = (1000, 280))

# %%
# Temporal profile through the two moving regions.
plot(
    [mean(abs.(unname(series))[roi_bolus, t]) for t in 1:nt];
    label = "bolus ROI", lw = 2, xlabel = "frame", ylabel = "mean |x|", size = (600, 300)
)
plot!([mean(abs.(unname(series))[roi_beat, t]) for t in 1:nt]; label = "pulsating ROI", lw = 2)

# %%
mask_y = falses(n)
mask_y[1:3:n] .= true
mask_y[(n ÷ 2 - 4):(n ÷ 2 + 4)] .= true
println("acceleration: ", round(n / sum(mask_y), digits = 2), "×")

smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(n, n, nc))
acq_dyn = AcquisitionInfo(;
    is3D = false, image_size = (n, n), sensitivity_maps = smaps, subsampling = (:, mask_y)
)
data_dyn = simulate_acquisition(series + 0.01f0 * randn(ComplexF32, n, n, nt), acq_dyn)
println("k-space: ", size(data_dyn.kspace_data), " ", dimnames(data_dyn.kspace_data))

nrmse_dyn(x̂) = norm(abs.(unname(x̂)) - abs.(unname(series))) / norm(abs.(unname(series)))
x_dyn_direct = reconstruct(data_dyn; verbosity = Silent())
println("direct NRMSE: ", round(nrmse_dyn(x_dyn_direct), digits = 4))

# %% [markdown]
# ## 2. Temporal regularizers
#
# ### `L1TemporalFourier`
#
# Sparsity along the temporal frequency axis — the k-t SPARSE transform. Ideal when the dynamics
# are periodic or smooth (cine, cardiac).
#
# ### `TemporalTotalVariation`
#
# Penalizes frame-to-frame differences instead. Right for irregular dynamics: free-breathing,
# real-time, first-pass perfusion. It is the temporal counterpart of spatial TV and, like it,
# falls back to ADMM.

# %%
x_tf = reconstruct(
    data_dyn, IterativeReconstruction(L1TemporalFourier(2.0f-2; time_dim = :time); maxit = 60);
    verbosity = Silent()
)
x_ttv = reconstruct(
    data_dyn, IterativeReconstruction(TemporalTotalVariation(2.0f-2; time_dim = :time); maxit = 60);
    verbosity = Silent()
)
x_spatiotemporal = reconstruct(
    data_dyn,
    IterativeReconstruction(
        TotalVariation2D(1.0f-3), TemporalTotalVariation(2.0f-2; time_dim = :time);
        algorithm = ADMM(), maxit = 60
    );
    verbosity = Silent()
)

for (label, x̂) in (
        ("direct", x_dyn_direct), ("L1TemporalFourier", x_tf),
        ("TemporalTotalVariation", x_ttv), ("spatial TV + temporal TV", x_spatiotemporal),
    )
    println(rpad(label, 26), " NRMSE ", round(nrmse_dyn(x̂), digits = 4))
end

# %%
frame = 8
jim(
    jim(abs.(unname(series))[:, :, frame]; title = "truth"),
    jim(abs.(unname(x_dyn_direct))[:, :, frame]; title = "direct"),
    jim(abs.(unname(x_tf))[:, :, frame]; title = "temporal Fourier"),
    jim(abs.(unname(x_ttv))[:, :, frame]; title = "temporal TV");
    layout = (2, 2), size = (800, 700)
)

# %%
# The temporal profile is the point of these terms — compare the bolus curve.
truth_curve = [mean(abs.(unname(series))[roi_bolus, t]) for t in 1:nt]
plot(truth_curve; label = "truth", lw = 3, xlabel = "frame", ylabel = "mean |x| in bolus ROI", size = (650, 350))
for (label, x̂) in (("direct", x_dyn_direct), ("temporal Fourier", x_tf), ("temporal TV", x_ttv))
    plot!([mean(abs.(unname(x̂))[roi_bolus, t]) for t in 1:nt]; label = label, lw = 2)
end
plot!()

# %% [markdown]
# ## 3. Low-rank regularizers
#
# A dynamic series reshaped as a Casorati matrix (space × time) is nearly low rank whenever the
# frames are correlated.
#
# - `LowRank(λ)` — nuclear norm of the whole matrix.
# - `RankLimit(k)` — a hard rank constraint instead of a penalty. (In this version of MRT the
#   solvers reject the resulting `rank(ℛx) ≤ k` term at parse time, so it is not run below.)
# - `LocallyLowRank(λ; block_size)` — nuclear norm per spatial block, for dynamics that vary
#   across the field of view.
# - `MultiScaleLowRank(λ; block_sizes)` — several block sizes at once, via a proximal average.

# %%
x_lr = reconstruct(
    data_dyn, IterativeReconstruction(LowRank(5.0f-2; time_dim = :time); maxit = 60); verbosity = Silent()
)
x_llr = reconstruct(
    data_dyn,
    IterativeReconstruction(LocallyLowRank(5.0f-2; block_size = 8, time_dim = :time); maxit = 60);
    verbosity = Silent()
)
x_mslr = reconstruct(
    data_dyn,
    IterativeReconstruction(MultiScaleLowRank(5.0f-2; block_sizes = (4, 8, 16), time_dim = :time); maxit = 60);
    verbosity = Silent()
)

for (label, x̂) in (
        ("LowRank", x_lr), ("LocallyLowRank(8)", x_llr), ("MultiScaleLowRank", x_mslr),
    )
    println(rpad(label, 20), " NRMSE ", round(nrmse_dyn(x̂), digits = 4))
end

# %%
# `shift = :random` redraws the block grid before every proximal step, which averages out the
# block boundaries a fixed grid can leave at large λ. It changes the objective from iteration to
# iteration, so it must not be combined with the line-search algorithms.
x_llr_shift = reconstruct(
    data_dyn,
    IterativeReconstruction(
        LocallyLowRank(5.0f-2; block_size = 8, time_dim = :time, shift = :random);
        algorithm = FISTA(), maxit = 60
    );
    verbosity = Silent()
)
println("LocallyLowRank, random grid: NRMSE ", round(nrmse_dyn(x_llr_shift), digits = 4))

jim(
    jim(abs.(unname(x_lr))[:, :, frame]; title = "LowRank"),
    jim(abs.(unname(x_llr))[:, :, frame]; title = "LocallyLowRank"),
    jim(abs.(unname(x_llr_shift))[:, :, frame]; title = "LLR, random grid"),
    jim(abs.(unname(x_mslr))[:, :, frame]; title = "MultiScaleLowRank");
    layout = (2, 2), size = (800, 700)
)

# %%
# How low-rank is the result? Singular values of the Casorati matrix.
casorati(x) = reshape(abs.(unname(x)), n * n, nt)
plot(
    svdvals(casorati(series))[1:12]; label = "truth", lw = 2, marker = :circle,
    yscale = :log10, xlabel = "index", ylabel = "singular value", size = (650, 350)
)
plot!(svdvals(casorati(x_dyn_direct))[1:12]; label = "direct", lw = 2, marker = :circle)
plot!(svdvals(casorati(x_lr))[1:12]; label = "LowRank", lw = 2, marker = :circle)

# %% [markdown]
# ## 4. Image decomposition — low-rank + sparse
#
# `Component(name, regularizers...)` declares one additive part of the image. The data term sees
# the *sum* of the components, so its cost is that of a single-image reconstruction; each part
# gets its own prior. The classic model (Otazo, Candès & Sodickson 2015) is a low-rank background
# plus a temporally sparse foreground.

# %%
img_ls = reconstruct(
    data_dyn,
    IterativeReconstruction(
        Component(:lowrank, LowRank(5.0f-2; time_dim = :time)),
        Component(:sparse, TemporalTotalVariation(2.0f-2; time_dim = :time));
        maxit = 80
    );
    verbosity = Silent()
)

println(typeof(img_ls).name.name)
println("components: ", keys(components(img_ls)))
println("L+S NRMSE ", round(nrmse_dyn(img_ls), digits = 4))

# %%
# The result behaves as an array equal to the sum of its parts …
println("sum of components == total: ", sum(values(components(img_ls))) ≈ total_image(img_ls))

# … and the parts stay accessible.
L = img_ls.components.lowrank
S = img_ls.components.sparse

jim(
    jim(abs.(unname(L))[:, :, frame]; title = "L — background"),
    jim(abs.(unname(S))[:, :, frame]; title = "S — dynamics"),
    jim(abs.(unname(img_ls))[:, :, frame]; title = "L + S");
    layout = (1, 3), size = (1100, 330)
)

# %%
# The separation is temporal: L is nearly constant, S carries the bolus and the pulsation.
plot(
    [mean(abs.(unname(L))[roi_bolus, t]) for t in 1:nt];
    label = "L in bolus ROI", lw = 2, xlabel = "frame", ylabel = "mean |x|", size = (650, 350)
)
plot!([mean(abs.(unname(S))[roi_bolus, t]) for t in 1:nt]; label = "S in bolus ROI", lw = 2)
plot!([mean(abs.(unname(S))[roi_beat, t]) for t in 1:nt]; label = "S in pulsating ROI", lw = 2)

# %%
# A component may carry several regularizers, exactly like the plain API.
img_multi = reconstruct(
    data_dyn,
    IterativeReconstruction(
        Component(:structured, LowRank(5.0f-2; time_dim = :time), TotalVariation2D(5.0f-4)),
        Component(:sparse, L1Image(5.0f-3));
        maxit = 40
    );
    verbosity = Silent()
)
println("two-regularizer component: NRMSE ", round(nrmse_dyn(img_multi), digits = 4))

# %%
# The initial guess can be given per component (a `NamedTuple` keyed by component name). By
# default the first component starts from the direct reconstruction and the rest from zero —
# the usual L+S/RPCA warm start.
img_warm = reconstruct(
    data_dyn,
    IterativeReconstruction(
        Component(:lowrank, LowRank(5.0f-2; time_dim = :time)),
        Component(:sparse, TemporalTotalVariation(2.0f-2; time_dim = :time));
        maxit = 40
    );
    x₀ = (lowrank = x_dyn_direct, sparse = zero(x_dyn_direct)), verbosity = Silent()
)
println("warm-started L+S: NRMSE ", round(nrmse_dyn(img_warm), digits = 4))

# %%
# A single component is rejected — that is just the plain regularization API.
try
    reconstruct(data_dyn, IterativeReconstruction(Component(:only, LowRank(5.0f-2))); verbosity = Silent())
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ### Infimal-convolution TV as a decomposition
#
# Splitting an image into a piecewise-constant "cartoon" and a piecewise-linear "ramp" needs no
# special regularizer: it is an additive decomposition with a first-order TV term on one part and
# a second-order term on the other. Unlike `TotalGeneralizedVariation2D`, this gives you the two
# parts separately — useful when the smooth part *is* the quantity of interest (a bias field, a
# background).

# %%
nimg = 96
x_ic_true = create_shepp_logan_phantom(nimg, nimg, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
ramp = ComplexF32[0.35f0 * (i + j) / (2nimg) for i in 1:nimg, j in 1:nimg]      # smooth shading
acq_ic = AcquisitionInfo(;
    is3D = false, image_size = (nimg, nimg), sensitivity_maps = coil_sensitivities(nimg, nimg, 4),
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.05), (nimg, nimg)
    ),
)
data_ic = simulate_acquisition(x_ic_true .+ ramp, acq_ic)

img_ic = reconstruct(
    data_ic,
    IterativeReconstruction(
        Component(:cartoon, TotalVariation2D(1.0f-3)),
        Component(:ramp, SecondOrderTotalVariation2D(1.0f-3));
        algorithm = ADMM(), maxit = 100
    );
    verbosity = Silent()
)

jim(
    jim(abs.(img_ic.components.cartoon); title = "cartoon part"),
    jim(abs.(img_ic.components.ramp); title = "ramp part"),
    jim(abs.(Array(img_ic)); title = "sum");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# ## 5. Problem decomposition and threading
#
# When the data has batch dimensions that nothing couples — slices, contrasts, or time if there
# is no temporal regularizer — `reconstruct` solves each one independently and in parallel.

# %%
nslices = 8
vol = create_shepp_logan_phantom(64, 64, nslices; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps_ms = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(64, 64, 4))

acq_ms = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky, :coil, :slice)}(zeros(ComplexF32, 64, 64, 4, nslices));
    is3D = false, sensitivity_maps = smaps_ms
)
data_ms = simulate_acquisition(NamedDimsArray{(:x, :y, :slice)}(vol), acq_ms)
println("multi-slice k-space: ", size(data_ms.kspace_data), " ", dimnames(data_ms.kspace_data))

# %%
method_ms = IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 40)

# Warm up first — otherwise the first call pays Julia's compilation and the comparison is
# meaningless. (On a shared or busy machine the numbers still move around by tens of percent.)
reconstruct(data_ms, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 2); verbosity = Silent())
reconstruct(
    data_ms, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 2);
    decomposition_executor = SequentialExecutor(), verbosity = Silent()
)
reconstruct(
    data_ms, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 2);
    disable_problem_decomposition = true, verbosity = Silent()
)

t_par = @elapsed x_par = reconstruct(
    data_ms, method_ms; decomposition_executor = MultiThreadingExecutor(), verbosity = Silent()
)
t_seq = @elapsed x_seq = reconstruct(
    data_ms, method_ms; decomposition_executor = SequentialExecutor(), verbosity = Silent()
)
t_none = @elapsed x_none = reconstruct(
    data_ms, method_ms; disable_problem_decomposition = true, verbosity = Silent()
)

println("threaded over slices : ", round(t_par, digits = 2), " s")
println("sequential slices    : ", round(t_seq, digits = 2), " s")
println("no decomposition     : ", round(t_none, digits = 2), " s   (Julia threads: ", Threads.nthreads(), ")")
println("same answer: ", x_par ≈ x_seq)

# %% [markdown]
# Whether decomposition happens at all depends on which dimensions the regularizers touch:
#
# | Regularizer | Couples | Decomposes over slices/time? |
# |---|---|---|
# | `L1Wavelet2D`, `TotalVariation2D`, `L1Image` | x, y | ✅ |
# | `L1Wavelet3D`, `TotalVariation3D` | x, y, z | ❌ over slices |
# | `L1TemporalFourier`, `TemporalTotalVariation` | time | ❌ over time |
# | `LowRank`, `LocallyLowRank` | space × time | ❌ over time |
#
# `get_affected_dims` is the interface function that answers this, and it is what a custom
# regularizer implements.

# %%
using MriReconstructionToolbox: get_affected_dims

for reg in (L1Wavelet2D(1.0f-3), L1Wavelet3D(1.0f-3), TotalVariation2D(1.0f-3),
        L1TemporalFourier(1.0f-2), LowRank(1.0f-1))
    println(rpad(string(typeof(reg).name.name), 22), get_affected_dims(reg, nothing, (:x, :y, :slice, :time)))
end

# %% [markdown]
# ### Threading notes
#
# MRT parallelizes *across* slices and keeps each slice's work single-threaded, because a 128²
# slice is small enough that splitting it costs more than it saves. The library-level thread
# pools underneath (BLAS, FFTW, NFFT) are managed for you during the solve — do not call
# `BLAS.set_num_threads` yourself.
#
# Two things are yours to set:
#
# - `julia -t N` with `N` = the number of physical cores you actually have (and, on Slurm,
#   `--cpus-per-task` to match, plus an explicit `--mem`).
# - `export KMP_BLOCKTIME=0` **before** starting Julia, if you use MKL. It cannot be set from
#   inside Julia, and without it MKL's spinning worker threads crowd out the reconstruction.

# %%
using LinearAlgebra: BLAS
using FFTW

@show Threads.nthreads()
@show BLAS.get_num_threads()
@show FFTW.get_num_threads()
@show get(ENV, "KMP_BLOCKTIME", "unset")
@show MriReconstructionToolbox.serial_blas_threshold_bytes()

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
# # 7 — Dynamic imaging and image decomposition
#
# Everything in this notebook needs more than one image: a time series, or a model that splits
# one image into additive parts.
#
# **Image decomposition** is the subject here: the image *is* a sum of components, each carrying
# its own regularizer (low-rank + sparse, cartoon + ramp), all solved together in one problem.
# It is unrelated to **task splitting**, which runs one *independent* reconstruction per batch
# element and lives in notebook 06 §8.
#
# **Contents**
# 1. A dynamic torso-phantom dataset
# 2. Temporal regularizers
# 3. Low-rank regularizers
# 4. Image decomposition — low-rank + sparse
# 5. Infimal-convolution TV as a decomposition

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using GeometricMedicalPhantoms: create_torso_phantom, TissueMask,
    generate_respiratory_signal, generate_cardiac_signals
using MIRTjim: jim
using Plots
using NamedDims
using LinearAlgebra
using Statistics: mean
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. A dynamic torso-phantom dataset
#
# The phantom is `create_torso_phantom` from
# [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl), driven
# by that package's physiological signal generators rather than by a hand-rolled bolus curve:
#
# - `generate_cardiac_signals(duration, fs, hr)` returns the four chamber volumes in millilitres
#   (`lv`, `rv`, `la`, `ra`); the phantom rescales its chambers to follow them.
# - `generate_respiratory_signal(duration, fs, rr)` returns lung volume in litres; the phantom
#   moves the diaphragm and the structures above it accordingly.
#
# Sixteen frames spanning one cardiac cycle at 60 bpm, with the respiratory signal sampled over
# the same one-second window, gives a short cine with a strongly moving heart and a slow
# through-plane drift. Slice 10 of 16 is the one that cuts through both ventricles.

# %%
n, nt, nc, nz, zslice = 64, 16, 4, 16, 10

_, cardiac = generate_cardiac_signals(1.0, Float64(nt), 60.0)     # one beat, nt frames
_, respiratory = generate_respiratory_signal(1.0, Float64(nt), 15.0)
println(
    "LV volume over the cine: ", round(minimum(cardiac.lv), digits = 1), " – ",
    round(maximum(cardiac.lv), digits = 1), " mL"
)
println(
    "lung volume:             ", round(minimum(respiratory), digits = 2), " – ",
    round(maximum(respiratory), digits = 2), " L"
)

volume = create_torso_phantom(
    n, n, nz; respiratory_signal = respiratory, cardiac_volumes = cardiac, eltype = ComplexF32
)
series = NamedDimsArray{(:x, :y, :time)}(volume[:, :, zslice, :])
println("series: ", size(series), " ", dimnames(series))

jim(unname(series)[:, :, 1:5:16]; title = "frames 1, 6, 11, 16", nrow = 1, size = (1000, 280))

# %% [markdown]
# ### Tissue masks
#
# Passing `ti = TissueMask(lv_blood = true)` builds the *same* phantom with one tissue set to 1
# and everything else to 0, frame by frame, so the masks move with the anatomy. That gives an
# error metric per tissue instead of a single global number — which matters here, because a
# dynamic reconstruction can be excellent everywhere except in the one structure that moves.

# %%
tissue_mask(mask) = create_torso_phantom(
    n, n, nz; respiratory_signal = respiratory, cardiac_volumes = cardiac, ti = mask
)[:, :, zslice, :]

masks = (
    lv_blood = tissue_mask(TissueMask(lv_blood = true)),
    rv_blood = tissue_mask(TissueMask(rv_blood = true)),
    heart = tissue_mask(TissueMask(heart = true)),
    lung = tissue_mask(TissueMask(lung = true)),
    bones = tissue_mask(TissueMask(bones = true)),
)

for (name, m) in pairs(masks)
    println(
        rpad(string(name), 10), " ", lpad(sum(m), 6), " voxel-frames; per-frame area ",
        extrema(sum(m[:, :, t]) for t in 1:nt)
    )
end

# %% [markdown]
# The left-ventricular blood pool shrinks from 124 pixels to 69 and back over the sixteen
# frames — that contraction is the moving structure every temporal method below is judged on.

# %%
plot(
    [sum(masks.lv_blood[:, :, t]) for t in 1:nt];
    label = "LV blood pool", lw = 2, marker = :circle, xlabel = "frame", ylabel = "area (pixels)",
    size = (650, 300)
)
plot!([sum(masks.rv_blood[:, :, t]) for t in 1:nt]; label = "RV blood pool", lw = 2, marker = :circle)

# %%
jim(
    jim(Float32.(masks.lv_blood[:, :, 1]); title = "LV mask, frame 1 (diastole)"),
    jim(Float32.(masks.lv_blood[:, :, 6]); title = "LV mask, frame 6 (systole)"),
    jim(Float32.(masks.lung[:, :, 1]); title = "lung mask");
    layout = (1, 3), size = (1050, 330)
)

# %% [markdown]
# ### Undersampling and the error metrics
#
# Phase encodes are undersampled with a fully sampled centre, the same pattern for every frame.

# %%
mask_y = falses(n)
mask_y[1:4:n] .= true
mask_y[(n ÷ 2 - 4):(n ÷ 2 + 4)] .= true
println("acceleration: ", round(n / sum(mask_y), digits = 2), "×")

smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(n, n, nc))
acq_dyn = AcquisitionInfo(;
    is3D = false, image_size = (n, n), sensitivity_maps = smaps, subsampling = (:, mask_y)
)
data_dyn = simulate_acquisition(series + 0.01f0 * randn(ComplexF32, n, n, nt), acq_dyn)
println("k-space: ", size(data_dyn.kspace_data), " ", dimnames(data_dyn.kspace_data))

# %%
# Global NRMSE comes from NotebookUtils; the per-tissue one restricts both arrays to a mask.
nrmse_dyn(x̂) = nrmse(unname(x̂), unname(series))
function tissue_nrmse(x̂, mask)
    a, b = abs.(unname(x̂)), abs.(unname(series))
    return norm(a[mask] - b[mask]) / norm(b[mask])
end

function report(label, x̂)
    print(rpad(label, 26), " global ", lpad(round(nrmse_dyn(x̂), digits = 4), 7))
    for (name, m) in pairs(masks)
        print("  ", name, " ", lpad(round(tissue_nrmse(x̂, m), digits = 4), 7))
    end
    return println()
end

x_dyn_direct = reconstruct(data_dyn; verbosity = Silent())
report("direct", x_dyn_direct)

# %% [markdown]
# ## 2. Temporal regularizers
#
# ### `L1TemporalFourier`
#
# Sparsity along the temporal frequency axis — the k-t SPARSE transform. Ideal when the dynamics
# are periodic or smooth (cine, cardiac), which is exactly what a one-beat cine is.
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

report("direct", x_dyn_direct)
report("L1TemporalFourier", x_tf)
report("TemporalTotalVariation", x_ttv)
report("spatial TV + temporal TV", x_spatiotemporal)

# %% [markdown]
# Read the columns, not just the first one. Every method improves the global number, but the
# ranking inside the *lung* — low signal, and moving with the diaphragm — is different from the
# ranking inside the blood pools, and the lung is where each method's error is largest.

# %%
frame = 8
side_by_side(
    unname(series)[:, :, frame], unname(x_dyn_direct)[:, :, frame],
    unname(x_tf)[:, :, frame], unname(x_ttv)[:, :, frame];
    titles = ("truth", "direct", "temporal Fourier", "temporal TV"),
    size = (1250, 330)
)

# %% [markdown]
# ### The temporal profile through the moving structure
#
# The point of a temporal term is the time course, not the single frame. Averaging the magnitude
# inside the (moving) LV blood-pool mask traces the ventricle filling and emptying; the direct
# reconstruction blurs that curve towards its mean, and the temporal terms restore it.

# %%
lv_curve(x) = [mean(abs.(unname(x))[:, :, t][masks.lv_blood[:, :, t]]) for t in 1:nt]

plot(
    lv_curve(series); label = "truth", lw = 3, color = :black,
    xlabel = "frame", ylabel = "mean |x| in the LV blood pool", size = (700, 360)
)
for (label, x̂) in (
        ("direct", x_dyn_direct), ("temporal Fourier", x_tf),
        ("temporal TV", x_ttv), ("spatial + temporal TV", x_spatiotemporal),
    )
    plot!(lv_curve(x̂); label = label, lw = 2)
end
plot!()

# %% [markdown]
# A y–t profile through a fixed column across the ventricle shows the same thing without any
# mask: the horizontal axis is time, so a moving wall is a slanted edge, and blurring it is
# immediately visible.

# %%
column = 34
profile(x) = abs.(unname(x))[:, column, :]
side_by_side(
    profile(series), profile(x_dyn_direct), profile(x_tf), profile(x_ttv);
    titles = ("truth", "direct", "temporal Fourier", "temporal TV"),
    size = (1250, 300)
)

# %% [markdown]
# ## 3. Low-rank regularizers
#
# A dynamic series reshaped as a Casorati matrix (space × time) is nearly low rank whenever the
# frames are correlated — and a cine is, since most of the field of view does not move at all.
#
# - `LowRank(λ)` — nuclear norm of the whole matrix.
# - `RankLimit(k)` — a hard rank constraint instead of a penalty.
# - `LocallyLowRank(λ; block_size)` — nuclear norm per spatial block, for dynamics that vary
#   across the field of view. A torso cine is exactly that case: the heart moves, the chest wall
#   barely does.
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

report("LowRank", x_lr)
report("LocallyLowRank(8)", x_llr)
report("MultiScaleLowRank", x_mslr)

# %%
# `shift = :random` redraws the block grid before every proximal step, which averages out the
# block boundaries a fixed grid can leave at large λ. It changes the objective from iteration to
# iteration, so it must not be combined with a line-search algorithm — hence the explicit
# `FISTA()`, whose step size is fixed by `Lf` (notebook 06 §7).
x_llr_shift = reconstruct(
    data_dyn,
    IterativeReconstruction(
        LocallyLowRank(5.0f-2; block_size = 8, time_dim = :time, shift = :random);
        algorithm = FISTA(), maxit = 60
    );
    verbosity = Silent()
)
report("LocallyLowRank, random", x_llr_shift)

side_by_side(
    unname(x_lr)[:, :, frame], unname(x_llr)[:, :, frame],
    unname(x_llr_shift)[:, :, frame], unname(x_mslr)[:, :, frame];
    titles = ("LowRank", "LocallyLowRank", "LLR, random grid", "MultiScaleLowRank"),
    size = (1250, 330)
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
# plus a temporally sparse foreground, and a cardiac cine is what it was designed for: the static
# chest is the background, the beating heart is the foreground.

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
report("L+S", img_ls)

# %%
# The result behaves as an array equal to the sum of its parts …
println("sum of components == total: ", sum(values(components(img_ls))) ≈ total_image(img_ls))

# … and the parts are reachable directly by name on the image itself.
L = img_ls.lowrank
S = img_ls.sparse

side_by_side(
    unname(L)[:, :, frame], unname(S)[:, :, frame], unname(img_ls)[:, :, frame];
    titles = ("L — background", "S — dynamics", "L + S")
)

# %% [markdown]
# The separation is temporal, and the tissue masks make that concrete: inside the LV blood pool
# `L` is nearly flat while `S` carries the whole cardiac cycle, and inside the (nearly static)
# bones both are flat.

# %%
plot(
    lv_curve(L); label = "L in the LV blood pool", lw = 2,
    xlabel = "frame", ylabel = "mean |x|", size = (700, 360)
)
plot!(lv_curve(S); label = "S in the LV blood pool", lw = 2)
plot!(
    [mean(abs.(unname(S))[:, :, t][masks.bones[:, :, t]]) for t in 1:nt];
    label = "S in bone", lw = 2, ls = :dash
)

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
report("two-reg component", img_multi)

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
report("warm-started L+S", img_warm)

# %%
# A single component is rejected — that is just the plain regularization API.
try
    reconstruct(data_dyn, IterativeReconstruction(Component(:only, LowRank(5.0f-2))); verbosity = Silent())
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ## 5. Infimal-convolution TV as a decomposition
#
# Splitting an image into a piecewise-constant "cartoon" and a piecewise-linear "ramp" needs no
# special regularizer: it is an additive decomposition with a first-order TV term on one part and
# a second-order term on the other. Unlike `TotalGeneralizedVariation2D`, this gives you the two
# parts separately — useful when the smooth part *is* the quantity of interest (a bias field, a
# background, a shading correction).
#
# The failure it fixes only shows up on data that *has* smooth gradients. Plain TV is minimized
# by piecewise-constant images, so it renders a linear ramp as a flight of steps — **staircasing**
# — and that is invisible on a phantom made of flat ellipses. The test image below is therefore
# built deliberately: two piecewise-constant shapes on top of a genuine linear ramp running
# across the whole field of view.

# %%
m_ic = 96
blocks = zeros(Float32, m_ic, m_ic)
for i in 1:m_ic, j in 1:m_ic
    hypot(i - 40, j - 40) < 22 && (blocks[i, j] = 0.55f0)
    (60 <= i <= 84 && 20 <= j <= 76) && (blocks[i, j] = 0.85f0)
end
ramp_true = Float32[0.15f0 + 0.6f0 * (j - 1) / (m_ic - 1) for i in 1:m_ic, j in 1:m_ic]
x_ic_true = ComplexF32.(blocks .+ ramp_true)

# Rows 1–15 sit above every shape, so they are pure ramp: the region where staircasing lives.
ramp_region = falses(m_ic, m_ic)
ramp_region[1:15, :] .= true

side_by_side(blocks, ramp_true, x_ic_true; titles = ("cartoon part", "ramp part", "test image"))

# %%
acq_ic = AcquisitionInfo(;
    is3D = false, image_size = (m_ic, m_ic), sensitivity_maps = coil_sensitivities(m_ic, m_ic, 4),
    subsampling = create_sampling_pattern(
        VariableDensitySampling(PolynomialDistribution(3), 5.0, 0.06), (m_ic, m_ic)
    ),
)
data_ic = simulate_acquisition(x_ic_true + 0.02f0 * randn(ComplexF32, m_ic, m_ic), acq_ic)

λ_ic = 1.0f-2
x_ic_tv = reconstruct(
    data_ic, IterativeReconstruction(TotalVariation2D(λ_ic); algorithm = ADMM(), maxit = 400);
    verbosity = Silent()
)
img_ic = reconstruct(
    data_ic,
    IterativeReconstruction(
        Component(:cartoon, TotalVariation2D(λ_ic)),
        Component(:ramp, SecondOrderTotalVariation2D(λ_ic));
        algorithm = ADMM(), maxit = 400
    );
    verbosity = Silent()
)

# The staircase metric: the total second difference along a profile through the pure-ramp band.
# A straight ramp has second differences of zero; every step contributes twice.
staircase(x) = sum(abs, diff(diff(abs.(unname(x))[8, :])))
region_nrmse(x) = nrmse(unname(x)[ramp_region], x_ic_true[ramp_region])

for (label, x̂) in (("plain TV", x_ic_tv), ("infimal-convolution TV", Array(img_ic)))
    println(
        rpad(label, 24),
        " global NRMSE ", lpad(round(nrmse(x̂, x_ic_true), digits = 4), 7),
        "   ramp-region NRMSE ", lpad(round(region_nrmse(x̂), digits = 4), 7),
        "   staircase ", lpad(round(staircase(x̂), digits = 3), 6)
    )
end

# %% [markdown]
# The global NRMSE barely separates the two — staircasing is a *structured* error confined to
# the smooth regions, and a whole-image average dilutes it against the edges, which plain TV
# renders slightly more crisply. Restricted to the pure-ramp band the gap is large, and the
# staircase metric is halved. The profile below is what those two numbers are measuring.

# %%
row = 8
plot(
    abs.(x_ic_true[row, :]); label = "truth", lw = 3, color = :black,
    xlabel = "column", ylabel = "|x|", title = "profile through the pure-ramp band", size = (750, 380)
)
plot!(abs.(unname(x_ic_tv))[row, :]; label = "plain TV — staircased", lw = 2)
plot!(abs.(unname(Array(img_ic)))[row, :]; label = "infimal-convolution TV", lw = 2)

# %%
side_by_side(
    x_ic_true, x_ic_tv, Array(img_ic);
    titles = ("truth", "plain TV", "infimal-convolution TV")
)

# %% [markdown]
# And the two components, which plain TV cannot give you at all: `img_ic.cartoon` holds the
# shapes and `img_ic.ramp` the smooth shading, reachable by name straight off the image.

# %%
side_by_side(
    abs.(img_ic.cartoon), abs.(img_ic.ramp), abs.(Array(img_ic));
    titles = ("cartoon component", "ramp component", "sum")
)

# %% [markdown]
# ---
#
# **Task splitting** — running one independent reconstruction per slice, contrast or frame, and
# the threading settings that control it — is a *run* setting rather than a model, and is covered
# in notebook 06 §8, together with the `get_affected_dims` table that says which of the terms
# above leave which dimensions splittable.

# %% [markdown]
# ## Environment

# %%
print_versions()

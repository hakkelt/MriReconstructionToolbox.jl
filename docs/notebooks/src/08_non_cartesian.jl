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
# # 8 — Non-Cartesian acquisitions
#
# Radial and spiral trajectories do not land on a grid, so the Fourier operator becomes an NFFT
# instead of an FFT. This notebook covers the trajectory generators, the NFFT encoding operator
# and its density-compensation (DCF) options, gradient-delay correction, and the NFFT
# accuracy/speed knobs.
#
# **Contents**
# 1. Trajectory families
# 2. Simulating a non-Cartesian acquisition
# 3. The NFFT encoding operator — density compensation is opt-in
# 4. Density compensation — Pipe–Menon and Voronoi
# 5. Iterative and regularized reconstruction
# 6. Gradient-delay correction
# 7. Accuracy vs. speed of the gridding

# %%
include("NotebookUtils.jl")
using .NotebookUtils

using MriReconstructionToolbox
using MriReconstructionToolbox: get_encoding_operator
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using MIRTjim: jim
using Plots
using NamedDims
using LinearAlgebra
using Random

Random.seed!(0)

# %% [markdown]
# ## 1. Trajectory families
#
# `MriReconstructionToolbox` ships generators for the common non-Cartesian sampling patterns.
# All of them return a `NamedDimsArray` with the coordinate axis first (`:coord`, one of the two
# names `AcquisitionInfo` accepts for non-Cartesian data), normalized to `[-0.5, 0.5)` (the NFFT.jl
# convention):
#
# - `radial_trajectory(nsamples, nspokes; ordering)` — 2D radial spokes through the k-space
#   center. `ordering` is `:linear` (uniform angle step), `:golden_angle` (successive spokes
#   rotated by ≈111.25°, so any prefix of the sequence covers k-space near-uniformly), or
#   `:tiny_golden_angle` (a smaller member of the golden-angle family, useful for view sharing).
#   A full golden-angle step swings the readout gradients through a large angle between any two
#   consecutive spokes, and each swing drives its own eddy currents in the gradient coils; tiny
#   golden angles keep successive spokes close together in angle, so the gradient waveform
#   changes little from one spoke to the next and eddy currents stay suppressed, at the cost of
#   less-uniform coverage for any short prefix of the sequence.
# - `stack_of_stars_trajectory(nsamples, nspokes, npartitions)` — the 2D radial pattern repeated
#   at Cartesian partition-encoding (`kz`) positions.
# - `kooshball_trajectory(nsamples, nspokes)` — full 3D radial, spoke directions distributed
#   quasi-uniformly over the sphere.
# - `spiral_trajectory(nsamples, ninterleaves; variant)` — rotated copies of one spiral arm;
#   `:archimedean` (uniform radial density) or `:variable_density` (denser at the center).

# %%
traj_linear = radial_trajectory(96, 13; ordering = :linear)
traj_golden = radial_trajectory(96, 13; ordering = :golden_angle)
traj_tiny = radial_trajectory(96, 13; ordering = :tiny_golden_angle, tiny_index = 3)
traj_sos = stack_of_stars_trajectory(96, 13, 6; ordering = :golden_angle)
traj_koosh = kooshball_trajectory(64, 89)
# Fewer arms than a real acquisition would use, so the density difference between the two
# variants is visible arm by arm rather than smeared into a filled disc.
traj_spiral_a = spiral_trajectory(512, 2; variant = :archimedean)
traj_spiral_vd = spiral_trajectory(512, 2; variant = :variable_density, density_exponent = 2.0)

function traj_scatter(traj; title = "", kwargs...)
    t = unname(traj)
    return scatter(
        vec(t[1, :, :]), vec(t[2, :, :]);
        markersize = 1.2, markerstrokewidth = 0, legend = false, aspect_ratio = 1,
        xlabel = "kx", ylabel = "ky", xlim = (-0.55, 0.55), ylim = (-0.55, 0.55),
        title, kwargs...
    )
end

function traj_scatter3d(traj; title = "", kwargs...)
    t = reshape(unname(traj), size(traj, 1), :)   # flatten sample/spoke/partition axes
    return scatter(
        t[1, :], t[2, :], t[3, :];
        markersize = 1.0, markerstrokewidth = 0, legend = false,
        xlabel = "kx", ylabel = "ky", zlabel = "kz", title, kwargs...
    )
end

plot(
    traj_scatter(traj_linear; title = "radial, linear"),
    traj_scatter(traj_golden; title = "radial, golden angle"),
    traj_scatter(traj_tiny; title = "radial, tiny golden angle"),
    traj_scatter(traj_spiral_a; title = "spiral, archimedean"),
    traj_scatter(traj_spiral_vd; title = "spiral, variable density");
    layout = (1, 5), size = (1650, 350)
)

# %%
# The two fully 3D families, shown in 3D rather than projected onto kx-ky: stack-of-stars is
# radial in-plane and Cartesian through-plane (the discrete kz "shells"), while kooshball spokes
# point quasi-uniformly over the whole sphere.
plot(
    traj_scatter3d(traj_sos; title = "stack of stars (6 partitions)"),
    traj_scatter3d(traj_koosh; title = "kooshball");
    layout = (1, 2), size = (1000, 480)
)

# %% [markdown]
# ## 2. Simulating a non-Cartesian acquisition
#
# `AcquisitionInfo(; trajectory, image_size, sensitivity_maps)` is the advertised constructor —
# no k-space placeholder is needed. `simulate_acquisition` handles non-Cartesian trajectories
# exactly like Cartesian ones: it builds the encoding operator, applies it to the image, and
# returns a new acquisition object with the k-space data filled in.

# %%
nx, ny = 96, 96
nsamp, nspokes = 128, 96

traj = radial_trajectory(nsamp, nspokes; ordering = :golden_angle)
x_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps = coil_sensitivities(nx, ny, 4)

acq_radial = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
data_radial = simulate_acquisition(x_true, acq_radial)

println("radial k-space: ", size(data_radial.kspace_data))
# Radial Nyquist needs about (π/2)·N spokes; fewer than that is undersampling.
println("spokes: ", nspokes, " of the ", ceil(Int, π / 2 * nx), " a fully sampled radial scan would need")
jim(log.(abs.(data_radial.kspace_data[:, :, 1]) .+ 1.0f-6); title = "log |k-space|, coil 1 (samples × spokes)", size = (450, 350))

# %% [markdown]
# ## 3. The NFFT encoding operator — density compensation is opt-in
#
# `get_encoding_operator` builds an NFFT-based operator (from `NFFTOperators.jl`) whenever the
# acquisition carries a trajectory; everything downstream — `reconstruct`, regularizers — works
# exactly as in the Cartesian case.
#
# `NFFTOp`'s `dcf` keyword (forwarded from `acq.dcf`) now defaults to `nothing`: **no** density
# compensation is applied, so the adjoint `𝒜'` is the *true* mathematical adjoint of `𝒜`, not an
# approximate inverse. `density_compensation(acq; method)` (below) computes weights and stores
# them on `acq.dcf`; only then does `𝒜'` become the weighted, gridded-adjoint approximation to
# the inverse. Passing `dcf = :auto` to `get_encoding_operator`/`get_fourier_operator` runs the
# same Pipe–Menon estimator inline instead.

# %%
𝒜_nodcf = get_encoding_operator(data_radial)
x_adjoint_nodcf = 𝒜_nodcf' * data_radial.kspace_data
println("no DCF: adjoint magnitude range = ", extrema(abs.(x_adjoint_nodcf)))
jim(abs.(x_adjoint_nodcf[:, :, 1]); title = "plain adjoint, no DCF (coil 1)", size = (380, 350))

# %% [markdown]
# The plain adjoint is dominated by the heavily oversampled k-space center — the image below is
# badly blurred. This is expected: without density compensation the adjoint is *not* an estimate
# of the inverse, it is the exact adjoint of an operator that oversamples low frequencies.

# %% [markdown]
# ## 4. Density compensation — Pipe–Menon and Voronoi
#
# - `PipeMenonDCF()` — iterative, works for any trajectory in 2D or 3D (the default method for
#   `density_compensation`).
# - `VoronoiDCF()` — geometric, exact areas of the Voronoi cells, 2D only. Cells at the edge of
#   the sampled disc are unbounded in an ordinary Voronoi diagram; `VoronoiDCF` clips every cell
#   against a bounding box (`bounds`, default `(-0.5, 0.5, -0.5, 0.5)`, the same domain the
#   trajectory itself is normalized to) before computing its area, so the outermost samples get a
#   finite, meaningful weight instead of an unbounded one.
#
# The weights are stored on the acquisition (`acq.dcf`) and forwarded to the Fourier operator, so
# `reconstruct` picks them up automatically.

# %%
acq_pipe = density_compensation(data_radial; method = PipeMenonDCF(maxit = 20))
acq_voronoi = density_compensation(data_radial; method = VoronoiDCF())

println("Pipe–Menon DCF: ", size(acq_pipe.dcf), " ", eltype(acq_pipe.dcf))
println("Voronoi DCF all finite: ", all(isfinite, acq_voronoi.dcf), ", range = ", extrema(acq_voronoi.dcf))

plot(
    acq_pipe.dcf[:, 1]; label = "Pipe–Menon", lw = 2, xlabel = "readout sample", ylabel = "weight",
    title = "density compensation along one spoke", size = (600, 320)
)
plot!(acq_voronoi.dcf[:, 1]; label = "Voronoi")

# %%
x_nodcf = reconstruct(data_radial; verbosity = Silent())
x_pipe = reconstruct(acq_pipe; verbosity = Silent())
x_voronoi = reconstruct(acq_voronoi; verbosity = Silent())

function aligned_scale(x̂)
    a = abs.(unname(x̂))
    return sum(a .* abs.(x_true)) / sum(abs2, a)
end
function aligned_nrmse(x̂)
    a = abs.(unname(x̂))
    return norm(aligned_scale(x̂) .* a - abs.(x_true)) / norm(abs.(x_true))
end

println("adjoint, no DCF   ", round(aligned_nrmse(x_nodcf), digits = 4))
println("adjoint, Pipe     ", round(aligned_nrmse(x_pipe), digits = 4))
println("adjoint, Voronoi  ", round(aligned_nrmse(x_voronoi), digits = 4))

# The no-DCF adjoint is ~10^5x the scale of the DCF-corrected ones (§3): a shared color scale
# across all three would render the corrected panels solid black. Bring each panel to the
# truth's own scale with the same least-squares factor aligned_nrmse uses, so the comparison
# is about structure, not units.
side_by_side(
    aligned_scale(x_nodcf) .* unname(x_nodcf), aligned_scale(x_pipe) .* unname(x_pipe),
    aligned_scale(x_voronoi) .* unname(x_voronoi);
    titles = ("no DCF", "Pipe-Menon", "Voronoi (clipped)"), size = (1050, 350)
)

# %% [markdown]
# The plain (no-DCF) adjoint is badly blurred by the oversampled k-space center, exactly as in
# §3. Both DCF methods correct for this. With the bounding-box clipping, `VoronoiDCF` gives a
# result close to `PipeMenonDCF` on this golden-angle trajectory rather than the "edge samples
# blow up" failure an unclipped Voronoi diagram would show.

# %% [markdown]
# ## 5. Iterative and regularized reconstruction
#
# An iterative solve does not need a DCF at all — it inverts the operator instead of
# approximating the inverse with a weighted adjoint — but a good DCF still makes a useful
# starting point, and the regularizers behave exactly as in the Cartesian case.

# %%
x_cg = reconstruct(
    data_radial, IterativeReconstruction(L2Image(1.0f-4); algorithm = CGNR(), maxit = 20);
    verbosity = Silent()
)
x_tv = reconstruct(
    data_radial, IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 40); verbosity = Silent()
)

println("CG-SENSE          ", round(aligned_nrmse(x_cg), digits = 4))
println("TV-regularized    ", round(aligned_nrmse(x_tv), digits = 4))

side_by_side(x_pipe, x_cg, x_tv; titles = ("gridded adjoint", "CG-SENSE", "TV compressed sensing"), size = (1050, 350))

# %%
# Fewer spokes: the regime where the regularizer earns its keep.
nspokes_us = 32
traj_us = unname(traj)[:, :, 1:nspokes_us]
acq_us = AcquisitionInfo(; trajectory = traj_us, image_size = (nx, ny), sensitivity_maps = smaps)
data_us = simulate_acquisition(x_true, acq_us)
println("spokes: ", nspokes_us, "  (acceleration ≈ ",
    round(ceil(π / 2 * nx) / nspokes_us, digits = 1), "× relative to radial Nyquist)")

x_us_adj = reconstruct(density_compensation(data_us); verbosity = Silent())
x_us_tv = reconstruct(data_us, IterativeReconstruction(TotalVariation2D(2.0f-3); maxit = 60); verbosity = Silent())

println("adjoint + DCF ", round(aligned_nrmse(x_us_adj), digits = 4))
println("TV            ", round(aligned_nrmse(x_us_tv), digits = 4))

side_by_side(
    x_us_adj, x_us_tv;
    titles = ("$(nspokes_us) spokes, gridded", "$(nspokes_us) spokes, TV"), size = (700, 350)
)

# %% [markdown]
# ## 6. Gradient-delay correction
#
# Gradient hardware delays and eddy currents shift the actual sampled k-space location along each
# spoke relative to its nominal position, which blurs the image and produces streak/ring
# artifacts. Two estimators:
#
# - `OpposingSpokes()` — cross-correlates spoke pairs, fitting an isotropic `(dx, dy)` shift from
#   the peak position of the readout signal.
# - `RING()` — fits the full anisotropic delay tensor `(dx, dy, dxy)` from the same peak-fitting
#   approach.
#
# Both estimators work by locating the peak of `|k-space|` along each spoke relative to the
# nominal sample grid — the classical self-navigator approach (Peters et al. 2003; Rosenzweig et
# al. 2019), calibrated here with an idealized point-source readout (a narrow Gaussian peak),
# exactly like the package's own tests. A real object's k-space is not this well-behaved, so in
# practice the estimate comes from a short dedicated calibration acquisition rather than from the
# imaging data itself — which is the workflow this section reproduces: a calibration trajectory
# estimates the delay, and the corrected trajectory is then applied to the imaging acquisition.

# %%
Nsamples, Nspokes = 64, 48
angles = range(0, 2π; length = Nspokes + 1)[1:Nspokes]
r = Float32.(range(-0.45, 0.45; length = Nsamples))

nxg, nyg = 64, 64
x_gd_true = create_shepp_logan_phantom(nxg, nyg, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

traj_gd_true = zeros(Float32, 2, Nsamples, Nspokes)     # matches the physical sample locations
for s in 1:Nspokes
    traj_gd_true[1, :, s] = r .* cos(angles[s])
    traj_gd_true[2, :, s] = r .* sin(angles[s])
end

delay_true = (0.02f0, -0.015f0)                          # ground truth, injected below
traj_gd_wrong = copy(traj_gd_true)                        # the naive/nominal trajectory, biased by the delay
for s in 1:Nspokes
    traj_gd_wrong[1, :, s] .+= delay_true[1] * cos(angles[s])
    traj_gd_wrong[2, :, s] .+= delay_true[2] * sin(angles[s])
end

plot(
    traj_scatter(NamedDimsArray{(:coord, :sample, :spoke)}(traj_gd_true[:, :, 1:6]); title = "true trajectory (first 6 spokes)"),
    traj_scatter(NamedDimsArray{(:coord, :sample, :spoke)}(traj_gd_wrong[:, :, 1:6]); title = "delay-biased trajectory (first 6 spokes)");
    layout = (1, 2), size = (800, 380)
)

# %% [markdown]
# Samples are truly acquired at `traj_gd_true` (the object does not know about the delay); a
# reconstruction that (wrongly) assumes the nominal, delay-biased trajectory shows the artefact.

# %%
acq_gd_true = AcquisitionInfo(;
    trajectory = NamedDimsArray{(:coord, :kx, :ky)}(traj_gd_true), image_size = (nxg, nyg)
)
data_gd = simulate_acquisition(x_gd_true, acq_gd_true)

acq_gd_naive = AcquisitionInfo(
    data_gd.kspace_data;
    trajectory = NamedDimsArray{(:coord, :kx, :ky)}(traj_gd_wrong), image_size = (nxg, nyg)
)
x_gd_naive = reconstruct(density_compensation(acq_gd_naive; method = PipeMenonDCF(maxit = 15)); verbosity = Silent())

# %% [markdown]
# Estimate the delay from a calibration acquisition (idealized point-source signal along the
# same spoke angles) and correct the nominal trajectory with it.

# %%
ksp_calib = zeros(ComplexF32, Nsamples, Nspokes)
for s in 1:Nspokes
    shift_s = delay_true[1] * cos(angles[s]) + delay_true[2] * sin(angles[s])
    ksp_calib[:, s] = exp.(-50.0f0 .* (r .- shift_s) .^ 2)
end
calib_acq = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky)}(ksp_calib);
    trajectory = NamedDimsArray{(:coord, :kx, :ky)}(traj_gd_wrong), image_size = (nxg, nyg)
)

est_opposing = estimate_gradient_delays(calib_acq; method = OpposingSpokes())
println("true delay          ", delay_true)
println("OpposingSpokes est.  ", (round(est_opposing[1], digits = 4), round(est_opposing[2], digits = 4)))

calib_corrected = correct_gradient_delays(calib_acq; method = OpposingSpokes())
corrected_traj = calib_corrected.trajectory
println("residual trajectory error vs. true: ", round(norm(unname(corrected_traj) - traj_gd_true), digits = 4))

acq_gd_fixed = AcquisitionInfo(data_gd.kspace_data; trajectory = corrected_traj, image_size = (nxg, nyg))
x_gd_fixed = reconstruct(density_compensation(acq_gd_fixed; method = PipeMenonDCF(maxit = 15)); verbosity = Silent())

function aligned_nrmse_gd(x̂)
    a = abs.(unname(x̂))
    α = sum(a .* abs.(x_gd_true)) / sum(abs2, a)
    return norm(α .* a - abs.(x_gd_true)) / norm(abs.(x_gd_true))
end
println("naive (uncorrected) nrmse ", round(aligned_nrmse_gd(x_gd_naive), digits = 4))
println("corrected nrmse           ", round(aligned_nrmse_gd(x_gd_fixed), digits = 4))

side_by_side(x_gd_naive, x_gd_fixed; titles = ("uncorrected (delay artefact)", "gradient-delay corrected"), size = (700, 350))

# %% [markdown]
# `RING()` recovers the full anisotropic tensor `(dx, dy, dxy)` the same way, including a
# nonzero cross-term.

# %%
Sxx, Syy, Sxy = 0.02, -0.015, 0.005
traj_ring_wrong = copy(traj_gd_true)
ksp_ring = zeros(ComplexF32, Nsamples, Nspokes)
for s in 1:Nspokes
    θ = angles[s]
    shift = Sxx * cos(θ)^2 + Syy * sin(θ)^2 + 2 * Sxy * cos(θ) * sin(θ)
    traj_ring_wrong[1, :, s] .+= shift * cos(θ)
    traj_ring_wrong[2, :, s] .+= shift * sin(θ)
    ksp_ring[:, s] = exp.(-50.0f0 .* (r .- shift) .^ 2)
end
acq_ring = AcquisitionInfo(
    NamedDimsArray{(:kx, :ky)}(ksp_ring);
    trajectory = NamedDimsArray{(:coord, :kx, :ky)}(traj_ring_wrong), image_size = (nxg, nyg)
)
delays_ring = estimate_gradient_delays(acq_ring; method = RING())
println("true (Sxx, Syy, Sxy) = ", (Sxx, Syy, Sxy))
println("RING estimate        = ", map(v -> round(v, digits = 4), delays_ring))

bar(
    ["Sxx", "Syy", "Sxy"], [Sxx, Syy, Sxy];
    label = "true", alpha = 0.6, xlabel = "delay component", ylabel = "value",
    title = "RING: true vs. estimated", size = (450, 320)
)
bar!(["Sxx", "Syy", "Sxy"], [delays_ring.dx, delays_ring.dy, delays_ring.dxy]; label = "RING estimate", alpha = 0.6, bar_width = 0.4)

# %%
# Correction is refused on Cartesian data — there is no trajectory to correct.
try
    correct_gradient_delays(AcquisitionInfo(zeros(ComplexF32, 16, 16); is3D = false))
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ## 7. Accuracy vs. speed of the gridding
#
# `get_encoding_operator` (and `get_fourier_operator`) forward `m`, `sigma` and `precompute`
# straight to NFFT.jl. Left at `nothing`, MRT uses its own default operating point
# (`DEFAULT_NFFT_M = 4`, `DEFAULT_NFFT_SIGMA = 1.5`, `DEFAULT_NFFT_PRECOMPUTE = NFFT.POLYNOMIAL`),
# a lower-accuracy, faster point than NFFT.jl's own default (`m = 5`, `σ = 2.0`), chosen because
# the accuracy loss is negligible for iterative reconstruction while the speed gain compounds
# over many forward/adjoint applications per solve.
#
# This HPC login node shows ±30-60% timing swings between runs (shared, contended cores), so the
# table below uses the minimum of several repeats ("best-of-N") rather than a single measurement,
# and the *ratios* are more meaningful than the absolute milliseconds.

# %%
using NFFT

configs = (
    ("NFFT.jl default (m=5, σ=2.0, polynomial)", (m = 5, sigma = 2.0, precompute = NFFT.POLYNOMIAL)),
    ("MRT default (m=4, σ=1.5, polynomial)", (m = nothing, sigma = nothing, precompute = nothing)),
    ("fast (m=3, σ=1.25, tensor)", (m = 3, sigma = 1.25, precompute = NFFT.TENSOR)),
)

𝒜_reference = get_encoding_operator(data_radial; m = 5, sigma = 2.0, precompute = NFFT.POLYNOMIAL)
y_reference = 𝒜_reference * x_true

results = map(configs) do (name, c)
    𝒜 = get_encoding_operator(data_radial; m = c.m, sigma = c.sigma, precompute = c.precompute)
    y = 𝒜 * x_true
    best_ms = 1000 * minimum(@elapsed(𝒜 * x_true) for _ in 1:10)
    rel_err = norm(y - y_reference) / norm(y_reference)
    (name = name, ms = round(best_ms, digits = 2), rel_err = round(rel_err, digits = 6))
end

println(rpad("configuration", 42), rpad("best-of-10 (ms)", 18), "rel. error vs. NFFT.jl default")
for r in results
    println(rpad(r.name, 42), rpad(r.ms, 18), r.rel_err)
end

# %% [markdown]
# The `rel_err` column is the forward-operator difference against NFFT.jl's own (most accurate)
# default, not against the true continuous Fourier transform — both MRT's default and the fast
# operating point stay within a fraction of a percent of it, at a fraction of the cost. Given the
# timing noise on this node, treat the exact millisecond values as illustrative; the ordering
# (accurate ≥ MRT default ≥ fast) is the reproducible part.

# %% [markdown]
# ## Environment

# %%
print_versions()

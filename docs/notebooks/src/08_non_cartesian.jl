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
# and the adjoint needs density compensation. This notebook covers trajectories, DCF,
# reconstruction, gradient-delay correction and the NFFT accuracy/speed knobs.
#
# **Contents**
# 1. A radial trajectory
# 2. The NFFT encoding operator
# 3. Density compensation — Pipe–Menon and Voronoi
# 4. Iterative and regularized reconstruction
# 5. Gradient-delay correction
# 6. Accuracy vs. speed of the gridding
# 7. A spiral trajectory

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
# ## 1. A radial trajectory
#
# `NonCartesianAcquisitionInfo` stores the trajectory with the coordinate axes in the **first**
# dimension: `(2, nsamples, nspokes)` for 2D, `(3, …)` for 3D. Coordinates are in normalized
# units, i.e. in `[-0.5, 0.5)`. The remaining dimensions have to match the k-space sample layout.

# %%
nx, ny = 96, 96
nsamp, nspokes = 128, 96

# Golden-angle radial: successive spokes rotated by 111.246°.
golden = Float32(π * (3 - sqrt(5)) / 2 * 2)     # ≈ 111.25° in radians
r = Float32.(range(-0.5, 0.5; length = nsamp) .* 0.99)

traj = zeros(Float32, 2, nsamp, nspokes)
for s in 1:nspokes
    θ = Float32((s - 1) * golden)
    traj[1, :, s] = r .* cos(θ)
    traj[2, :, s] = r .* sin(θ)
end

scatter(
    vec(traj[1, :, 1:16]), vec(traj[2, :, 1:16]);
    markersize = 1.5, legend = false, aspect_ratio = 1, title = "first 16 spokes",
    xlabel = "kx", ylabel = "ky", size = (420, 400)
)

# %% [markdown]
# ## 2. The NFFT encoding operator
#
# With a trajectory in the configuration, `get_encoding_operator` builds an NFFT-based operator
# (from `NFFTOperators.jl`); everything downstream — `simulate_acquisition`, `reconstruct`,
# regularizers — works exactly as in the Cartesian case.

# %%
x_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps = coil_sensitivities(nx, ny, 4)

# `simulate_acquisition` is Cartesian-only in this version, so the non-Cartesian forward model is
# applied explicitly: build the operator from a configuration with a correctly-shaped k-space
# placeholder, apply it, and put the result back into a new configuration.
function simulate_noncartesian(image, trajectory, image_size; sensitivity_maps = nothing, noise = 0)
    nsamples = size(trajectory)[2:end]
    ncoil = isnothing(sensitivity_maps) ? () : (size(sensitivity_maps)[end],)
    placeholder = zeros(Complex{eltype(trajectory)}, nsamples..., ncoil...)
    acq = NonCartesianAcquisitionInfo(
        placeholder; trajectory, image_size, sensitivity_maps
    )
    y = get_encoding_operator(acq) * image
    noise > 0 && (y .+= noise .* randn(eltype(y), size(y)))
    return NonCartesianAcquisitionInfo(acq; kspace_data = y)
end

acq_radial = NonCartesianAcquisitionInfo(;
    trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps
)
data_radial = simulate_noncartesian(x_true, traj, (nx, ny); sensitivity_maps = smaps)

println("radial k-space: ", size(data_radial.kspace_data))
# Radial Nyquist needs about (π/2)·N spokes; fewer than that is undersampling.
println("spokes: ", nspokes, " of the ", ceil(Int, π / 2 * nx), " a fully sampled radial scan would need")
jim(log.(abs.(data_radial.kspace_data[:, :, 1]) .+ 1.0f-6); title = "log |k-space|, coil 1 (samples × spokes)", size = (450, 350))

# %% [markdown]
# ## 3. Density compensation
#
# Radial sampling oversamples the centre of k-space by a large factor, so the plain adjoint
# $\mathcal{A}^H y$ is dominated by low frequencies and looks badly blurred. Density compensation
# weights each sample by the inverse of the local sampling density.
#
# - `PipeMenonDCF()` — iterative, works for any trajectory in 2D or 3D (the default).
# - `VoronoiDCF()` — geometric, exact areas of the Voronoi cells, 2D only.
#
# The weights are stored on the acquisition (`acq.dcf`) and folded into the Fourier operator, so
# `reconstruct` picks them up automatically. On this golden-angle trajectory the gridded adjoint
# built without an explicit DCF is numerically indistinguishable from the Pipe–Menon one, while
# the Voronoi weights are visibly worse — Voronoi cells at the edge of the sampled disc are
# unbounded, so the outermost samples are mis-weighted.

# %%
acq_pipe = density_compensation(data_radial; method = PipeMenonDCF(maxit = 20))
acq_voronoi = density_compensation(data_radial; method = VoronoiDCF())

println("Pipe–Menon DCF: ", size(acq_pipe.dcf), " ", eltype(acq_pipe.dcf))
plot(
    acq_pipe.dcf[:, 1]; label = "Pipe–Menon", lw = 2, xlabel = "readout sample", ylabel = "weight",
    title = "density compensation along one spoke", size = (600, 320)
)
plot!(acq_voronoi.dcf[:, 1]; label = "Voronoi", lw = 2)

# %%
x_nodcf = reconstruct(data_radial; verbosity = Silent())
x_pipe = reconstruct(acq_pipe; verbosity = Silent())
x_voronoi = reconstruct(acq_voronoi; verbosity = Silent())

function aligned_nrmse(x̂)
    a = abs.(unname(x̂))
    α = sum(a .* abs.(x_true)) / sum(abs2, a)
    return norm(α .* a - abs.(x_true)) / norm(abs.(x_true))
end

println("adjoint, no DCF   ", round(aligned_nrmse(x_nodcf), digits = 4))
println("adjoint, Pipe     ", round(aligned_nrmse(x_pipe), digits = 4))
println("adjoint, Voronoi  ", round(aligned_nrmse(x_voronoi), digits = 4))
println("difference between the no-DCF and Pipe–Menon images: ",
    round(norm(x_nodcf - x_pipe) / norm(x_nodcf), digits = 6))

jim(
    jim(abs.(x_nodcf); title = "no DCF"),
    jim(abs.(x_pipe); title = "Pipe–Menon"),
    jim(abs.(x_voronoi); title = "Voronoi");
    layout = (1, 3), size = (1100, 330)
)

# %% [markdown]
# ## 4. Iterative and regularized reconstruction
#
# An iterative solve does not need the DCF at all — it inverts the operator instead of
# approximating the inverse with a weighted adjoint — but a good DCF still makes a useful
# preconditioner-like starting point, and the regularizers behave exactly as in the Cartesian
# case.

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

jim(
    jim(abs.(x_pipe); title = "gridded adjoint"),
    jim(abs.(x_cg); title = "CG-SENSE"),
    jim(abs.(x_tv); title = "TV compressed sensing");
    layout = (1, 3), size = (1100, 330)
)

# %%
# Fewer spokes: the regime where the regularizer earns its keep.
nspokes_us = 32
traj_us = traj[:, :, 1:nspokes_us]
data_us = simulate_noncartesian(x_true, traj_us, (nx, ny); sensitivity_maps = smaps)
println("spokes: ", nspokes_us, "  (acceleration ≈ ",
    round(ceil(π / 2 * nx) / nspokes_us, digits = 1), "× relative to radial Nyquist)")

x_us_adj = reconstruct(density_compensation(data_us); verbosity = Silent())
x_us_tv = reconstruct(data_us, IterativeReconstruction(TotalVariation2D(2.0f-3); maxit = 60); verbosity = Silent())

println("adjoint + DCF ", round(aligned_nrmse(x_us_adj), digits = 4))
println("TV            ", round(aligned_nrmse(x_us_tv), digits = 4))

jim(
    jim(abs.(x_us_adj); title = "$(nspokes_us) spokes, gridded"),
    jim(abs.(x_us_tv); title = "$(nspokes_us) spokes, TV");
    layout = (1, 2), size = (800, 350)
)

# %% [markdown]
# ## 5. Gradient-delay correction
#
# Gradient hardware delays and eddy currents shift the actual sample positions along each spoke,
# which blurs the image and produces ring artifacts. Two estimators:
#
# - `OpposingSpokes()` — cross-correlates spoke pairs 180° apart; isotropic `(dx, dy)`.
# - `RING()` — fits the full anisotropic delay tensor `(dx, dy, dxy)` from spoke intersections.

# %%
nsd, nsp = 64, 32
angles = range(0, 2π; length = nsp + 1)[1:nsp]
rr = Float32.(range(-0.5, 0.5; length = nsd))

traj_true = zeros(Float32, 2, nsd, nsp)
for s in 1:nsp
    traj_true[1, :, s] = rr .* cos(angles[s])
    traj_true[2, :, s] = rr .* sin(angles[s])
end

# A known delay: each spoke's samples slide along its own direction.
delay = (0.02, -0.015)
traj_delayed = copy(traj_true)
ksp_delayed = zeros(ComplexF32, nsd, nsp)
for s in 1:nsp
    shift = delay[1] * cos(angles[s]) + delay[2] * sin(angles[s])
    traj_delayed[1, :, s] .+= delay[1] * cos(angles[s])
    traj_delayed[2, :, s] .+= delay[2] * sin(angles[s])
    ksp_delayed[:, s] = exp.(-50.0f0 .* (rr .- Float32(shift)) .^ 2)     # centred readout peak
end

acq_delayed = NonCartesianAcquisitionInfo(
    NamedDimsArray{(:kx, :ky)}(ksp_delayed);
    trajectory = NamedDimsArray{(:dim, :kx, :ky)}(traj_delayed), image_size = (32, 32)
)

est = estimate_gradient_delays(acq_delayed; method = OpposingSpokes())
println("true delay      ", delay)
println("estimated       ", (round(est[1], digits = 4), round(est[2], digits = 4)))

acq_corrected = correct_gradient_delays(acq_delayed; method = OpposingSpokes())
println("residual trajectory error: ",
    round(norm(unname(acq_corrected.trajectory) - traj_true), digits = 5))

# %%
# `RING` estimates the anisotropic tensor, including the cross term.
Sxx, Syy, Sxy = 0.02, -0.015, 0.005
traj_ring = copy(traj_true)
ksp_ring = zeros(ComplexF32, nsd, nsp)
for s in 1:nsp
    θ = angles[s]
    shift = Sxx * cos(θ)^2 + Syy * sin(θ)^2 + 2 * Sxy * cos(θ) * sin(θ)
    traj_ring[1, :, s] .+= shift * cos(θ)
    traj_ring[2, :, s] .+= shift * sin(θ)
    ksp_ring[:, s] = exp.(-50.0f0 .* (rr .- Float32(shift)) .^ 2)
end

acq_ring = NonCartesianAcquisitionInfo(
    NamedDimsArray{(:kx, :ky)}(ksp_ring);
    trajectory = NamedDimsArray{(:dim, :kx, :ky)}(traj_ring), image_size = (32, 32)
)
delays_ring = estimate_gradient_delays(acq_ring; method = RING())
println("true (Sxx, Syy, Sxy) = ", (Sxx, Syy, Sxy))
println("RING estimate        = ", map(v -> round(v, digits = 4), delays_ring))

# %%
# Correction is refused on Cartesian data — there is no trajectory to correct.
try
    correct_gradient_delays(AcquisitionInfo(zeros(ComplexF32, 16, 16); is3D = false))
catch e
    println(sprint(showerror, e))
end

# %% [markdown]
# ## 6. Accuracy vs. speed of the gridding
#
# `get_encoding_operator` (and `get_fourier_operator`) forward `m`, `sigma` and `precompute`
# straight to NFFT.jl. Left at `nothing`, MRT uses NFFT.jl's own defaults (`m = 5`, `σ = 2.0`,
# polynomial precomputation) — considerably more accurate than the operating point other
# toolboxes default to, and correspondingly slower.

# %%
using NFFT

𝒜_default = get_encoding_operator(data_radial)
𝒜_fast = get_encoding_operator(data_radial; m = 3, sigma = 1.25, precompute = NFFT.TENSOR)

y_default = 𝒜_default * x_true
y_fast = 𝒜_fast * x_true
println("relative difference in the forward model: ", round(norm(y_fast - y_default) / norm(y_default), digits = 6))

t_default = @elapsed for _ in 1:5; 𝒜_default * x_true; end
t_fast = @elapsed for _ in 1:5; 𝒜_fast * x_true; end
println("forward apply: default ", round(1000t_default / 5, digits = 2), " ms, fast ", round(1000t_fast / 5, digits = 2), " ms")

# %% [markdown]
# ## 7. A spiral trajectory
#
# Nothing about the pipeline is radial-specific: any set of sample coordinates works.

# %%
nturns, npoints, ninterleaves = 12, 1024, 8
traj_spiral = zeros(Float32, 2, npoints, ninterleaves)
for i in 1:ninterleaves
    ϕ0 = Float32(2π * (i - 1) / ninterleaves)
    for k in 1:npoints
        t = Float32((k - 1) / (npoints - 1))
        ρ = 0.49f0 * t
        ϕ = ϕ0 + Float32(2π * nturns) * t
        traj_spiral[1, k, i] = ρ * cos(ϕ)
        traj_spiral[2, k, i] = ρ * sin(ϕ)
    end
end

plot(
    traj_spiral[1, :, 1], traj_spiral[2, :, 1];
    legend = false, aspect_ratio = 1, xlabel = "kx", ylabel = "ky",
    title = "spiral interleaves", size = (420, 400)
)
for i in 2:ninterleaves
    plot!(traj_spiral[1, :, i], traj_spiral[2, :, i])
end
plot!()

# %%
data_spiral = simulate_noncartesian(x_true, traj_spiral, (nx, ny); sensitivity_maps = smaps)

x_spiral_adj = reconstruct(density_compensation(data_spiral; method = PipeMenonDCF(maxit = 20)); verbosity = Silent())
x_spiral_tv = reconstruct(data_spiral, IterativeReconstruction(TotalVariation2D(1.0f-3); maxit = 40); verbosity = Silent())

println("spiral, gridded adjoint ", round(aligned_nrmse(x_spiral_adj), digits = 4))
println("spiral, TV              ", round(aligned_nrmse(x_spiral_tv), digits = 4))

jim(
    jim(abs.(x_spiral_adj); title = "spiral, gridded"),
    jim(abs.(x_spiral_tv); title = "spiral, TV");
    layout = (1, 2), size = (800, 350)
)

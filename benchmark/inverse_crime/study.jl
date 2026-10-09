# How fine a phantom must be for simulated data to be free of the inverse crime.
#
#   julia --project=benchmark --threads=8 benchmark/inverse_crime/study.jl
#
# The modified Shepp–Logan phantom (Toft) is a sum of ellipses, so its continuous Fourier
# transform is known in closed form. Data simulated from that transform are the reference: no
# grid is involved. The study compares them with data simulated from rasterized phantoms of
# `s` times the reconstruction size (128 × 128), point-sampled (each voxel takes the value at its
# centre) and area-sampled (the mean of 4 × 4 sub-voxel samples), in two ways:
#
# 1. k-space error: the relative L2 distance of the simulated samples from the analytic ones,
#    on a Cartesian grid and on a radial trajectory;
# 2. reconstruction bias: the signal-to-error ratio (SER) of a total-variation reconstruction
#    from 3-fold undersampled Cartesian data, simulated from the rasterized phantom, minus that
#    of the same reconstruction from analytic data. A positive bias is an optimistic result.
#
# Every rasterized phantom and the ground truth use the geometry `simulate_acquisition` assumes:
# the phantom and the reconstruction grid cover the same field of view, [-1, 1]², voxel edge to
# voxel edge.

using Ristretto
using SpecialFunctions: besselj1
using LinearAlgebra, Printf, Random

# intensity, semi-axes a and b, centre (x0, y0), rotation in degrees
const SHEPP_LOGAN = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0), (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0), (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0.0), (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0.0), (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0.0), (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
]

function phantom_value(x, y)
    v = 0.0
    for (ρ, a, b, x0, y0, θ) in SHEPP_LOGAN
        c, s = cosd(θ), sind(θ)
        xr = (x - x0) * c + (y - y0) * s
        yr = -(x - x0) * s + (y - y0) * c
        (xr / a)^2 + (yr / b)^2 <= 1 && (v += ρ)
    end
    return v
end

# The phantom on an `n × n` grid over [-1, 1]²: the value at each voxel centre (`sub = 1`) or the
# mean over `sub × sub` points spread evenly over the voxel.
function rasterize(n::Int; sub::Int = 1)
    Δ = 2 / n
    img = zeros(ComplexF64, n, n)
    Threads.@threads for i in 1:n
        for j in 1:n, p in 1:sub, q in 1:sub
            x = -1 + (i - 1 + (p - 0.5) / sub) * Δ
            y = -1 + (j - 1 + (q - 0.5) / sub) * Δ
            img[i, j] += phantom_value(x, y) / sub^2
        end
    end
    return img
end

# The continuous Fourier transform `∫ f(r) exp(-2πi u·r) dr` at spatial frequency `u`.
function phantom_spectrum(u, v)
    S = zero(ComplexF64)
    for (ρ, a, b, x0, y0, θ) in SHEPP_LOGAN
        c, s = cosd(θ), sind(θ)
        up, vp = u * c + v * s, -u * s + v * c
        q = sqrt((a * up)^2 + (b * vp)^2)
        F = q < 1.0e-12 ? π * a * b : a * b * besselj1(2π * q) / q
        S += ρ * F * cispi(-2 * (u * x0 + v * y0))
    end
    return S
end

# Analytic data in the convention of a reconstruction grid of `n` voxels per axis whose Fourier
# operator has its phase origin at voxel `o`: `Σ x[i] exp(-2πi k (i - o) / n)` becomes
# `F(k / 2) exp(2πi k p_o / 2) / Δ²`, with `p_o` the centre of voxel `o` and frequencies `k` in
# cycles per field of view (2 units).
analytic_sample(kx, ky, n, o) = phantom_spectrum(kx / 2, ky / 2) * cispi((kx + ky) * (-1 + (o - 0.5) * 2 / n)) / (2 / n)^2

const N = 128
const RATIOS = (1.0, 1.3, 1.5, 202 / 128, 1.7, 2.0, 3.0)
fine_size(s) = round(Int, s * N)
relerr(a, b) = norm(a - b) / norm(b)
ser(x, ref) = 20 * log10(norm(ref) / norm(x - ref))

# ─── 1. k-space error ────────────────────────────────────────────────────────────────────────

# Cartesian, no shifts: centred k-space (zero frequency at `n ÷ 2 + 1`) and the image's phase
# origin at voxel 1.
freqs(n) = [p - (n ÷ 2 + 1) for p in 1:n]
cartesian_reference = [analytic_sample(kx, ky, N, 1) for kx in freqs(N), ky in freqs(N)]
cartesian_acq = AcquisitionInfo(is3D = false, image_size = (N, N))

traj = Float64.(radial_trajectory(2N, 64))
radial_acq = AcquisitionInfo(; trajectory = traj, image_size = (N, N))
# The non-Cartesian operator places the phase origin at voxel `n ÷ 2 + 1`.
radial_reference = [analytic_sample(traj[1, I] * N, traj[2, I] * N, N, N ÷ 2 + 1) for I in CartesianIndices(size(traj)[2:end])]

println("1. k-space error of the simulated data against the analytic transform")
@printf("%8s %8s %14s %14s %14s %14s\n", "ratio", "phantom", "Cart. point", "Cart. area", "radial point", "radial area")
for s in RATIOS
    n = fine_size(s)
    row = Float64[]
    for (acq, ref) in ((cartesian_acq, cartesian_reference), (radial_acq, radial_reference)), sub in (1, 4)
        data = simulate_acquisition(rasterize(n; sub), acq; inverse_crime_check = false).kspace_data
        push!(row, relerr(data, ref))
    end
    @printf("%8.3f %8d %13.1f%% %13.1f%% %13.1f%% %13.1f%%\n", s, n, (100 .* row[[1, 2, 3, 4]])...)
end

# ─── 2. Reconstruction bias ──────────────────────────────────────────────────────────────────

Random.seed!(0)
pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.08), (N, N))
undersampled = AcquisitionInfo(is3D = false, image_size = (N, N), subsampling = pattern)
method = IterativeReconstruction(TotalVariation2D(2.0e-3); maxit = 200)
truth = rasterize(N; sub = 8)
# The analytic samples in the layout of the undersampled acquisition.
template = simulate_acquisition(truth, undersampled; inverse_crime_check = false)
ℳ = Ristretto.get_subsampling_operator(template; threaded = false)
analytic_data = AcquisitionInfo(template; kspace_data = ℳ * cartesian_reference)
ser_analytic = ser(reconstruct(analytic_data, method; verbosity = Silent()), truth)

println()
println("2. TV reconstruction from 3-fold undersampled data: SER against the area-sampled truth")
@printf("analytic data: SER %.2f dB\n", ser_analytic)
@printf("%8s %8s %16s %16s\n", "ratio", "phantom", "bias, point [dB]", "bias, area [dB]")
for s in RATIOS
    n = fine_size(s)
    bias = map((1, 4)) do sub
        data = simulate_acquisition(rasterize(n; sub), undersampled; inverse_crime_check = false)
        ser(reconstruct(data, method; verbosity = Silent()), truth) - ser_analytic
    end
    @printf("%8.3f %8d %16.2f %16.2f\n", s, n, bias...)
end

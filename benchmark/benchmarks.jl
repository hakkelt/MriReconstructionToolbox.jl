using BenchmarkTools
using MriReconstructionToolbox
using GeometricMedicalPhantoms
using Random

const SUITE = BenchmarkGroup()

# -----------------------------------------------------------------------------
# Setup data for benchmarks
# -----------------------------------------------------------------------------
Random.seed!(42)
nx, ny, nc = 64, 64, 8
phantom_2d = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps_2d = coil_sensitivities(nx, ny, nc)

# Sampling pattern (3x acceleration)
subsampling_mask = rand(Float32, nx, ny) .> 0.66
subsampling_mask[28:36, 28:36] .= true

acq_2d_base = CartesianAcquisitionInfo(;
    is3D = false,
    image_size = (nx, ny),
    sensitivity_maps = smaps_2d,
    subsampling = subsampling_mask,
)
acq_2d_ksp = simulate_acquisition(phantom_2d, acq_2d_base)

# Multi-slice data (4 slices)
num_slices = 4
smaps_ms = repeat(reshape(smaps_2d, nx, ny, nc, 1), 1, 1, 1, num_slices)
acq_ms_base = CartesianAcquisitionInfo(;
    is3D = false,
    image_size = (nx, ny),
    sensitivity_maps = smaps_ms,
    subsampling = subsampling_mask,
)
phantom_ms = repeat(reshape(phantom_2d, nx, ny, 1), 1, 1, num_slices)
acq_ms_ksp = simulate_acquisition(phantom_ms, acq_ms_base)

# -----------------------------------------------------------------------------
# Group 1: operator (𝒜*x, 𝒜'*y, ‖𝒜‖)
# -----------------------------------------------------------------------------
SUITE["operator"] = BenchmarkGroup()
op_enc = get_encoding_operator(acq_2d_ksp)
x_in = copy(phantom_2d)
y_in = op_enc * x_in

SUITE["operator"]["forward"] = @benchmarkable $op_enc * $x_in
SUITE["operator"]["adjoint"] = @benchmarkable $(op_enc') * $y_in
# The step-size estimate every proximal solve pays once, up front. `𝒜` itself is no
# longer rescaled by it -- see "Operator norm, step size and λ" in the docs.
SUITE["operator"]["estimate_opnorm"] =
    @benchmarkable MriReconstructionToolbox.AbstractOperators.estimate_opnorm($op_enc)

# -----------------------------------------------------------------------------
# Group 2: reconstruct
# -----------------------------------------------------------------------------
SUITE["reconstruct"] = BenchmarkGroup()

# FISTA maxit=20 with L1Wavelet2D
reg_wavelet = L1Wavelet2D(0.01)
fista_alg = FISTA(maxit = 20)
method_fista = IterativeReconstruction(reg_wavelet; algorithm = fista_alg)

SUITE["reconstruct"]["2D_CS_FISTA"] = @benchmarkable reconstruct(
    $acq_2d_ksp,
    $method_fista;
    verbose = false,
)

# Multi-slice 2D CS
SUITE["reconstruct"]["multi_slice_FISTA"] = @benchmarkable reconstruct(
    $acq_ms_ksp,
    $method_fista;
    verbose = false,
)

# -----------------------------------------------------------------------------
# Group 3: prox (allocs / memory for main regularizers)
# -----------------------------------------------------------------------------
SUITE["prox"] = BenchmarkGroup()

reg_tv = TotalVariation2D(0.01)
reg_lr = LowRank(0.01; time_dim = 3)
reg_tf = TemporalFourier(0.01; time_dim = 3)

# Test arrays
x_2d = copy(phantom_2d)
x_3d = randn(ComplexF32, nx, ny, 8)

op_tv = MriReconstructionToolbox.get_operator(reg_tv, x_2d)
op_wavelet = MriReconstructionToolbox.get_operator(reg_wavelet, x_2d)

SUITE["prox"]["TotalVariation_op"] = @benchmarkable $op_tv * $x_2d
SUITE["prox"]["Wavelet_op"] = @benchmarkable $op_wavelet * $x_2d
SUITE["prox"]["LowRank_op"] = @benchmarkable MriReconstructionToolbox.get_operator($reg_lr, $x_3d) * $x_3d
SUITE["prox"]["TemporalFourier_op"] = @benchmarkable MriReconstructionToolbox.get_operator($reg_tf, $x_3d) * $x_3d

# -----------------------------------------------------------------------------
# Group 4: real scanner data (opt-in — ENV["MRT_BENCH_REAL_DATA"] = "1")
# -----------------------------------------------------------------------------
# Real Cartesian k-space via MRITestData.jl (RealData.jl is shared with benchmarking/). The
# dataset is downloaded and cached on first use; a failure is logged and the group skipped.
if get(ENV, "MRT_BENCH_REAL_DATA", "0") == "1"
    include(joinpath(@__DIR__, "..", "benchmarking", "src", "RealData.jl"))
    using .RealData: load_real_case
    try
        real_case = load_real_case()
        acq_real = CartesianAcquisitionInfo(
            real_case.kspace; is3D = false, sensitivity_maps = real_case.smaps, shifted_image_dims = (:x, :y),
        )
        method_cg = IterativeReconstruction(
            regularization = (), algorithm = MriReconstructionToolbox.CGNR(maxit = 10, tol = 1.0e-14),
        )
        SUITE["real_data"] = BenchmarkGroup()
        SUITE["real_data"]["CG_SENSE"] = @benchmarkable reconstruct(
            $acq_real, $method_cg; maxit = 10, tol = 1.0e-14, verbose = false,
        )
        SUITE["real_data"]["FISTA_wavelet"] = @benchmarkable reconstruct(
            $acq_real, $method_fista; verbose = false,
        )
    catch e
        @warn "MRT_BENCH_REAL_DATA set but real-data benchmark setup failed" exception = (e, catch_backtrace())
    end
end

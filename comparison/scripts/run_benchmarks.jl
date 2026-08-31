# comparison/scripts/run_benchmarks.jl
#
# Comprehensive performance & accuracy benchmark suite comparing:
# - MriReconstructionToolbox (OpenBLAS / MKL)
# - BART (OpenBLAS / MKL with startup-time subtracted)
# - SigPy (Python)
# - MRIReco (Julia)
#
# Evaluated at various thread counts (e.g. 1, 8, 16) with strict hardware thread pinning.

using Printf
using JSON

# 1. Parse CLI options
use_mkl = "--use-mkl" in ARGS
num_threads_arg = findfirst(a -> startswith(a, "--threads="), ARGS)
num_threads = if isnothing(num_threads_arg)
    Threads.nthreads()
else
    parse(Int, split(ARGS[num_threads_arg], "=")[2])
end

if use_mkl
    @info "Enabling Intel MKL backend via MKL.jl"
    using MKL
    bart_binary = "/project/c_mrrecon/bart_mkl"
else
    bart_binary = "/project/c_mrrecon/bart_openblas"
end

using ThreadPinning
# Query SLURM / OS affinity mask and pin Julia threads to physical cores
mask = getaffinity()
allowed_cpus = findall(==(1), mask) .- 1
if isempty(allowed_cpus)
    allowed_cpus = collect(0:(Threads.nthreads() - 1))
end
pinned_cpus = allowed_cpus[1:min(length(allowed_cpus), Threads.nthreads())]
pinthreads(pinned_cpus)
cpu_str = join(pinned_cpus, ",")
@info "Julia threads pinned to CPUs: $cpu_str"

# Configure subprocess pinning and threading for BART, Python, and C extensions
ENV["TOOLBOX_PATH"] = bart_binary
ENV["BART_USE_FFTW_WISDOM"] = "1"
ENV["OMP_NUM_THREADS"] = string(num_threads)
ENV["OPENBLAS_NUM_THREADS"] = string(num_threads)
ENV["MKL_NUM_THREADS"] = string(num_threads)
ENV["GOMP_CPU_AFFINITY"] = cpu_str
ENV["KMP_AFFINITY"] = "granularity=fine,proclist=[$cpu_str],explicit"
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "{$cpu_str}"

using MriReconstructionToolbox
using GeometricMedicalPhantoms
using BenchmarkTools
using LinearAlgebra
using Statistics
using Random
using FFTW
using BartIO
using PyCall
using MRIReco

include("../src/ComparisonHarness.jl")
using .ComparisonHarness: check_nrmse, nrmse, run_bart, generate_multicoil_brain, generate_dynamic_multicoil_brain

sigpy = pyimport("sigpy")
sp_mri = pyimport("sigpy.mri")

println("=================================================================")
println("    MRI RECONSTRUCTION PERFORMANCE & ACCURACY BENCHMARK SUITE    ")
println("=================================================================")
println(" Hostname:         ", gethostname())
println(" Julia Version:    ", VERSION)
println(" Julia Threads:    ", Threads.nthreads())
println(" BLAS Vendor:      ", BLAS.get_config().loaded_libs[1].libname)
println(" MKL Enabled:      ", use_mkl)
println(" BART Binary:      ", bart_binary)
println(" Pinned CPU IDs:   ", cpu_str)
println(" OpenMP Threads:   ", ENV["OMP_NUM_THREADS"])
println("=================================================================\n")

# Measure BART startup overhead (e.g. `bart version`) to subtract from BART timings
function measure_bart_startup(; num_runs=10)
    times = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        read(pipeline(ignorestatus(`$bart_binary version`)), String)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    return minimum(times)
end

bart_startup_time = measure_bart_startup()
println(@sprintf("Measured BART startup overhead: %.2f ms (will be subtracted from BART raw timings)\n", bart_startup_time * 1000))

# Utility to time functions after a warmup pass
function time_reconstruction(f; num_runs=3, is_bart=false)
    # Warmup
    res = f()
    times = Float64[]
    for _ in 1:num_runs
        t0 = time_ns()
        res = f()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    t_min = minimum(times)
    t_med = median(times)
    if is_bart
        t_min = max(1e-5, t_min - bart_startup_time)
        t_med = max(1e-5, t_med - bart_startup_time)
    end
    return t_min, t_med, res
end

# Benchmark Results Table
struct BenchResult
    category::String
    method::String
    framework::String
    threads::Int
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_mrt::Float64
end

results = BenchResult[]

# -------------------------------------------------------------
# 1. Base Setup: 2D Multi-Coil & 1-Coil Brain Phantom
# -------------------------------------------------------------
N = 128
Nc = 8
img_mc, kspace_mc, cmap = generate_multicoil_brain(N=N, num_coils=Nc)

# 1-Coil
img_1c = img_mc
kspace_1c = ifftshift(fft(fftshift(img_1c)))
kdata_1c = NamedDimsArray(kspace_1c, (:kx, :ky))
acq_1c = CartesianAcquisitionInfo(kdata_1c; is3D=false, shifted_image_dims=(:x, :y))

# --- 1-Coil Adjoint ---
println("--> Benchmarking Cartesian 1-Coil Adjoint...")
E_1c = MriReconstructionToolbox.get_encoding_operator(acq_1c)
t_min_mrt_1c, _, mrt_1c_adj_raw = time_reconstruction(() -> E_1c' * kdata_1c)
mrt_1c_adj = mrt_1c_adj_raw .* (norm(abs.(img_1c)) / norm(abs.(mrt_1c_adj_raw)))
e_gt_mrt_1c = nrmse(mrt_1c_adj, img_1c)
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_1c * 1000, e_gt_mrt_1c, 0.0))

# SigPy 1-Coil Adjoint
kdata_sp_1c = parent(permutedims(kspace_1c, (2, 1)))
F_sp_1c = sp_mri.linop.Sense(ones(ComplexF64, 1, N, N), ishape=(N, N))
t_min_sp_1c, _, sp_1c_raw = time_reconstruction(() -> F_sp_1c.H(reshape(kdata_sp_1c, 1, N, N)))
sp_1c_adj = permutedims(sp_1c_raw, (2, 1))
sp_1c_adj = sp_1c_adj .* (norm(abs.(img_1c)) / norm(abs.(sp_1c_adj)))
e_gt_sp_1c = nrmse(sp_1c_adj, img_1c)
e_mrt_sp_1c = nrmse(mrt_1c_adj, sp_1c_adj)
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", "SigPy", num_threads, t_min_sp_1c * 1000, e_gt_sp_1c, e_mrt_sp_1c))

# BART 1-Coil Adjoint
kdata_bart_1c = reshape(kspace_1c, N, N, 1, 1)
t_min_bart_1c, _, bart_1c_raw = time_reconstruction(() -> run_bart(1, "fft -i 3", ComplexF32.(kdata_bart_1c)), is_bart=true)
bart_1c_adj = bart_1c_raw[:, :, 1, 1]
bart_1c_adj = bart_1c_adj .* (norm(abs.(img_1c)) / norm(abs.(bart_1c_adj)))
e_gt_bart_1c = nrmse(bart_1c_adj, img_1c)
e_mrt_bart_1c = nrmse(mrt_1c_adj, bart_1c_adj)
push!(results, BenchResult("Base 1C", "1-Coil Adjoint", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_1c * 1000, e_gt_bart_1c, e_mrt_bart_1c))

# --- Multi-Coil Adjoint ---
println("--> Benchmarking Cartesian Multi-Coil Adjoint...")
kdata_mc = NamedDimsArray(kspace_mc, (:kx, :ky, :coil))
smaps_mc = NamedDimsArray(cmap, (:x, :y, :coil))
acq_mc = CartesianAcquisitionInfo(kdata_mc; is3D=false, sensitivity_maps=smaps_mc, shifted_image_dims=(:x, :y))

E_mc = MriReconstructionToolbox.get_encoding_operator(acq_mc)
t_min_mrt_adj, _, mrt_adj = time_reconstruction(() -> E_mc' * kdata_mc)
recon_mrt_adj = mrt_adj ./ sum(abs2.(smaps_mc), dims=3)[:, :, 1]
e_gt_mrt_adj = nrmse(recon_mrt_adj .* (norm(abs.(img_mc)) / norm(abs.(recon_mrt_adj))), img_mc)
push!(results, BenchResult("Base MC", "Cartesian Adjoint", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_adj * 1000, e_gt_mrt_adj, 0.0))

# SigPy MC Adjoint
kdata_sp_mc = parent(permutedims(kspace_mc, (3, 2, 1)))
smaps_sp_mc = parent(permutedims(cmap, (3, 2, 1)))
S_sp_mc = sp_mri.linop.Sense(smaps_sp_mc, ishape=(N, N))
t_min_sp_adj, _, sp_adj_raw = time_reconstruction(() -> S_sp_mc.H(kdata_sp_mc))
sp_adj = permutedims(sp_adj_raw, (2, 1))
recon_sp_adj = sp_adj ./ sum(abs2.(cmap), dims=3)[:, :, 1]
e_gt_sp_adj = nrmse(recon_sp_adj .* (norm(abs.(img_mc)) / norm(abs.(recon_sp_adj))), img_mc)
e_mrt_sp_adj = nrmse(mrt_adj .* (norm(abs.(sp_adj)) / norm(abs.(mrt_adj))), sp_adj)
push!(results, BenchResult("Base MC", "Cartesian Adjoint", "SigPy", num_threads, t_min_sp_adj * 1000, e_gt_sp_adj, e_mrt_sp_adj))

# BART MC Adjoint
kdata_bart_cart = reshape(kspace_mc, N, N, 1, Nc)
t_min_bart_adj, _, bart_ifft = time_reconstruction(() -> run_bart(1, "fft -i 3", ComplexF32.(kdata_bart_cart)), is_bart=true)
bart_adj = sum(bart_ifft[:, :, 1, :] .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims=3)[:, :, 1]
recon_bart_adj = bart_adj ./ sum(abs2.(ComplexF32.(cmap)), dims=3)[:, :, 1]
e_gt_bart_adj = nrmse(recon_bart_adj .* (norm(abs.(img_mc)) / norm(abs.(recon_bart_adj))), img_mc)
e_mrt_bart_adj = nrmse(mrt_adj .* (norm(abs.(bart_adj)) / norm(abs.(mrt_adj))), bart_adj)
push!(results, BenchResult("Base MC", "Cartesian Adjoint", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_adj * 1000, e_gt_bart_adj, e_mrt_bart_adj))

# --- Non-Cartesian DCF Adjoint (Gridding) ---
println("--> Benchmarking Non-Cartesian DCF Adjoint (Gridding)...")
t = RadialTrajectory(Float32, N, N; TE=0.0f0, AQ=1.0f-3)
traj_named = NamedDimsArray(t.nodes, (:dim, :k))
smaps_nc = NamedDimsArray(ComplexF32.(cmap), (:x, :y, :coil))

kdata_nc_zeros = NamedDimsArray(zeros(ComplexF32, 16384, Nc), (:k, :coil))
acq_nc_sim = NonCartesianAcquisitionInfo(kdata_nc_zeros; trajectory=traj_named, image_size=(N, N), sensitivity_maps=smaps_nc, shifted_image_dims=(:x, :y))
E_nc_sim = MriReconstructionToolbox.get_encoding_operator(acq_nc_sim)
kdata_nc_sim = E_nc_sim * NamedDimsArray(ComplexF32.(img_mc), (:x, :y))

acq_nc_dcf = NonCartesianAcquisitionInfo(kdata_nc_sim; trajectory=traj_named, image_size=(N, N), sensitivity_maps=smaps_nc, shifted_image_dims=(:x, :y))
E_nc_dcf = MriReconstructionToolbox.get_encoding_operator(acq_nc_dcf)
t_min_mrt_dcf, _, mrt_adj_dcf_raw = time_reconstruction(() -> E_nc_dcf' * kdata_nc_sim)
mrt_adj_dcf = mrt_adj_dcf_raw .* (norm(abs.(img_mc)) / norm(abs.(mrt_adj_dcf_raw)))
e_gt_mrt_dcf = nrmse(mrt_adj_dcf, img_mc)
push!(results, BenchResult("Non-Cartesian", "DCF Adjoint (Gridding)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_dcf * 1000, e_gt_mrt_dcf, 0.0))

# MRIReco Non-Cartesian Direct Reconstruction
kdata_mr_nc = reshape(kdata_nc_sim, 16384, Nc, 1, 1)
acq_mr_nc = AcquisitionData(t, fill(kdata_mr_nc[:, :, 1, 1], 1, 1, 1))
smaps_mr_f32 = reshape(ComplexF32.(cmap), N, N, 1, Nc)
recoParams_nc = Dict{Symbol, Any}(:reco => "direct", :reconSize => (N, N), :senseMaps => smaps_mr_f32)
t_min_mr_dcf, _, img_mr_nc_direct = time_reconstruction(() -> MRIReco.reconstruction(acq_mr_nc, recoParams_nc)[:, :, 1, 1, :])
mr_adj_nc_raw = sum(img_mr_nc_direct .* conj.(reshape(ComplexF32.(cmap), N, N, Nc)), dims=3)[:, :, 1]
mr_adj_nc = mr_adj_nc_raw .* (norm(abs.(img_mc)) / norm(abs.(mr_adj_nc_raw)))
e_gt_mr_dcf = nrmse(mr_adj_nc, img_mc)
e_mrt_mr_dcf = nrmse(mrt_adj_dcf, mr_adj_nc)
push!(results, BenchResult("Non-Cartesian", "DCF Adjoint (Gridding)", "MRIReco", num_threads, t_min_mr_dcf * 1000, e_gt_mr_dcf, e_mrt_mr_dcf))

# --- CG-SENSE (10 Iterations) ---
println("--> Benchmarking CG-SENSE (10 Iterations)...")
method_cg = IterativeReconstruction(regularization=(), algorithm=MriReconstructionToolbox.CGNR(maxit=10, tol=1e-14))
t_min_mrt_cg, _, x_mrt_cg = time_reconstruction(() -> reconstruct(acq_mc, method_cg; tol=1e-14, maxit=10))
e_gt_mrt_cg = nrmse(x_mrt_cg .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_cg))), img_mc)
push!(results, BenchResult("Base MC", "CG-SENSE (10 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_cg * 1000, e_gt_mrt_cg, 0.0))

# SigPy CG-SENSE
t_min_sp_cg, _, sp_cg_raw = time_reconstruction(() -> sp_mri.app.SenseRecon(kdata_sp_mc, smaps_sp_mc, max_iter=10, show_pbar=false).run())
sp_cg = permutedims(sp_cg_raw, (2, 1))
e_gt_sp_cg = nrmse(sp_cg .* (norm(abs.(img_mc)) / norm(abs.(sp_cg))), img_mc)
e_mrt_sp_cg = nrmse(x_mrt_cg .* (norm(abs.(sp_cg)) / norm(abs.(x_mrt_cg))), sp_cg)
push!(results, BenchResult("Base MC", "CG-SENSE (10 it)", "SigPy", num_threads, t_min_sp_cg * 1000, e_gt_sp_cg, e_mrt_sp_cg))

# BART CG-SENSE
smaps_bart_cart = reshape(cmap, N, N, 1, Nc)
t_min_bart_cg, _, bart_cg_raw = time_reconstruction(() -> run_bart(1, "pics -S -i 10", ComplexF32.(kdata_bart_cart), ComplexF32.(smaps_bart_cart)), is_bart=true)
bart_cg = bart_cg_raw[:, :, 1]
e_gt_bart_cg = nrmse(bart_cg .* (norm(abs.(img_mc)) / norm(abs.(bart_cg))), img_mc)
e_mrt_bart_cg = nrmse(x_mrt_cg .* (norm(abs.(bart_cg)) / norm(abs.(x_mrt_cg))), bart_cg)
push!(results, BenchResult("Base MC", "CG-SENSE (10 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_cg * 1000, e_gt_bart_cg, e_mrt_bart_cg))

# -------------------------------------------------------------
# 2. Sparsity-Based Regularization: 2x Undersampled Brain
# -------------------------------------------------------------
mask_reg = rand(MersenneTwister(42), Bool, N, N)
mask_reg[(N ÷ 2 - 8):(N ÷ 2 + 8), :] .= true

kdata_reg_us = NamedDimsArray(kspace_mc[mask_reg, :], (:kxy, :coil))
acq_reg_us = CartesianAcquisitionInfo(kdata_reg_us; is3D=false, image_size=(N, N), sensitivity_maps=smaps_mc, shifted_image_dims=(:x, :y), subsampling=mask_reg)

kspace_reg_sp = copy(kspace_mc)
kspace_reg_sp[.!mask_reg, :] .= 0
kdata_sp_reg = parent(permutedims(kspace_reg_sp, (3, 2, 1)))
kdata_bart_reg = reshape(kspace_reg_sp, N, N, 1, Nc)

# --- Total Variation (TV, 30 Iterations) ---
println("--> Benchmarking Total Variation (30 Iterations)...")
method_tv = IterativeReconstruction(regularization=TotalVariation2D(0.01))
t_min_mrt_tv, _, x_mrt_tv = time_reconstruction(() -> reconstruct(acq_reg_us, method_tv; maxit=30, tol=1e-5))
e_gt_mrt_tv = nrmse(x_mrt_tv .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_tv))), img_mc)
push!(results, BenchResult("Sparsity", "Total Variation (30 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_tv * 1000, e_gt_mrt_tv, 0.0))

# BART TV
t_min_bart_tv, _, bart_tv_raw = time_reconstruction(() -> run_bart(1, "pics -S -i 30 -R T:3:0:0.01", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_cart)), is_bart=true)
bart_tv = bart_tv_raw[:, :, 1]
e_gt_bart_tv = nrmse(bart_tv .* (norm(abs.(img_mc)) / norm(abs.(bart_tv))), img_mc)
e_mrt_bart_tv = nrmse(x_mrt_tv .* (norm(abs.(bart_tv)) / norm(abs.(x_mrt_tv))), bart_tv)
push!(results, BenchResult("Sparsity", "Total Variation (30 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_tv * 1000, e_gt_bart_tv, e_mrt_bart_tv))

# --- L1-Wavelet (30 Iterations) ---
println("--> Benchmarking L1-Wavelet (30 Iterations)...")
method_wav = IterativeReconstruction(regularization=L1Wavelet2D(0.005))
t_min_mrt_wav, _, x_mrt_wav = time_reconstruction(() -> reconstruct(acq_reg_us, method_wav; maxit=30, tol=1e-5))
e_gt_mrt_wav = nrmse(x_mrt_wav .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_wav))), img_mc)
push!(results, BenchResult("Sparsity", "L1-Wavelet (30 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_wav * 1000, e_gt_mrt_wav, 0.0))

# BART L1-Wavelet
t_min_bart_wav, _, bart_wav_raw = time_reconstruction(() -> run_bart(1, "pics -m -l1 -r 0.005 -n -S -i 30", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_cart)), is_bart=true)
bart_wav = bart_wav_raw[:, :, 1]
e_gt_bart_wav = nrmse(bart_wav .* (norm(abs.(img_mc)) / norm(abs.(bart_wav))), img_mc)
e_mrt_bart_wav = nrmse(x_mrt_wav .* (norm(abs.(bart_wav)) / norm(abs.(x_mrt_wav))), bart_wav)
push!(results, BenchResult("Sparsity", "L1-Wavelet (30 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_wav * 1000, e_gt_bart_wav, e_mrt_bart_wav))

# --- Total Generalized Variation (TGV, 30 Iterations) ---
println("--> Benchmarking Total Generalized Variation (30 Iterations)...")
method_tgv = IterativeReconstruction(regularization=TotalGeneralizedVariation2D(0.01; ratio=2.0))
t_min_mrt_tgv, _, x_mrt_tgv = time_reconstruction(() -> reconstruct(acq_reg_us, method_tgv; maxit=30, tol=1e-5))
e_gt_mrt_tgv = nrmse(x_mrt_tgv .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_tgv))), img_mc)
push!(results, BenchResult("Sparsity", "TGV (30 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_tgv * 1000, e_gt_mrt_tgv, 0.0))

# BART TGV
t_min_bart_tgv, _, bart_tgv_raw = time_reconstruction(() -> run_bart(1, "pics -S -i 30 -R G:3:0:0.01", ComplexF32.(kdata_bart_reg), ComplexF32.(smaps_bart_cart)), is_bart=true)
bart_tgv = bart_tgv_raw[:, :, 1]
e_gt_bart_tgv = nrmse(bart_tgv .* (norm(abs.(img_mc)) / norm(abs.(bart_tgv))), img_mc)
e_mrt_bart_tgv = nrmse(x_mrt_tgv .* (norm(abs.(bart_tgv)) / norm(abs.(x_mrt_tgv))), bart_tgv)
push!(results, BenchResult("Sparsity", "TGV (30 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_tgv * 1000, e_gt_bart_tgv, e_mrt_bart_tgv))

# -------------------------------------------------------------
# 3. Dynamic & Low-Rank: 2D+t Brain Dataset (64x64, 4 coils, 8 frames)
# -------------------------------------------------------------
Nd = 64
Ncd = 4
Td = 8
img_dyn, kspace_dyn, cmap_dyn = generate_dynamic_multicoil_brain(N=Nd, num_coils=Ncd, num_frames=Td)

mask_pe = rand(MersenneTwister(42), Bool, Nd)
mask_pe[(Nd ÷ 2 - 4):(Nd ÷ 2 + 4)] .= true

kspace_dyn_us = kspace_dyn[:, mask_pe, :, :]
kdata_dyn_us = NamedDimsArray(permutedims(kspace_dyn_us, (1, 2, 4, 3)), (:kx, :ky, :coil, :time))
smaps_dyn_named = NamedDimsArray(cmap_dyn, (:x, :y, :coil))

acq_dyn = CartesianAcquisitionInfo(
    kdata_dyn_us;
    is3D=false,
    image_size=(Nd, Nd),
    sensitivity_maps=smaps_dyn_named,
    subsampling=(:, mask_pe),
    shifted_image_dims=(:x, :y)
)

kdata_bart_dyn = zeros(ComplexF32, Nd, Nd, 1, Ncd, 1, Td)
for t = 1:Td
    kdata_bart_dyn[:, mask_pe, 1, :, 1, t] .= ComplexF32.(kspace_dyn[:, mask_pe, t, :])
end
smaps_bart_dyn = reshape(ComplexF32.(cmap_dyn), Nd, Nd, 1, Ncd)

# --- Global Low-Rank (20 Iterations) ---
println("--> Benchmarking Global Low-Rank (20 Iterations)...")
method_lr = IterativeReconstruction(regularization=LowRank(0.01; time_dim=:time))
t_min_mrt_lr, _, x_mrt_lr = time_reconstruction(() -> reconstruct(acq_dyn, method_lr; maxit=20, tol=1e-4))
e_gt_mrt_lr = nrmse(x_mrt_lr .* (norm(abs.(img_dyn)) / norm(abs.(x_mrt_lr))), img_dyn)
push!(results, BenchResult("Dynamic", "Global Low-Rank (20 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_lr * 1000, e_gt_mrt_lr, 0.0))

# --- Locally Low-Rank (LLR, 20 Iterations) ---
println("--> Benchmarking Locally Low-Rank (20 Iterations)...")
method_llr = IterativeReconstruction(regularization=LocallyLowRank(0.01; block_size=(8, 8), time_dim=:time))
t_min_mrt_llr, _, x_mrt_llr = time_reconstruction(() -> reconstruct(acq_dyn, method_llr; maxit=20, tol=1e-4))
e_gt_mrt_llr = nrmse(x_mrt_llr .* (norm(abs.(img_dyn)) / norm(abs.(x_mrt_llr))), img_dyn)
push!(results, BenchResult("Dynamic", "Locally Low-Rank (20 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_llr * 1000, e_gt_mrt_llr, 0.0))

# BART LLR
t_min_bart_llr, _, bart_llr_raw = time_reconstruction(() -> run_bart(1, "pics -S -i 20 -b 8 -R L:3:3:0.01", kdata_bart_dyn, smaps_bart_dyn), is_bart=true)
bart_llr = dropdims(bart_llr_raw, dims=(3, 4, 5))
e_gt_bart_llr = nrmse(bart_llr .* (norm(abs.(img_dyn)) / norm(abs.(bart_llr))), img_dyn)
e_mrt_bart_llr = nrmse(x_mrt_llr .* (norm(abs.(bart_llr)) / norm(abs.(x_mrt_llr))), bart_llr)
push!(results, BenchResult("Dynamic", "Locally Low-Rank (20 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_llr * 1000, e_gt_bart_llr, e_mrt_bart_llr))

# --- Temporal TV (20 Iterations) ---
println("--> Benchmarking Temporal TV (20 Iterations)...")
method_ttv = IterativeReconstruction(regularization=TemporalTotalVariation(0.01; time_dim=:time))
t_min_mrt_ttv, _, x_mrt_ttv = time_reconstruction(() -> reconstruct(acq_dyn, method_ttv; maxit=20, tol=1e-4))
e_gt_mrt_ttv = nrmse(x_mrt_ttv .* (norm(abs.(img_dyn)) / norm(abs.(x_mrt_ttv))), img_dyn)
push!(results, BenchResult("Dynamic", "Temporal TV (20 it)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_ttv * 1000, e_gt_mrt_ttv, 0.0))

# BART Temporal TV
t_min_bart_ttv, _, bart_ttv_raw = time_reconstruction(() -> run_bart(1, "pics -S -i 20 -R T:32:0:0.01", kdata_bart_dyn, smaps_bart_dyn), is_bart=true)
bart_ttv = dropdims(bart_ttv_raw, dims=(3, 4, 5))
e_gt_bart_ttv = nrmse(bart_ttv .* (norm(abs.(img_dyn)) / norm(abs.(bart_ttv))), img_dyn)
e_mrt_bart_ttv = nrmse(x_mrt_ttv .* (norm(abs.(bart_ttv)) / norm(abs.(x_mrt_ttv))), bart_ttv)
push!(results, BenchResult("Dynamic", "Temporal TV (20 it)", "BART ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_bart_ttv * 1000, e_gt_bart_ttv, e_mrt_bart_ttv))

# -------------------------------------------------------------
# 4. K-Space Methods: GRAPPA
# -------------------------------------------------------------
mask_grappa = falses(N, N)
mask_grappa[:, 1:2:N] .= true
mask_grappa[:, (N ÷ 2 - 12):(N ÷ 2 + 11)] .= true

kdata_grappa_us = NamedDimsArray(kspace_mc[mask_grappa, :], (:kxy, :coil))
acq_grappa = CartesianAcquisitionInfo(
    kdata_grappa_us;
    is3D=false,
    image_size=(N, N),
    sensitivity_maps=smaps_mc,
    subsampling=mask_grappa,
    shifted_image_dims=(:x, :y)
)

println("--> Benchmarking GRAPPA (RSS)...")
method_grappa_rss = GRAPPA(kernel_size=(4, 3), calib_size=(24, 24), coil_combination=RootSumSquares())
t_min_mrt_grappa_rss, _, x_mrt_grappa_rss = time_reconstruction(() -> reconstruct(acq_grappa, method_grappa_rss))
e_gt_mrt_grappa_rss = nrmse(x_mrt_grappa_rss .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_grappa_rss))), img_mc)
push!(results, BenchResult("K-Space", "GRAPPA (RSS)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_grappa_rss * 1000, e_gt_mrt_grappa_rss, 0.0))

println("--> Benchmarking GRAPPA (Sensitivity)...")
method_grappa_sens = GRAPPA(kernel_size=(4, 3), calib_size=(24, 24), coil_combination=AdjointSensitivity())
t_min_mrt_grappa_sens, _, x_mrt_grappa_sens = time_reconstruction(() -> reconstruct(acq_grappa, method_grappa_sens))
e_gt_mrt_grappa_sens = nrmse(x_mrt_grappa_sens .* (norm(abs.(img_mc)) / norm(abs.(x_mrt_grappa_sens))), img_mc)
push!(results, BenchResult("K-Space", "GRAPPA (Sensitivity)", "MRT ($(use_mkl ? "MKL" : "OpenBLAS"))", num_threads, t_min_mrt_grappa_sens * 1000, e_gt_mrt_grappa_sens, 0.0))

# -------------------------------------------------------------
# Display Benchmark Results Table
# -------------------------------------------------------------
println("\n\n========================================================================================================")
println("                                        FINAL BENCHMARK RESULTS                                         ")
println("========================================================================================================")
@printf("%-14s | %-26s | %-24s | %-7s | %-12s | %-10s | %-10s\n", "Category", "Method", "Framework", "Threads", "Time (ms)", "NRMSE (GT)", "Diff (MRT)")
println("--------------------------------------------------------------------------------------------------------")
for r in results
    @printf("%-14s | %-26s | %-24s | %7d | %10.2f ms | %10.2e | %10.2e\n", r.category, r.method, r.framework, r.threads, r.time_ms, r.nrmse_gt, r.nrmse_mrt)
end
println("========================================================================================================\n")

# -------------------------------------------------------------
# Save to JSON File
# -------------------------------------------------------------
results_dir = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(results_dir)
json_filename = "benchmark_$(use_mkl ? "mkl" : "openblas")_$(num_threads)threads.json"
json_path = joinpath(results_dir, json_filename)

json_data = Dict(
    "hostname" => gethostname(),
    "julia_version" => string(VERSION),
    "julia_threads" => Threads.nthreads(),
    "blas_vendor" => BLAS.get_config().loaded_libs[1].libname,
    "use_mkl" => use_mkl,
    "bart_binary" => bart_binary,
    "pinned_cpus" => cpu_str,
    "bart_startup_time_ms" => bart_startup_time * 1000,
    "benchmarks" => [
        Dict(
            "category" => r.category,
            "method" => r.method,
            "framework" => r.framework,
            "threads" => r.threads,
            "time_ms" => r.time_ms,
            "nrmse_gt" => r.nrmse_gt,
            "nrmse_mrt" => r.nrmse_mrt
        ) for r in results
    ]
)

open(json_path, "w") do io
    JSON.print(io, json_data, 4)
end
println("Benchmark results saved to: $json_path\n")

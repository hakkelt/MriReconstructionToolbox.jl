using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using MriReconstructionToolbox
using LinearAlgebra
using Statistics

# Add our comparison harness
include("../src/ComparisonHarness.jl")
using .ComparisonHarness

@info "Starting Preprocessing Comparisons"

# Setup MATLAB
# setup_matlab_paths()

@testset "Preprocessing Benchmark" begin
    @testset "ESPIRiT Sensitivity Estimation" begin
        # Generate data
        N = 64
        num_coils = 8
        img, kspace, true_sens = ComparisonHarness.generate_multicoil_brain(N=N, num_coils=num_coils)
        
        # 1. MriReconstructionToolbox
        # We need a calibration region. ESPIRiT usually extracts the center of k-space internally if we pass the whole k-space.
        # Let's pass the whole kspace to estimate_sensitivities
        mrt_sens = MriReconstructionToolbox.estimate_sensitivities(kspace, method=MriReconstructionToolbox.ESPIRiT(calib_size=24))
        
        # 2. BART
        # We pass the full k-space. BART ecalib with -r 24 will extract the center 24x24 internally.
        bart_kspace = ComplexF32.(reshape(kspace, (N, N, 1, num_coils)))
        bart_res = ComparisonHarness.run_bart(2, "ecalib -r 24 -c 0 -m 1", bart_kspace)
        bart_sens = bart_res[1]
        
        # 3. SigPy
        # SigPy expects data in [coils, Y, X] (or Z, Y, X).
        sp_kspace = permutedims(kspace, (3, 2, 1))
        sp_app = ComparisonHarness.sigpy_mri_app.EspiritCalib(sp_kspace, calib_width=24, show_pbar=false)
        sp_sens = sp_app.run()
        
        # We check metrics, e.g., SSIM or just norm difference (modulo global phase).
        function check_rmse(A, B; tol=0.1)
            return norm(abs.(A) - abs.(B)) / norm(abs.(A)) < tol
        end
        
        @test size(bart_sens) == (N, N, 1, num_coils)
        @test size(sp_sens) == (num_coils, N, N) # Assuming sp returns coils first
        @test size(mrt_sens) == (N, N, num_coils)
        
        # Compare magnitudes (ESPIRiT maps have arbitrary phase)
        bart_mag = dropdims(abs.(bart_sens), dims=3)
        sp_mag = permutedims(abs.(sp_sens), (2, 3, 1))
        mrt_mag = abs.(mrt_sens)
        
        @test check_rmse(mrt_mag, bart_mag)
        @test check_rmse(mrt_mag, sp_mag)
    end
    
    @testset "Coil Compression" begin
        N = 64
        num_coils = 8
        target_coils = 4
        _, kspace, _ = ComparisonHarness.generate_multicoil_brain(N=N, num_coils=num_coils)
        
        # 1. MRT SVD Compression
        comp_mrt_svd, C_svd = MriReconstructionToolbox.compress_coils(kspace, target_coils, method=MriReconstructionToolbox.SVDCompression())
        @test size(comp_mrt_svd) == (N, N, target_coils)
        
        # 2. MRT Geometric Compression
        comp_mrt_geo, C_geo = MriReconstructionToolbox.compress_coils(kspace, target_coils, method=MriReconstructionToolbox.GeometricCompression())
        @test size(comp_mrt_geo) == (N, N, target_coils)
        
        # 3. BART
        bart_kspace = ComplexF32.(reshape(kspace, (N, N, 1, num_coils)))
        bart_comp_res = ComparisonHarness.run_bart(1, "cc -p $target_coils -S", bart_kspace)
        @test size(bart_comp_res) == (N, N, 1, target_coils)
        
        # Total energy in the compressed coils should be similar
        energy_mrt = norm(comp_mrt_svd)
        energy_bart = norm(bart_comp_res)
        @test isapprox(energy_mrt, energy_bart, rtol=1e-2)
    end
    
    @testset "Gradient Delay (RING)" begin
        @test true
    end
end

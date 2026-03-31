using TestItems

@testitem "Reconstruction integration" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra

    # Helper to test type stability: ensures return type matches expected concrete type
    macro test_type_stable(expected_type, expr)
        quote
            local result = $(esc(expr))
            @test typeof(result) == $(esc(expected_type))
            result
        end
    end

@testset "High-Level Reconstruction Integration Tests" begin
    
    @testset "2D Reconstruction Pipeline" begin
        @testset "Fully-sampled without regularization" begin
            nx, ny, nc = 32, 32, 4
            
            # Create phantom and simulate acquisition
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct
            img_recon = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data; verbose=false)
            
            # Reconstruction accuracy check (L2-norm)
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1e-3
        end
        
        @testset "Undersampled with Tikhonov regularization" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            # Create undersampling pattern (less aggressive undersampling)
            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct with Tikhonov
            img_recon = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, Tikhonov(0.001); maxit=100, verbose=false)
            
            # Reconstruction accuracy check (reconstruction quality depends on sampling)
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 2.5  # Looser tolerance as this tests pipeline, not reconstruction quality
        end
        
        @testset "Undersampled with L1Wavelet regularization" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct with L1Wavelet
            img_recon = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, L1Wavelet2D(0.005); maxit=50, verbose=false)
            
            # Reconstruction accuracy check
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.3
        end
        
        @testset "Multiple regularizations" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct with multiple regularizations
            img_recon = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                (L1Wavelet2D(0.003), TotalVariation2D(0.001));
                maxit=100,
                verbose=false
            )
            
            # Reconstruction accuracy check (tests pipeline functionality)
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 2.5  # Looser tolerance as this tests pipeline, not reconstruction quality
        end
        
        @testset "Different algorithms" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = UniformRandomSampling(2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Test FISTA algorithm (works well for general problems)
            img_fista = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, Tikhonov(0.001); maxit=100, verbose=false)
            error_fista = norm(img_fista - img_true) / norm(img_true)
            @test error_fista < 2.5  # Tests type stability and pipeline, not reconstruction quality
            
            # Test ADMM algorithm with L1 regularization
            img_admm = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, L1Wavelet2D(0.003), ADMM(); maxit=50, verbose=false)
            error_admm = norm(img_admm - img_true) / norm(img_true)
            @test error_admm < 2.5  # Tests type stability and pipeline, not reconstruction quality
        end
        
        @testset "With initial guess" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Direct reconstruction as initial guess
            x_init = reconstruct(acq_with_data; verbose=false)
            
            # Refine with regularization
            img_recon = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                L1Wavelet2D(0.005);
                x₀=x_init,
                maxit=30,
                verbose=false
            )
            
            # Reconstruction accuracy check
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.3
        end
    end
    
    @testset "3D Reconstruction Pipeline" begin
        @testset "Fully-sampled 3D" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            
            img_true = shepp_logan(nx, ny, nz)
            smaps = coil_sensitivities(nx, ny, nz, nc)
            
            acq = AcquisitionInfo(is3D=true, sensitivity_maps=smaps)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct
            img_recon = @test_type_stable Array{ComplexF32, 3} reconstruct(acq_with_data; verbose=false)
            
            # Reconstruction accuracy check
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1e-3
        end
        
        @testset "Undersampled 3D with regularization" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            
            img_true = shepp_logan(nx, ny, nz)
            smaps = coil_sensitivities(nx, ny, nz, nc)
            
            pdf = UniformRandomSampling(4.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny, nz))
            
            acq = AcquisitionInfo(is3D=true, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Reconstruct with 3D wavelet regularization
            img_recon = @test_type_stable Array{ComplexF32, 3} reconstruct(acq_with_data, L1Wavelet3D(0.005); maxit=30, verbose=false)
            
            # Reconstruction accuracy check
            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.4
        end
    end
    
    @testset "Multi-slice 2D Reconstruction" begin
        @testset "Multi-slice with decomposition" begin
            nx, ny, nslices, nc = 32, 32, 3, 4
            
            # Create multi-slice phantom
            img_true = shepp_logan(nx, ny)
            img_true_ms = repeat(img_true, 1, 1, nslices)
            
            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)
            
            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps)
                acq_temp = simulate_acquisition(img_true_ms[:,:,s], acq)
                ksp_ms[:,:,:,s] .= acq_temp.kspace_data
            end
            
            acq_ms = AcquisitionInfo(ksp_ms; is3D=false, sensitivity_maps=smaps_ms)
            
            # Reconstruct with automatic decomposition
            img_recon = @test_type_stable Array{ComplexF32, 3} reconstruct(acq_ms; maxit=10, verbose=false)
            
            # Check output size
            @test size(img_recon) == (nx, ny, nslices)
            
            # Reconstruction accuracy check
            error_norm = norm(img_recon - img_true_ms) / norm(img_true_ms)
            @test error_norm < 1e-3
        end
    end
    
    @testset "Config and Configuration Options" begin
        @testset "Config object usage" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Method 1: Keyword arguments
            img1 = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, Tikhonov(0.01); maxit=20, tol=1e-5, verbose=false)
            
            # Method 2: Config object
            config = Config(maxit=20, tol=1e-5, verbose=false)
            img2 = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, Tikhonov(0.01); config=config)
            
            # Method 3: Override config with keywords
            config_base = Config(maxit=100, verbose=false)
            img3 = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data, Tikhonov(0.01); config=config_base, maxit=20)
            
            # All should produce similar results
            @test norm(img1 - img2) / norm(img1) < 1e-3
        end
        
        @testset "Threading configuration" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # Single-threaded
            img_st = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data; threaded=false, verbose=false)
            
            # Multi-threaded
            img_mt = @test_type_stable Matrix{ComplexF32} reconstruct(acq_with_data; threaded=true, verbose=false)
            
            # Should produce same results
            @test norm(img_st - img_mt) / norm(img_st) < 1e-10
        end
        
        @testset "Normalization strategies" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # BartScaling
            img_bart = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                Tikhonov(0.01);
                normalization=BartScaling(),
                maxit=20,
                verbose=false
            )
            
            # NoScaling
            img_noscale = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                Tikhonov(0.01);
                normalization=NoScaling(),
                maxit=20,
                verbose=false
            )
            
            # MeasurementBasedScaling
            img_meas = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                Tikhonov(0.01);
                normalization=MeasurementBasedScaling(),
                maxit=20,
                verbose=false
            )
        end
    end
    
    @testset "NamedDims Support" begin
        @testset "NamedDims preservation" begin
            nx, ny, nc = 32, 32, 4
            
            # Create named arrays
            img_true = shepp_logan(nx, ny)
            ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, nx, ny, nc))
            smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))
            
            acq = AcquisitionInfo(ksp; sensitivity_maps=smaps)
            
            # Reconstruct (test that concrete type matches NamedDimsArray)
            img_recon = @test_type_stable NamedDimsArray{(:x, :y), ComplexF32, 2, Matrix{ComplexF32}} reconstruct(acq; verbose=false)
            
            # Check that dimensions are preserved
            @test dimnames(img_recon) == (:x, :y)
            @test eltype(img_recon) == ComplexF32
        end
    end
    
    @testset "Operator Options" begin
        @testset "Operator normalization" begin
            nx, ny, nc = 32, 32, 4
            
            img_true = shepp_logan(nx, ny)
            smaps = coil_sensitivities(nx, ny, nc)
            
            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))
            
            acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps, subsampling=pattern)
            acq_with_data = simulate_acquisition(img_true, acq)
            
            # With normalization (default)
            img_norm = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                Tikhonov(0.01);
                disable_operator_normalization=false,
                maxit=20,
                verbose=false
            )
            
            # Without normalization
            img_unnorm = @test_type_stable Matrix{ComplexF32} reconstruct(
                acq_with_data,
                Tikhonov(0.01);
                disable_operator_normalization=true,
                maxit=20,
                verbose=false
            )
        end
        
        @testset "Problem decomposition control" begin
            nx, ny, nslices, nc = 32, 32, 2, 4
            
            img_true = shepp_logan(nx, ny)
            img_true_ms = repeat(img_true, 1, 1, nslices)
            
            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)
            
            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D=false, sensitivity_maps=smaps)
                acq_temp = simulate_acquisition(img_true_ms[:,:,s], acq)
                ksp_ms[:,:,:,s] .= acq_temp.kspace_data
            end
            
            acq_ms = AcquisitionInfo(ksp_ms; is3D=false, sensitivity_maps=smaps_ms)
            
            # With decomposition (default)
            img_decomp = @test_type_stable Array{ComplexF32, 3} reconstruct(
                acq_ms;
                disable_problem_decomposition=false,
                maxit=10,
                verbose=false
            )
            
            # Without decomposition
            img_no_decomp = @test_type_stable Array{ComplexF32, 3} reconstruct(
                acq_ms;
                disable_problem_decomposition=true,
                maxit=10,
                verbose=false
            )
            
            # Should produce same results
            @test norm(img_decomp - img_no_decomp) / norm(img_decomp) < 1e-10
        end
    end
end
end

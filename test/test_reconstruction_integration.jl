@testitem "2D Reconstruction Pipeline" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "2D Reconstruction Pipeline" begin
        @testset "Fully-sampled without regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; verbose = false))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1.0e-3
        end

        @testset "Undersampled with Tikhonov regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.001); maxit = 100, verbose = false))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 2.5
        end

        @testset "Undersampled with L1Wavelet regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, L1Wavelet2D(0.005); maxit = 50, verbose = false))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.3
        end

        @testset "Multiple regularizations" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, (L1Wavelet2D(0.003), TotalVariation2D(0.001)); maxit = 100, verbose = false),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 2.5
        end

        @testset "Different algorithms" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_fista = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.001); maxit = 100, verbose = false))
            error_fista = norm(img_fista - img_true) / norm(img_true)
            @test error_fista < 2.5

            img_admm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, L1Wavelet2D(0.003), ADMM(); maxit = 50, verbose = false))
            error_admm = norm(img_admm - img_true) / norm(img_true)
            @test error_admm < 2.5
        end

        @testset "With initial guess" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            x_init = reconstruct(acq_with_data; verbose = false)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, L1Wavelet2D(0.005); x₀ = x_init, maxit = 30, verbose = false),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.5
        end
    end
end

@testitem "3D Reconstruction Pipeline" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "3D Reconstruction Pipeline" begin
        @testset "Fully-sampled 3D" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            img_true = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nz, nc)

            acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data; verbose = false))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1.0e-3
        end

        @testset "Undersampled 3D with regularization" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            img_true = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nz, nc)

            pdf = UniformRandomSampling(4.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny, nz))

            acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data, L1Wavelet3D(0.005); maxit = 30, verbose = false))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.7
        end
    end
end

@testitem "Multi-slice 2D Reconstruction" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "Multi-slice 2D Reconstruction" begin
        @testset "Multi-slice with decomposition" begin
            nx, ny, nslices, nc = 32, 32, 3, 4

            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = repeat(img_true, 1, 1, nslices)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end

            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; maxit = 10, verbose = false))

            @test size(img_recon) == (nx, ny, nslices)
            error_norm = norm(img_recon - img_true_ms) / norm(img_true_ms)
            @test error_norm < 1.0e-3
        end

        @testset "Multi-slice with regularization and decomposition" begin
            nx, ny, nslices, nc = 16, 16, 3, 2

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            # Tikhonov regularization + multislice exercises problem decomposition with regularization
            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms, Tikhonov(0.01); maxit = 5, verbose = false))
            @test size(img_recon) == (nx, ny, nslices)
        end
    end
end

@testitem "Config and Configuration Options" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "Config and Configuration Options" begin
        @testset "Config object usage" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img1 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); maxit = 20, tol = 1.0e-5, verbose = false))
            config = Config(maxit = 20, tol = 1.0e-5, verbose = false)
            img2 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); config = config))
            config_base = Config(maxit = 100, verbose = false)
            img3 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); config = config_base, maxit = 20))

            # Loose tolerance: threaded FFTs make repeated solver runs agree only to ~1e-3
            @test norm(img1 - img2) / norm(img1) < 5.0e-3
            @test norm(img1 - img3) / norm(img1) < 5.0e-3
        end

        @testset "Threading configuration" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_st = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = false, verbose = false))
            img_mt = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = true, verbose = false))

            @test norm(img_st - img_mt) / norm(img_st) < 1.0e-10
        end

        @testset "Normalization strategies" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); normalization = BartScaling(), maxit = 20, verbose = false))
            img_noscale = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); normalization = NoScaling(), maxit = 20, verbose = false))
            img_meas = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); normalization = MeasurementBasedScaling(), maxit = 20, verbose = false))

            @test size(img_bart) == size(img_noscale) == size(img_meas)
        end

        @testset "FixedScaling" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            @test MriReconstructionToolbox.get_scale(FixedScaling(2.5), acq_with_data, nothing) == 2.5
            @test_throws ArgumentError FixedScaling(0.0)
            @test_throws ArgumentError FixedScaling(-1.0)

            scale = MriReconstructionToolbox.get_scale(BartScaling(), acq_with_data, img_true)
            img_fixed = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); normalization = FixedScaling(scale), maxit = 20, verbose = false, tol = 0.0))
            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); normalization = BartScaling(), maxit = 20, verbose = false, tol = 0.0))
            @test size(img_fixed) == size(img_bart)
        end
    end
end

@testitem "NamedDims Support" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "NamedDims Support" begin
        @testset "NamedDims preservation" begin
            nx, ny, nc = 32, 32, 4

            ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, nx, ny, nc))
            smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_recon = test_type_stable(NamedDimsArray{(:x, :y), ComplexF32, 2, Matrix{ComplexF32}}, reconstruct(acq; verbose = false))

            @test dimnames(img_recon) == (:x, :y)
            @test eltype(img_recon) == ComplexF32
        end

        @testset "NamedDims with problem decomposition" begin
            nx, ny, nslices, nc = 16, 16, 3, 2

            ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, nx, ny, nc, nslices))
            smaps = NamedDimsArray{(:x, :y, :coil, :z)}(repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_direct = reconstruct(acq; verbose = false)
            @test img_direct isa NamedDimsArray
            @test dimnames(img_direct) == (:x, :y, :z)
            @test size(img_direct) == (nx, ny, nslices)

            img_reg = reconstruct(acq, Tikhonov(0.01); maxit = 5, verbose = false)
            @test img_reg isa NamedDimsArray
            @test dimnames(img_reg) == (:x, :y, :z)
            @test size(img_reg) == (nx, ny, nslices)
        end
    end
end

@testitem "Operator Options" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "Operator Options" begin
        @testset "Operator normalization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_norm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); disable_operator_normalization = false, maxit = 20, verbose = false))
            img_unnorm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, Tikhonov(0.01); disable_operator_normalization = true, maxit = 20, verbose = false))

            @test size(img_norm) == size(img_unnorm)
        end

        @testset "Problem decomposition control" begin
            nx, ny, nslices, nc = 32, 32, 2, 4

            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = repeat(img_true, 1, 1, nslices)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
            for s in 1:nslices
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end

            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_decomp = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_problem_decomposition = false, maxit = 10, verbose = false))
            img_no_decomp = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_problem_decomposition = true, maxit = 10, verbose = false))

            @test norm(img_decomp - img_no_decomp) / norm(img_decomp) < 1.0e-10

            # Regularized case: slices are identical, so the per-slice median scale equals
            # the global scale and both paths must converge to the same solution.
            img_decomp_reg = reconstruct(acq_ms, Tikhonov(0.01); disable_problem_decomposition = false, maxit = 30, verbose = false)
            img_no_decomp_reg = reconstruct(acq_ms, Tikhonov(0.01); disable_problem_decomposition = true, maxit = 30, verbose = false)

            @test norm(img_decomp_reg - img_no_decomp_reg) / norm(img_no_decomp_reg) < 1.0e-3
        end

        @testset "x₀ with problem decomposition" begin
            nx, ny, nslices, nc = 16, 16, 3, 2
            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)
            ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            x₀ = zeros(ComplexF32, nx, ny, nslices)
            img_recon = reconstruct(acq_ms, Tikhonov(0.01); x₀, maxit = 5, verbose = false)
            @test size(img_recon) == (nx, ny, nslices)

            x₀_wrong = zeros(ComplexF32, nx, ny)
            @test_throws ArgumentError reconstruct(acq_ms, Tikhonov(0.01); x₀ = x₀_wrong, maxit = 5, verbose = false)
        end
    end
end

@testitem "Verbose and MultiThreading Decomposition" tags = [:reconstruction, :integration] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra
    using GeometricMedicalPhantoms

    test_type_stable(::Type{T}, value) where {T} = (@test typeof(value) == T; value)

    @testset "Verbose progress output" begin
        nx, ny, nc = 16, 16, 2
        img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        smaps = coil_sensitivities(nx, ny, nc)
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        acq_with_data = simulate_acquisition(img_true, acq)

        output = IOBuffer()
        printfunc = (args...) -> print(output, args...)
        config = Config(; verbose = true, printfunc = printfunc, maxit = 5)
        img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; config = config))

        @test size(img_recon) == (nx, ny)
        @test length(take!(output)) > 0
    end

    @testset "MultiThreadingExecutor decomposition" begin
        nx, ny, nslices, nc = 16, 16, 4, 2
        smaps = coil_sensitivities(nx, ny, nc)
        smaps_ms = repeat(smaps, 1, 1, 1, nslices)
        ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
        acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

        # Force MultiThreadingExecutor to cover that path in decomposition.jl
        executor = MriReconstructionToolbox.MultiThreadingExecutor()
        img_recon = test_type_stable(
            Array{ComplexF32, 3},
            reconstruct(acq_ms, Tikhonov(0.01); maxit = 5, verbose = false, decomposition_executor = executor),
        )
        @test size(img_recon) == (nx, ny, nslices)
    end
end

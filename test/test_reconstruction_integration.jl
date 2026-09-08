@testitem "2D Reconstruction Pipeline" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using Random

    @testset "2D Reconstruction Pipeline" begin
        @testset "Fully-sampled without regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; verbosity = Silent()))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 1.0e-3
        end

        @testset "Undersampled with L2Image regularization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.001); maxit = 100); verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.4  # measured ≈0.20
        end

        @testset "Undersampled with L1Wavelet regularization" begin
            # The sampling pattern is drawn at random, and the achievable error genuinely varies
            # with the draw: over 20 seeds this lands between 0.178 and 0.331, so an unseeded run
            # cleared the 0.3 bound only about 85% of the time. The reconstruction is converged by
            # maxit = 50 (200 and 1000 give the same answer), so the spread is the pattern, not the
            # solver. Fix the draw so the bound tests the reconstruction rather than the dice.
            Random.seed!(20260829)
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.005); maxit = 50); verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.3
        end

        @testset "Multiple regularizations" begin
            # Seeded for the same reason as the L1Wavelet case above.
            Random.seed!(20260829)
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(
                    acq_with_data,
                    IterativeReconstruction(L1Wavelet2D(0.003), TotalVariation2D(0.001); maxit = 100); verbosity = Silent()
                ),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            # Tight enough to catch a sign-flipped solution (which lands at ≈2.0); measured ≈0.02.
            # The sampling pattern is drawn randomly per run, so the bound keeps a wide margin.
            @test error_norm < 0.3
        end

        @testset "Different algorithms" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(2.0, 0.15)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_fista = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.001); maxit = 100); verbosity = Silent()),
            )
            error_fista = norm(img_fista - img_true) / norm(img_true)
            @test error_fista < 0.3  # measured ≈0.01-0.10 depending on the random sampling pattern

            img_admm = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.003); algorithm = ADMM(), maxit = 50); verbosity = Silent()),
            )
            error_admm = norm(img_admm - img_true) / norm(img_true)
            # A sign error in the ADMM data term lands at ≈2.0; measured ≈0.06 at 50 iterations.
            @test error_admm < 0.3
        end

        @testset "With initial guess" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            x_init = reconstruct(acq_with_data; verbosity = Silent())

            img_recon = test_type_stable(
                Matrix{ComplexF32},
                reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(0.005); maxit = 30); x₀ = x_init, verbosity = Silent()),
            )

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.5
        end
    end
end

@testitem "3D Reconstruction Pipeline" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "3D Reconstruction Pipeline" begin
        @testset "Fully-sampled 3D" begin
            nx, ny, nz, nc = 16, 16, 16, 4
            img_true = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nz, nc)

            acq = AcquisitionInfo(is3D = true, sensitivity_maps = smaps)
            acq_with_data = simulate_acquisition(img_true, acq)

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data; verbosity = Silent()))

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

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet3D(0.005); maxit = 30); verbosity = Silent()))

            error_norm = norm(img_recon - img_true) / norm(img_true)
            @test error_norm < 0.7
        end
    end
end

@testitem "Multi-slice 2D Reconstruction" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using StructuredOptimization

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

            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; verbosity = Silent()))

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

            # L2Image regularization + multislice exercises problem decomposition with regularization
            img_recon = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Silent()))
            @test size(img_recon) == (nx, ny, nslices)
        end

        @testset "Regularized decomposition with varying slice intensities" begin
            # Slices with wildly different signal levels: regularized decomposition solves each
            # slice normalized by its own scale (so λ is applied consistently) but uses one shared
            # scale to convert every slice back to image units, matching a joint (non-decomposed)
            # solve of the whole stack. See scale_regularization.
            nx, ny, nc = 16, 16, 2
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            intensities = ComplexF32[0.1, 1.0, 5.0, 20.0]
            img_true_ms = cat((intensities[s] .* img_true for s in eachindex(intensities))...; dims = 3)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, length(intensities))

            ksp_ms = zeros(ComplexF32, nx, ny, nc, length(intensities))
            for s in eachindex(intensities)
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_decomp = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 20); disable_problem_decomposition = false, verbosity = Silent())
            img_no_decomp = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 20); disable_problem_decomposition = true, verbosity = Silent())

            # Decomposed vs jointly-solved must agree closely regardless of the intensity spread.
            @test norm(img_decomp - img_no_decomp) / norm(img_no_decomp) < 1.0e-3

            # Regularization strength must stay consistent across slices: relative error against
            # the true image should not blow up for the low- or high-intensity slices.
            rel_errors = [
                norm(img_decomp[:, :, s] - img_true_ms[:, :, s]) / norm(img_true_ms[:, :, s])
                    for s in eachindex(intensities)
            ]
            @test maximum(rel_errors) / minimum(rel_errors) < 1.1
        end

        @testset "Regularized decomposition with an all-zero slice" begin
            # A slice whose own scale estimate is (near) zero must not have its regularization
            # collapse to zero (which would leave noise unregularized); safe_scale_ratio guards this.
            nx, ny, nc = 16, 16, 2
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            img_true_ms = cat(img_true, zeros(ComplexF32, nx, ny), img_true; dims = 3)

            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, 3)

            ksp_ms = zeros(ComplexF32, nx, ny, nc, 3)
            for s in 1:3
                acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
                acq_temp = simulate_acquisition(img_true_ms[:, :, s], acq)
                ksp_ms[:, :, :, s] .= acq_temp.kspace_data
            end
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            img_recon = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.05); maxit = 15); disable_problem_decomposition = false, verbosity = Silent())
            @test all(isfinite, img_recon)
            @test norm(img_recon[:, :, 2]) / norm(img_recon[:, :, 1]) < 0.1
        end
    end
end

@testitem "ReconstructionConfig and Configuration Options" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "ReconstructionConfig and Configuration Options" begin
        @testset "ReconstructionConfig object usage" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            img1 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, tol = 1.0e-5); verbosity = Silent()))
            config = ReconstructionConfig(; verbosity = Silent())
            img2 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, tol = 1.0e-5); config = config))
            # Iteration control lives on the method, so extending a `ReconstructionConfig` cannot change it;
            # only the run settings come from the config.
            config_base = ReconstructionConfig(; verbosity = Verbose())
            img3 = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 20, tol = 1.0e-5); config = config_base, verbosity = Silent()))

            # `maxit`/`tol`/`verbose` at `reconstruct` are rejected, not silently ignored.
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); maxit = 5)
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); tol = 1.0e-5)
            @test_throws ArgumentError reconstruct(acq_with_data, DirectReconstruction(); verbose = false)

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

            img_st = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = false, verbosity = Silent()))
            img_mt = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data; threaded = true, verbosity = Silent()))

            @test norm(img_st - img_mt) / norm(img_st) < 1.0e-10
        end

        @testset "Scaling strategies" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = BartScaling(), verbosity = Silent()))
            img_noscale = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = NoScaling(), verbosity = Silent()))
            img_meas = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1); scaling = MeasurementBasedScaling(), verbosity = Silent()))

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
            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_fixed = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1, tol = 0.0); scaling = FixedScaling(scale), verbosity = Silent()))
            img_bart = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); maxit = 1, tol = 0.0); scaling = BartScaling(), verbosity = Silent()))
            @test size(img_fixed) == size(img_bart)
        end
    end
end

@testitem "NamedDims Support" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using NamedDims

    @testset "NamedDims Support" begin
        @testset "NamedDims preservation" begin
            nx, ny, nc = 32, 32, 4

            ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, nx, ny, nc))
            smaps = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(nx, ny, nc))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_recon = test_type_stable(NamedDimsArray{(:x, :y), ComplexF32, 2, Matrix{ComplexF32}}, reconstruct(acq; verbosity = Silent()))

            @test dimnames(img_recon) == (:x, :y)
            @test eltype(img_recon) == ComplexF32
        end

        @testset "NamedDims with problem decomposition" begin
            nx, ny, nslices, nc = 16, 16, 3, 2

            ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, nx, ny, nc, nslices))
            smaps = NamedDimsArray{(:x, :y, :coil, :z)}(repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices))

            acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)

            img_direct = reconstruct(acq; verbosity = Silent())
            @test img_direct isa NamedDimsArray
            @test dimnames(img_direct) == (:x, :y, :z)
            @test size(img_direct) == (nx, ny, nslices)

            img_reg = reconstruct(acq, IterativeReconstruction(L2Image(0.01); maxit = 5); verbosity = Silent())
            @test img_reg isa NamedDimsArray
            @test dimnames(img_reg) == (:x, :y, :z)
            @test size(img_reg) == (nx, ny, nslices)
        end
    end
end

@testitem "Operator Options" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms
    using StructuredOptimization

    @testset "Operator Options" begin
        @testset "Operator normalization" begin
            nx, ny, nc = 32, 32, 4
            img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
            smaps = coil_sensitivities(nx, ny, nc)

            pdf = UniformRandomSampling(3.0, 0.1)
            pattern = create_sampling_pattern(pdf, (nx, ny))

            acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
            acq_with_data = simulate_acquisition(img_true, acq)

            # Only the output shape is checked here, so a single iteration is enough -- 20 iterations
            # bought no extra coverage, just a slower test.
            img_norm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); disable_operator_normalization = false, maxit = 1); verbosity = Silent()))
            img_unnorm = test_type_stable(Matrix{ComplexF32}, reconstruct(acq_with_data, IterativeReconstruction(L2Image(0.01); disable_operator_normalization = true, maxit = 1); verbosity = Silent()))

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

            img_decomp = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_problem_decomposition = false, verbosity = Silent()))
            img_no_decomp = test_type_stable(Array{ComplexF32, 3}, reconstruct(acq_ms; disable_problem_decomposition = true, verbosity = Silent()))

            # The two paths are the same computation in a different summation order, so they can
            # only agree to Float32 precision (eps ≈ 1.2e-7), and the order the reductions actually
            # take depends on threading. 1e-10 was below what the element type can deliver and made
            # this assertion flaky; 1e-5 still catches any real divergence between the paths.
            @test norm(img_decomp - img_no_decomp) / norm(img_decomp) < 1.0e-5

            # Regularized case: slices are identical, so the per-slice median scale equals
            # the global scale and both paths must converge to the same solution.
            img_decomp_reg = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 30); disable_problem_decomposition = false, verbosity = Silent())
            img_no_decomp_reg = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 30); disable_problem_decomposition = true, verbosity = Silent())

            @test norm(img_decomp_reg - img_no_decomp_reg) / norm(img_no_decomp_reg) < 1.0e-3
        end

        @testset "x₀ with problem decomposition" begin
            nx, ny, nslices, nc = 16, 16, 3, 2
            smaps = coil_sensitivities(nx, ny, nc)
            smaps_ms = repeat(smaps, 1, 1, 1, nslices)
            ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
            acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

            x₀ = zeros(ComplexF32, nx, ny, nslices)
            img_recon = reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀, verbosity = Silent())
            @test size(img_recon) == (nx, ny, nslices)

            x₀_wrong = zeros(ComplexF32, nx, ny)
            @test_throws ArgumentError reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀ = x₀_wrong, verbosity = Silent())

            # `reconstruct` must not write its solution back through the caller's `x₀`. The
            # component path handed the arrays straight to `Variable`, which stores them by
            # reference, so `solve`'s final write-back landed in the caller's arrays -- visible
            # whenever `scale == 1` skips the reallocating rescale, and through the `@view`s the
            # decomposition path passes down.
            x₀_keep = rand(ComplexF32, nx, ny, nslices)
            x₀_ref = copy(x₀_keep)
            reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); x₀ = x₀_keep, scaling = NoScaling(), verbosity = Silent())
            @test x₀_keep == x₀_ref
        end
    end
end

@testitem "Verbose and MultiThreading Decomposition" tags = [:reconstruction, :integration] setup = [TestHelpers] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: scale_regularization, Regularization, Scaling
    using LinearAlgebra
    using GeometricMedicalPhantoms

    @testset "Verbose progress output" begin
        nx, ny, nc = 16, 16, 2
        img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        smaps = coil_sensitivities(nx, ny, nc)
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        acq_with_data = simulate_acquisition(img_true, acq)

        output = IOBuffer()
        printfunc = (args...) -> print(output, args...)
        config = ReconstructionConfig(; verbosity = Verbose(; printfunc = printfunc))
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

        # Force MultiThreadingExecutor to cover that path in decomposition/execution.jl
        executor = MriReconstructionToolbox.MultiThreadingExecutor()
        img_recon = test_type_stable(
            Array{ComplexF32, 3},
            reconstruct(acq_ms, IterativeReconstruction(L2Image(0.01); maxit = 5); decomposition_executor = executor, verbosity = Silent()),
        )
        @test size(img_recon) == (nx, ny, nslices)
    end
end

@testitem "Per-slice threading gate" tags = [:reconstruction, :integration] begin
    using MriReconstructionToolbox
    using GeometricMedicalPhantoms
    const MRT = MriReconstructionToolbox

    nx, ny, nslices, nc = 128, 128, 2, 4
    smaps = repeat(coil_sensitivities(nx, ny, nc), 1, 1, 1, nslices)
    ksp = rand(ComplexF32, nx, ny, nc, nslices)
    acq = AcquisitionInfo(ksp; is3D = false, sensitivity_maps = smaps)
    method = IterativeReconstruction(L2Image(0.01f0); maxit = 5)

    config = ReconstructionConfig(; threaded = true, verbosity = Silent())
    plan = MRT.get_problem_decomposition_plan(acq, method, config)
    @test plan !== nothing

    # Two slices on an 8-thread process take the sequential executor, and one 128² slice is far
    # below `serial_blas_threshold_bytes()`, so the work inside a slice must stay serial even
    # though `config.threaded` is on.
    @test MRT.slice_bytes(plan, acq) == nx * ny * sizeof(ComplexF32)
    @test MRT.slice_bytes(plan, acq) < MRT.serial_blas_threshold_bytes()
    @test MRT.slice_threading(plan, acq, config, MRT.SequentialExecutor()) == false
    @test MRT.slice_threading(plan, acq, config, MRT.MultiThreadingExecutor()) == false

    # A slice large enough to pay for threading keeps it -- but only under the sequential
    # executor, and only when `config.threaded` is on.
    big_size = (2048, 2048, nslices)
    big = MRT.ProblemDecompositionPlan(
        big_size, (3,), (2048, 2048, nc, nslices), (4,), false, big_size
    )
    @test MRT.slice_bytes(big, acq) >= MRT.serial_blas_threshold_bytes()
    @test MRT.slice_threading(big, acq, config, MRT.SequentialExecutor()) == true
    @test MRT.slice_threading(big, acq, config, MRT.MultiThreadingExecutor()) == false
    @test MRT.slice_threading(
        big, acq, ReconstructionConfig(config; threaded = false), MRT.SequentialExecutor()
    ) == false

    # The gate is a performance decision only: the result may not depend on it.
    img_threaded = reconstruct(acq, method; threaded = true, verbosity = Silent())
    img_serial = reconstruct(acq, method; threaded = false, verbosity = Silent())
    @test size(img_threaded) == (nx, ny, nslices)
    @test img_threaded == img_serial
end

@testitem "Serial-BLAS threshold is settable" tags = [:reconstruction] begin
    using MriReconstructionToolbox
    const MRT = MriReconstructionToolbox

    @test MRT.serial_blas_threshold_bytes() == MRT.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES
    config = ReconstructionConfig(; threaded = true)
    @test MRT._should_thread_work_item(config, MRT.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES)
    @test !MRT._should_thread_work_item(config, 4 * 2^20)

    try
        MRT.set_serial_blas_threshold_bytes!(2^20)
        @test MRT.serial_blas_threshold_bytes() == 2^20
        # The gate follows the new value, which is the whole point of it being settable.
        @test MRT._should_thread_work_item(config, 4 * 2^20)
        @test !MRT._should_thread_work_item(config, 2^19)
        # `threaded = false` still vetoes, at any size.
        @test !MRT._should_thread_work_item(ReconstructionConfig(config; threaded = false), 2^30)
    finally
        MRT.set_serial_blas_threshold_bytes!(MRT.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES)
    end
    @test MRT.serial_blas_threshold_bytes() == MRT.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES

    @test_throws Exception MRT.set_serial_blas_threshold_bytes!(-1)
end

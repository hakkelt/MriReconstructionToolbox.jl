using TestItems

@testitem "Regularization terms in reconstruction" tags = [:regularization, :integration] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using NamedDims
    using Random

    Random.seed!(42)
    nx, ny, nt = 16, 16, 6
    img = zeros(ComplexF32, nx, ny, nt)
    for t in 1:nt
        img[4:12, 4:12, t] .= 1 + 0.2f0 * t
        img[6:8, 6:8, t] .= 2
    end

    ksp = zeros(ComplexF32, nx, ny, nt)
    for t in 1:nt
        acq_t = AcquisitionInfo(is3D = false, image_size = (nx, ny))
        ksp[:, :, t] = simulate_acquisition(img[:, :, t], acq_t).kspace_data
    end
    acq = AcquisitionInfo(NamedDimsArray{(:kx, :ky, :time)}(ksp); is3D = false)

    relerr(x) = norm(Array(x) .- img) / norm(img)

    @testset "LocallyLowRank" begin
        @test relerr(reconstruct(acq, LocallyLowRank(0.02f0; block_size = 4, time_dim = 3);
            maxit = 60, verbose = false)) < 0.1
    end

    @testset "TemporalTotalVariation" begin
        @test relerr(reconstruct(acq, TemporalTotalVariation(0.02f0; time_dim = 3);
            maxit = 100, verbose = false)) < 0.1
    end

    @testset "L+S with LocallyLowRank and TemporalTotalVariation components" begin
        components = (
            Component(:lowrank, LocallyLowRank(0.02f0; block_size = 4, time_dim = 3)),
            Component(:sparse, TemporalTotalVariation(0.02f0; time_dim = 3)),
        )
        img_recon = reconstruct(acq, components; maxit = 100, verbose = false)
        @test img_recon isa DecomposedImage
        @test relerr(img_recon) < 0.2
    end

    @testset "L+S with LowRank and JointSparsity components" begin
        components = (
            Component(:lowrank, LowRank(0.02f0; time_dim = 3)),
            Component(:sparse, L1Image(0.02f0)),
        )
        img_recon = reconstruct(acq, components; maxit = 100, verbose = false)
        @test img_recon isa DecomposedImage
        @test relerr(img_recon) < 0.2
    end

    # Regression: two components whose operators are both non-tight (here `LowRank`'s reshape and
    # the finite difference of temporal TV) used to fail to prepare for any algorithm, because
    # `Reshape` claimed `is_AAc_diagonal == true` without a `diag_AAc` method to back it up, so any
    # code path that trusted the trait crashed instead of falling back to a different assumption.
    @testset "LowRank + TemporalTotalVariation components" begin
        components = (
            Component(:lowrank, LowRank(0.02f0; time_dim = 3)),
            Component(:sparse, TemporalTotalVariation(0.02f0; time_dim = 3)),
        )
        img_recon = reconstruct(acq, components; maxit = 100, verbose = false)
        @test img_recon isa DecomposedImage
        @test relerr(img_recon) < 0.2
    end
end

@testitem "add_noise on plain arrays and NamedDimsArrays" tags = [:simulation] begin
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra
    using Statistics: std
    using Random

    Random.seed!(0)

    @testset "snr_db on a plain array" begin
        k = 5.0f0 .* ones(ComplexF32, 32, 32)
        noisy = add_noise(k; snr_db = 20)
        @test size(noisy) == size(k)
        @test eltype(noisy) == ComplexF32
        @test noisy != k
        # Empirical SNR should be roughly in the right ballpark.
        measured_snr_db = 20 * log10(norm(k) / norm(noisy - k))
        @test isapprox(measured_snr_db, 20; atol = 3)
    end

    @testset "noise_std on a plain array" begin
        k = zeros(ComplexF32, 64, 64)
        noisy = add_noise(k; noise_std = 0.1)
        measured_std = std(vec(noisy); corrected = false)
        @test isapprox(measured_std, 0.1; atol = 0.02)
    end

    @testset "NamedDimsArray preserves dimension names" begin
        k = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 16, 16, 4))
        noisy = add_noise(k; snr_db = 15)
        @test noisy isa NamedDimsArray
        @test dimnames(noisy) == (:kx, :ky, :coil)
        @test unname(noisy) != unname(k)
    end

    @testset "exactly one of snr_db / noise_std required" begin
        k = rand(ComplexF32, 8, 8)
        @test_throws ArgumentError add_noise(k)
        @test_throws ArgumentError add_noise(k; snr_db = 10, noise_std = 0.1)
    end

    @testset "reproducible with an explicit rng" begin
        k = rand(ComplexF32, 16, 16)
        a = add_noise(k; snr_db = 20, rng = MersenneTwister(42))
        b = add_noise(k; snr_db = 20, rng = MersenneTwister(42))
        @test a == b
    end
end

@testitem "add_noise on AcquisitionInfo (Cartesian and non-Cartesian)" tags = [:simulation, :acquisition, :nfft] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo
    using Random

    Random.seed!(1)

    @testset "CartesianAcquisitionInfo: returns a copy, original untouched" begin
        ksp = rand(ComplexF32, 32, 32)
        acq = AcquisitionInfo(ksp; is3D = false)
        noisy_acq = add_noise(acq; snr_db = 25)
        @test noisy_acq isa CartesianAcquisitionInfo
        @test noisy_acq.kspace_data != acq.kspace_data
        @test acq.kspace_data == ksp # original untouched
    end

    @testset "NonCartesianAcquisitionInfo: returns a copy" begin
        traj = radial_trajectory(24, 8)
        ksp = rand(ComplexF32, 24, 8)
        acq = NonCartesianAcquisitionInfo(ksp; trajectory = traj, image_size = (16, 16))
        noisy_acq = add_noise(acq; noise_std = 0.01)
        @test noisy_acq isa NonCartesianAcquisitionInfo
        @test noisy_acq.kspace_data != acq.kspace_data
        @test acq.kspace_data == ksp
        @test noisy_acq.trajectory === acq.trajectory
    end

    @testset "errors without k-space data" begin
        traj = radial_trajectory(16, 4)
        acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (16, 16))
        @test_throws ArgumentError add_noise(acq; snr_db = 20)
    end
end

@testitem "radial_trajectory" tags = [:simulation, :nfft] begin
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        traj = radial_trajectory(64, 32)
        @test size(traj) == (2, 64, 32)
        @test dimnames(traj) == (:coord, :sample, :spoke)
        @test all(x -> -0.5 <= x < 0.5, unname(traj))
    end

    @testset "orderings differ and stay within bounds" begin
        for ordering in (:linear, :golden_angle, :tiny_golden_angle)
            traj = radial_trajectory(32, 16; ordering)
            @test size(traj) == (2, 32, 16)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
        @test unname(radial_trajectory(8, 4; ordering = :linear)) != unname(radial_trajectory(8, 4; ordering = :golden_angle))
    end

    @testset "unknown ordering errors" begin
        @test_throws ArgumentError radial_trajectory(8, 4; ordering = :bogus)
    end

    @testset "simulate_acquisition + direct NFFT reconstruction is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny = 48, 48
        img = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = radial_trajectory(72, 150; ordering = :golden_angle)
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny))
        data = simulate_acquisition(img, acq)
        @test size(data.kspace_data) == (72, 150)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 15))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        @test norm(α .* a .- r) / norm(r) < 0.35
    end
end

@testitem "stack_of_stars_trajectory" tags = [:simulation, :nfft] begin
    using MriReconstructionToolbox
    using NamedDims

    @testset "shape, dimension names and partition grid" begin
        traj = stack_of_stars_trajectory(48, 24, 6)
        @test size(traj) == (3, 48, 24, 6)
        @test dimnames(traj) == (:coord, :sample, :spoke, :partition)
        raw = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, raw)
        # in-plane pattern is identical across partitions
        @test raw[1:2, :, :, 1] == raw[1:2, :, :, end]
        # partition (kz) coordinate varies across partitions and is constant within one
        @test length(unique(raw[3, 1, 1, :])) == 6
        @test all(==(raw[3, 1, 1, 3]), raw[3, :, :, 3])
    end
end

@testitem "kooshball_trajectory" tags = [:simulation, :nfft] begin
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        traj = kooshball_trajectory(32, 500)
        @test size(traj) == (3, 32, 500)
        @test dimnames(traj) == (:coord, :sample, :spoke)
        raw = unname(traj)
        @test all(x -> -0.5 <= x < 0.5, raw)
        # spoke directions should cover the sphere roughly isotropically: the mean direction of
        # the outermost sample over many spokes should be close to zero.
        outer = raw[:, end, :]
        dirs = outer ./ mapslices(norm, outer; dims = 1)
        @test norm(sum(dirs; dims = 2)) / size(dirs, 2) < 0.15
    end

    @testset "simulate_acquisition + direct NFFT reconstruction of a 3D phantom is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny, nz = 24, 24, 24
        img = create_shepp_logan_phantom(nx, ny, nz; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = kooshball_trajectory(32, 2000)
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny, nz))
        data = simulate_acquisition(img, acq)
        @test size(data.kspace_data) == (32, 2000)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 10))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        # A coarse gridded (non-iterative) reconstruction of a heavily undersampled 3D kooshball
        # is not expected to be very accurate; this only checks it is not garbage.
        @test norm(α .* a .- r) / norm(r) < 0.7
    end
end

@testitem "spiral_trajectory" tags = [:simulation, :nfft] begin
    using MriReconstructionToolbox
    using NamedDims
    using LinearAlgebra

    @testset "shape, dimension names and k-space extent" begin
        for variant in (:archimedean, :variable_density)
            traj = spiral_trajectory(128, 6; variant, nturns = 8)
            @test size(traj) == (2, 128, 6)
            @test dimnames(traj) == (:coord, :sample, :interleave)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
    end

    @testset "variable density oversamples the center relative to archimedean" begin
        arch = unname(spiral_trajectory(64, 1; variant = :archimedean, nturns = 8))
        vd = unname(spiral_trajectory(64, 1; variant = :variable_density, density_exponent = 0.5, nturns = 8))
        r_arch = sqrt.(arch[1, :, 1] .^ 2 .+ arch[2, :, 1] .^ 2)
        r_vd = sqrt.(vd[1, :, 1] .^ 2 .+ vd[2, :, 1] .^ 2)
        # variable-density spiral grows its radius more slowly at the start of the arm
        @test r_vd[8] < r_arch[8]
    end

    @testset "unknown variant errors" begin
        @test_throws ArgumentError spiral_trajectory(16, 2; variant = :bogus)
    end

    @testset "simulate_acquisition + direct NFFT reconstruction is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny = 48, 48
        img = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = spiral_trajectory(1024, 6; nturns = 12)
        acq = AcquisitionInfo(; trajectory = traj, image_size = (nx, ny))
        data = simulate_acquisition(img, acq)
        acq_dcf = density_compensation(data; method = PipeMenonDCF(maxit = 15))
        rec = reconstruct(acq_dcf; verbosity = Silent())

        a, r = abs.(rec), abs.(img)
        α = sum(a .* r) / sum(abs2, a)
        @test norm(α .* a .- r) / norm(r) < 0.35
    end
end

@testitem "AcquisitionInfo non-Cartesian construction needs no k-space placeholder" tags = [:acquisition, :nfft] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo

    traj = radial_trajectory(16, 4)
    smaps = coil_sensitivities(16, 16, 3)

    # No positional k-space argument at all, and `kspace_data = nothing` explicitly: both work
    # without inventing a placeholder array.
    acq1 = AcquisitionInfo(; trajectory = traj, image_size = (16, 16), sensitivity_maps = smaps)
    @test acq1 isa NonCartesianAcquisitionInfo
    @test isnothing(acq1.kspace_data)

    acq2 = AcquisitionInfo(nothing; trajectory = traj, image_size = (16, 16))
    @test isnothing(acq2.kspace_data)
end

@testitem "subsampling: tuple of indexing expressions (default idiom)" tags = [:acquisition, :simulation] begin
    using MriReconstructionToolbox

    @testset "partial Fourier as a UnitRange" begin
        nx, ny = 32, 32
        pf_pattern = (:, 1:round(Int, 0.65 * ny))
        acq = AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), subsampling = pf_pattern)
        @test acq.subsampling === pf_pattern

        img = rand(ComplexF32, nx, ny)
        data = simulate_acquisition(img, acq)
        @test size(data.kspace_data) == (nx, round(Int, 0.65 * ny))
        rec = reconstruct(data, DirectReconstruction(); verbosity = Silent())
        @test size(rec) == (nx, ny)
    end

    @testset "GRAPPA-style uniform undersampling with an ACS block, as a StepRange ∪ UnitRange" begin
        nx, ny = 32, 32
        R, acs_half_width = 4, 3
        center = div(ny, 2)
        grappa_lines = sort(union(1:R:ny, (center - acs_half_width):(center + acs_half_width)))
        grappa_pattern = (:, grappa_lines)
        acq = AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), subsampling = grappa_pattern)

        img = rand(ComplexF32, nx, ny)
        data = simulate_acquisition(img, acq)
        @test size(data.kspace_data) == (nx, length(grappa_lines))
        rec = reconstruct(data, DirectReconstruction(); verbosity = Silent())
        @test size(rec) == (nx, ny)
    end
end

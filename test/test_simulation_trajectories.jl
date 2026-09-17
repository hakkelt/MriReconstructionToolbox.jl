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

@testitem "estimate_snr and the image-domain `snr` keyword" tags = [:simulation, :analysis] begin
    using MriReconstructionToolbox
    using NamedDims
    using Random
    using Statistics: std

    Random.seed!(0)

    # A disc of signal on an empty background, wide enough to contain the centred signal box and
    # far enough from the corners to leave them signal-free.
    n = 128
    image = ComplexF32[
        (i - n ÷ 2)^2 + (j - n ÷ 2)^2 < (n ÷ 3)^2 ? 1 : 0 for i in 1:n, j in 1:n
    ]

    @testset "add_noise(; snr) round-trips through estimate_snr" begin
        # Fixed boxes measure the noise where there is no signal, so there is no threshold to clip
        # the background tail and the round trip holds at low SNR too.
        for target in (5, 10, 20, 40, 100)
            noisy = add_noise(image; snr = target, rng = MersenneTwister(1))
            @test isapprox(estimate_snr(noisy), target; rtol = 0.1)
        end
    end

    @testset "the box sizes are in voxels and the corners can be chosen" begin
        noisy = add_noise(image; snr = 20, rng = MersenneTwister(4))
        @test isapprox(estimate_snr(noisy; signal_box = 24, noise_box = 12), 20; rtol = 0.15)
        @test isapprox(estimate_snr(noisy; signal_box = (24, 16)), 20; rtol = 0.15)
        # One corner holds a quarter of the noise samples and still measures the same noise.
        @test isapprox(estimate_snr(noisy; corners = 1), estimate_snr(noisy); rtol = 0.15)
        @test isapprox(estimate_snr(noisy; corners = (1, 4)), estimate_snr(noisy); rtol = 0.15)
    end

    @testset "a noiseless image has no background noise" begin
        @test estimate_snr(image) == Inf
    end

    @testset "NamedDimsArray input" begin
        noisy = add_noise(NamedDimsArray{(:x, :y)}(image); snr = 50, rng = MersenneTwister(2))
        @test noisy isa NamedDimsArray
        @test isapprox(estimate_snr(noisy), 50; rtol = 0.1)
    end

    @testset "snr_masks is the pair of regions estimate_snr measures over" begin
        noisy = add_noise(image; snr = 50, rng = MersenneTwister(3))
        sig, noise = snr_masks(noisy)
        # A centred box of n ÷ 8 a side, and four corner boxes of the same size.
        @test count(sig) == (n ÷ 8)^2
        @test count(noise) == 4 * (n ÷ 8)^2
        @test !any(sig .& noise)
        # The signal box lands on the disc and the corner boxes land off it.
        @test all(!iszero, image[sig])
        @test all(iszero, image[noise])
        # Measuring by hand over those masks reproduces `estimate_snr`.
        mag = abs.(noisy)
        by_hand = sqrt(2 - π / 2) * (sum(mag[sig]) / count(sig)) / std(mag[noise])
        @test by_hand ≈ estimate_snr(noisy)
    end

    @testset "argument checking" begin
        @test_throws ArgumentError add_noise(image; snr = 20, noise_std = 0.1)
        @test_throws ArgumentError add_noise(image; snr = -1)
        # A box that does not fit twice along a dimension would meet its opposite number.
        @test_throws ArgumentError estimate_snr(image; noise_box = 100)
        # A signal box that reaches into the corners is not a signal box.
        @test_throws ArgumentError estimate_snr(image; signal_box = 120, noise_box = 16)
        @test_throws ArgumentError estimate_snr(image; corners = 5)
        @test_throws ArgumentError estimate_snr(image; signal_box = (8, 8, 8))
        # `snr` is image-domain, so it is rejected on an acquisition's k-space.
        acq = AcquisitionInfo(; is3D = false, image_size = (16, 16))
        data = simulate_acquisition(image[1:16, 1:16], acq)
        @test_throws ArgumentError add_noise(data; snr = 20)
    end
end

@testitem "add_noise on AcquisitionInfo (Cartesian and non-Cartesian)" tags = [:simulation, :acquisition, :nfft] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
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
        for ordering in (LinearOrdering(), GoldenAngle(), TinyGoldenAngle(), TinyGoldenAngle(3))
            traj = radial_trajectory(32, 16; ordering)
            @test size(traj) == (2, 32, 16)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
        @test unname(radial_trajectory(8, 4; ordering = LinearOrdering())) != unname(radial_trajectory(8, 4; ordering = GoldenAngle()))
        # Index 1 of the tiny family *is* the standard golden angle.
        @test unname(radial_trajectory(8, 4; ordering = TinyGoldenAngle(1))) ≈ unname(radial_trajectory(8, 4; ordering = GoldenAngle()))
        # Consecutive tiny-golden-angle spokes stay closer together than golden-angle ones.
        tiny = unname(radial_trajectory(2, 2; ordering = TinyGoldenAngle(4)))
        golden = unname(radial_trajectory(2, 2; ordering = GoldenAngle()))
        @test norm(tiny[:, :, 2] - tiny[:, :, 1]) < norm(golden[:, :, 2] - golden[:, :, 1])
    end

    @testset "an ordering is a type, not a symbol" begin
        @test_throws TypeError radial_trajectory(8, 4; ordering = :golden_angle)
        @test_throws ArgumentError TinyGoldenAngle(0)
    end

    @testset "simulate_acquisition + direct NFFT reconstruction is sane" begin
        using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
        nx, ny = 48, 48
        img = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
        traj = radial_trajectory(72, 150; ordering = GoldenAngle())
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
        for variant in (Archimedean(), VariableDensity(), VariableDensity(0.5))
            traj = spiral_trajectory(128, 6; variant, nturns = 8)
            @test size(traj) == (2, 128, 6)
            @test dimnames(traj) == (:coord, :sample, :interleave)
            @test all(x -> -0.5 <= x < 0.5, unname(traj))
        end
    end

    @testset "variable density oversamples the center relative to archimedean" begin
        arch = unname(spiral_trajectory(64, 1; variant = Archimedean(), nturns = 8))
        vd = unname(spiral_trajectory(64, 1; variant = VariableDensity(2.0), nturns = 8))
        r_arch = sqrt.(arch[1, :, 1] .^ 2 .+ arch[2, :, 1] .^ 2)
        r_vd = sqrt.(vd[1, :, 1] .^ 2 .+ vd[2, :, 1] .^ 2)
        # A variable-density spiral with exponent > 1 grows its radius more slowly at the start
        # of the arm, packing more samples near the center than the Archimedean spiral.
        @test r_vd[8] < r_arch[8]
        # Exponent 1 is the Archimedean spiral.
        @test unname(spiral_trajectory(64, 1; variant = VariableDensity(1), nturns = 8)) ≈ arch
    end

    @testset "a variant is a type, not a symbol" begin
        @test_throws TypeError spiral_trajectory(16, 2; variant = :archimedean)
        @test_throws ArgumentError VariableDensity(0)
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

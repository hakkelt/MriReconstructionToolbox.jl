@testitem "Density compensation on Cartesian errors" tags = [:acquisition] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    nx, ny = 16, 16
    ksp = zeros(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(ksp; is3D = false)
    @test_throws ArgumentError density_compensation(acq)
    @test_throws ArgumentError density_compensation(acq; method = VoronoiDCF())
end

@testitem "PipeMenonDCF density compensation" tags = [:acquisition, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, NonCartesianAcquisitionInfo
    using NamedDims

    nsamp, nspokes = 32, 16
    traj = zeros(Float32, 2, nsamp, nspokes)
    for s in 1:nspokes
        θ = Float32((s - 1) * π / nspokes)
        for k in 1:nsamp
            r = Float32((k - 1 - nsamp / 2) / nsamp * 0.99)
            traj[1, k, s] = r * cos(θ)
            traj[2, k, s] = r * sin(θ)
        end
    end

    acq = NonCartesianAcquisitionInfo(
        trajectory = traj,
        image_size = (32, 32),
    )
    @test isnothing(acq.dcf)

    acq_dcf = density_compensation(acq; method = PipeMenonDCF(maxit = 15))
    @test !isnothing(acq_dcf.dcf)
    @test size(acq_dcf.dcf) == (nsamp, nspokes)
    @test eltype(acq_dcf.dcf) === Float32
    @test all(acq_dcf.dcf .> 0)

    # NamedDims preservation
    traj_named = NamedDimsArray{(:coord, :kx, :ky)}(traj)
    acq_named = NonCartesianAcquisitionInfo(
        trajectory = traj_named,
        image_size = (32, 32),
    )
    acq_named_dcf = density_compensation(acq_named)
    @test acq_named_dcf.dcf isa NamedDimsArray
    @test dimnames(acq_named_dcf.dcf) == (:kx, :ky)
end

@testitem "VoronoiDCF density compensation" tags = [:acquisition] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, NonCartesianAcquisitionInfo
    using NamedDims

    nsamp, nspokes = 32, 16
    traj = zeros(Float32, 2, nsamp, nspokes)
    for s in 1:nspokes
        θ = Float32((s - 1) * π / nspokes)
        for k in 1:nsamp
            r = Float32((k - 1 - nsamp / 2) / nsamp * 0.99)
            traj[1, k, s] = r * cos(θ)
            traj[2, k, s] = r * sin(θ)
        end
    end

    acq = NonCartesianAcquisitionInfo(
        trajectory = traj,
        image_size = (32, 32),
    )
    acq_vor = density_compensation(acq; method = VoronoiDCF())
    @test !isnothing(acq_vor.dcf)
    @test size(acq_vor.dcf) == (nsamp, nspokes)
    @test eltype(acq_vor.dcf) === Float32
    @test all(acq_vor.dcf .> 0)

    # 3D trajectory with VoronoiDCF throws ArgCheck error
    traj_3d = zeros(Float32, 3, nsamp, nspokes)
    acq_3d = NonCartesianAcquisitionInfo(
        trajectory = traj_3d,
        image_size = (32, 32, 32),
    )
    @test_throws ArgumentError density_compensation(acq_3d; method = VoronoiDCF())
end

@testitem "DCF radial reconstruction accuracy" tags = [:reconstruction, :nfft, :quality] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, NonCartesianAcquisitionInfo
    using LinearAlgebra
    using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities

    nx, ny = 32, 32
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

    nsamp, nspokes = 64, 32
    traj = zeros(Float32, 2, nsamp, nspokes)
    for s in 1:nspokes
        θ = Float32((s - 1) * π / nspokes)
        for k in 1:nsamp
            r = Float32((k - 1 - nsamp / 2) / nsamp * 0.99)
            traj[1, k, s] = r * cos(θ)
            traj[2, k, s] = r * sin(θ)
        end
    end

    acq_dummy = NonCartesianAcquisitionInfo(
        zeros(ComplexF32, nsamp, nspokes);
        trajectory = traj,
        image_size = (nx, ny),
    )
    𝒜 = get_encoding_operator(acq_dummy)
    ksp = 𝒜 * img_true

    acq = NonCartesianAcquisitionInfo(
        ksp;
        trajectory = traj,
        image_size = (nx, ny),
    )

    acq_pm = density_compensation(acq; method = PipeMenonDCF(maxit = 20))
    @test !isnothing(acq_pm.dcf)
    rec_pm = reconstruct(acq_pm, DirectReconstruction(); verbosity = Silent())
    @test size(rec_pm) == (nx, ny)

    acq_vor = density_compensation(acq; method = VoronoiDCF())
    @test !isnothing(acq_vor.dcf)
    rec_vor = reconstruct(acq_vor, DirectReconstruction(); verbosity = Silent())
    @test size(rec_vor) == (nx, ny)

    # Both DCF reconstructions should correlate highly with ground truth
    corr_pm = abs(dot(vec(rec_pm), vec(img_true))) / (norm(rec_pm) * norm(img_true))
    corr_vor = abs(dot(vec(rec_vor), vec(img_true))) / (norm(rec_vor) * norm(img_true))
    @test corr_pm > 0.85
    @test corr_vor > 0.5
end

@testitem "DCF edge correction" tags = [:acquisition, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using NamedDims

    traj = radial_trajectory(128, 64; ordering = GoldenAngle())
    unnamed(a) = a isa NamedDimsArray ? parent(a) : a

    # How far the outermost weight of each spoke sits from the ramp the interior is on, as a
    # fraction of it: a smooth continuation gives ≈ 0.
    function edge_deviation(w)
        return [
            let predicted = 2 * Float64(w[6, s]) - Float64(w[7, s])
                    abs(Float64(w[1, s]) - predicted) / abs(predicted)
            end for s in axes(w, 2)
        ]
    end

    @testset "the jump at the end of each readout is removed" begin
        for method in (VoronoiDCF, PipeMenonDCF)
            raw = unnamed(MriReconstructionToolbox.compute_dcf(traj, (128, 128), method(; edge_correction = false)))
            fixed = unnamed(MriReconstructionToolbox.compute_dcf(traj, (128, 128), method()))

            # Voronoi's unbounded outer cell is the extreme case (tens of times the right weight);
            # Pipe-Menon's one-sided neighbourhood is milder but still tens of percent off.
            @test maximum(edge_deviation(raw)) > 0.4
            @test maximum(edge_deviation(fixed)) < 0.1
            @test all(>=(0), fixed)
        end
    end

    @testset "only the ends of a readout are touched" begin
        raw = unnamed(MriReconstructionToolbox.compute_dcf(traj, (128, 128), VoronoiDCF(; edge_correction = false)))
        fixed = unnamed(MriReconstructionToolbox.compute_dcf(traj, (128, 128), VoronoiDCF()))
        @test fixed[4:(end - 3), :] == raw[4:(end - 3), :]
    end

    @testset "correct_dcf_edges extrapolates the trend and leaves a short readout alone" begin
        ramp = repeat(Float64.(1:20), 1, 3)
        ramp[1, :] .= 99.0
        ramp[end, :] .= -5.0
        fixed = correct_dcf_edges(ramp; edge_samples = 1, fit_samples = 5)
        @test fixed[1, :] ≈ fill(1.0, 3)
        @test fixed[end, :] ≈ fill(20.0, 3)
        @test fixed[2:(end - 1), :] == ramp[2:(end - 1), :]

        short = Float64.(1:8)
        @test correct_dcf_edges(short; edge_samples = 3, fit_samples = 8) === short
    end
end

@testitem "Scaling rules" tags = [:reconstruction] setup = [TestHelpers] begin
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_scale, get_encoding_operator, _measurement, _quantile_select!
    using LinearAlgebra, Random, Statistics
    using GeometricMedicalPhantoms

    nx, ny, nc = 32, 32, 4
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    pattern = create_sampling_pattern(UniformRandomSampling(3.0, 0.1), (nx, ny))
    acq = simulate_acquisition(img_true, AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern))
    A = get_encoding_operator(acq; threaded = false)
    x̂ = A' * _measurement(acq.kspace_data)

    @testset "the default is QuantileScaling" begin
        @test ReconstructionConfig().scaling == QuantileScaling()
        @test QuantileScaling().p == 0.99
        @test_throws ArgumentError QuantileScaling(0.0)
        @test_throws ArgumentError QuantileScaling(1.5)
        @test_throws ArgumentError KSpaceNormScaling(0)
    end

    @testset "every data-dependent scale is degree-1 homogeneous in the data" begin
        # A power of two scales every Float32 exactly, so no rounding separates the two sides.
        c = 4.0f0
        acq_c = AcquisitionInfo(acq; kspace_data = c .* acq.kspace_data)
        x̂_c = A' * _measurement(acq_c.kspace_data)
        for s in (
                QuantileScaling(), QuantileScaling(0.9), BartScaling(), MaxScaling(), StdScaling(),
                NoiseLevelScaling(), MeasurementBasedScaling(), KSpaceNormScaling(),
            )
            @test get_scale(s, acq_c, x̂_c, A) ≈ c * get_scale(s, acq, x̂, A) rtol = 1.0e-5
        end
        # The system-matrix rule reads the operator only.
        @test get_scale(SystemMatrixBasedScaling(), acq_c, x̂_c, A) == get_scale(SystemMatrixBasedScaling(), acq, x̂, A)
    end

    @testset "each rule computes what it names" begin
        a = abs.(vec(x̂))
        @test get_scale(QuantileScaling(), acq, x̂, A) == quantile(a, 0.99)
        @test get_scale(MaxScaling(), acq, x̂, A) == maximum(a)
        @test get_scale(StdScaling(), acq, x̂, A) ≈ std(vec(x̂)) rtol = 1.0e-5
        @test get_scale(KSpaceNormScaling(), acq, x̂, A) ≈ norm(acq.kspace_data) / 100 rtol = 1.0e-6
        # A unitary operator has trace(𝒜ᴴ𝒜)/N = 1, whatever the probes.
        @test get_scale(SystemMatrixBasedScaling(), acq, x̂, MriReconstructionToolbox.AbstractOperators.Eye(ComplexF32, size(x̂))) ≈ 1
        # The noise level of pure noise is its (complex) standard deviation.
        σ = 0.3f0
        noise = σ .* randn(Xoshiro(1), ComplexF32, 256, 256)
        @test get_scale(NoiseLevelScaling(), acq, noise, nothing) ≈ σ rtol = 0.03
    end

    @testset "the subsampled quantile tracks the exact one, and ignores a few hot voxels" begin
        v = abs.(randn(Xoshiro(2), ComplexF32, 128, 64, 64))
        q = get_scale(QuantileScaling(), acq, v, nothing)
        @test q ≈ quantile(vec(v), 0.99) rtol = 0.01
        hot = copy(v)
        hot[randperm(Xoshiro(3), length(v))[1:(length(v) ÷ 1000)]] .= 100 * maximum(v)
        @test get_scale(QuantileScaling(), acq, hot, nothing) ≈ q rtol = 0.02
        @test get_scale(MaxScaling(), acq, hot, nothing) > 50 * q
        @test MriReconstructionToolbox._next_prime.(0:12) == [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13]
    end

    @testset "reconstruction is equivariant to the data's intensity under the default" begin
        method = IterativeReconstruction(L1Image(0.01); maxit = 10, reltol = 0.0)
        x1 = reconstruct(acq, method; verbosity = Silent(), threaded = false)
        c = 250.0f0
        x2 = reconstruct(AcquisitionInfo(acq; kspace_data = c .* acq.kspace_data), method; verbosity = Silent(), threaded = false)
        @test x2 ≈ c .* x1 rtol = 1.0e-5
    end
end

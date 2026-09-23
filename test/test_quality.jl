@testitem "Aqua" tags = [:quality, :aqua] begin
    using Aqua
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    Aqua.test_all(
        MriReconstructionToolbox;
        ambiguities = false,
        piracies = false,
        persistent_tasks = false,
        stale_deps = false,
    )
    # The vendored dependencies live in submodules of this package, so a recursive ambiguity
    # check would report their ambiguities as ours. They are checked -- and, where they are
    # unavoidable consequences of the operator syntax, tolerated -- by their own test suites,
    # so only this package's own methods are checked here.
    Aqua.test_ambiguities(MriReconstructionToolbox; recursive = false)
end

@testitem "JET test_package" tags = [:quality, :jet] begin
    using JET
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    # `LastFrameModuleExact`, not the bare module: the vendored dependencies are submodules of
    # this one, and a plain module target reports their inference problems as ours.
    JET.test_package(
        MriReconstructionToolbox;
        target_modules = (JET.LastFrameModuleExact(MriReconstructionToolbox),),
    )
end

@testitem "JET exported API @test_opt" tags = [:quality, :jet] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using JET
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate, NonCartesianAcquisitionInfo
    using MriReconstructionToolbox.AbstractOperators
    using NamedDims

    # The target of every `@test_call`/`@test_opt` below: this package's own frames only. A bare
    # module would also match the vendored dependencies, which are submodules of it, and report
    # their inference problems -- such as the keyword splat in `override_parameters` -- as ours.
    const MRT = JET.LastFrameModuleExact(MriReconstructionToolbox)
    nx, ny, nc = 8, 8, 2

    img = rand(ComplexF32, nx, ny)
    ksp = rand(ComplexF32, nx, ny)
    ksp_coil = rand(ComplexF32, nx, ny, nc)
    smaps = coil_sensitivities(nx, ny, nc)
    wrapped_img = NamedDimsArray{(:x, :y)}(img)
    wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
    wrapped_ksp_coil = NamedDimsArray{(:kx, :ky, :coil)}(ksp_coil)
    wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
    mask = trues(nx, ny)
    cart_info = AcquisitionInfo(ksp_coil; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps)
    cart_info_named = AcquisitionInfo(wrapped_ksp_coil; image_size = (nx, ny), sensitivity_maps = wrapped_smaps)
    trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
    dcf = rand(Float32, 12, 2)
    noncart_ksp = rand(ComplexF32, 12, 2)
    noncart_info = AcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_opt target_modules = (MRT,) AcquisitionInfo(ksp; is3D = false, image_size = (nx, ny))
    @test_opt target_modules = (MRT,) CartesianAcquisitionInfo(ksp; is3D = false, image_size = (nx, ny), subsampling = mask)
    @test_opt target_modules = (MRT,) NonCartesianAcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_opt target_modules = (MRT,) get_subsampling_operator(rand(ComplexF32, count(mask)), (nx, ny), mask)

    @test_opt target_modules = (MRT,) L2Image(0.1)
    @test_opt target_modules = (MRT,) L1Image(0.1)
    @test_opt target_modules = (MRT,) L1Wavelet2D(0.1)
    @test_opt target_modules = (MRT,) L1Wavelet3D(0.1)
    @test_opt target_modules = (MRT,) TotalVariation2D(0.1)
    @test_opt target_modules = (MRT,) TotalVariation3D(0.1)
    @test_opt target_modules = (MRT,) L1TemporalFourier(0.1; time_dim = 2)
    @test_opt target_modules = (MRT,) LowRank(0.1; time_dim = 2)
    @test_opt target_modules = (MRT,) RankLimit(2; time_dim = 2)

    @test_opt target_modules = (MRT,) UniformRandomSampling(2.0, 0.1)
    @test_opt target_modules = (MRT,) VariableDensitySampling(GaussianDistribution(), 2.0)
    @test_opt target_modules = (MRT,) VariableDensitySampling(PolynomialDistribution(), 2.0)
    @test_opt target_modules = (MRT,) PoissonDiskSampling(2.0)
    @test_opt target_modules = (MRT,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_opt target_modules = (MRT,) coil_sensitivities(nx, ny, nc)
    @test_opt target_modules = (MRT,) ReconstructionConfig(; verbosity = Silent())
    @test_opt target_modules = (MRT,) SequentialExecutor()
    @test_opt target_modules = (MRT,) MultiThreadingExecutor()
    @test_opt target_modules = (MRT,) BartScaling()
    @test_opt target_modules = (MRT,) MeasurementBasedScaling()
    @test_opt target_modules = (MRT,) NoScaling()
end

@testitem "JET exported API @test_call" tags = [:quality, :jet] begin
    using JET
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator, calculate
    using MriReconstructionToolbox.AbstractOperators
    using NamedDims

    # The target of every `@test_call`/`@test_opt` below: this package's own frames only. A bare
    # module would also match the vendored dependencies, which are submodules of it, and report
    # their inference problems -- such as the keyword splat in `override_parameters` -- as ours.
    const MRT = JET.LastFrameModuleExact(MriReconstructionToolbox)
    nx, ny, nc = 8, 8, 2

    img = rand(ComplexF32, nx, ny)
    ksp = rand(ComplexF32, nx, ny)
    ksp_coil = rand(ComplexF32, nx, ny, nc)
    smaps = coil_sensitivities(nx, ny, nc)
    wrapped_ksp = NamedDimsArray{(:kx, :ky)}(ksp)
    wrapped_smaps = NamedDimsArray{(:x, :y, :coil)}(smaps)
    mask = trues(nx, ny)
    cart_info = AcquisitionInfo(ksp_coil; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps)
    trajectory = rand(Float32, 2, 12, 2) .- 0.5f0
    dcf = rand(Float32, 12, 2)
    noncart_ksp = rand(ComplexF32, 12, 2)
    noncart_info = AcquisitionInfo(noncart_ksp; trajectory, dcf, image_size = (nx, ny))

    @test_call target_modules = (MRT,) AcquisitionInfo(ksp; is3D = false, image_size = (nx, ny))
    @test_call target_modules = (MRT,) get_fourier_operator(ksp, false; threaded = false)
    @test_call target_modules = (MRT,) get_fourier_operator(wrapped_ksp; threaded = false)
    @test_call target_modules = (MRT,) get_fourier_operator(noncart_info; threaded = false)
    @test_call target_modules = (MRT,) get_sensitivity_map_operator(smaps, false; threaded = false)
    @test_call target_modules = (MRT,) get_sensitivity_map_operator(wrapped_smaps; threaded = false)
    @test_call target_modules = (MRT,) get_subsampling_operator(rand(ComplexF32, count(mask)), (nx, ny), mask)
    @test_call target_modules = (MRT,) get_encoding_operator(cart_info; threaded = false)
    @test_call target_modules = (MRT,) get_encoding_operator(noncart_info; threaded = false)
    @test_call target_modules = (MRT,) calculate(L2Image(0.1), img; threaded = false)
    @test_call target_modules = (MRT,) build_model(Eye(img), img, L2Image(0.1); threaded = false)
    @test_call target_modules = (MRT,) create_sampling_pattern(UniformRandomSampling(2.0, 0.1), (nx, ny))
    @test_call target_modules = (MRT,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_call target_modules = (MRT,) coil_sensitivities(nx, ny, nc)
    @test_call target_modules = (MRT,) simulate_acquisition(img, AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps))
    @test_call target_modules = (MRT,) ReconstructionConfig(; verbosity = Silent())
    @test_call target_modules = (MRT,) DirectReconstruction()
    @test_call target_modules = (MRT,) IterativeReconstruction(L2Image(0.01))
    @test_call target_modules = (MRT,) reconstruct(cart_info, IterativeReconstruction(L2Image(0.01); maxit = 2); threaded = false, verbosity = Silent())
    @test_call target_modules = (MRT,) reconstruct(cart_info; threaded = false, verbosity = Silent())
    @test_call target_modules = (MRT,) SequentialExecutor()
    @test_call target_modules = (MRT,) MultiThreadingExecutor()
    @test_call target_modules = (MRT,) BartScaling()
    @test_call target_modules = (MRT,) MeasurementBasedScaling()
    @test_call target_modules = (MRT,) NoScaling()
end

@testitem "Benchmark case catalog smoke test" tags = [:quality] begin
    using MriReconstructionToolbox

    # The case catalog and MRT's reconstruction of each method, at the reduced size, one method per
    # case family: this guards against the harness (benchmark/run.jl) and the comparison suite
    # breaking on an API change, without paying for a benchmark run.
    include(joinpath(pkgdir(MriReconstructionToolbox), "benchmark", "utils", "bench_utils.jl"))
    const BU = BenchUtils
    withenv("MRT_BENCH_SMALL" => "1", "MRT_BENCH_REAL_DATA" => "0") do
        for (id, method) in (
                ("shepp_logan_2d_1ch_cartesian", :tv),
                ("shepp_logan_2d_8ch_cartesian", :cgsense),
                ("shepp_logan_2d_8ch_radial", :gridding),
                ("shepp_logan_multislice_8ch_cartesian", :adjoint),
                ("shepp_logan_3d_8ch_cartesian", :adjoint),
                ("torso_cine_8ch_cartesian", :lowrank),
                ("torso_cine_8ch_radial", :gridding),
            )
            c = BU.get_case(id)
            @test method in BU.applicable_methods(c)
            tmin, tmed, x = BU.time_run(BU.mrt_reconstructor(c, method; maxit = 2); warmup = 0, runs = 1)
            @test tmin > 0
            @test size(parent(x)) == size(c.reference)
            @test isfinite(BU.mag_nrmse(Array(parent(x)), c.reference))
        end
    end
end

@testitem "Aqua" tags = [:quality, :aqua] begin
    using Aqua
    using MriReconstructionToolbox
    Aqua.test_all(
        MriReconstructionToolbox;
        ambiguities = true,
        piracies = false,
        persistent_tasks = false,
        stale_deps = false,
    )
end

@testitem "JET test_package" tags = [:quality, :jet] begin
    using JET
    using MriReconstructionToolbox
    JET.test_package(MriReconstructionToolbox; target_modules = (MriReconstructionToolbox,))
end

@testitem "JET exported API @test_opt" tags = [:quality, :jet] begin
    using JET
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    const MRT = MriReconstructionToolbox
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

    @test_opt target_modules = (MRT,) Tikhonov(0.1)
    @test_opt target_modules = (MRT,) L1Image(0.1)
    @test_opt target_modules = (MRT,) L1Wavelet2D(0.1)
    @test_opt target_modules = (MRT,) L1Wavelet3D(0.1)
    @test_opt target_modules = (MRT,) TotalVariation2D(0.1)
    @test_opt target_modules = (MRT,) TotalVariation3D(0.1)
    @test_opt target_modules = (MRT,) TemporalFourier(0.1; time_dim = 2)
    @test_opt target_modules = (MRT,) LowRank(0.1; time_dim = 2)
    @test_opt target_modules = (MRT,) RankLimit(2; time_dim = 2)

    @test_opt target_modules = (MRT,) UniformRandomSampling(2.0, 0.1)
    @test_opt target_modules = (MRT,) VariableDensitySampling(GaussianDistribution(), 2.0)
    @test_opt target_modules = (MRT,) VariableDensitySampling(PolynomialDistribution(), 2.0)
    @test_opt target_modules = (MRT,) PoissonDiskSampling(2.0)
    @test_opt target_modules = (MRT,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_opt target_modules = (MRT,) coil_sensitivities(nx, ny, nc)
    @test_opt target_modules = (MRT,) Config(maxit = 2, verbose = false)
    @test_opt target_modules = (MRT,) SequentialExecutor()
    @test_opt target_modules = (MRT,) MultiThreadingExecutor()
    @test_opt target_modules = (MRT,) BartScaling()
    @test_opt target_modules = (MRT,) MeasurementBasedScaling()
    @test_opt target_modules = (MRT,) NoScaling()
end

@testitem "JET exported API @test_call" tags = [:quality, :jet] begin
    using JET
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    const MRT = MriReconstructionToolbox
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
    @test_call target_modules = (MRT,) calculate(Tikhonov(0.1), img; threaded = false)
    @test_call target_modules = (MRT,) build_model(Eye(img), img, Tikhonov(0.1); threaded = false)
    @test_call target_modules = (MRT,) create_sampling_pattern(UniformRandomSampling(2.0, 0.1), (nx, ny))
    @test_call target_modules = (MRT,) to_displayable_mask((:, trues(ny)), (nx, ny))
    @test_call target_modules = (MRT,) coil_sensitivities(nx, ny, nc)
    @test_call target_modules = (MRT,) simulate_acquisition(img, AcquisitionInfo(nothing; is3D = false, image_size = (nx, ny), sensitivity_maps = smaps))
    @test_call target_modules = (MRT,) Config(maxit = 2, verbose = false)
    @test_call target_modules = (MRT,) reconstruct(cart_info, Tikhonov(0.01); maxit = 2, verbose = false, threaded = false)
    @test_call target_modules = (MRT,) SequentialExecutor()
    @test_call target_modules = (MRT,) MultiThreadingExecutor()
    @test_call target_modules = (MRT,) BartScaling()
    @test_call target_modules = (MRT,) MeasurementBasedScaling()
    @test_call target_modules = (MRT,) NoScaling()
end

@testitem "Benchmark suite smoke test" tags = [:quality] begin
    using BenchmarkTools
    using MriReconstructionToolbox

    benchmark_file = joinpath(pkgdir(MriReconstructionToolbox), "benchmark", "benchmarks.jl")
    @test isfile(benchmark_file)
    include(benchmark_file)
    @test haskey(SUITE, "operator")
    @test haskey(SUITE, "reconstruct")
    @test haskey(SUITE, "prox")

    # Quick smoke test execution (1 sample, 1 eval)
    results = run(SUITE, samples = 1, evals = 1)
    @test results isa BenchmarkTools.BenchmarkGroup
end

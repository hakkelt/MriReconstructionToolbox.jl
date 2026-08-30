@testitem "Component: construction, show, validation" tags = [:components] begin
    using Test
    using MriReconstructionToolbox

    c = Component(:sparse, L1Image(0.1))
    @test c.name === :sparse
    @test length(c.regularizations) == 1
    @test occursin("Component(:sparse", sprint(show, c))

    c2 = Component(:smooth, Tikhonov(0.01), L1Image(0.2))
    @test length(c2.regularizations) == 2

    @test_throws ArgumentError Component(:empty)

    @test_throws ArgumentError MriReconstructionToolbox.check_components((c,))
    @test_throws ArgumentError MriReconstructionToolbox.check_components((c, Component(:sparse, L1Image(0.1))))
    @test isnothing(MriReconstructionToolbox.check_components((c, c2)))
end

@testitem "Component: forwarders" tags = [:components] begin
    using Test
    using MriReconstructionToolbox

    c = Component(:lowrank, LowRank(0.05))
    factor = 2.0
    scaled = MriReconstructionToolbox.scale_regularization(c, factor)
    @test scaled.regularizations[1] isa LowRank
    @test scaled.regularizations[1].λ ≈ c.regularizations[1].λ * factor

    c3 = Component(:sparse, L1Image(0.1))
    x = Variable(rand(8, 8))
    t1 = MriReconstructionToolbox.materialize(c3, x; threaded = false)
    t2 = MriReconstructionToolbox.materialize(c3.regularizations[1], x; threaded = false)
    @test t1 isa typeof(t2)
end

@testitem "DecomposedImage: array semantics" tags = [:components] begin
    using Test
    using MriReconstructionToolbox

    a = rand(4, 4)
    b = rand(4, 4)
    img = MriReconstructionToolbox.DecomposedImage(a + b, (lowrank = a, sparse = b))

    @test size(img) == (4, 4)
    @test img[2, 3] ≈ (a + b)[2, 3]
    @test img.components.lowrank == a
    @test img.components.sparse == b
    @test total(img) === img.total
    @test components(img) === img.components
    @test sum(values(components(img))) ≈ total(img)
    @test Array(img) ≈ a + b
    @test_throws ErrorException img[1, 1] = 1.0
end

@testitem "build_model: two-component model matches hand-computed objective" tags = [:components, :minimizer] begin
    using Test
    using MriReconstructionToolbox

    x_true = rand(8, 8)
    y_true = rand(8, 8)
    𝒜 = Eye(x_true)
    y = 𝒜 * (x_true + y_true) .+ 0.01 .* randn(size(x_true))

    reg1 = L1Image(0.2)
    reg2 = Tikhonov(0.1)
    components = (Component(:sparse, reg1), Component(:smooth, reg2))

    terms, vars, _ = build_model(𝒜, y, components; threaded = false, x₀s = (copy(x_true), copy(y_true)))
    @test length(StructuredOptimization.extract_variables(terms)) == 2
    @test terms isa StructuredOptimization.TermSet

    f = StructuredOptimization.extract_functions(terms)
    op = StructuredOptimization.extract_operators(vars, terms)
    combined = StructuredOptimization.ArrayPartition(~vars[1], ~vars[2])
    model_val = f(op * combined)

    data_fidelity = 0.5 * sum(abs2, (𝒜 * (x_true + y_true)) .- y)
    reg1_val = MriReconstructionToolbox.calculate(reg1, x_true; threaded = false)
    reg2_val = MriReconstructionToolbox.calculate(reg2, y_true; threaded = false)
    @test isapprox(model_val, data_fidelity + reg1_val + reg2_val; rtol = 1.0e-8, atol = 1.0e-10)
end

@testitem "reconstruct: recovers additive smooth + sparse components" tags = [:components, :integration] begin
    using Test
    using LinearAlgebra
    using GeometricMedicalPhantoms

    using Random
    rng = MersenneTwister(1234)

    nx, ny, nc = 32, 32, 4
    smooth_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    sparse_true = zeros(ComplexF32, nx, ny)
    for _ in 1:20
        i, j = rand(rng, 1:nx), rand(rng, 1:ny)
        sparse_true[i, j] = 1.0 + 0im
    end
    img_true = smooth_true + sparse_true
    smaps = coil_sensitivities(nx, ny, nc)

    pdf = VariableDensitySampling(PolynomialDistribution(3), 2.0, 0.15)
    pattern = create_sampling_pattern(pdf, (nx, ny))
    acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps, subsampling = pattern)
    acq_with_data = simulate_acquisition(img_true, acq)

    img_recon = reconstruct(
        acq_with_data,
        IterativeReconstruction(Component(:smooth, Tikhonov(0.01)), Component(:sparse, L1Image(0.05)));
        maxit = 150, verbose = false,
    )

    @test img_recon isa DecomposedImage
    error_norm = norm(Array(img_recon) - img_true) / norm(img_true)
    @test error_norm < 0.5

    smooth_component = img_recon.components.smooth
    sparse_component = img_recon.components.sparse
    @test all(isfinite, smooth_component) && all(isfinite, sparse_component)
    @test Array(img_recon) ≈ smooth_component .+ sparse_component
end

@testitem "reconstruct: Lf=n_components required for convergence" tags = [:components] begin
    using Test

    x_true = rand(6, 6)
    y_true = rand(6, 6)
    𝒜 = Eye(x_true)
    y = 𝒜 * (x_true + y_true)

    components = (Component(:a, L1Image(0.01)), Component(:b, L1Image(0.01)))

    terms, vars, _ = build_model(𝒜, y, components; threaded = false, x₀s = (zero(x_true), zero(y_true)))
    default_alg = MriReconstructionToolbox.patch_algorithm_with_default_values(FISTA(), 2)
    @test default_alg.kwargs[:Lf] == 2
    solve(terms, default_alg; maxit = 500)
    @test all(isfinite, ~vars[1]) && all(isfinite, ~vars[2])

    terms2, vars2, _ = build_model(𝒜, y, components; threaded = false, x₀s = (zero(x_true), zero(y_true)))
    bad_alg = FISTA(Lf = 1)
    solve(terms2, bad_alg; maxit = 500)
    @test any(!isfinite, ~vars2[1]) || any(!isfinite, ~vars2[2]) || norm(~vars2[1]) > 1.0e6

    # ADMM only gets `cg_maxit`. Pinning `rho` would override ADMM's adaptive penalty, which
    # converges much faster here, and pinning `cg_tol` would break its coupling to the outer `tol`.
    admm_alg = MriReconstructionToolbox.patch_algorithm_with_default_values(ADMM(), 2)
    @test admm_alg.kwargs[:cg_maxit] == 10
    @test :rho ∉ keys(admm_alg.kwargs)
    @test admm_alg.kwargs[:cg_tol] == ADMM().kwargs[:cg_tol]
    tight = MriReconstructionToolbox.patch_algorithm_with_default_values(ADMM(tol = 1.0e-12), 2)
    @test tight.kwargs[:cg_tol] == ADMM(tol = 1.0e-12).kwargs[:cg_tol]
    @test tight.kwargs[:cg_tol] < admm_alg.kwargs[:cg_tol]
    # An explicitly passed value always wins, and the other defaults still get filled in.
    admm_user = MriReconstructionToolbox.patch_algorithm_with_default_values(ADMM(rho = 0.25), 2)
    @test admm_user.kwargs[:rho] == 0.25
    @test admm_user.kwargs[:cg_maxit] == 10
end

@testitem "reconstruct: multi-regularization component falls back to ADMM" tags = [:components] begin
    using Test
    using GeometricMedicalPhantoms

    nx, ny, nc = 16, 16, 2
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
    acq_with_data = simulate_acquisition(img_true, acq)

    components = (
        Component(:multi, L1Wavelet2D(0.001), TotalVariation2D(0.001)),
        Component(:sparse, L1Image(0.001)),
    )

    img_recon = reconstruct(acq_with_data, IterativeReconstruction(components...); maxit = 20, verbose = false)
    @test img_recon isa DecomposedImage

    @test_throws ErrorException reconstruct(acq_with_data, IterativeReconstruction(components...; algorithm = FISTA()); maxit = 20, verbose = false)
end

@testitem "reconstruct: components interact with problem decomposition" tags = [:components, :integration] begin
    using Test
    using GeometricMedicalPhantoms

    nx, ny, nslices, nc = 16, 16, 3, 2
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    smaps_ms = repeat(smaps, 1, 1, 1, nslices)
    ksp_ms = zeros(ComplexF32, nx, ny, nc, nslices)
    for s in 1:nslices
        acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
        ksp_ms[:, :, :, s] = simulate_acquisition(img_true, acq).kspace_data
    end
    acq_ms = AcquisitionInfo(ksp_ms; is3D = false, sensitivity_maps = smaps_ms)

    components = (Component(:smooth, Tikhonov(0.005)), Component(:sparse, L1Image(0.005)))

    img_decomposed = reconstruct(acq_ms, IterativeReconstruction(components...); maxit = 30, verbose = false)
    img_joint = reconstruct(acq_ms, IterativeReconstruction(components...); maxit = 30, verbose = false, disable_problem_decomposition = true)

    @test img_decomposed isa DecomposedImage
    @test size(img_decomposed) == (nx, ny, nslices)
    @test isapprox(Array(img_decomposed), Array(img_joint); rtol = 0.1)

    # Per-component `x₀`s must not be written through either: `build_model` hands them to
    # `Variable`, which stores by reference, and `solve` writes the solution back through it.
    x₀s = (rand(ComplexF32, nx, ny, nslices), rand(ComplexF32, nx, ny, nslices))
    x₀s_ref = map(copy, x₀s)
    reconstruct(acq_ms, IterativeReconstruction(components...); x₀ = x₀s, normalization = NoScaling(), maxit = 5, verbose = false)
    @test all(x₀s .== x₀s_ref)
end

@testitem "reconstruct: a NamedTuple x₀ with an unknown component name is rejected" tags = [:components] begin
    using Test
    using GeometricMedicalPhantoms

    # Regression: get_component_x0s falls back to copy(x̂)/zero(x̂) for any key it does not find, so a
    # mistyped name silently discarded the caller's initial guess and warm-started from zero instead.
    nx, ny, nc = 16, 16, 2
    img_true = create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
    smaps = coil_sensitivities(nx, ny, nc)
    acq = simulate_acquisition(img_true, AcquisitionInfo(is3D = false, sensitivity_maps = smaps))

    components = (Component(:lowrank, Tikhonov(0.01)), Component(:sparse, L1Image(0.01)))
    good = (lowrank = zeros(ComplexF32, nx, ny), sparse = zeros(ComplexF32, nx, ny))
    typo = (lowrnak = zeros(ComplexF32, nx, ny), sparse = zeros(ComplexF32, nx, ny))

    @test isnothing(MriReconstructionToolbox.check_x₀_components_size(good, components, (nx, ny)))
    @test_throws ArgumentError MriReconstructionToolbox.check_x₀_components_size(typo, components, (nx, ny))
    @test_throws ArgumentError reconstruct(acq, IterativeReconstruction(components...); x₀ = typo, maxit = 5, verbose = false)

    # A partial NamedTuple is still legal: the unnamed components fall back to their defaults.
    partial = (sparse = zeros(ComplexF32, nx, ny),)
    @test isnothing(MriReconstructionToolbox.check_x₀_components_size(partial, components, (nx, ny)))
end

@testitem "reconstruct: LowRank + Sparse with NamedDimsArray and symbol time_dim" tags = [:components, :integration] begin
    using Test
    using NamedDims
    using MriReconstructionToolbox

    nx, ny, nt = 16, 16, 4
    img_true = NamedDimsArray{(:x, :y, :time)}(rand(ComplexF32, nx, ny, nt))
    smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, nx, ny, 2))
    acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
    ksp = simulate_acquisition(img_true, acq).kspace_data
    acq_data = AcquisitionInfo(acq, kspace_data = ksp)

    components = (
        Component(:lowrank, LowRank(0.01; time_dim = :time)),
        Component(:sparse, L1Image(0.01)),
    )

    img_recon = reconstruct(acq_data, IterativeReconstruction(components...); maxit = 5, verbose = false)
    @test img_recon isa DecomposedImage
    @test size(img_recon) == (nx, ny, nt)
    @test dimnames(img_recon) == (:x, :y, :time)
    @test haskey(img_recon.components, :lowrank)
    @test haskey(img_recon.components, :sparse)
    @test all(isfinite, img_recon.components.lowrank)
    @test all(isfinite, img_recon.components.sparse)
end

@testitem "reconstruct: components with problem decomposition and NamedDimsArray" tags = [:components, :integration] begin
    using Test
    using NamedDims
    using MriReconstructionToolbox

    nx, ny, nslices, nt = 16, 16, 2, 4
    img_true = NamedDimsArray{(:x, :y, :z, :time)}(rand(ComplexF32, nx, ny, nslices, nt))
    smaps = NamedDimsArray{(:x, :y, :coil, :z)}(rand(ComplexF32, nx, ny, 2, nslices))
    acq = AcquisitionInfo(is3D = false, sensitivity_maps = smaps)
    ksp = simulate_acquisition(img_true, acq).kspace_data
    acq_data = AcquisitionInfo(acq, kspace_data = ksp)

    # Both components affect only :time, so :z stays a batch dimension: the problem-decomposition
    # planner must resolve the symbol `time_dim` against the named image dims and pick :z to slice.
    components = (
        Component(:fourier, TemporalFourier(0.01; time_dim = :time)),
        Component(:tv, TemporalTotalVariation(0.01; time_dim = :time)),
    )

    image_dims = MriReconstructionToolbox.get_image_dims(acq_data)
    @test image_dims == (:x, :y, :z, :time)

    # bind_dimensions is the symbol-resolution step the fix moved ahead of unnaming: every
    # component's regularizer must come back with an integer `time_dim` (4, here).
    bound = MriReconstructionToolbox.bind_dimensions(components, image_dims)
    @test bound[1].regularizations[1].time_dim == 4
    @test bound[2].regularizations[1].time_dim == 4

    plan = MriReconstructionToolbox.get_problem_decomposition_plan(acq_data, IterativeReconstruction(bound...), Config(verbose = false))
    @test plan !== nothing
    @test plan.image_batch_dims == (3,)  # slice over :z
end

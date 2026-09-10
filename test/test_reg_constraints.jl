using TestItems

@testitem "NonNegative and BoxConstraint regularizations" tags = [:regularization] setup = [RegTestSetup] begin

    @testset "NonNegative" for threaded in [false, true]
        @test MriReconstructionToolbox.calculate(NonNegative(), abs.(randn(4, 4)); threaded) == 0
        @test MriReconstructionToolbox.calculate(NonNegative(), [-1.0 1.0; 1.0 1.0]; threaded) == Inf
    end

    @testset "BoxConstraint" begin
        @test MriReconstructionToolbox.calculate(BoxConstraint(0.0, 1.0), rand(4, 4); threaded = false) == 0
        @test MriReconstructionToolbox.calculate(BoxConstraint(0.0, 1.0), [2.0 0.5]; threaded = false) == Inf
        @test MriReconstructionToolbox.calculate(
            BoxConstraint(zeros(1, 2), ones(1, 2)), [0.5 0.5]; threaded = false
        ) == 0
        @test_throws ArgumentError BoxConstraint(1.0, 0.0)
    end

    @testset "complex data is rejected by default" begin
        x = Variable(randn(ComplexF64, 4, 4))
        @test_throws ArgumentError MriReconstructionToolbox.materialize(NonNegative(), x; threaded = false)
        @test_throws ArgumentError MriReconstructionToolbox.materialize(
            BoxConstraint(0.0, 1.0), x; threaded = false
        )
    end

    @testset "complex_handling = :real projects onto the real orthant/box" begin
        using ProximalCore
        x_complex = [1.0 - 2.0im 0.5 + 0.0im; -3.0 + 1.0im 2.0 + 0.0im]
        x_var = Variable(copy(x_complex))

        reg_nn = NonNegative(; complex_handling = :real)
        @test MriReconstructionToolbox.calculate(reg_nn, x_complex; threaded = false) == Inf
        term_nn = MriReconstructionToolbox.materialize(reg_nn, x_var; threaded = false)
        y_nn = similar(x_complex)
        v_nn = ProximalCore.prox!(y_nn, term_nn.f, x_complex, 1.0)
        @test v_nn == 0
        @test y_nn ≈ [1.0 0.5; 0.0 2.0]

        reg_box = BoxConstraint(0.0, 1.0; complex_handling = :real)
        term_box = MriReconstructionToolbox.materialize(reg_box, x_var; threaded = false)
        y_box = similar(x_complex)
        v_box = ProximalCore.prox!(y_box, term_box.f, x_complex, 1.0)
        @test v_box == 0
        @test y_box ≈ [1.0 0.5; 0.0 1.0]

        @test MriReconstructionToolbox.scale_regularization(reg_box, 2.0).complex_handling == :real
    end

    @testset "invalid complex_handling is rejected" begin
        @test_throws ArgumentError NonNegative(; complex_handling = :bogus)
        @test_throws ArgumentError BoxConstraint(0.0, 1.0; complex_handling = :bogus)
    end

    @testset "get_operator and get_affected_dims" begin
        x = randn(4, 4)
        @test get_operator(NonNegative(), x; threaded = false) isa Eye
        @test get_operator(BoxConstraint(0.0, 1.0), x; threaded = false) isa Eye
        ksp = randn(ComplexF32, 4, 4)
        info = AcquisitionInfo(ksp; image_size = (4, 4))
        @test MriReconstructionToolbox.get_affected_dims(NonNegative(), info, 1:2) == ()
        @test MriReconstructionToolbox.get_affected_dims(BoxConstraint(0.0, 1.0), info, 1:2) == ()
        # array-valued bounds have the size of the full image, so task splitting must be blocked
        @test MriReconstructionToolbox.get_affected_dims(
            BoxConstraint(zeros(4, 4), ones(4, 4)), info, 1:2
        ) == (1, 2)
    end

    @testset "scale_regularization" begin
        @test MriReconstructionToolbox.scale_regularization(NonNegative(), 2.5) isa NonNegative
        scaled = MriReconstructionToolbox.scale_regularization(BoxConstraint(0.5, 1.0), 2.0)
        @test scaled.lower ≈ 1.0
        @test scaled.upper ≈ 2.0
    end
end

@testitem "TotalVariation2D + NonNegative(:real) selects ADMM and beats TV alone" tags = [:regularization, :reconstruction, :minimizer] begin
    using Test
    using MriReconstructionToolbox
    using LinearAlgebra: norm
    using Random

    Random.seed!(1)
    nx, ny = 32, 32
    x_true = zeros(ComplexF32, nx, ny)
    x_true[10:22, 10:22] .= 1
    # undersampled + noisy: TV alone leaves negative Gibbs-ringing overshoot for NonNegative to
    # visibly correct (a clean, fully-sampled block phantom converges to non-negative on its own,
    # which would make the constraint's effect untestable)
    pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.1)
    pattern = create_sampling_pattern(pdf, (nx, ny))
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny), subsampling = pattern)
    data = add_noise(simulate_acquisition(x_true, acq); noise_std = 0.08f0)

    nrmse(rec) = norm(rec .- x_true) / norm(x_true)

    tv_only = reconstruct(
        data, IterativeReconstruction(TotalVariation2D(3.0f-2); algorithm = ADMM(maxit = 50));
        verbosity = Silent(),
    )
    @test minimum(real.(tv_only)) < 0  # TV alone overshoots negative -- the premise this test checks

    metrics_seen = Symbol[]
    cb = info -> append!(metrics_seen, collect(keys(info)))
    tv_plus_nn = reconstruct(
        data,
        IterativeReconstruction(
            TotalVariation2D(3.0f-2), NonNegative(; complex_handling = :real);
            algorithm = ADMM(maxit = 50), on_iteration = cb,
        );
        verbosity = Silent(),
    )

    # ADMM's iteration metrics (primal_residual/dual_residual/iterate_change) are how solver
    # selection is observed indirectly: only ADMM reports these (solve_core.jl's
    # `_iteration_metrics(::ADMMIteration, ...)`).
    @test :primal_residual in metrics_seen
    @test :dual_residual in metrics_seen

    # ADMM only drives the auxiliary variable z (the constraint's own copy) to feasibility; the
    # returned x approaches it at the primal-residual rate, so at a finite iteration count these
    # checks are directional (less negative / more real / lower error), not near-zero.
    @test minimum(real.(tv_plus_nn)) > minimum(real.(tv_only))
    @test maximum(abs.(imag.(tv_plus_nn))) < maximum(abs.(imag.(tv_only)))
    @test nrmse(tv_plus_nn) < nrmse(tv_only)
end

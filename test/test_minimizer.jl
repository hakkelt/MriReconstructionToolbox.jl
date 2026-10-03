@testitem "Model builder: Eye + L1Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using MriReconstructionToolbox.AbstractOperators

    @testset "Eye + L1Image" for threaded in (false, true)
        x = rand(8, 8)
        𝒜 = Eye(x)
        y = 𝒜 * x .+ 0.01 .* randn(size(x))
        reg = L1Image(0.2)
        terms = build_model(𝒜, y, reg; threaded)

        model_val = eval_term(terms)

        x̂ = 𝒜' * y
        data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
        reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
        @test isapprox(model_val, data_fidelity + reg_val; rtol = 1.0e-10, atol = 1.0e-12)
    end
end

@testitem "Model builder: Eye + L1Image + L2Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using MriReconstructionToolbox.AbstractOperators

    @testset "Eye + L1Image + L2Image" for threaded in (false, true)
        x = rand(6, 6)
        𝒜 = Eye(x)
        y = copy(x)
        regs = (L1Image(0.1), L2Image(0.05))
        terms = build_model(𝒜, y, regs; threaded)

        model_val = eval_term(terms)

        x̂ = 𝒜' * y
        data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
        reg1 = MriReconstructionToolbox.calculate(regs[1], y; threaded)
        reg2 = MriReconstructionToolbox.calculate(regs[2], y; threaded)
        @test isapprox(model_val, data_fidelity + reg1 + reg2; rtol = 1.0e-10, atol = 1.0e-12)
    end
end

@testitem "Model builder: Linear op + L2Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using MriReconstructionToolbox.AbstractOperators

    @testset "Linear op + L2Image" for threaded in (false, true)
        x = rand(8, 8)
        y = rand(8, 8)
        𝒜 = Eye(x)
        reg = L2Image(0.3)
        terms = build_model(𝒜, y, reg; threaded)

        model_val = eval_term(terms)

        x̂ = 𝒜' * y
        data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
        reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
        @test isapprox(model_val, data_fidelity + reg_val; rtol = 1.0e-10, atol = 1.0e-12)

        vars = StructuredOptimization.extract_variables(terms)
        xvar = vars[1]
        x0 = copy(~xvar)
        δ = 0.01 .* randn(size(x0))
        ~xvar .= x0 .+ δ
        model_val2 = eval_term(terms)

        data = 0.5 * sum(abs2, (𝒜 * (x0 .+ δ)) .- y)
        reg2 = MriReconstructionToolbox.calculate(reg, x0 .+ δ; threaded)
        @test isapprox(model_val2, data + reg2; rtol = 1.0e-8, atol = 1.0e-10)
    end
end

@testitem "Model builder: NamedDims y and A" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using NamedDims
    using MriReconstructionToolbox.AbstractOperators

    @testset "NamedDims y and A" for threaded in (false, true)
        x = rand(8, 8)
        𝒜 = MriReconstructionToolbox.NamedDimsOp{(:x, :y), (:x, :y)}(Eye(x))
        y = NamedDimsArray(copy(x), (:x, :y))
        reg = L1Image(0.15)
        terms = build_model(𝒜, y, reg; threaded)

        model_val = eval_term(terms)

        x̂ = 𝒜' * y
        data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
        reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
        @test isapprox(model_val, data_fidelity + reg_val; rtol = 1.0e-10, atol = 1.0e-12)
    end
end

@testitem "Model builder: overload parity" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using MriReconstructionToolbox.AbstractOperators

    @testset "overload parity" for threaded in (false, true)
        x = rand(5, 5)
        𝒜 = Eye(x)
        y = copy(x)
        reg = L2Image(0.2)
        t1 = build_model(𝒜, y, reg; threaded)
        t2 = build_model(𝒜, y, (reg,); threaded)
        @test isapprox(eval_term(t1), eval_term(t2); atol = 1.0e-12)
    end
end

@testitem "DouglasRachford default parameter patching" tags = [:minimizer] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    alg = DouglasRachford(maxit = 100)
    patched = MriReconstructionToolbox.patch_algorithm_with_default_values(alg, 2.0)
    @test patched.kwargs[:gamma] == 0.5

    patched_no_lf = MriReconstructionToolbox.patch_algorithm_with_default_values(alg, nothing)
    @test patched_no_lf.kwargs[:gamma] == 1.0

    explicit = DouglasRachford(gamma = 0.1)
    patched_explicit = MriReconstructionToolbox.patch_algorithm_with_default_values(explicit, 5.0)
    @test patched_explicit.kwargs[:gamma] == 0.1
end

@testitem "HardConsistency projection fast path vs inner-CG" tags = [:minimizer] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra
    using ProximalCore
    using MriReconstructionToolbox.ProximalOperators

    nx, ny = 16, 16
    x = rand(ComplexF32, nx, ny)
    mask = rand(Bool, nx, ny)
    mask[1, 1] = true
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny), subsampling = mask)
    acq_data = simulate_acquisition(x, acq)
    𝒜 = unname(get_encoding_operator(acq_data))
    y = acq_data.kspace_data

    @test MriReconstructionToolbox.is_AAc_diagonal(𝒜)

    # Fast diagonal projection: `hard_consistency_prox` hands `diag_AAc(𝒜)` to `IndAffineCG`, which
    # then divides instead of iterating.
    x_test = rand(ComplexF32, nx, ny)
    f_fast = MriReconstructionToolbox.hard_consistency_prox(𝒜, y, 50, 1.0e-6)
    @test f_fast.AAc_diag !== nothing
    proj_fast = similar(x_test)
    ProximalCore.prox!(proj_fast, f_fast, x_test, 1.0)

    # Inner-CG projection
    v_cg = ProximalOperators._cg_solve_AAc(𝒜, 𝒜 * x_test - y; maxit = 100, tol = 1.0e-6)
    proj_cg = x_test .- 𝒜' * v_cg

    @test isapprox(proj_fast, proj_cg; rtol = 1.0e-4, atol = 1.0e-5)
    @test isapprox(𝒜 * proj_fast, y; rtol = 1.0e-4, atol = 1.0e-5)
end

@testitem "Reconstruction with HardConsistency + DouglasRachford" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra

    nx, ny = 16, 16
    x_true = zeros(ComplexF32, nx, ny)
    x_true[4:8, 4:8] .= 1.0f0 + 0.5f0im
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    method = IterativeReconstruction(
        L1Image(1.0e-6);
        algorithm = DouglasRachford(maxit = 50, tol = 1.0e-5),
        fidelity = HardConsistency(),
    )
    rec = reconstruct(acq_data, method; verbosity = Silent())
    @test isapprox(rec, x_true; rtol = 1.0e-4, atol = 1.0e-4)
end

@testitem "Unregularized Iterative Least-Squares with CGNR" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra

    nx, ny = 16, 16
    x_true = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    method = IterativeReconstruction(;
        algorithm = CGNR(maxit = 20, tol = 1.0e-6),
        fidelity = L2Loss(),
    )
    rec = reconstruct(acq_data, method; verbosity = Silent())
    @test isapprox(rec, x_true; rtol = 1.0e-4, atol = 1.0e-4)
end

@testitem "ChambollePock (PDHG) reaches the ADMM solution of total variation" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using LinearAlgebra, Random

    @test PDHG === ChambollePock
    Random.seed!(3)
    nx, ny = 24, 24
    img = zeros(ComplexF32, nx, ny)
    img[6:18, 8:16] .= 1
    img[10:13, 10:13] .= 2
    smaps = coil_sensitivities(nx, ny, 4)
    mask = rand(nx, ny) .< 0.5
    mask[:, (ny ÷ 2 - 2):(ny ÷ 2 + 3)] .= true
    acq = CartesianAcquisitionInfo(; is3D = false, image_size = (nx, ny), subsampling = mask, sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)

    reg = TotalVariation2D(1.0e-2)
    admm = IterativeReconstruction(reg; algorithm = ADMM(rho = 0.05, maxit = 1000, tol = 0.0, cg_tol = 0.0, cg_maxit = 10), maxit = 1000, reltol = 0.0)
    pdhg = IterativeReconstruction(reg; algorithm = PDHG(maxit = 5000, tol = 0.0), maxit = 5000, reltol = 0.0)
    x_admm = reconstruct(data, admm; verbosity = Silent())
    x_pdhg = reconstruct(data, pdhg; verbosity = Silent())
    @test norm(x_pdhg - x_admm) / norm(x_admm) < 1.0e-2
end

@testitem "VuCondat reaches the ADMM solution of total variation" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using LinearAlgebra, Random

    Random.seed!(3)
    nx, ny = 24, 24
    img = zeros(ComplexF32, nx, ny)
    img[6:18, 8:16] .= 1
    img[10:13, 10:13] .= 2
    smaps = coil_sensitivities(nx, ny, 4)
    mask = rand(nx, ny) .< 0.5
    mask[:, (ny ÷ 2 - 2):(ny ÷ 2 + 3)] .= true
    acq = CartesianAcquisitionInfo(; is3D = false, image_size = (nx, ny), subsampling = mask, sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)

    reg = TotalVariation2D(1.0e-2)
    admm = IterativeReconstruction(reg; algorithm = ADMM(rho = 0.05, maxit = 1000, tol = 0.0, cg_tol = 0.0, cg_maxit = 10), maxit = 1000, reltol = 0.0)
    vc = IterativeReconstruction(reg; algorithm = VuCondat(maxit = 5000, tol = 0.0), maxit = 5000, reltol = 0.0)
    x_admm = reconstruct(data, admm; verbosity = Silent())
    x_vc = reconstruct(data, vc; verbosity = Silent())
    @test norm(x_vc - x_admm) / norm(x_admm) < 1.0e-2
end

@testitem "Preconditioned ChambollePock on radial data reaches the VuCondat solution" tags = [:minimizer, :reconstruction, :nfft] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo
    using LinearAlgebra

    nx, ny = 32, 32
    img = zeros(ComplexF32, nx, ny)
    img[10:22, 8:20] .= 1
    img[14:17, 12:15] .= 2
    traj = radial_trajectory(64, 24; ordering = GoldenAngle())
    smaps = coil_sensitivities(nx, ny, 4)
    acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)

    reg = AnisotropicTotalVariation2D(1.0e-3)
    ref = IterativeReconstruction(reg; algorithm = VuCondat(maxit = 6000, tol = 0.0), maxit = 6000, reltol = 0.0)
    x_ref = reconstruct(data, ref; verbosity = Silent())
    # The density-compensated, block-preconditioned steps reconstruct derives...
    err(x) = norm(x - x_ref) / norm(x_ref)
    cp = IterativeReconstruction(reg; algorithm = ChambollePock(maxit = 300, tol = 0.0), maxit = 300, reltol = 0.0)
    @test err(reconstruct(data, cp; verbosity = Silent())) < 1.0e-2
    # ...against one scalar step, which a given `ratio` leaves in place: further from the
    # solution after the same number of iterations.
    short(alg) = err(reconstruct(data, IterativeReconstruction(reg; algorithm = alg, maxit = 60, reltol = 0.0); verbosity = Silent()))
    @test short(ChambollePock(maxit = 60, tol = 0.0)) < short(ChambollePock(maxit = 60, tol = 0.0, ratio = 1))
end

@testitem "NCG and LBFGS reach the POGM solution of a smooth problem" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using LinearAlgebra, Random

    Random.seed!(5)
    nx, ny = 24, 24
    img = zeros(ComplexF32, nx, ny)
    img[6:18, 8:16] .= 1
    img[10:13, 10:13] .= 2
    smaps = coil_sensitivities(nx, ny, 4)
    mask = rand(nx, ny) .< 0.5
    mask[:, (ny ÷ 2 - 2):(ny ÷ 2 + 3)] .= true
    acq = CartesianAcquisitionInfo(; is3D = false, image_size = (nx, ny), subsampling = mask, sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)

    # Huber with a small threshold: a smooth problem, but not a quadratic one.
    reg = EdgePreservingRoughness2D(1.0e-2; δ = 0.01)
    pogm = IterativeReconstruction(reg; algorithm = POGM(maxit = 3000, tol = 0.0), maxit = 3000, reltol = 0.0)
    x_ref = reconstruct(data, pogm; verbosity = Silent())
    for alg in (NCG(maxit = 300, tol = 0.0), LBFGS(maxit = 300, tol = 0.0))
        m = IterativeReconstruction(reg; algorithm = alg, maxit = 300, reltol = 0.0)
        @test norm(reconstruct(data, m; verbosity = Silent()) - x_ref) / norm(x_ref) < 1.0e-3
    end

    # The default tuple sends a smooth, non-quadratic problem to LBFGS (whose callback payload
    # carries `gradient_norm`), and a non-smooth one past it.
    selected(reg) = (trace = IterationTrace(x -> 0.0); reconstruct(data, IterativeReconstruction(reg; maxit = 2, on_iteration = trace); verbosity = Silent()); keys(trace.metrics[1]))
    @test :gradient_norm in selected(reg)
    @test :gradient_norm ∉ selected(TotalVariation2D(1.0e-2))
end

@testitem "CGNR on radial data beats the plain adjoint" tags = [:reconstruction, :nfft, :quality] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, NonCartesianAcquisitionInfo
    using LinearAlgebra

    nx, ny = 32, 32
    img = zeros(ComplexF32, nx, ny)
    img[10:22, 10:22] .= 1
    traj = radial_trajectory(64, 64; ordering = GoldenAngle())
    smaps = coil_sensitivities(nx, ny, 4)
    acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)

    𝒜 = get_encoding_operator(data)
    nrmse(rec) = norm(rec .- img) / norm(img)
    nrmse_adjoint = nrmse(𝒜' * data.kspace_data)

    # The bare adjoint `𝒜'y` is an un-normalized-NFFT-scale warm start (off by orders of
    # magnitude), which a finite-`maxit` CG-SENSE solve does not correct on its own -- it used to
    # score *worse* than the plain adjoint. The scale-correct `𝒜'y/‖𝒜‖²` warm start
    # (`_direct_reconstruct`) fixes that.
    method = IterativeReconstruction(; algorithm = CGNR(maxit = 20, tol = 1.0e-6), fidelity = L2Loss())
    rec = reconstruct(data, method; verbosity = Silent())
    @test nrmse(rec) < nrmse_adjoint
end

@testitem "POGM matches FISTA on a single L1 regularizer" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox

    nx, ny = 32, 32
    x_true = zeros(ComplexF32, nx, ny)
    x_true[10:22, 10:22] .= 1
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    reg = L1Image(1.0e-3)
    fista_rec = reconstruct(
        acq_data, IterativeReconstruction(reg; algorithm = FISTA(maxit = 100));
        verbosity = Silent(),
    )
    pogm_rec = reconstruct(
        acq_data, IterativeReconstruction(reg; algorithm = POGM(maxit = 100));
        verbosity = Silent(),
    )
    @test isapprox(pogm_rec, fista_rec; rtol = 1.0e-2, atol = 1.0e-3)
end

@testitem "POGM survives an under-estimated Lf" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo, get_encoding_operator
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox.AbstractOperators: estimate_opnorm
    using LinearAlgebra: norm

    # POGM's worst-case rate is tight, so a stepsize above `1/Lf` makes it diverge rather than
    # converge slowly — and `estimate_opnorm`'s power iteration, which is where MRT's `Lf` comes
    # from, converges from *below*. The adaptive restart of Kim & Fessler (2018) is what keeps
    # that safe; this pins it, by handing POGM an `Lf` deliberately 15% too small.
    nx, ny = 32, 32
    x_true = zeros(ComplexF32, nx, ny)
    x_true[10:22, 10:22] .= 1
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    reg = L1Image(1.0e-3)
    L = estimate_opnorm(get_encoding_operator(acq_data))
    too_small = Float32(0.85 * L^2)

    reference = reconstruct(
        acq_data, IterativeReconstruction(reg; algorithm = FISTA(maxit = 300), reltol = 0.0);
        verbosity = Silent(),
    )
    with_restart = reconstruct(
        acq_data,
        IterativeReconstruction(reg; algorithm = POGM(Lf = too_small, maxit = 300), reltol = 0.0);
        verbosity = Silent(),
    )
    without_restart = reconstruct(
        acq_data,
        IterativeReconstruction(
            reg; algorithm = POGM(Lf = too_small, adaptive_restart = false, maxit = 300),
            reltol = 0.0,
        );
        verbosity = Silent(),
    )

    @test all(isfinite, with_restart)
    @test isapprox(with_restart, reference; rtol = 5.0e-2, atol = 1.0e-2)
    # Without the restart the same run leaves the neighbourhood of the solution entirely.
    @test norm(without_restart .- reference) > 10 * norm(with_restart .- reference)
end

@testitem "NoFidelity and error handling" tags = [:minimizer] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    nx, ny = 8, 8
    x = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x, acq)
    𝒜 = get_encoding_operator(acq_data)
    y = acq_data.kspace_data

    # NoFidelity with empty regularizations throws ArgumentError
    @test_throws ArgumentError build_model(𝒜, y, (); fidelity = NoFidelity())
end

@testitem "Diagnostic ArgumentError on single-solver parse failure" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    nx, ny = 16, 16
    x = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D = false, image_size = (nx, ny))
    acq_data = simulate_acquisition(x, acq)

    # Incompatible single solver (DouglasRachford with L2Loss and 2 L1 terms) throws informative ArgumentError
    method = IterativeReconstruction(L1Image(0.1), L1Image(0.2); algorithm = DouglasRachford(), fidelity = L2Loss())
    err = try
        reconstruct(acq_data, method; verbosity = Silent())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Cannot parse problem for algorithm", err.msg)
    @test occursin("DouglasRachford", err.msg)
    @test occursin("L2Loss", err.msg)
end

@testitem "Preconditioned CGNR: λ is honoured and convergence accelerates" tags = [:minimizer, :reconstruction] begin
    using MriReconstructionToolbox: CartesianAcquisitionInfo
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using MriReconstructionToolbox.AbstractOperators: DiagOp
    using LinearAlgebra
    using Random

    PA = MriReconstructionToolbox.ProximalAlgorithms

    @testset "the preconditioned iterations solve the regularized system" begin
        # `PCGIteration`/`PCGNRIteration` used to drop the λ term from both the initial residual
        # and the `Ap` update, so a `CGNR(; P, λ)` solve silently returned the *unregularized*
        # minimizer -- a wrong answer, not an error.
        Random.seed!(1)
        n, m = 20, 30
        A = randn(m, n)
        b = randn(m)
        λ = 0.5
        x_reg = (A'A + λ * I) \ (A'b)
        x_unreg = (A'A) \ (A'b)
        @test norm(x_reg - x_unreg) > 1.0e-2  # the two references are far apart, so the test can tell

        P = Diagonal(diag(A'A) .+ λ)
        function run_iter(iter)
            state = nothing
            for s in Iterators.take(iter, 200)
                state = s
            end
            return state.x
        end

        x_pcgnr = run_iter(PA.PCGNRIteration(; x0 = zeros(n), A = A, b = b, P = P, λ = λ))
        @test isapprox(x_pcgnr, x_reg; atol = 1.0e-10)

        x_pcg = run_iter(PA.PCGIteration(; x0 = zeros(n), A = A'A, b = A'b, P = P, λ = λ))
        @test isapprox(x_pcg, x_reg; atol = 1.0e-10)
    end

    @testset "CGNR(; P) through reconstruct" begin
        nx, ny = 32, 32
        Random.seed!(5)
        x_true = ComplexF32.(rand(Float32, nx, ny))
        smaps = unname(coil_sensitivities(nx, ny, 4))
        # A smooth intensity ramp makes the coil coverage Σ|S_c|² span two orders of magnitude,
        # which is exactly what a diagonal image-domain preconditioner is for.
        smaps = ComplexF32.(smaps .* reshape(range(0.05f0, 1.0f0, length = nx), nx, 1, 1))
        sub = create_sampling_pattern(VariableDensitySampling(GaussianDistribution(), 2.0), (nx, ny))
        acq = CartesianAcquisitionInfo(
            is3D = false, image_size = (nx, ny), sensitivity_maps = smaps, subsampling = sub
        )
        data = simulate_acquisition(x_true, acq)

        λ = 1.0f-3
        coverage = real(sum(abs2, smaps; dims = 3)[:, :, 1])
        # An `AbstractOperator` supports `mul!` but not `ldiv!`, so the *inverse* preconditioner is
        # what is passed, with `P_is_inverse = true`.
        Pinv = DiagOp(ComplexF32.(1 ./ (coverage .+ λ)))

        nrmse(r) = norm(r .- x_true) / norm(x_true)
        solve_at(k; kwargs...) = nrmse(
            reconstruct(
                data,
                IterativeReconstruction(
                    L2Image(λ); algorithm = CGNR(; tol = 1.0e-14, kwargs...), maxit = k, reltol = nothing,
                    fidelity = L2Loss(),
                );
                verbosity = Silent(),
            )
        )

        plain_32 = solve_at(32)
        pc_8 = solve_at(8; P = Pinv, P_is_inverse = true)
        # Same error in a quarter of the iterations: the point of the preconditioner is convergence
        # speed, not a different answer.
        @test pc_8 <= plain_32
        @test pc_8 < solve_at(8)
    end

    @testset "a preconditioned solver is still recognized as Krylov" begin
        @test MriReconstructionToolbox._is_krylov_solver(CGNR(P = Diagonal(ones(4)), P_is_inverse = true))
        @test MriReconstructionToolbox._is_krylov_solver(CG(P = Diagonal(ones(4)), P_is_inverse = true))
        @test MriReconstructionToolbox._is_krylov_solver(CGNR())
        @test !MriReconstructionToolbox._is_krylov_solver(FISTA())
    end
end

@testitem "The operator-norm margin is chosen per algorithm" tags = [:minimizer, :reconstruction] begin
    using LinearAlgebra, Random
    using MriReconstructionToolbox: opnorm_rel_margin, get_encoding_operator
    import MriReconstructionToolbox.AbstractOperators as AbstractOperators

    # POGM is the algorithm that diverges on an under-estimated `Lf`, so it asks for the tightest
    # margin; everything else takes the conservative default.
    @test opnorm_rel_margin(POGM()) == 1.0e-3
    @test opnorm_rel_margin(FISTA()) == 0.01
    @test opnorm_rel_margin(ISTA()) == 0.01
    @test opnorm_rel_margin(CGNR()) == 0.01
    # A tuple of algorithms takes the tightest of them: the number is handed to all of them.
    @test opnorm_rel_margin((FISTA(), POGM())) == 1.0e-3

    # Whatever the margin, the value is at or above `‖𝒜‖` — that is what a fixed step `1/Lf` needs.
    Random.seed!(19)
    nx, ny, nc = 32, 32, 4
    maps = NamedDimsArray{(:x, :y, :coil)}(randn(ComplexF32, nx, ny, nc))
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(randn(ComplexF32, nx, ny, nc))
    E = get_encoding_operator(
        AcquisitionInfo(ksp; sensitivity_maps = maps, image_size = (nx, ny)); threaded = false
    )
    truth = AbstractOperators.powerit(E; maxit = 500, rel_margin = 1.0e-12)
    for margin in (1.0e-3, 0.01, 0.05)
        @test AbstractOperators.estimate_opnorm(E; rel_margin = margin) >= truth
    end
    # A name for the axes is an isometry, so the bound sees through it rather than giving up.
    @test isfinite(AbstractOperators.opnorm_bound(E))
end

@testitem "ADMM's penalty is relative to the curvature of the data term" tags = [:minimizer] begin
    using MriReconstructionToolbox: _scale_admm_penalty
    import MriReconstructionToolbox.ProximalAlgorithms as PA

    method = IterativeReconstruction(; regularization = TotalVariation2D(0.01))
    config = ReconstructionConfig()
    scaled(alg; m = method) = _scale_admm_penalty(alg, nothing, nothing, 2.0, m, config; eltype_real = Float32)

    # `L = 2`, so every given penalty is multiplied by `L² = 4`, which ADMM applies as `rho_scale`.
    for alg in (ADMM(; rho = 0.05f0), ADMM(; rho = (0.05f0, 1.0f0)), ADMM(; penalty_sequence = PA.FixedPenalty([0.1f0])))
        @test scaled(alg).kwargs[:rho_scale] ≈ 4.0f0
    end
    kw = scaled(ADMM(; rho = 0.05f0)).kwargs
    g = MriReconstructionToolbox.ProximalOperators.NormL1(0.1f0)
    iter = PA.ADMMIteration(; x0 = zeros(Float32, 3), A = randn(Float32, 4, 3), b = randn(Float32, 4), g, rho = kw[:rho], rho_scale = kw[:rho_scale])
    @test iter.penalty_sequence.rho ≈ [0.2f0]
    # A penalty that was not given -- the default adaptive sequence, or a sequence without an
    # initial value -- is left to ADMM, and so are other algorithms and an explicit opt-out.
    @test scaled(ADMM(; maxit = 3)).kwargs == ADMM(; maxit = 3).kwargs
    alg = scaled(ADMM(; penalty_sequence = PA.ResidualBalancingPenalty()))
    @test !haskey(alg.kwargs, :rho_scale)
    @test scaled(FISTA(; maxit = 3)).kwargs == FISTA(; maxit = 3).kwargs
    opt_out = IterativeReconstruction(; regularization = TotalVariation2D(0.01), disable_operator_normalization = true)
    @test !haskey(scaled(ADMM(; rho = 0.05f0); m = opt_out).kwargs, :rho_scale)
end

@testitem "Fixed-penalty ADMM does not depend on the scale of the encoding" tags = [:minimizer, :reconstruction, :nfft] begin
    using LinearAlgebra
    using MriReconstructionToolbox: NonCartesianAcquisitionInfo

    # Multiplying the sensitivity maps and the data by `c` multiplies the encoding by `c` and
    # leaves the problem unchanged, once the data scaling is recomputed. A penalty relative to the
    # curvature makes ADMM's iterates unchanged too; an absolute one does not, which is how a
    # radial NFFT encoding (curvature ~10⁶) used to make the result independent of `λ`.
    nx, ny = 32, 32
    img = zeros(ComplexF32, nx, ny)
    img[10:22, 10:22] .= 1
    traj = radial_trajectory(64, 32; ordering = GoldenAngle())
    smaps = coil_sensitivities(nx, ny, 4)
    acq = NonCartesianAcquisitionInfo(nothing; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps)
    data = simulate_acquisition(img, acq)
    c = 1.0f3
    scaled_data = NonCartesianAcquisitionInfo(
        data.kspace_data .* c; trajectory = traj, image_size = (nx, ny), sensitivity_maps = smaps .* c
    )
    rec(d, λ) = reconstruct(
        d,
        IterativeReconstruction(;
            regularization = TotalVariation2D(λ), maxit = 10, reltol = 0,
            algorithm = ADMM(; rho = 0.05, maxit = 10, tol = 0, cg_tol = 0, cg_maxit = 5),
        );
        verbosity = Silent(),
    )
    x1, xc = rec(data, 0.01), rec(scaled_data, 0.01)
    @test norm(xc - x1) / norm(x1) < 1.0e-5
    # And the penalty reaches the image: λ changes the result.
    @test norm(rec(data, 1.0) - x1) / norm(x1) > 1.0e-2
end

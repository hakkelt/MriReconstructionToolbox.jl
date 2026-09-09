@testitem "Model builder: Eye + L1Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using AbstractOperators

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
        @test isapprox(model_val, data_fidelity + reg_val; rtol=1.0e-10, atol=1.0e-12)
    end
end

@testitem "Model builder: Eye + L1Image + L2Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using AbstractOperators

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
        @test isapprox(model_val, data_fidelity + reg1 + reg2; rtol=1.0e-10, atol=1.0e-12)
    end
end

@testitem "Model builder: Linear op + L2Image" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using AbstractOperators

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
        @test isapprox(model_val, data_fidelity + reg_val; rtol=1.0e-10, atol=1.0e-12)

        vars = StructuredOptimization.extract_variables(terms)
        xvar = vars[1]
        x0 = copy(~xvar)
        δ = 0.01 .* randn(size(x0))
        ~xvar .= x0 .+ δ
        model_val2 = eval_term(terms)

        data = 0.5 * sum(abs2, (𝒜 * (x0 .+ δ)) .- y)
        reg2 = MriReconstructionToolbox.calculate(reg, x0 .+ δ; threaded)
        @test isapprox(model_val2, data + reg2; rtol=1.0e-8, atol=1.0e-10)
    end
end

@testitem "Model builder: NamedDims y and A" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using NamedDims
    using AbstractOperators

    @testset "NamedDims y and A" for threaded in (false, true)
        x = rand(8, 8)
        𝒜 = MriReconstructionToolbox.NamedDimsOp{(:x, :y),(:x, :y)}(Eye(x))
        y = NamedDimsArray(copy(x), (:x, :y))
        reg = L1Image(0.15)
        terms = build_model(𝒜, y, reg; threaded)

        model_val = eval_term(terms)

        x̂ = 𝒜' * y
        data_fidelity = 0.5 * sum(abs2, (𝒜 * x̂) .- y)
        reg_val = MriReconstructionToolbox.calculate(reg, y; threaded)
        @test isapprox(model_val, data_fidelity + reg_val; rtol=1.0e-10, atol=1.0e-12)
    end
end

@testitem "Model builder: overload parity" tags = [:minimizer] setup = [ModelEval] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using AbstractOperators

    @testset "overload parity" for threaded in (false, true)
        x = rand(5, 5)
        𝒜 = Eye(x)
        y = copy(x)
        reg = L2Image(0.2)
        t1 = build_model(𝒜, y, reg; threaded)
        t2 = build_model(𝒜, y, (reg,); threaded)
        @test isapprox(eval_term(t1), eval_term(t2); atol=1.0e-12)
    end
end

@testitem "DouglasRachford default parameter patching" tags = [:minimizer] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    alg = DouglasRachford(maxit=100)
    patched = MriReconstructionToolbox.patch_algorithm_with_default_values(alg, 2.0)
    @test patched.kwargs[:gamma] == 0.5

    patched_no_lf = MriReconstructionToolbox.patch_algorithm_with_default_values(alg, nothing)
    @test patched_no_lf.kwargs[:gamma] == 1.0

    explicit = DouglasRachford(gamma=0.1)
    patched_explicit = MriReconstructionToolbox.patch_algorithm_with_default_values(explicit, 5.0)
    @test patched_explicit.kwargs[:gamma] == 0.1
end

@testitem "HardConsistency projection fast path vs inner-CG" tags = [:minimizer] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra
    using ProximalCore
    using ProximalOperators

    nx, ny = 16, 16
    x = rand(ComplexF32, nx, ny)
    mask = rand(Bool, nx, ny)
    mask[1, 1] = true
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny), subsampling=mask)
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
    v_cg = ProximalOperators._cg_solve_AAc(𝒜, 𝒜 * x_test - y; maxit=100, tol=1.0e-6)
    proj_cg = x_test .- 𝒜' * v_cg

    @test isapprox(proj_fast, proj_cg; rtol=1.0e-4, atol=1.0e-5)
    @test isapprox(𝒜 * proj_fast, y; rtol=1.0e-4, atol=1.0e-5)
end

@testitem "Reconstruction with HardConsistency + DouglasRachford" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra

    nx, ny = 16, 16
    x_true = zeros(ComplexF32, nx, ny)
    x_true[4:8, 4:8] .= 1.0f0 + 0.5f0im
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    method = IterativeReconstruction(
        L1Image(1.0e-6);
        algorithm=DouglasRachford(maxit=50, tol=1.0e-5),
        fidelity=HardConsistency(),
    )
    rec = reconstruct(acq_data, method; verbosity=Silent())
    @test isapprox(rec, x_true; rtol=1.0e-4, atol=1.0e-4)
end

@testitem "Unregularized Iterative Least-Squares with CGNR" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using LinearAlgebra

    nx, ny = 16, 16
    x_true = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    for disable_normalop in (false, true)
        method = IterativeReconstruction(;
            algorithm=CGNR(maxit=20, tol=1.0e-6),
            fidelity=L2Loss(),
            disable_normalop_optimization=disable_normalop,
        )
        rec = reconstruct(acq_data, method; verbosity=Silent())
        @test isapprox(rec, x_true; rtol=1.0e-4, atol=1.0e-4)
    end
end

@testitem "CGNR on radial data beats the plain adjoint" tags = [:reconstruction, :nfft, :quality] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator, NonCartesianAcquisitionInfo
    using LinearAlgebra

    nx, ny = 32, 32
    img = zeros(ComplexF32, nx, ny)
    img[10:22, 10:22] .= 1
    traj = radial_trajectory(64, 64; ordering=:golden_angle)
    smaps = coil_sensitivities(nx, ny, 4)
    acq = NonCartesianAcquisitionInfo(nothing; trajectory=traj, image_size=(nx, ny), sensitivity_maps=smaps)
    data = simulate_acquisition(img, acq)

    𝒜 = get_encoding_operator(data)
    nrmse(rec) = norm(rec .- img) / norm(img)
    nrmse_adjoint = nrmse(𝒜' * data.kspace_data)

    # The bare adjoint `𝒜'y` is an un-normalized-NFFT-scale warm start (off by orders of
    # magnitude), which a finite-`maxit` CG-SENSE solve does not correct on its own -- it used to
    # score *worse* than the plain adjoint. The scale-correct `𝒜'y/‖𝒜‖²` warm start
    # (`_direct_reconstruct`) fixes that.
    method = IterativeReconstruction(; algorithm=CGNR(maxit=20, tol=1.0e-6), fidelity=L2Loss())
    rec = reconstruct(data, method; verbosity=Silent())
    @test nrmse(rec) < nrmse_adjoint
end

@testitem "POGM matches FISTA on a single L1 regularizer" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox

    nx, ny = 32, 32
    x_true = zeros(ComplexF32, nx, ny)
    x_true[10:22, 10:22] .= 1
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny))
    acq_data = simulate_acquisition(x_true, acq)

    reg = L1Image(1.0e-3)
    fista_rec = reconstruct(
        acq_data, IterativeReconstruction(reg; algorithm=FISTA(maxit=100));
        verbosity=Silent(),
    )
    pogm_rec = reconstruct(
        acq_data, IterativeReconstruction(reg; algorithm=POGM(maxit=100));
        verbosity=Silent(),
    )
    @test isapprox(pogm_rec, fista_rec; rtol=1.0e-2, atol=1.0e-3)
end

@testitem "NoFidelity and error handling" tags = [:minimizer] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    nx, ny = 8, 8
    x = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny))
    acq_data = simulate_acquisition(x, acq)
    𝒜 = get_encoding_operator(acq_data)
    y = acq_data.kspace_data

    # NoFidelity with empty regularizations throws ArgumentError
    @test_throws ArgumentError build_model(𝒜, y, (); fidelity=NoFidelity())
end

@testitem "Diagnostic ArgumentError on single-solver parse failure" tags = [:minimizer, :reconstruction] begin
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator

    nx, ny = 16, 16
    x = rand(ComplexF32, nx, ny)
    acq = CartesianAcquisitionInfo(is3D=false, image_size=(nx, ny))
    acq_data = simulate_acquisition(x, acq)

    # Incompatible single solver (DouglasRachford with L2Loss and 2 L1 terms) throws informative ArgumentError
    method = IterativeReconstruction(L1Image(0.1), L1Image(0.2); algorithm=DouglasRachford(), fidelity=L2Loss())
    err = try
        reconstruct(acq_data, method; verbosity=Silent())
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
    using Test
    using MriReconstructionToolbox
    using MriReconstructionToolbox: get_encoding_operator
    using AbstractOperators: DiagOp
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

        x_pcgnr = run_iter(PA.PCGNRIteration(; x0=zeros(n), A=A, b=b, P=P, λ=λ))
        @test isapprox(x_pcgnr, x_reg; atol=1.0e-10)

        x_pcg = run_iter(PA.PCGIteration(; x0=zeros(n), A=A'A, b=A'b, P=P, λ=λ))
        @test isapprox(x_pcg, x_reg; atol=1.0e-10)
    end

    @testset "CGNR(; P) through reconstruct" begin
        nx, ny = 32, 32
        Random.seed!(5)
        x_true = ComplexF32.(rand(Float32, nx, ny))
        smaps = unname(coil_sensitivities(nx, ny, 4))
        # A smooth intensity ramp makes the coil coverage Σ|S_c|² span two orders of magnitude,
        # which is exactly what a diagonal image-domain preconditioner is for.
        smaps = ComplexF32.(smaps .* reshape(range(0.05f0, 1.0f0, length=nx), nx, 1, 1))
        sub = create_sampling_pattern(VariableDensitySampling(GaussianDistribution(), 2.0), (nx, ny))
        acq = CartesianAcquisitionInfo(
            is3D=false, image_size=(nx, ny), sensitivity_maps=smaps, subsampling=sub
        )
        data = simulate_acquisition(x_true, acq)

        λ = 1.0f-3
        coverage = real(sum(abs2, smaps; dims=3)[:, :, 1])
        # An `AbstractOperator` supports `mul!` but not `ldiv!`, so the *inverse* preconditioner is
        # what is passed, with `P_is_inverse = true`.
        Pinv = DiagOp(ComplexF32.(1 ./ (coverage .+ λ)))

        nrmse(r) = norm(r .- x_true) / norm(x_true)
        solve_at(k; kwargs...) = nrmse(
            reconstruct(
                data,
                IterativeReconstruction(
                    L2Image(λ); algorithm=CGNR(; tol=1.0e-14, kwargs...), maxit=k, tol=nothing,
                    fidelity=L2Loss(),
                );
                verbosity=Silent(),
            )
        )

        plain_32 = solve_at(32)
        pc_8 = solve_at(8; P=Pinv, P_is_inverse=true)
        # Same error in a quarter of the iterations: the point of the preconditioner is convergence
        # speed, not a different answer.
        @test pc_8 <= plain_32
        @test solve_at(8; P=Pinv, P_is_inverse=true) < solve_at(8)
    end

    @testset "a preconditioned solver is still recognized as Krylov" begin
        @test MriReconstructionToolbox._is_krylov_solver(CGNR(P=Diagonal(ones(4)), P_is_inverse=true))
        @test MriReconstructionToolbox._is_krylov_solver(CG(P=Diagonal(ones(4)), P_is_inverse=true))
        @test MriReconstructionToolbox._is_krylov_solver(CGNR())
        @test !MriReconstructionToolbox._is_krylov_solver(FISTA())
    end
end

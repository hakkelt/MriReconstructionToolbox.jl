using LinearAlgebra
using Test

using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: NormL1, LeastSquares, SqrNormL2, ElasticNet, Translate, SlicedSeparableSum
using ProximalAlgorithms
using ProximalAlgorithms:
    LBFGS,
    Broyden,
    AndersonAcceleration,
    NesterovExtrapolation,
    FixedNesterovSequence,
    SimpleNesterovSequence

@testset "Lasso small ($T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    lam = R(0.1) * norm(A' * b, Inf)
    @test typeof(lam) == R

    f_autodiff =
        ProximalAlgorithms.AutoDifferentiable(x -> (norm(x - b)^2) / 2, AutoZygote())
    fA_autodiff =
        ProximalAlgorithms.AutoDifferentiable(x -> (norm(A * x - b)^2) / 2, AutoZygote())
    f_prox = Translate(SqrNormL2(R(1)), -b)
    fA_prox = LeastSquares(A, b)
    g = NormL1(lam)

    Lf = opnorm(A)^2

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step, regret)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = R(1.01),
        )
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step, regret)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = R(1.01),
        )
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "POGM (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.POGM(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 400
        @test x0 == x0_backup
    end

    @testset "POGM restart test off Array storage" begin
        w, z, x = randn(T, n), randn(T, n), randn(T, n)
        for (a, b, c) in ((w, z, x), (z, w, x), (x, z, w))
            expected = ProximalAlgorithms._pogm_restart(a, b, c)
            @test ProximalAlgorithms._pogm_restart(view(a, :), view(b, :), view(c, :)) == expected
        end
    end

    @testset "FastForwardBackward (custom extrapolation)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        x, it = @inferred solver(
            x0 = x0,
            f = fA_autodiff,
            g = g,
            Lf = Lf,
            extrapolation_sequence = FixedNesterovSequence(real(T)),
        )
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOC (fixed step)" begin

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

    @testset "PANOC (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOCplus (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOCplus (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "DouglasRachford" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DouglasRachford(gamma = R(10) / opnorm(A)^2, tol = TOL)
        y, it = @inferred solver(x0 = x0, f = fA_prox, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 30
        @test x0 == x0_backup
    end

    @testset "DouglasRachford line search ($acc)" for (acc, maxit) in [
        (LBFGS(5), 17),
        (Broyden(), 19),
        (AndersonAcceleration(5), 12),
        (NesterovExtrapolation(FixedNesterovSequence), 36),
        (NesterovExtrapolation(SimpleNesterovSequence), 36),
    ]
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DRLS(tol = 10 * TOL, directions = acc)
        z, it = @inferred solver(x0 = x0, f = fA_prox, g = g, Lf = Lf)
        @test eltype(z) == T
        @test norm(z - x_star, Inf) <= 10 * TOL
        @test it < maxit
        @test x0 == x0_backup
    end

    @testset "AFBA dual start" begin
        # The dual starts at zero, not at `L * x0`: a warm start at the solution stays close to it
        # after one iteration instead of being pulled towards zero.
        iter = ProximalAlgorithms.AFBAIteration(x0 = copy(x_star), f = fA_autodiff, h = g, beta_f = opnorm(A)^2)
        @test iszero(iter.y0)
        state, _ = iterate(iter)
        @test norm(state.x - x_star) < norm(x_star) / 10
    end

    @testset "AFBA" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.AFBA(theta = 1, mu = 1, tol = R(1e-6))
        (x_afba, y_afba), it_afba = @inferred solver(
            x0 = x0,
            y0 = zeros(T, n),
            f = fA_autodiff,
            g = g,
            beta_f = opnorm(A)^2,
        )
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 80
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = 1, mu = 1, tol = R(1e-6))
        (x_afba, y_afba), it_afba = @inferred solver(
            x0 = x0,
            y0 = zeros(T, n),
            f = fA_autodiff,
            h = g,
            beta_f = opnorm(A)^2,
        )
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 100
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = 1, mu = 1, tol = R(1e-6))
        (x_afba, y_afba), it_afba =
            @inferred solver(x0 = x0, y0 = zeros(T, m), h = f_prox, L = A, g = g)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 150
        @test x0 == x0_backup
    end

    @testset "ChambollePock" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)

        # g = λ‖x‖₁, h(Ax) = ½‖Ax - b‖²
        solver = ProximalAlgorithms.ChambollePock(tol = R(1e-7), maxit = 10_000)
        (x_cp, y_cp), it_cp = @inferred solver(x0 = x0, g = g, h = f_prox, L = A)
        @test eltype(x_cp) == T
        @test eltype(y_cp) == T
        @test norm(x_cp - x_star, Inf) <= 1e-4
        @test x0 == x0_backup

        # Both terms through the dual: h = ½‖· - b‖² ⊕ λ‖·‖₁ of [A; I] x
        h = SlicedSeparableSum((f_prox, g), ((1:m,), ((m + 1):(m + n),)))
        K = vcat(A, Matrix{T}(I, n, n))
        solver = ProximalAlgorithms.ChambollePock(tol = R(1e-7), maxit = 10_000, ratio = 4)
        (x_cp, y_cp), it_cp = @inferred solver(x0 = x0, h = h, L = K)
        @test eltype(x_cp) == T
        @test length(y_cp) == m + n
        @test norm(x_cp - x_star, Inf) <= 1e-4
        @test x0 == x0_backup
    end

    @testset "SFISTA" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.SFISTA(tol = 10 * TOL)
        y, it = @inferred solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= 10 * TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "ADMM" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        @testset "$(typeof(ps).name.name)" for ps in [
            ProximalAlgorithms.FixedPenalty(),
            # ProximalAlgorithms.ResidualBalancingPenalty(adp_freq = 5), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.WohlbergPenalty(), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.BarzilaiBorweinSpectralPenalty(), # TODO: This does not converge, needs debugging
            ProximalAlgorithms.SpectralRadiusBoundPenalty(),
            ProximalAlgorithms.SpectralRadiusApproximationPenalty(),
        ]
            solver = ProximalAlgorithms.ADMM(tol = 1e-5, maxit=500, penalty_sequence = ps)
            x_admm, it_admm = @inferred solver(; x0, A, b, g)
            @test eltype(x_admm) == T
            @test norm(x_admm - x_star, Inf) <= 1e-3
            @test it_admm ≤ 500
            @test x0 == x0_backup

            # A caller that already holds `AᴴA` may hand it over instead of having ADMM
            # build a second one; the iterates must be identical.
            solver_aha = ProximalAlgorithms.ADMM(tol = 1e-5, maxit=500, penalty_sequence = ps)
            x_aha, it_aha = solver_aha(; x0, A, b, g, AHA = A' * A)
            @test x_aha ≈ x_admm
            @test it_aha == it_admm

            # Likewise for `Aᴴb`.
            solver_ahb = ProximalAlgorithms.ADMM(tol = 1e-5, maxit=500, penalty_sequence = ps)
            x_ahb, it_ahb = solver_ahb(; x0, A, b, g, AHb = A' * b)
            @test x_ahb ≈ x_admm
            @test it_ahb == it_admm
        end

        # `rho_scale` multiplies a given `rho`, wherever it was given, and leaves the default
        # sequence's starting value alone.
        R = real(T)
        @testset "rho_scale" begin
            it(; kw...) = ProximalAlgorithms.ADMMIteration(; x0, A, b, g, kw...)
            @test it(rho = 0.5, rho_scale = 4).penalty_sequence.rho == [R(2)]
            @test it(penalty_sequence = ProximalAlgorithms.SpectralRadiusBoundPenalty(rho = 0.5), rho_scale = 4).penalty_sequence.rho ==
                it(penalty_sequence = ProximalAlgorithms.SpectralRadiusBoundPenalty(rho = 2.0)).penalty_sequence.rho
            @test it(rho_scale = 4).penalty_sequence.rho == it().penalty_sequence.rho
            x_scaled, _ = ProximalAlgorithms.ADMM(tol = 1e-5, maxit = 500, rho = 0.25, rho_scale = 4)(; x0, A, b, g)
            x_plain, _ = ProximalAlgorithms.ADMM(tol = 1e-5, maxit = 500, rho = 1.0)(; x0, A, b, g)
            @test x_scaled ≈ x_plain
        end
    end

end

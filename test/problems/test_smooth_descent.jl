using LinearAlgebra
using Test

using ProximalOperators: SqrNormL2, HuberLoss, Translate, LeastSquares
using ProximalAlgorithms

ProximalAlgorithms.value_and_gradient!(y, f::Union{SqrNormL2, HuberLoss, Translate, LeastSquares}, x) =
    ProximalAlgorithms.ProximalCore.gradient!(y, f, x)

# A linear map whose `'` is its adjoint divided by `s`, as a Fourier transform normalised so that
# `'` is its inverse is.
struct ScaledAdjointMap{M, R}
    A::M
    s::R
end
struct ScaledAdjointMapT{M, R}
    A::M
    s::R
end
Base.:*(m::ScaledAdjointMap, x) = m.A * x
Base.adjoint(m::ScaledAdjointMap) = ScaledAdjointMapT(m.A, m.s)
LinearAlgebra.mul!(y, m::ScaledAdjointMap, x) = mul!(y, m.A, x)
LinearAlgebra.mul!(y, m::ScaledAdjointMapT, x) = (mul!(y, m.A', x); y ./= m.s; y)

@testset "Smooth descent ($T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    R = real(T)
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
        0.5 1.0 -2.0 0.0 1.0
        3.0 0.0 1.0 2.0 -1.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0, -1.0, 0.5]
    m, n = size(A)
    lam = R(0.3)
    TOL = R(1e-4)

    # ½‖Ax - b‖² + (λ/2)‖x‖², by its normal equations
    x_ridge = (A' * A + lam * I) \ (A' * b)
    f = (Translate(SqrNormL2(R(1)), -b), SqrNormL2(lam))

    @testset "$(nameof(alg)) ridge" for alg in (ProximalAlgorithms.NonlinearCG, ProximalAlgorithms.LimitedMemoryBFGS)
        x0 = zeros(T, n)
        solver = alg(tol = R(1e-7), maxit = 200)
        x, it = solver(x0 = x0, f = f, L = (A, I))
        @test eltype(x) == T
        @test norm(x - x_ridge, Inf) <= TOL
        @test it <= 50
        @test x0 == zeros(T, n)
    end

    # The least-squares term carrying its own operator (L = I): a quadratic, so its gradient is
    # evaluated once per iteration. On a quadratic cost NCG with exact line searches is linear
    # CG, done in about n steps; L-BFGS accepts inexact steps (`eta = 0.9`) and takes more.
    @testset "$(nameof(alg)) ridge, operator inside f" for alg in (ProximalAlgorithms.NonlinearCG, ProximalAlgorithms.LimitedMemoryBFGS)
        solver = alg(tol = R(1e-7), maxit = 200)
        x, it = solver(x0 = zeros(T, n), f = (LeastSquares(A, b), SqrNormL2(lam)))
        @test norm(x - x_ridge, Inf) <= TOL
        @test it <= (alg === ProximalAlgorithms.NonlinearCG ? 3n : 10n)
    end

    # The same cost with `A'` scaled down by 4: the gradient followed is that of
    # ¼·½‖Ax - b‖² + (λ/2)‖x‖², and the line search minimises that cost along it.
    x_scaled = (A' * A / 4 + lam * I) \ (A' * b / 4)
    @testset "$(nameof(alg)) scaled adjoint" for alg in (ProximalAlgorithms.NonlinearCG, ProximalAlgorithms.LimitedMemoryBFGS)
        solver = alg(tol = R(1e-7), maxit = 200)
        x, it = solver(x0 = zeros(T, n), f = f, L = (ScaledAdjointMap(A, R(4)), I))
        @test norm(x - x_scaled, Inf) <= TOL
    end

    # Huber data fit with a ridge term: no closed form, so checked against a long gradient descent
    # run and through the optimality condition.
    delta = R(0.5)
    fh = (Translate(HuberLoss(delta), -b), SqrNormL2(lam))
    grad_h(x) = A' * ProximalAlgorithms.ProximalCore.gradient(fh[1], A * x)[1] + lam * x
    @testset "$(nameof(alg)) Huber" for alg in (ProximalAlgorithms.NonlinearCG, ProximalAlgorithms.LimitedMemoryBFGS)
        solver = alg(tol = R(1e-8), maxit = 500)
        x, it = solver(x0 = zeros(T, n), f = fh, L = (A, I))
        @test norm(grad_h(x), Inf) <= 10 * TOL
        @test it <= 200
    end
end

@testset "Smooth descent assumptions" begin
    @test length(ProximalAlgorithms.get_assumptions(ProximalAlgorithms.NonlinearCGIteration)) == 1
    @test length(ProximalAlgorithms.get_assumptions(ProximalAlgorithms.LimitedMemoryBFGSIteration)) == 1
end

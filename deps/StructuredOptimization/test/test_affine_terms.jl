# A term `h(L·x + d)` hands an algorithm's operator slot the linear `L`, with `d` moved into the
# function, so algorithms whose updates need a linear operator solve affine terms correctly.

using ProximalAlgorithms: PANOC, ChambollePock, FastForwardBackward

@testset "affine terms in operator slots" begin
    Random.seed!(310)
    n, m = 8, 12
    A = randn(m, n)
    b = randn(m)

    @testset "split" begin
        x = Variable(n)
        term = ls(A * x - b)
        vars = StructuredOptimization.extract_variables(term)
        f, op = StructuredOptimization.split_affine(vars, term)
        @test is_linear(op)
        @test f isa ProximalOperators.Translate
        v = randn(n)
        @test f(op * v) ≈ 0.5 * norm(A * v - b)^2
        @test is_affine(term) && is_linear(term)
    end

    @testset "PANOC on a lasso" begin
        x = Variable(n)
        λ = 0.05
        ~x .= 0
        solve(problem(ls(A * x - b), λ * norm(x, 1)), FastForwardBackward(maxit = 20000, tol = 1.0e-12))
        reference = copy(~x)
        ~x .= 0
        _, kwargs, _ = StructuredOptimization.parse_problem(problem(ls(A * x - b), λ * norm(x, 1)), PANOC())
        @test is_linear(kwargs[:A])
        solve(problem(ls(A * x - b), λ * norm(x, 1)), PANOC(maxit = 5000, tol = 1.0e-10))
        @test norm(~x - reference) / norm(reference) < 1.0e-5
    end

    @testset "Chambolle-Pock with a displaced operator" begin
        D = randn(n, n) + 3I
        c = randn(n)
        y = randn(n)
        λ = 0.3
        # min ½‖x - y‖² + λ‖Dx - c‖₁, and the same problem in w = x - D⁻¹c, which has no
        # displacement inside the ℓ1 term.
        xc = D \ c
        w = Variable(n)
        ~w .= 0
        # `ChambollePock` is an AFBA iteration, which takes the least-squares term as its smooth
        # `f` and then needs that term's Lipschitz constant.
        cp() = ChambollePock(maxit = 20000, tol = 1.0e-10, beta_f = 1.0)
        solve(problem(ls(w - (y - xc)), λ * norm(D * w, 1)), cp())
        reference = ~w + xc
        x = Variable(n)
        ~x .= 0
        _, kwargs, _ = StructuredOptimization.parse_problem(problem(ls(x - y), λ * norm(D * x - c, 1)), cp())
        @test is_linear(kwargs[:L])
        solve(problem(ls(x - y), λ * norm(D * x - c, 1)), cp())
        @test norm(~x - reference) / norm(reference) < 1.0e-5
    end
end

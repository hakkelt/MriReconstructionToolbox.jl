# Regression tests for the correctness fixes in PLAN.md Phase 1.
# Each test targets one defect and asserts a *value*, not just a code path.

const SO = StructuredOptimization

# Capture the stdout of a diagnostics call as a String. `redirect_stdout` needs a real
# file descriptor, so route through a temp file rather than an IOBuffer.
function capture_stdout(f)
    return mktemp() do _path, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
end

@testset "Phase 1 regressions" begin

    # 1.1 — sum of smooth terms containing a nonlinear composition must not
    # double-count the displacement or λ (the ProximalOperators.Sum branch).
    @testset "1.1 no double displacement/λ in Sum branch" begin
        Random.seed!(11)
        x = Variable(3)
        c = randn(3)
        b = randn(3)
        lam = 3.0
        # nonlinear (sin) term with displacement c and λ=lam, plus a linear term
        ts = lam * ls(sin(x) + c) + ls(x - b)
        vars = SO.extract_variables(ts)
        asm = ProximalAlgorithms.SimpleTerm(:f => (SO.is_smooth,))
        res = SO.prepare(ts, asm, vars)
        @test res !== nothing
        f = res[1].second                     # ProximalOperators.Sum
        xt = randn(3)
        true_val = lam * 0.5 * norm(sin.(xt) + c)^2 + 0.5 * norm(xt - b)^2
        @test abs(f(xt) - true_val) < 1.0e-10
    end

    # 1.2 — OperatorTerm TermSet path must carry displacement only in the operator
    # (via weighted_function), never fold it into f as well.
    @testset "1.2 no double displacement in OperatorTerm TermSet path" begin
        Random.seed!(12)
        x = Variable(3)
        A1, A2 = randn(4, 3), randn(4, 3)
        b1, b2 = randn(4), randn(4)
        lam = 2.0
        ts = lam * ls(A1 * x - b1) + ls(A2 * x - b2)
        vars = SO.extract_variables(ts)
        # empty func/operator-properties so the affine (AffineAdd) operator is accepted,
        # forcing the OperatorTerm branch that used to double-count displacement.
        asm = ProximalAlgorithms.OperatorTerm(:f => (), :A => ())
        res = SO.prepare(ts, asm, vars)
        @test res !== nothing
        f = res[1].second
        op = res[2].second
        xt = randn(3)
        true_val = lam * 0.5 * norm(A1 * xt - b1)^2 + 0.5 * norm(A2 * xt - b2)^2
        @test abs(f(op * xt) - true_val) < 1.0e-9
    end

    # 1.3 — the func₂ branch of the InfConv TermSet path must return the same
    # operator that was checked (the full stacked op), not only the first term's.
    @testset "1.3 InfConv func₂ returns the full checked operator" begin
        Random.seed!(13)
        x = Variable(3)
        A1, A2 = randn(4, 3), randn(5, 3)
        b1, b2 = randn(4), randn(5)
        ts = ls(A1 * x - b1) + ls(A2 * x - b2)
        vars = SO.extract_variables(ts)
        # func₁ unsatisfiable (indicator), func₂ trivially satisfiable => func₂ branch.
        asm = ProximalAlgorithms.OperatorTermWithInfimalConvolution(
            :h => (SO.is_set_indicator,), :l => (), :A => ()
        )
        res = SO.prepare(ts, asm, vars)
        @test res !== nothing
        returned_op = res[2].second
        full_op = SO.extract_affines(vars, ts)
        # codomain must span BOTH terms (4+5), not just the first (4).
        @test size(returned_op, 1) == size(full_op, 1)
    end

    # 1.4 — least-squares λ scaling is by √λ, so CG-family solvers weight the data
    # term correctly relative to the SquaredL2 regularizer.
    @testset "1.4 least-squares √λ scaling" begin
        Random.seed!(14)
        A = randn(8, 5)
        b = randn(8)
        a = 4.0
        # direct: the prepared operator is scaled by √a, not a.
        xu = Variable(5)
        t = a * ls(A * xu - b)
        vars = SO.extract_variables(t)
        res = SO.prepare(t, ProximalAlgorithms.LeastSquaresTerm(:A => (is_linear,), :b), vars)
        opres = res[1].second
        v = randn(5)
        @test norm(opres * v - sqrt(a) * (A * v)) < 1.0e-9

        # end-to-end: CGNR (LeastSquaresTerm+SquaredL2Term) must agree with PANOCplus
        # (smooth path, unaffected by this bug) on the same weighted ridge problem.
        r = 0.3
        xc = Variable(5)
        solve(problem(a * ls(A * xc - b) + r * norm(xc, 2)^2), ProximalAlgorithms.CGNR(maxit = 5000, tol = 1.0e-12))
        xp = Variable(5)
        solve(problem(a * ls(A * xp - b) + r * norm(xp, 2)^2), ProximalAlgorithms.PANOCplus(maxit = 8000, tol = 1.0e-10))
        @test norm(~xc - ~xp) < 1.0e-4
    end

    # 1.5 — weighted SqrNormL2WithNormalOp gradient applies weights in the codomain
    # (Aᴴ·diag(λ)·A·x), and strong convexity requires full column rank.
    @testset "1.5 weighted normal-op gradient and strong convexity" begin
        Random.seed!(15)
        Lm = randn(7, 4)
        L = MatrixOp(Lm)
        lam = rand(7) .+ 0.5           # array weights
        f = SO.SqrNormL2WithNormalOp(L, lam)
        xv = randn(4)
        yv = zero(xv)
        v = gradient!(yv, f, xv)
        @test norm(yv - Lm' * (lam .* (Lm * xv))) < 1.0e-9
        @test abs(f(xv) - 0.5 * sum(lam .* (Lm * xv) .^ 2)) < 1.0e-10
        # finite-difference check of the gradient
        g_fd = similar(xv)
        h = 1.0e-6
        for k in eachindex(xv)
            e = zero(xv); e[k] = h
            g_fd[k] = (f(xv + e) - f(xv - e)) / (2h)
        end
        @test norm(yv - g_fd) / norm(g_fd) < 1.0e-4

        # tall, full-column-rank operator with positive weights => strongly convex
        @test SO.is_strongly_convex(typeof(f))
        # fat operator cannot have full column rank => not strongly convex
        fw = SO.SqrNormL2WithNormalOp(MatrixOp(randn(4, 7)), rand(4) .+ 0.5)
        @test !SO.is_strongly_convex(typeof(fw))
    end

    # 1.6 — solve with a Vector of a concrete algorithm type must dispatch.
    @testset "1.6 solve with a vector of solvers" begin
        Random.seed!(16)
        A = randn(6, 4)
        b = randn(6)
        x = Variable(4)
        p = problem(ls(A * x - b))
        sol = solve(p, [ProximalAlgorithms.PANOCplus(tol = 1.0e-6, maxit = 2000)])
        @test sol !== nothing
        # also a tuple of heterogeneous solvers
        x2 = Variable(4)
        p2 = problem(ls(A * x2 - b))
        sol2 = solve(p2, (ProximalAlgorithms.PANOCplus(tol = 1.0e-6, maxit = 2000),))
        @test sol2 !== nothing
    end

    # 1.7 — the no-solver auto-select path handles a Tuple minimizer (multi-variable).
    @testset "1.7 auto-select multi-variable solve" begin
        Random.seed!(17)
        A1 = randn(6, 4)
        A2 = randn(6, 4)
        bb = randn(6)
        x1 = Variable(4)
        x2 = Variable(4)
        p = problem(ls(A1 * x1 - A2 * x2 - bb) + 1.0e-2 * norm(x1, 1))
        # Should not throw regardless of whether the minimizer comes back as a Tuple.
        sol = solve(p)
        @test sol !== nothing
    end

    # 1.8 — scalar * TermSet stays a TermSet; scalar * Term preserves repr.
    @testset "1.8 scalar-* on TermSet and repr preservation" begin
        x = Variable(3)
        ts = ls(x) + norm(x, 1)
        @test 2.0 * ts isa SO.TermSet
        @test length(2.0 * ts) == length(ts)
        t = SO.Term(norm(x, 1), "custom_repr")
        @test (3.0 * t).repr == "custom_repr"
    end

    # 1.1/1.2 residue — the *diagnostics* printed for the OperatorTerm and InfConv paths
    # must show the same decomposition the matching `prepare` would build. They used to
    # print a displacement-folded `PrecomposeDiagonal` next to an operator that still
    # carried the same displacement, i.e. a decomposition with the displacement applied
    # twice, which is not the problem that would have been solved.
    @testset "1.1/1.2 diagnostics do not double-count displacement" begin
        Random.seed!(112)
        x = Variable(3)
        A1, A2 = randn(4, 3), randn(4, 3)
        b1, b2 = randn(4), randn(4)
        ts = 2.0 * ls(A1 * x - b1) + ls(A2 * x - b2)
        vars = SO.extract_variables(ts)

        # An assumption whose operator side cannot be satisfied, so the decomposition is
        # printed rather than accepted.
        op_asm = ProximalAlgorithms.OperatorTerm(:f => (SO.is_proximable,), :A => (is_eye,))
        out = capture_stdout(() -> SO.print_diagnostics(ts, op_asm, vars))
        @test occursin("A possible decomposition", out)
        @test !occursin("PrecomposeDiagonal", out)

        inf_asm = ProximalAlgorithms.OperatorTermWithInfimalConvolution(
            :f => (SO.is_proximable,), :g => (SO.is_proximable,), :A => (is_eye,)
        )
        out2 = capture_stdout(() -> SO.print_diagnostics(ts, inf_asm, vars))
        @test !occursin("PrecomposeDiagonal", out2)

        # The single-term InfConv diagnostics path uses the same convention.
        t = ls(A1 * x - b1)
        out3 = capture_stdout(() -> SO.print_diagnostics(t, inf_asm, (x,)))
        @test !occursin("PrecomposeDiagonal", out3)
    end

    # 1.9 — UnregularIndex length counts iterator states (prod), not sum.
    @testset "1.9 UnregularIndex length" begin
        idx = SO.UnregularIndex((2, 3))
        @test length(idx) == 6
        @test length(collect(idx)) == 6
        idx2 = SO.UnregularIndex((2, 2, 2))
        @test length(idx2) == 8
        @test length(collect(idx2)) == length(idx2)
    end

end

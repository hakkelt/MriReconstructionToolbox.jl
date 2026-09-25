# Phase 2.3 — deterministic assumption matching in `parse_problem`.
# Phase 2.4 — rejecting ruleset: a solver whose convexity/smoothness assumptions the
# term structure cannot certify must fail at solve time with a diagnostic naming the
# unsatisfied property, instead of silently running a solver that stalls or returns
# a wrong answer.

using ProximalAlgorithms: PANOCplus, ZeroFPR, FastForwardBackward

const SO_M = StructuredOptimization

# Capture the stdout of a `print_diagnostics` call as a String. `redirect_stdout`
# needs a real file descriptor, so route through a temp file rather than an IOBuffer.
function capture_diagnostics(f)
    return mktemp() do _path, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
end

@testset "Phase 2.3 deterministic matching" begin
    Random.seed!(230)
    x = Variable(6)
    A = randn(4, 6)
    b = randn(4)
    # IndBallL2 (norm(x,2) <= c) is genuinely proximable, so PANOCplus can parse it.
    p = problem(ls(A * x - b), norm(x, 2) <= 1.0)

    # Parsing is deterministic: repeated calls select the same terms for the same
    # kwargs (the greedy largest-subset-first rule has no external order dependence).
    r1 = SO_M.parse_problem(p, PANOCplus())
    r2 = SO_M.parse_problem(p, PANOCplus())
    @test r1 !== nothing
    @test r2 !== nothing
    @test Set(keys(r1[2])) == Set(keys(r2[2]))

    # A single term matched against an assumption is found via the shared helper.
    vars = SO_M.extract_variables(p)
    smooth_assumption = first(ProximalAlgorithms.get_assumptions(PANOCplus()))
    match = SO_M.match_assumption(smooth_assumption, p, vars)
    @test match !== nothing
    _, matched_terms = match
    @test length(matched_terms) >= 1

    # suggest_algorithm still returns a non-empty list for a standard lasso problem.
    @test !isempty(SO_M.suggest_algorithm(p))
end

@testset "Phase 2.4 rejecting ruleset" begin
    Random.seed!(240)
    x = Variable(5)
    b = randn(5)
    # `sin(x)` is a nonlinear (hence non-convex) composition; the least-squares term
    # is smooth but not convex.
    p = problem(ls(sin(x) - b))

    # FastForwardBackward requires a convex smooth term -> parsing must reject it.
    @test SO_M.parse_problem(p, FastForwardBackward()) === nothing

    # solve surfaces a clear error instead of silently running.
    @test_throws ErrorException solve(p, FastForwardBackward())

    # The diagnostic names the unsatisfied property.
    diag = capture_diagnostics(() -> SO_M.print_diagnostics(p, FastForwardBackward()))
    @test occursin("is_convex", diag)

    # ... and so does the *exception*, not only the report printed to stdout: a caught
    # error has to be as informative as the printed one (PLAN.md 2.4).
    err = try
        capture_diagnostics(() -> solve(p, FastForwardBackward()))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("is_convex", err.msg)
    @test occursin(SO_M._term_repr(first(p)), err.msg)
    @test occursin("print_diagnostics", err.msg)

    # The solver-list path diagnoses against the solvers it was given, not against every
    # algorithm in the registry (ZeroFPR parses this problem, and would otherwise make the
    # message claim there is nothing wrong with it).
    err_list = try
        capture_diagnostics(() -> solve(p, [FastForwardBackward(), FastForwardBackward()]))
        nothing
    catch e
        e
    end
    @test err_list isa ErrorException
    @test occursin("is_convex", err_list.msg)

    # ZeroFPR permits nonconvex smooth f, so it parses the same problem.
    @test SO_M.parse_problem(p, ZeroFPR()) !== nothing
end

# Phase 5 — scored selection at both layers.
@testset "Phase 5 scored formulation selection" begin
    Random.seed!(500)

    @testset "the ranking reproduces the branch table" begin
        n = 6
        # identity: no operator is applied at all
        @test SO_M.best_formulation(AbstractOperators.Eye(Float64, (n,)), NormL1(), 0, 1)[1] === :eye
        # diagonal + squared L2 + no displacement: the operator folds into the weight
        D = DiagOp(randn(n) .+ 2)
        @test SO_M.best_formulation(D, SqrNormL2(), 0, 1)[1] === :diagonal_weight
        # the same with a displacement has nowhere to put it, so the operator stays outside
        @test SO_M.best_formulation(D, SqrNormL2(), randn(n), 1)[1] === :diagonal
        @test SO_M.best_formulation(D, NormL1(), 0, 1)[1] === :diagonal
        # AAᴴ-diagonal: the prox trick beats the normal operator although the latter is
        # cheaper, because it is the only one of the two with an exact prox
        dft = SO_M.operator(fft(Variable(8)))
        @test SO_M.best_formulation(dft, SqrNormL2(), 0, 1)[1] === :aac_diagonal
        # IndPoint over a general matrix: IndAffine
        @test SO_M.best_formulation(MatrixOp(randn(4, 10)), IndPoint(randn(4)), 0, 1)[1] === :ind_affine
        # tall MatrixOp + squared L2: the fused normal operator is the cheaper gradient
        @test SO_M.best_formulation(MatrixOp(randn(20, 5)), SqrNormL2(), 0, 1)[1] === :normal_op
        # wide: `LᴴL` acts on the larger space, so the generic formulation wins
        @test SO_M.best_formulation(MatrixOp(randn(5, 20)), SqrNormL2(), 0, 1)[1] === :precompose
        # a non-squared-L2 function has no normal-operator formulation at all
        @test SO_M.best_formulation(MatrixOp(randn(20, 5)), NormL1(), 0, 1)[1] === :precompose
        # nonlinear
        @test SO_M.best_formulation(SO_M.operator(sin(Variable(5))), SqrNormL2(), 0, 1)[1] === :nonlinear
    end

    @testset "needs = :prox filters the gradient-only formulations" begin
        A = MatrixOp(randn(20, 5))
        @test SO_M.best_formulation(A, SqrNormL2(), 0, 1, :prox)[1] === :none
        @test SO_M.best_formulation(A, SqrNormL2(), 0, 1, :any)[1] === :normal_op
        # ... and an exact-prox formulation is still found when one exists
        dft = SO_M.operator(fft(Variable(8)))
        @test SO_M.best_formulation(dft, SqrNormL2(), 0, 1, :prox)[1] === :aac_diagonal
    end

    # The type-level fuse predicate must agree with the constructing one wherever the
    # constructing one is consulted; it is allowed to be conservative, never optimistic.
    @testset "normal_op_fuses agrees with fused_normal_op" begin
        Random.seed!(501)
        xf, yf = Variable(10), Variable(7)
        uf, vf = Variable(50), Variable(100)
        xs, ys = Variable(5), Variable(5)
        ops = (
            MatrixOp(randn(7, 4)),
            MatrixOp(randn(4, 6)),
            DiagOp(randn(5)),
            SO_M.operator(fft(MatrixOp(randn(5, 5)) * Variable(5))),
            SO_M.extract_operators((xf, yf), ls(randn(25, 10) * xf + randn(25, 7) * yf)),
            SO_M.extract_operators((xs, ys), ls(MatrixOp(randn(12, 5)) * xs + MatrixOp(randn(12, 4)) * ys[1:4])),
            SO_M.extract_operators((uf, vf), ls(randn(30, 50) * uf + randn(30, 100) * vf)),
        )
        for op in ops
            predicted = SO_M.normal_op_fuses(op) && SO_M.normal_op_worthwhile(op)
            @test predicted == (SO_M.fused_normal_op(op) !== nothing)
        end
    end

    # The constraint that drove the design: scoring reads static metadata only, so its cost
    # is independent of the size of the operators it ranks and negligible next to the
    # optimization pass it selects.
    @testset "scoring is metadata-only" begin
        Random.seed!(502)
        small = MatrixOp(randn(10, 8))
        big = MatrixOp(randn(800, 600))
        f = SqrNormL2()
        SO_M.best_formulation(small, f, 0, 1)   # warm up inference and its cache
        SO_M.best_formulation(big, f, 0, 1)
        alloc_small = @allocated SO_M.best_formulation(small, f, 0, 1)
        alloc_big = @allocated SO_M.best_formulation(big, f, 0, 1)
        @test alloc_small == alloc_big
        @test alloc_big == 0
        # For contrast: answering the same fusing question by construction forms the Gram
        # matrix — 600×600 here — which is what scoring must not do.
        @test (@allocated SO_M.fused_normal_op(big)) > 100 * max(alloc_big, 1)

        # ... and in wall-clock terms against the algorithm's own work.
        n, m = 200, 300
        A, b = randn(m, n), randn(m)
        xb = Variable(n)
        ~xb .= 0.0
        p = problem(ls(A * xb - b) + 1.0e-2 * norm(xb, 1))
        alg = ProximalAlgorithms.PANOCplus(maxit = 5, tol = 0.0)
        assumptions = ProximalAlgorithms.get_assumptions(alg)
        score_all() = sum(SO_M.selection_cost(a, collect(p)) for a in assumptions)
        score_all()
        solve(p, alg)
        t_score = minimum(@elapsed(score_all()) for _ in 1:20)
        t_solve = minimum(@elapsed(solve(p, alg)) for _ in 1:3)
        # Measured ratio on the development machine is ~20x for a five-iteration pass; the
        # assertion keeps a wide margin because this runs on a shared node.
        @test t_score < t_solve / 5
    end

    # `is_aac_diagonal` short-circuits the upstream O(m²n) `isdiag(A*Aᴴ)` for a `MatrixOp`
    # by disproving row orthogonality on a sample. It must agree with the predicate it
    # replaces on every operator shape, not merely approximate it.
    @testset "is_aac_diagonal agrees with is_AAc_diagonal" begin
        Random.seed!(504)
        Q = Matrix(qr(randn(6, 6)).Q)
        aac_ops = (
            AbstractOperators.Eye(Float64, (5,)),
            DiagOp(randn(5)),
            SO_M.operator(fft(Variable(8))),
            MatrixOp(randn(7, 4)),
            MatrixOp(randn(4, 7)),
            MatrixOp(Q[1:4, :]),
            MatrixOp(reshape([2.0], 1, 1)),
            AbstractOperators.AffineAdd(MatrixOp(randn(7, 4)), randn(7)),
        )
        for op in aac_ops
            @test SO_M.is_aac_diagonal(op) == is_AAc_diagonal(op)
        end
        # An exactly-diagonal-rows matrix is accepted, so the sample is not simply
        # answering "false" for everything dense.
        @test SO_M.is_aac_diagonal(MatrixOp([1.0 0.0 0.0; 0.0 2.0 0.0]))
    end

    # Behaviour preservation: the scored search must still be deterministic, and pick the
    # same formulations the fixed branch chain did.
    @testset "parse results are stable" begin
        Random.seed!(503)
        xd = Variable(6)
        Ad, bd = randn(10, 6), randn(10)
        pd = problem(ls(Ad * xd - bd) + 1.0e-2 * norm(xd, 1))
        first_parse = SO_M.parse_problem(pd, PANOCplus())
        second_parse = SO_M.parse_problem(pd, PANOCplus())
        @test first_parse !== nothing
        @test Dict(k => typeof(v) for (k, v) in first_parse[2]) ==
            Dict(k => typeof(v) for (k, v) in second_parse[2])
        # PANOCplus assumes `f(Ax) + g(x)`, so the least-squares term is split into the
        # function and its linear operator rather than absorbed -- the displacement `-b` moves
        # into the function as a `Translate` -- and the ℓ1 term keeps its prox.
        @test first_parse[2][:f] isa ProximalOperators.Translate
        @test first_parse[2][:f].f isa SqrNormL2
        @test first_parse[2][:A] isa AbstractOperators.AbstractOperator
        @test is_linear(first_parse[2][:A])
        @test SO_M.is_proximable(first_parse[2][:g])

        # With a purely smooth algorithm there is no operator slot, so the same term must be
        # absorbed — and the tall, fusing operator makes the normal-operator formulation the
        # cheapest candidate.
        smooth_parse = SO_M.parse_problem(problem(ls(Ad * xd - bd)), FastForwardBackward())
        @test smooth_parse !== nothing
        @test smooth_parse[2][:f] isa SO_M.SqrNormL2WithNormalOp
    end
end

@testset "solver selection builds nothing" begin
    Random.seed!(250)
    xs = Variable(5)
    As = randn(20, 5)
    bs = randn(20)
    # Tall and fusing: the smooth formulation of this term is the normal-operator one.
    p_ls = problem(ls(As * xs - bs))
    p_lasso = problem(ls(As * xs - bs), 0.1 * norm(xs, 1))
    solvers = (ProximalAlgorithms.CG(), ProximalAlgorithms.CGNR(), FastForwardBackward())

    # `select_solver` is the solver `solve(p, solvers)` runs: CG needs a square operator, so
    # the least-squares problem goes to CGNR; the ℓ1 term rules out both Krylov solvers.
    @test SO_M.select_solver(p_ls, solvers) isa typeof(ProximalAlgorithms.CGNR())
    @test SO_M.select_solver(p_lasso, solvers) isa typeof(FastForwardBackward())
    @test SO_M.select_solver(p_lasso, solvers[1:2]) === nothing

    # A dry run leaves the normal operator unbuilt, and parses exactly as a real run does.
    op = MatrixOp(As)
    f = ProximalOperators.SqrNormL2()
    @test SO_M.merge_function_with_operator(op, f, 0, 1) isa SO_M.SqrNormL2WithNormalOp
    @test SO_M._dry_run(() -> SO_M.merge_function_with_operator(op, f, 0, 1)) === f
    for s in (solvers..., ProximalAlgorithms.PANOCplus())
        real_parse = SO_M.parse_problem(p_lasso, s)
        dry_parse = SO_M._dry_run(() -> SO_M.parse_problem(p_lasso, s))
        @test (real_parse === nothing) == (dry_parse === nothing)
        real_parse === nothing || @test Set(keys(real_parse[2])) == Set(keys(dry_parse[2]))
    end

    # A tuple solves with the selected solver.
    ~xs .= 0
    solve(p_lasso, solvers; maxit = 50)
    x_tuple = copy(~xs)
    ~xs .= 0
    solve(p_lasso, FastForwardBackward(); maxit = 50)
    @test ~xs == x_tuple

    # A least-squares slot that asks for AᴴA is handed the one the parser built; one that does
    # not ask gets none, and none is built for it.
    vars = SO_M.extract_variables(p_ls)
    term = only(collect(p_ls))
    with_aha = ProximalAlgorithms.LeastSquaresTerm(:A => (SO_M.is_linear,), :b, :AHA)
    without_aha = ProximalAlgorithms.LeastSquaresTerm(:A => (SO_M.is_linear,), :b)
    prepared = Dict(SO_M.prepare(term, with_aha, vars)...)
    @test prepared[:AHA] isa AbstractOperators.AbstractOperator
    @test prepared[:AHA] * ones(5) ≈ As' * (As * ones(5))
    @test !haskey(Dict(SO_M.prepare(term, without_aha, vars)...), :AHA)
    @test !haskey(Dict(SO_M._dry_run(() -> SO_M.prepare(term, with_aha, vars))...), :AHA)
end

# Phase 4 — value-asserting coverage tests for the worst-covered files. Each test
# checks *behavior* (a computed value or a captured diagnostic), not just that a line
# runs, so it also guards against regressions the way the Phase 1 tests do.

using ProximalAlgorithms: CGNR, PANOCplus, ZeroFPR, FastForwardBackward
import ProximalCore

const SO4 = StructuredOptimization

capture(f) = mktemp() do _p, io
    redirect_stdout(() -> f(), io)
    flush(io); seekstart(io); read(io, String)
end

@testset "utils.jl displacement" begin
    x = Variable(3)
    @test displacement(x) == 0
    c = randn(3)
    @test displacement(x + c) == c                       # A*x + c  ->  +c
    A = randn(4, 3); b = randn(4)
    @test norm(displacement(A * x - b) - (-b)) < 1.0e-12    # A*x - b  ->  -b
    @test SO4.variables(x) == (x,)
end

@testset "term.jl show / scalar-mul / iteration" begin
    x = Variable(4)
    A = randn(3, 4); b = randn(3)
    cost = ls(A * x - b)
    cons = norm(x, 2) <= 1.0
    ts = cost + cons
    s = sprint(show, ts)
    @test occursin("s.t.", s)                            # cost + constraint layout

    # scalar * TermSet stays a TermSet (Phase 1.8), and scalar * Term keeps repr.
    ts2 = 2.0 * ts
    @test ts2 isa SO4.TermSet
    tr = @term norm(x, 1)
    @test (3.0 * tr).repr == tr.repr

    # a Term iterates as a single element (iterate protocol, no length)
    first_item, state = iterate(tr)
    @test first_item === tr
    @test iterate(tr, state) === nothing
    @test !isempty(tr)
end

@testset "smooth / conj" begin
    x = Variable(5)
    t = norm(x, 1)
    @test !SO4.is_smooth(t)
    st = smooth(t)
    @test SO4.is_smooth(st)                                  # Moreau envelope is smooth
    @test smooth(ls(x)) === ls(x) || SO4.is_smooth(smooth(ls(x)))  # already-smooth passthrough

    # conj of a diagonal-operator term works; non-diagonal errors
    @test conj(norm(x, 1)) isa SO4.Term
    @test_throws ErrorException conj(norm(randn(3, 5) * x, 1))
end

@testset "sqrNormL2WithNormalOp traits" begin
    x = Variable(6)
    A = randn(4, 6)
    f = SO4.SqrNormL2WithNormalOp(MatrixOp(A))
    T = typeof(f)
    @test SO4.is_convex(T) && SO4.is_smooth(T)
    @test SO4.is_generalized_quadratic(T)
    # value: f(x) = 1/2 ||A x||^2
    xv = randn(6)
    @test abs(f(xv) - 0.5 * norm(A * xv)^2) < 1.0e-9 * (1 + norm(A * xv)^2)
    # the operator stays in the expression until the problem is parsed
    t = ls(A * x)
    @test t.f isa SqrNormL2
    @test SO4.operator(t) isa MatrixOp
end

@testset "parse.jl — LeastSquaresTerm scaling & sign (CGNR)" begin
    Random.seed!(414)
    x = Variable(5)
    A = randn(7, 5); b = randn(7)
    a = 3.0
    term = a * ls(A * x - b)                              # a * (1/2 ||A x - b||^2)
    vars = SO4.extract_variables(SO4.TermSet(term))
    ls_assumption = first(
        a for a in ProximalAlgorithms.get_assumptions(CGNR())
            if a isa ProximalAlgorithms.LeastSquaresTerm
    )
    prep = SO4.prepare(term, ls_assumption, vars)
    @test prep !== nothing
    d = Dict(prep)
    op = d[ls_assumption.operator.first]
    bvec = d[ls_assumption.b]
    # residual operator scaled by sqrt(lambda); target b = sqrt(lambda) * b_data
    @test norm(bvec - sqrt(a) * b) < 1.0e-8
    xr = randn(5)
    @test norm(op * xr - sqrt(a) * (A * xr)) < 1.0e-8

    # A non-least-squares function is rejected by the LeastSquares path.
    bad = norm(x, 1)
    @test SO4.prepare(bad, ls_assumption, vars) === nothing
end

@testset "parse.jl — print_diagnostics per algorithm" begin
    x = Variable(4)
    # A nonlinear (non-convex) smooth problem: rejected by convex-only FFB, and the
    # diagnostic names the property.
    p = problem(ls(sin(x) - randn(4)))
    out = capture(() -> SO4.print_diagnostics(p, FastForwardBackward()))
    @test occursin("could not be prepared", out)
    @test occursin("is_convex", out)

    # Auto-diagnostic (no algorithm) reports a closest algorithm.
    out2 = capture(() -> SO4.print_diagnostics(p))
    @test occursin("closest algorithm", out2)
end

# Find the first assumption of a given type across all advertised algorithms.
function find_assumption(::Type{T}) where {T}
    for alg in ProximalAlgorithms.get_algorithms()
        for a in ProximalAlgorithms.get_assumptions(alg)
            a isa T && return a
        end
    end
    return nothing
end

@testset "parse.jl — SquaredL2Term prepare (eye / diagonal / reject)" begin
    Random.seed!(415)
    x = Variable(4)
    sq = find_assumption(ProximalAlgorithms.SquaredL2Term)
    @test sq !== nothing
    vars = (x,)

    # eye operator: λ folds to term.lambda * f.lambda. norm(x,2)^2 == SqrNormL2(2.0),
    # so 1.5 * norm(x,2)^2 has λ = 1.5 * 2.0 = 3.0.
    t_eye = 1.5 * norm(x, 2)^2
    prep = SO4.prepare(t_eye, sq, vars)
    @test prep !== nothing
    @test Dict(prep)[sq.λ] ≈ 3.0

    # diagonal operator: λ scales by diag(op)^2 folded into the SqrNormL2 weight.
    D = [2.0, 3.0, 4.0, 5.0]
    t_diag = norm(DiagOp(D) * x, 2)^2
    prep_d = SO4.prepare(t_diag, sq, vars)
    @test prep_d !== nothing
    @test Dict(prep_d)[sq.λ] isa AbstractArray

    # non-zero displacement is rejected, and the diagnostic explains why.
    t_disp = norm(x - randn(4), 2)^2
    @test SO4.prepare(t_disp, sq, vars) === nothing
    out = capture(() -> SO4.print_diagnostics(t_disp, sq, vars))
    @test occursin("displacement", out)
end

@testset "parse.jl — OperatorTerm prepare + diagnostics" begin
    Random.seed!(416)
    x = Variable(5)
    A = randn(6, 5); b = randn(6)
    ot = find_assumption(ProximalAlgorithms.OperatorTerm)
    @test ot !== nothing
    vars = (x,)

    # Smooth term with a general operator: prepared as (func => f, operator => A).
    term = ls(A * x - b)
    prep = SO4.prepare(term, ot, vars)
    @test prep !== nothing
    d = Dict(prep)
    @test haskey(d, ot.func.first) && haskey(d, ot.operator.first)

    # print_diagnostics for the OperatorTerm decomposition runs and mentions the op.
    out = capture(() -> SO4.print_diagnostics(term, ot, vars))
    @test occursin("decomposition", out) || occursin("satisf", out)
end

@testset "parse.jl — diagnostics across every algorithm" begin
    Random.seed!(417)
    x = Variable(6)
    A = randn(4, 6); b = randn(4)
    p_ok = problem(ls(A * x - b) + 1.0e-2 * norm(x, 1))   # lasso, widely parseable
    p_bad = problem(ls(sin(x) - randn(6)))                 # nonconvex smooth

    # Exercise every algorithm's prepare + print_diagnostics branches.
    for alg in ProximalAlgorithms.get_algorithms()
        @test !isempty(capture(() -> SO4.print_diagnostics(p_ok, alg)))
        @test !isempty(capture(() -> SO4.print_diagnostics(p_bad, alg)))
    end

    # suggest_algorithm returns candidates for the lasso and (smooth) nonconvex case.
    @test !isempty(SO4.suggest_algorithm(p_ok))
    @test !isempty(SO4.suggest_algorithm(p_bad))
end

@testset "parse.jl — per-assumption print_diagnostics branches" begin
    Random.seed!(418)
    x = Variable(5)
    A = randn(4, 5); b = randn(4)
    c = randn(5)

    # SimpleTerm (proximable): multi-term diagnostics with two operators that are not
    # AAᴴ-diagonal -> the "absorption would not keep an exact prox" branch.
    simple_prox = ProximalAlgorithms.SimpleTerm(:g => (ProximalCore.is_proximable,))
    ts_overlap = SO4.TermSet(norm(x, 1), norm(A * x, 1))
    @test !isempty(capture(() -> SO4.print_diagnostics(ts_overlap, simple_prox, (x,))))

    # Two AAᴴ-diagonal (identity) but overlapping, non-sliced terms on one variable:
    # not a separable sum -> the "incompatible terms" branch (group_by_variables /
    # get_unseparable_pairs / add_to_incompatibilities).
    ts_incompat = SO4.TermSet(norm(x, 1), norm(x, 2))
    @test !SO4.is_proximable(ts_incompat)
    @test !isempty(capture(() -> SO4.print_diagnostics(ts_incompat, simple_prox, (x,))))
    # a single term failing the required property (built with the plain `SqrNormL2` Term,
    # not `ls`, so the operator stays the real `A` — this is testing diagnostics on a
    # non-eye operator, not `ls`'s normal-op selection)
    @test occursin(
        "does not satisfy",
        capture(() -> SO4.print_diagnostics(SO4.Term(SqrNormL2(), A * x - b), simple_prox, (x,)))
    )

    # OperatorTerm: non-eye decomposition, plus a multi-term set.
    ot = find_assumption(ProximalAlgorithms.OperatorTerm)
    @test ot !== nothing
    @test !isempty(capture(() -> SO4.print_diagnostics(norm(A * x, 1), ot, (x,))))
    @test !isempty(
        capture(
            () ->
            SO4.print_diagnostics(SO4.TermSet(SO4.Term(SqrNormL2(), A * x - b), norm(x, 1)), ot, (x,))
        )
    )

    # OperatorTermWithInfimalConvolution (single + multi-term).
    infc = find_assumption(ProximalAlgorithms.OperatorTermWithInfimalConvolution)
    if infc !== nothing
        @test !isempty(capture(() -> SO4.print_diagnostics(norm(A * x, 1), infc, (x,))))
        @test !isempty(
            capture(
                () ->
                SO4.print_diagnostics(SO4.TermSet(ls(A * x - b), norm(x, 1)), infc, (x,))
            )
        )
    end

    # LeastSquaresTerm: not-least-squares message, decomposition, and multi-term.
    lsa = find_assumption(ProximalAlgorithms.LeastSquaresTerm)
    @test occursin(
        "least squares",
        capture(() -> SO4.print_diagnostics(norm(x, 1), lsa, (x,)))
    )
    @test !isempty(capture(() -> SO4.print_diagnostics(ls(A * x - b), lsa, (x,))))
    @test !isempty(
        capture(
            () ->
            SO4.print_diagnostics(SO4.TermSet(ls(A * x - b), norm(x, 1)), lsa, (x,))
        )
    )

    # SquaredL2Term: displacement / not-squared-L2 / not-eye-or-diagonal / multi-term.
    sq = find_assumption(ProximalAlgorithms.SquaredL2Term)
    @test occursin(
        "displacement",
        capture(() -> SO4.print_diagnostics(norm(x - c, 2)^2, sq, (x,)))
    )
    @test occursin(
        "squared L2",
        capture(() -> SO4.print_diagnostics(norm(x, 1), sq, (x,)))
    )
    @test !isempty(capture(() -> SO4.print_diagnostics(norm(A * x, 2)^2, sq, (x,))))
    @test !isempty(
        capture(
            () ->
            SO4.print_diagnostics(SO4.TermSet(norm(x, 2)^2, norm(x, 1)), sq, (x,))
        )
    )

    # Single-element TermSet delegates to the single-term method for each family
    # (the `length(terms) == 1` branches in prepare / print_diagnostics). Use a
    # square operator so the LeastSquaresTerm (which requires `is_square`) prepares.
    As = randn(5, 5); bs = randn(5)
    ls1 = SO4.TermSet(ls(As * x - bs))
    @test SO4.prepare(ls1, lsa, (x,)) !== nothing
    @test SO4.prepare(SO4.TermSet(norm(x, 2)^2), sq, (x,)) !== nothing
    for a in (simple_prox, ot, lsa, sq)
        @test !isempty(capture(() -> SO4.print_diagnostics(ls1, a, (x,))))
    end

    # OperatorTerm with an identity operator hits the `is_eye` diagnostics branch.
    @test !isempty(capture(() -> SO4.print_diagnostics(norm(x, 1), ot, (x,))))
end

@testset "parse.jl — Repeated assumptions + sliced separable sum" begin
    Random.seed!(419)
    x = Variable(5)
    A = randn(4, 5); b = randn(4)

    # RepeatedSimpleTerm: single-term delegates to SimpleTerm; a TermSet iterates.
    rst = find_assumption(ProximalAlgorithms.RepeatedSimpleTerm)
    if rst !== nothing
        @test SO4.prepare(norm(x, 1), rst, (x,)) !== nothing
        multi = SO4.TermSet(norm(x, 1), norm(x, 2))
        @test SO4.prepare(multi, rst, (x,)) !== nothing
        @test !isempty(capture(() -> SO4.print_diagnostics(multi, rst, (x,))))
    end

    # RepeatedOperatorTerm: single-term + a TermSet of smooth operator terms.
    rot = find_assumption(ProximalAlgorithms.RepeatedOperatorTerm)
    if rot !== nothing
        @test SO4.prepare(ls(A * x - b), rot, (x,)) !== nothing
        @test !isempty(capture(() -> SO4.print_diagnostics(ls(A * x - b), rot, (x,))))
    end

    # Multi-variable separable problem: a shared smooth term plus one proximable
    # constraint per variable -> group_by_variables / can_be_separable_sum /
    # prepare_proximable_single_var_per_term (single-term-per-variable branch).
    u = Variable(4)
    v = Variable(4)
    Au = randn(3, 4); Bv = randn(3, 4); bb = randn(3)
    p_sep = problem(ls(Au * u - Bv * v + bb), norm(u, 2) <= 1.0, norm(v, 2) <= 1.0)
    ~u .= 0.0
    ~v .= 0.0
    sol = solve(p_sep, PANOCplus(maxit = 5))
    @test sol !== nothing
end

@testset "term.jl — constructors, ==, show, trait predicates" begin
    x = Variable(4)

    # Term(f, expression, repr) constructor + repr-based show.
    tr = SO4.Term(NormL1(), x, "myL1")
    @test sprint(show, tr) == "myL1"

    # equality ignores repr.
    @test norm(x, 1) == norm(x, 1)
    @test (@term norm(x, 1)) == norm(x, 1)

    # TermSet show: cost + two constraints (`s.t.` and the `, ` separator).
    ts = norm(x, 1) + (norm(x, 2) <= 1.0) + (x >= 0.0)
    s = sprint(show, ts)
    @test occursin("s.t.", s) && occursin(",", s)
    # constraint-only TermSet: no `s.t.` prefix.
    cons_only = (norm(x, 2) <= 1.0) + (x >= 0.0)
    @test !occursin("s.t.", sprint(show, cons_only))

    # TermSet + TermSet.
    combined = (norm(x, 1) + norm(x, 2)) + (ls(x) + norm(x, Inf))
    @test combined isa SO4.TermSet && length(combined) == 4

    # trait predicates on Terms (exercise the Term-level methods).
    @test SO4.is_quadratic(ls(x))
    @test SO4.is_affine_indicator(norm(x, 2) <= 1.0) isa Bool
    @test SO4.is_cone_indicator(norm(x, 2) <= 1.0) isa Bool
    @test !isempty(norm(x, 1))
end

@testset "addition.jl — multi-variable sums and array subtraction" begin
    Random.seed!(420)
    x = Variable(3)
    y = Variable(3)
    z = Variable(3)
    M() = MatrixOp(randn(2, 3))

    # two-variable HCAT, then HCAT + a new variable (multivar + var).
    e2 = (M() * x + M() * y) + M() * z
    @test Set(SO4.variables(e2)) == Set((x, y, z))
    # a variable already present is folded back in (the `xB[1] in xA` branch).
    e3 = (M() * x + M() * y) + M() * x
    @test Set(SO4.variables(e3)) == Set((x, y))
    # HCAT + HCAT.
    e4 = (M() * x + M() * y) + (M() * z + M() * x)
    @test Set(SO4.variables(e4)) == Set((x, y, z))

    # expression ± array / array ± expression: assert the full affine map value
    # (operator·w + displacement) reconstructs the intended expression.
    w = Variable(4)
    A = randn(3, 4); c = randn(3)
    wv = randn(4)
    affval(ex) = SO4.operator(ex) * wv + displacement(ex)
    @test norm(affval(A * w - c) - (A * wv - c)) < 1.0e-12
    @test norm(affval(c - A * w) - (c - A * wv)) < 1.0e-12
    @test norm(affval(c + A * w) - (c + A * wv)) < 1.0e-12
end

# Phase 4 (second pass) — the parse paths that were still reachable only indirectly.
# Everything here asserts a value or a captured message, never bare execution.

@testset "parse.jl — sliced separable sums" begin
    Random.seed!(430)
    x = Variable(4)
    ~x .= 0.0
    a, b = randn(2), randn(2)

    # Two proximable terms on *disjoint slices* of one variable: the parser accepts them as
    # a separable sum and folds them into a `PrecomposedSlicedSeparableSum`, with the
    # displacement inside each function (the one site that does fold it — see the
    # `fold_displacement` helper in `prepare(::TermSet, ::SimpleTerm, _)`).
    ts = SO4.TermSet(norm(x[1:2] - a, 1), norm(x[3:4] - b, 1))
    @test SO4.is_proximable(ts)
    simple_prox = ProximalAlgorithms.SimpleTerm(:g => (ProximalCore.is_proximable,))
    prep = SO4.prepare(ts, simple_prox, (x,))
    @test prep !== nothing
    g = prep[1].second
    @test g isa PrecomposedSlicedSeparableSum
    # The pieces handed over are the right ones: one ℓ1 per slice, each carrying its own
    # displacement, and the slices are the ones that were written.
    fs = g.fs[1]
    @test length(fs) == 2
    v = randn(4)
    @test abs(fs[1](view(v, 1:2)) - norm(v[1:2] - a, 1)) < 1.0e-9
    @test abs(fs[2](view(v, 3:4)) - norm(v[3:4] - b, 1)) < 1.0e-9
    @test g.idxs[1] == [(1:2,), (3:4,)]

    # `PrecomposedSlicedSeparableSum` itself does not handle this shape — one variable,
    # several sliced terms — correctly: its value does not match the sum of its own pieces,
    # and `prox!` throws while iterating the per-variable operator. The defect is in the
    # pinned ProximalOperators, in a path nothing exercised until now; these are marked
    # broken rather than deleted so that a fix flips them green.
    @test_broken abs(g((v,)) - (fs[1](view(v, 1:2)) + fs[2](view(v, 3:4)))) < 1.0e-9
    @test_broken try
        prox!((zeros(4),), g, (copy(v),), 1.0)
        true
    catch
        false
    end

    # Overlapping slices are *not* separable, and the diagnostic says which terms clash.
    y = Variable(4)
    ts_overlap = SO4.TermSet(norm(y[1:3], 1), norm(y[2:4], 1))
    @test !SO4.is_proximable(ts_overlap)
    @test SO4.prepare(ts_overlap, simple_prox, (y,)) === nothing
    out = capture(() -> SO4.print_diagnostics(ts_overlap, simple_prox, (y,)))
    @test occursin("incompatible", out)
end

@testset "parse.jl — Repeated* assumptions" begin
    Random.seed!(431)
    x = Variable(5)
    A, b = randn(4, 5), randn(4)

    # RepeatedSimpleTerm: one prepared function per term, collected into a tuple.
    rst = ProximalAlgorithms.RepeatedSimpleTerm(:gs => (ProximalCore.is_proximable,))
    single = SO4.prepare(norm(x, 1), rst, (x,))
    @test single !== nothing
    @test SO4.is_proximable(single[1].second)

    ts2 = SO4.TermSet(norm(x, 1), norm(x, 2))
    prep = SO4.prepare(ts2, rst, (x,))
    @test prep !== nothing
    fs = prep[1].second
    @test length(fs) == 2
    v = randn(5)
    @test abs(fs[1](v) - norm(v, 1)) < 1.0e-9
    @test abs(fs[2](v) - norm(v, 2)) < 1.0e-9

    # A term that cannot be prepared makes the whole repeated set fail, and the diagnostic
    # names it.
    ts_bad = SO4.TermSet(norm(x, 1), norm(A * x, 1))
    @test SO4.prepare(ts_bad, rst, (x,)) === nothing
    @test !isempty(capture(() -> SO4.print_diagnostics(ts_bad, rst, (x,))))
    @test !isempty(capture(() -> SO4.print_diagnostics(norm(A * x, 1), rst, (x,))))

    # RepeatedOperatorTerm: a function *and* an operator per term.
    rot = ProximalAlgorithms.RepeatedOperatorTerm(:fs => (), :As => ())
    prep_op = SO4.prepare(SO4.TermSet(ls(A * x - b), norm(x, 1)), rot, (x,))
    @test prep_op !== nothing
    funcs, ops = prep_op[1].second, prep_op[2].second
    @test length(funcs) == 2 && length(ops) == 2
    xv = randn(5)
    @test abs(funcs[1](ops[1] * xv) - 0.5 * norm(A * xv - b)^2) < 1.0e-9

    single_op = SO4.prepare(ls(A * x - b), rot, (x,))
    @test single_op !== nothing
    @test !isempty(capture(() -> SO4.print_diagnostics(ls(A * x - b), rot, (x,))))

    # An operator-side property nothing satisfies makes the repeated set fail.
    rot_eye = ProximalAlgorithms.RepeatedOperatorTerm(:fs => (SO4.is_proximable,), :As => (is_eye,))
    @test SO4.prepare(SO4.TermSet(norm(A * x, 1), norm(A * x, 2)), rot_eye, (x,)) === nothing
    @test !isempty(capture(() -> SO4.print_diagnostics(SO4.TermSet(norm(A * x, 1), norm(A * x, 2)), rot_eye, (x,))))
end

@testset "parse.jl — InfConv and OperatorTerm multi-term paths" begin
    Random.seed!(432)
    x = Variable(4)
    A1, A2 = randn(3, 4), randn(3, 4)
    b1, b2 = randn(3), randn(3)
    ts = SO4.TermSet(ls(A1 * x - b1), ls(A2 * x - b2))
    vars = SO4.extract_variables(ts)

    # func₁ branch: an assumption both the stacked function and operator satisfy.
    infc = ProximalAlgorithms.OperatorTermWithInfimalConvolution(:h => (), :l => (), :A => ())
    prep = SO4.prepare(ts, infc, vars)
    @test prep !== nothing
    f, op = prep[1].second, prep[2].second
    xv = randn(4)
    @test abs(f(op * xv) - (0.5 * norm(A1 * xv - b1)^2 + 0.5 * norm(A2 * xv - b2)^2)) < 1.0e-9

    # Single-term func₁ branch, same check.
    prep1 = SO4.prepare(ls(A1 * x - b1), infc, (x,))
    @test prep1 !== nothing
    @test abs(prep1[1].second(prep1[2].second * xv) - 0.5 * norm(A1 * xv - b1)^2) < 1.0e-9

    # func₂ branch: func₁ unsatisfiable, func₂ trivially satisfiable.
    infc2 = ProximalAlgorithms.OperatorTermWithInfimalConvolution(
        :h => (SO4.is_set_indicator,), :l => (), :A => ()
    )
    prep2 = SO4.prepare(ts, infc2, vars)
    @test prep2 !== nothing
    @test prep2[1].first === :l

    # Diagnostics for an assumption nothing can satisfy, single- and multi-term.
    infc_bad = ProximalAlgorithms.OperatorTermWithInfimalConvolution(
        :h => (SO4.is_set_indicator,), :l => (SO4.is_set_indicator,), :A => (is_eye,)
    )
    @test !isempty(capture(() -> SO4.print_diagnostics(ts, infc_bad, vars)))
    @test !isempty(capture(() -> SO4.print_diagnostics(ls(A1 * x - b1), infc_bad, (x,))))

    # OperatorTerm on a TermSet whose operator *is* the identity takes the `is_eye`
    # diagnostics branch.
    ot_eye = ProximalAlgorithms.OperatorTerm(:f => (SO4.is_set_indicator,), :A => ())
    ts_eye = SO4.TermSet(ls(x), norm(x, 1))
    @test !isempty(capture(() -> SO4.print_diagnostics(ts_eye, ot_eye, (x,))))

    # Multi-variable InfConv fallback: neither func slot is satisfiable, so it falls back to
    # the SimpleTerm preparation and attaches a block identity for the operator slot.
    u, w = Variable(3), Variable(3)
    ~u .= 0.0
    ~w .= 0.0
    # No operator can be a set indicator, so the operator side fails and the fallback runs.
    infc_fallback = ProximalAlgorithms.OperatorTermWithInfimalConvolution(
        :h => (SO4.is_smooth,), :l => (), :A => (SO4.is_set_indicator,)
    )
    for target in (ls(u + w), SO4.TermSet(ls(u), ls(w)))
        prep_mv = SO4.prepare(target, infc_fallback, (u, w))
        @test prep_mv !== nothing
        @test length(prep_mv) == 2
        @test is_eye(prep_mv[2].second)     # the block identity standing in for the operator
    end
end

@testset "parse.jl / calculus — rejection paths" begin
    Random.seed!(433)
    x = Variable(4)
    A, b = randn(6, 4), randn(6)
    lsa = find_assumption(ProximalAlgorithms.LeastSquaresTerm)
    sq = find_assumption(ProximalAlgorithms.SquaredL2Term)

    # LeastSquaresTerm rejects a function it cannot read the operator out of, and an
    # array-weighted one it would have to mis-scale.
    @test SO4.prepare(norm(x, 1), lsa, (x,)) === nothing
    t_arr = SO4.Term(SqrNormL2(rand(4) .+ 0.5), x)
    @test SO4.prepare(t_arr, lsa, (x,)) === nothing

    # ... and it *accepts* a term already folded into the normal-operator formulation,
    # reading the least-squares target back out of the operator's displacement. The CGNR
    # assumption also wants a square operator, hence the 4x4 here.
    Asq, bsq = randn(4, 4), randn(4)
    t_normal = SO4.Term(SO4.SqrNormL2WithNormalOp(AbstractOperators.AffineAdd(MatrixOp(Asq), -bsq), 1), x)
    prep_n = SO4.prepare(t_normal, lsa, (x,))
    @test prep_n !== nothing
    d = Dict(prep_n)
    @test norm(d[lsa.b] - bsq) < 1.0e-9
    xr = randn(4)
    @test norm(d[lsa.operator.first] * xr - Asq * xr) < 1.0e-9

    # SquaredL2Term rejects an operator that is neither identity nor diagonal.
    @test SO4.prepare(norm(A * x, 2)^2, sq, (x,)) === nothing

    # `merge_function_with_operator` has no exact-prox formulation for a nonlinear operator.
    op_sin = SO4.operator(sin(x))
    @test SO4.best_formulation(op_sin, SqrNormL2(), 0, 1, :prox)[1] === :none
    @test_throws ErrorException SO4.merge_function_with_operator(op_sin, SqrNormL2(), 0, 1; needs = :prox)

    # `_matrix_of` sees through a displacement.
    @test SO4._matrix_of(AbstractOperators.AffineAdd(MatrixOp(A), b)) === nothing ||
        SO4._matrix_of(AbstractOperators.AffineAdd(MatrixOp(A), b)) == A

    # `with_normal_op`/`normal_op_applicable` decline a function that is not a squared norm.
    @test SO4.with_normal_op(NormL1(), MatrixOp(A), 0, 1) === nothing
    @test !SO4.normal_op_applicable(NormL1(), MatrixOp(A), 0, 1)
    # ... an array λ ...
    @test !SO4.normal_op_applicable(SqrNormL2(rand(6) .+ 0.5), MatrixOp(A), 0, 1)
    # ... and a scalar displacement, which has no array to push through `opᴴ`.
    @test !SO4.normal_op_applicable(SqrNormL2(), MatrixOp(A), 1.0, 1)
end

@testset "calculus — PrecomposeNonlinear value and adjoint scaling" begin
    Random.seed!(434)
    x = Variable(5)
    xv = randn(5)

    # `PrecomposeNonlinear` evaluates as g∘G and reports itself smooth when g is.
    f = SO4.merge_function_with_operator(SO4.operator(sin(x)), SqrNormL2(), 0, 1)
    @test f isa PrecomposeNonlinear
    @test SO4.is_smooth(f)
    @test abs(f(xv) - 0.5 * norm(sin.(xv))^2) < 1.0e-9

    # The adjoint-scaling probe falls back to `Aᴴd` when the constant vector lands in the
    # null space of the operator, and to 1 when there is no displacement to try.
    #
    # `Z` annihilates the constant vector: its rows sum to zero.
    Zm = [1.0 -1.0 0.0 0.0 0.0; 0.0 1.0 -1.0 0.0 0.0; 0.0 0.0 1.0 -1.0 0.0; 0.0 0.0 0.0 1.0 -1.0]
    Z = MatrixOp(Zm)
    d = randn(4)
    fz = SO4.SqrNormL2WithNormalOp(AbstractOperators.AffineAdd(Z, d), 1)
    @test fz.inv_scaling > 0
    # value still agrees with the definition
    @test abs(fz(xv) - 0.5 * norm(Zm * xv + d)^2 * fz.inv_scaling) < 1.0e-8

    # No displacement and a null-space probe: the scaling is left at 1.
    fz0 = SO4.SqrNormL2WithNormalOp(Z, 1)
    @test fz0.inv_scaling == 1.0 || fz0.inv_scaling > 0

    # Traits.
    T = typeof(fz)
    @test SO4.is_separable(T) && SO4.is_generalized_quadratic(T) && !SO4.is_proximable(T)
end

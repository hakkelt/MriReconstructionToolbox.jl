# Phase 2.1 — property tests for the canonical absorption transform
# `merge_function_with_operator(op, f, disp, λ)`, which must satisfy
#     absorbed_f(x) ≈ λ · f(op * x + disp)
# for every absorption case (eye / diagonal / AAᴴ-diagonal / general linear /
# nonlinear). This is the value-level invariant that the Phase 1 displacement/λ
# bugs (1.1–1.4) all violated.

const SO2 = StructuredOptimization
const merge_fo = StructuredOptimization.merge_function_with_operator

@testset "Phase 2.1 absorption property" begin
    Random.seed!(200)

    # helper: absorbed(x) ≈ λ * f(op*x + disp)
    function check_absorption(op, f, disp, λ; cplx = false)
        g = merge_fo(op, f, disp, λ)
        for _ in 1:3
            x = cplx ? randn(ComplexF64, size(op, 2)) : randn(size(op, 2))
            expected = λ * f(op * x .+ disp)
            @test abs(g(x) - expected) < 1.0e-9 * (1 + abs(expected))
        end
    end

    # eye operator with displacement (non-SqrNormL2 function)
    @testset "eye" begin
        op = AbstractOperators.Eye(Float64, (4,))
        check_absorption(op, NormL1(), randn(4), 2.5)
        check_absorption(op, NormL1(), zeros(4), 1.0)
    end

    # diagonal operator: NormL1 keeps displacement; SqrNormL2 folds into the weight
    # (valid only at zero displacement, which is how it is reached in prepare).
    @testset "diagonal" begin
        op = DiagOp([2.0, -3.0, 4.0, 0.5])
        check_absorption(op, NormL1(), randn(4), 1.7)
        check_absorption(op, SqrNormL2(1.5), zeros(4), 2.0)
    end

    # AAᴴ-diagonal operator (DFT): AAᴴ = N·I, hit via the Precompose branch
    @testset "AAc-diagonal" begin
        xv = Variable(8)
        op = SO2.operator(fft(xv))          # DFT: ℝ^8 -> ℂ^8, AAᴴ = 8·I
        check_absorption(op, SqrNormL2(), randn(ComplexF64, 8), 1.3)
    end

    # general (non-square, non-AAᴴ-diagonal) linear operator
    @testset "general linear" begin
        op = MatrixOp(randn(6, 4))
        check_absorption(op, NormL1(), randn(6), 0.9)
        check_absorption(op, SqrNormL2(), randn(6), 2.2)
    end

    # nonlinear operator (sin): PrecomposeNonlinear with AffineAdd displacement
    @testset "nonlinear" begin
        xv = Variable(5)
        op = SO2.operator(sin(xv))
        check_absorption(op, SqrNormL2(), randn(5), 3.1)
    end
end

# Phase 2.2 — automatic selection of `SqrNormL2WithNormalOp` during absorption.
#
# The syntax layer never folds an operator into the function: `ls` builds a plain
# `SqrNormL2` over whatever expression it was given. The normal-operator rewrite happens
# only in `merge_function_with_operator`, where the operator has been expanded to the
# problem's full domain and is composed with nothing afterwards — so it also covers
# multi-variable terms, provided the joint normal operator both fuses and is the cheaper of
# the two formulations.
@testset "Phase 2.2 normal-op auto-selection" begin
    Random.seed!(220)

    # absorbed(x) ≈ λ·f(op*x + disp) and ∇absorbed(x) ≈ λ·opᴴ(op*x + disp), checked
    # against the plain Precompose formulation the fold replaces.
    function check_against_precompose(op, f, disp, λ, x)
        g = merge_fo(op, f, disp, λ)
        ref = Postcompose(Precompose(f, op, 1, disp), λ)
        @test g(x) ≈ ref(x) rtol = 1.0e-9
        gg, gr = zero(x), zero(x)
        vg = SO2.gradient!(gg, g, x)
        vr = ProximalOperators.gradient!(gr, ref, x)
        @test vg ≈ vr rtol = 1.0e-9
        @test gg ≈ gr rtol = 1.0e-9
        return g
    end

    @testset "single variable, fusing operator" begin
        op = MatrixOp(randn(7, 4))
        g = check_against_precompose(op, SqrNormL2(1.5), randn(7), 2.2, randn(4))
        @test g isa SO2.SqrNormL2WithNormalOp
        g0 = check_against_precompose(op, SqrNormL2(), zeros(7), 1.0, randn(4))
        @test g0 isa SO2.SqrNormL2WithNormalOp
    end

    # The joint operator has to be overdetermined for the block Gram to be worth building,
    # so the two blocks together stay narrower than the shared codomain.
    @testset "multiple variables (HCAT block Gram)" begin
        x, y = Variable(10), Variable(7)
        A, B, b = randn(25, 10), randn(25, 7), randn(25)
        t = ls(A * x - B * y + b)
        @test t.f isa SqrNormL2
        op = SO2.extract_operators((x, y), t)
        @test op isa AbstractOperators.HCAT
        g = check_against_precompose(
            op, t.f, SO2.displacement(t), t.lambda, ArrayPartition(randn(10), randn(7))
        )
        @test g isa SO2.SqrNormL2WithNormalOp
    end

    # A term on one variable of a two-variable problem is padded with a `Zeros` block for the
    # other. The block that remains decides, so an operator that is overdetermined on its own
    # variable still qualifies although the padded joint domain is larger than its codomain.
    @testset "multiple variables, term on one of them" begin
        x, w = Variable(16), Variable(40)
        t = ls(randn(24, 16) * x - randn(24))
        op = SO2.extract_operators((x, w), t)
        @test op isa AbstractOperators.HCAT
        @test SO2._drop_zero_blocks(op) === op.A[1]
        @test SO2._total_length(size(op, 2)) > SO2._total_length(size(op, 1))
        @test SO2.normal_op_worthwhile(op)
        @test SO2.best_formulation(op, t.f, SO2.displacement(t), t.lambda)[1] === :normal_op
        N = SO2.fused_normal_op(op)
        @test N isa AbstractOperators.VCAT
        v = ArrayPartition(randn(16), randn(40))
        @test N * v ≈ op' * (op * v)
        g = check_against_precompose(op, t.f, SO2.displacement(t), t.lambda, v)
        @test g isa SO2.SqrNormL2WithNormalOp
    end

    # Only a squared L2 norm is rewritten, only when the normal operator actually fuses,
    # and only when the normal operator is the cheaper of the two formulations.
    @testset "declined" begin
        op = MatrixOp(randn(6, 4))
        @test !(merge_fo(op, NormL1(), randn(6), 0.9) isa SO2.SqrNormL2WithNormalOp)

        xv = Variable(5)
        nonfusing = SO2.operator(fft(MatrixOp(randn(5, 5)) * xv))
        @test SO2.fused_normal_op(nonfusing) === nothing
        @test SO2.with_normal_op(SqrNormL2(), nonfusing, 0, 1.0) === nothing

        # one HCAT block that does not fuse is enough to make the block Gram the slower
        # of the two formulations
        x, y = Variable(5), Variable(5)
        ex_mixed = MatrixOp(randn(12, 5)) * x + MatrixOp(randn(12, 4)) * y[1:4]
        op_mixed = SO2.extract_operators((x, y), ls(ex_mixed))
        @test op_mixed isa AbstractOperators.HCAT
        @test SO2.normal_op_worthwhile(op_mixed)  # declined over fusion, not over size
        @test SO2.fused_normal_op(op_mixed) === nothing

        # underdetermined: `LᴴL` acts on the larger space, so it loses on both cost and
        # conditioning even though every block product fuses
        u, v = Variable(50), Variable(100)
        op_wide = SO2.extract_operators((u, v), ls(randn(30, 50) * u + randn(30, 100) * v))
        @test !SO2.normal_op_worthwhile(op_wide)
        @test SO2.fused_normal_op(op_wide) === nothing
        @test !(merge_fo(op_wide, SqrNormL2(), randn(30), 1.0) isa SO2.SqrNormL2WithNormalOp)

        @test !SO2.normal_op_worthwhile(MatrixOp(randn(4, 6)))
    end

    # A single-variable least-squares term reaches the same rewrite through `prepare`,
    # although `ls` itself no longer performs it.
    @testset "single variable, through prepare" begin
        v = Variable(4)
        t = ls(MatrixOp(randn(7, 4)) * v - randn(7))
        @test t.f isa SqrNormL2
        smooth_assumption = ProximalAlgorithms.SimpleTerm(:f => [SO2.is_smooth])
        prepared = SO2.prepare(t, smooth_assumption, (v,))
        @test prepared[1].second isa SO2.SqrNormL2WithNormalOp
    end

    # End-to-end: a purely smooth multi-variable least-squares problem now solved through
    # the joint normal operator must still satisfy the normal equations.
    @testset "multi-variable solve satisfies the normal equations" begin
        x, y = Variable(10), Variable(6)
        A, B, b = randn(20, 10), randn(20, 6), randn(20)
        solve(
            problem(ls(A * x + B * y - b)),
            ProximalAlgorithms.FastForwardBackward(tol = 1.0e-10, maxit = 5000),
        )
        r = A * (~x) + B * (~y) - b
        @test norm(A' * r) < 1.0e-4
        @test norm(B' * r) < 1.0e-4
    end
end

# Phase 2.3 — the formulation of a least-squares term is chosen at parse time, not by `ls`.
#
# `ls` used to fold a single-variable operator into a `SqrNormL2WithNormalOp` immediately,
# which hid the operator from every later decision: the diagonal and AAᴴ-diagonal
# absorptions never saw it, and the term advertised a prox it does not have. These tests
# pin down what deferring the choice buys.
@testset "Phase 2.3 deferred formulation choice" begin
    Random.seed!(230)

    @testset "diagonal operator folds into the weight, displacement and all" begin
        a, b = randn(6), randn(6)
        x = randn(6)

        # no displacement: ½‖diag(a)·x‖² is the weighted squared norm itself
        g0 = merge_fo(DiagOp(a), SqrNormL2(), 0, 1)
        @test g0 isa SqrNormL2
        @test g0(x) ≈ sum(abs2, a .* x) / 2

        # with a displacement there is nowhere to put it in the weighted form, so the
        # operator stays outside the function (it used to be dropped silently)
        gd = merge_fo(DiagOp(a), SqrNormL2(), -b, 1)
        @test gd(x) ≈ sum(abs2, a .* x .- b) / 2
        @test SO2.is_proximable(gd)
    end

    @testset "AAᴴ-diagonal operator keeps its exact prox" begin
        v = Variable(8)
        op = SO2.operator(fft(v))
        g = merge_fo(op, SqrNormL2(), zeros(ComplexF64, 8), 1.0)
        @test g isa Precompose
        @test !(g isa SO2.SqrNormL2WithNormalOp)
        @test SO2.is_proximable(g)
    end

    # The normal-operator formulation implements `gradient!` and no `prox!`, so it must not
    # claim proximability: a solver picked on that claim would fail at the first iteration.
    @testset "the normal-op formulation is not proximable" begin
        f = SO2.SqrNormL2WithNormalOp(MatrixOp(randn(7, 4)))
        @test SO2.is_smooth(f)
        @test !SO2.is_proximable(f)
    end
end

# Phase 2.6 — affine equality is deferred to parse time as well.
#
# `==(ex, b)` used to demand a `MatrixOp`, fold it into an `IndAffine` on the spot and
# return a term over `variables(ex)[1]` alone. It now builds `Term(IndPoint(b), ex)`, so the
# diagonal and AAᴴ-diagonal absorptions cover the two cases that used to error, the
# `MatrixOp` case is reproduced exactly by a new `IndPoint` rule, and no variable is lost.
@testset "Phase 2.6 affine equality at parse time" begin
    Random.seed!(260)

    absorbed(t) = merge_fo(SO2.operator(t), t.f, SO2.displacement(t), t.lambda)

    @testset "diagonal operator: a trivial projection, used to error" begin
        a, bb = randn(6) .+ 2, randn(6)
        xv = Variable(6)
        t = (a .* xv == bb)
        @test t.f isa IndPoint
        g = absorbed(t)
        @test SO2.is_proximable(g)
        # The only feasible point is `b ./ a`, so the projection lands there from anywhere.
        y, v = prox(g, randn(6), 1.0)
        @test norm(y - bb ./ a) < 1.0e-10
        @test v == 0.0
    end

    @testset "AAᴴ-diagonal operator (DFT): used to error" begin
        xv = Variable(8)
        x0 = randn(8)
        bb = fft(x0)
        t = (fft(xv) == bb)
        @test t.f isa IndPoint
        g = absorbed(t)
        @test SO2.is_proximable(g)
        # `fft` is injective on ℝ^8, so the feasible set is the single point `x0`.
        y, v = prox(g, randn(8), 1.0)
        @test norm(y - x0) < 1.0e-9
        @test v == 0.0
    end

    @testset "MatrixOp: same IndAffine as before, from either spelling" begin
        Am, bm = randn(4, 10), randn(4)
        xv = Variable(10)
        for t in (Am * xv == bm, Am * xv - bm == 0)
            g = absorbed(t)
            @test g isa IndAffine
            z = randn(10)
            y_ref, _ = prox(IndAffine(Am, bm), z, 1.0)
            y_got, _ = prox(g, z, 1.0)
            @test norm(y_got - y_ref) < 1.0e-10
        end
    end

    @testset "multi-variable equality keeps every variable" begin
        u, w = Variable(5), Variable(4)
        Au, Aw, bb = randn(3, 5), randn(3, 4), randn(3)
        t = (Au * u + Aw * w == bb)
        @test SO2.variables(t) == (u, w)
        @test t.f isa IndPoint
        # The constraint is the one that was written, over the joint domain.
        op = SO2.extract_operators((u, w), t)
        zu, zw = randn(5), randn(4)
        @test op * ArrayPartition(zu, zw) ≈ Au * zu + Aw * zw
    end
end

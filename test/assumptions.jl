using ProximalAlgorithms: get_assumptions

@testset "get_assumptions function" begin
    @test length(get_assumptions(ProximalAlgorithms.CGIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ADMMIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.DavisYinIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.DouglasRachfordIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.FastForwardBackwardIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.FastProximalGradientIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ForwardBackwardIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ProximalGradientIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.LiLinIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.PANOCIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.PANOCplusIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.AFBAIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.VuCondatIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.ChambollePockIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.SFISTAIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ZeroFPRIteration)) == 2
end

# An operator with a displacement: affine, not linear.
struct AffineOnly end
ProximalAlgorithms.is_affine(::AffineOnly) = true

@testset "affine operators" begin
    operator_predicates(T) =
        only(a.operator.second for a in get_assumptions(T) if hasproperty(a, :operator))
    accepts(T, op) = all(p -> p(op), operator_predicates(T))
    @test !ProximalAlgorithms.is_linear(AffineOnly())

    # PANOCplus only applies `A` and its adjoint, so an affine `A` is exact there.
    @test accepts(ProximalAlgorithms.PANOCplusIteration, AffineOnly())
    # These combine `A` linearly (or form `AᴴA`, or its norm), so they need a linear one.
    for T in (
            ProximalAlgorithms.PANOCIteration, ProximalAlgorithms.ZeroFPRIteration,
            ProximalAlgorithms.ChambollePockIteration, ProximalAlgorithms.CGNRIteration,
        )
        @test !accepts(T, AffineOnly())
    end
end
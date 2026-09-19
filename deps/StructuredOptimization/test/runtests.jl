using StructuredOptimization
using AbstractOperators, DSPOperators, FFTWOperators
using ProximalOperators
using ProximalAlgorithms
using RecursiveArrayTools
using LinearAlgebra, Random
using DSP, FFTW
using Test
using Aqua

Random.seed!(0)

@testset "StructuredOptimization" begin
    @testset "Calculus" begin
        include("test_proxstuff.jl")
    end

    @testset "Syntax" begin
        include("test_variables.jl")
        include("test_expressions.jl")
        include("test_AbstractOp_binding.jl")
        include("test_terms.jl")
    end

    @testset "Problem construction" begin
        include("test_problem.jl")
        include("test_build_minimize.jl")
    end

    @testset "Phase 1 regressions" begin
        include("test_phase1_regressions.jl")
    end

    @testset "Phase 2 absorption" begin
        include("test_phase2_absorption.jl")
    end

    @testset "Phase 2 matching" begin
        include("test_phase2_matching.jl")
    end

    @testset "Phase 4 coverage" begin
        include("test_phase4_coverage.jl")
    end

    @testset "End-to-end tests" begin
        include("test_usage_small.jl")
        include("test_usage.jl")
    end

    @testset "GPU" begin
        include("test_gpu.jl")
    end


    @testset "Aqua" begin
        Aqua.test_all(StructuredOptimization; ambiguities = false, piracies = false, persistent_tasks = false)
        Aqua.test_ambiguities(
            StructuredOptimization; exclude = [Base.:(+), Base.:<=, Base.:>=], broken = true
        )
        Aqua.test_persistent_tasks(StructuredOptimization)
        Aqua.test_piracies(
            StructuredOptimization;
            treat_as_own = [
                # Intentional cross-interface bridges (see StructuredOptimization.jl):
                # these adapt ProximalOperators-style gradients to ProximalAlgorithms'
                # value_and_gradient interface for the composite smooth functions this
                # package builds, and cannot be restricted to owned types.
                ProximalAlgorithms.value_and_gradient,
                ProximalAlgorithms.value_and_gradient!,
                ProximalOperators.prox,
                ProximalOperators.prox!,
                ProximalOperators.gradient,
                ProximalOperators.gradient!,
            ],
        )
    end
end

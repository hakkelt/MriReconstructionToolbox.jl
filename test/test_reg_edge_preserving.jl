using TestItems

@testitem "EdgePreservingRoughness regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    huber(t, λ, δ) = abs(t) <= δ ? λ * abs(t)^2 / (2δ) : λ * (abs(t) - δ / 2)

    function functions_of(reg, x)
        term = MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
        return SO.extract_functions(term)
    end

    @testset "Constructor" begin
        reg = EdgePreservingRoughness2D(0.1; δ = 0.05)
        @test reg.λ == 0.1
        @test reg.δ == 0.05
        @test EdgePreservingRoughness2D(0.1).δ == 0.01
        @test_throws ArgumentError EdgePreservingRoughness2D(-0.1)
        @test_throws ArgumentError EdgePreservingRoughness2D(0.1; δ = 0)
        @test_throws ArgumentError EdgePreservingRoughness3D(0.1; δ = -1)
    end

    @testset "get_operator is the gradient operator" for threaded in [false, true]
        x = randn(6, 6)
        @test get_operator(EdgePreservingRoughness2D(0.1), x; threaded) * x ≈
            get_operator(TotalVariation2D(0.1), x; threaded) * x
        x3 = randn(4, 4, 4)
        @test get_operator(EdgePreservingRoughness3D(0.1), x3; threaded) * x3 ≈
            get_operator(TotalVariation3D(0.1), x3; threaded) * x3
    end

    @testset "calculate is the Huber potential of every difference" begin
        x = randn(6, 6)
        λ, δ = 0.3, 0.4
        gradient = get_operator(EdgePreservingRoughness2D(λ), x; threaded = false) * x
        expected = sum(t -> huber(t, λ, δ), gradient)
        @test MriReconstructionToolbox.calculate(EdgePreservingRoughness2D(λ; δ), x) ≈ expected
    end

    @testset "δ interpolates between Tikhonov-like and TV-like behaviour" begin
        x = randn(6, 6)
        λ = 0.5
        gradient = get_operator(EdgePreservingRoughness2D(λ), x; threaded = false) * x

        # δ → 0: λ ∑|∂x| − λδ/2 per element, i.e. anisotropic total variation
        tiny = 1.0e-8
        anisotropic_tv = λ * sum(abs, gradient)
        @test MriReconstructionToolbox.calculate(EdgePreservingRoughness2D(λ; δ = tiny), x) ≈
            anisotropic_tv rtol = 1.0e-6

        # δ → ∞: λ/(2δ) ∑|∂x|², i.e. a quadratic roughness penalty
        huge = 1.0e8
        @test MriReconstructionToolbox.calculate(EdgePreservingRoughness2D(λ; δ = huge), x) ≈
            λ / (2 * huge) * sum(abs2, gradient) rtol = 1.0e-8
    end

    @testset "the term is smooth, so solvers can use its gradient" begin
        x = randn(6, 6)
        f = functions_of(EdgePreservingRoughness2D(0.3; δ = 0.4), x)
        @test PC.is_smooth(f)
        gradient = get_operator(EdgePreservingRoughness2D(0.3), x; threaded = false) * x
        g = similar(gradient)
        value = PC.gradient!(g, f, gradient)
        @test value ≈ f(gradient)
        # finite-difference check on a single coordinate
        h = 1.0e-6
        perturbed = copy(gradient)
        perturbed[3] += h
        @test (f(perturbed) - f(gradient)) / h ≈ g[3] rtol = 1.0e-4
    end

    @testset "complex input" begin
        x = randn(ComplexF64, 6, 6)
        λ, δ = 0.2, 0.3
        gradient = get_operator(EdgePreservingRoughness2D(λ), x; threaded = false) * x
        @test MriReconstructionToolbox.calculate(EdgePreservingRoughness2D(λ; δ), x) ≈
            sum(t -> huber(t, λ, δ), gradient)
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y)}(randn(6, 6))
        reg = EdgePreservingRoughness2D(0.1; δ = 0.2)
        @test get_operator(reg, x; threaded = false) isa MriReconstructionToolbox.NamedDimsOp
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈
            MriReconstructionToolbox.calculate(reg, unname(x); threaded = false)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 4)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test MriReconstructionToolbox.get_affected_dims(EdgePreservingRoughness2D(0.1f0), info, 1:3) == 1:2
        @test MriReconstructionToolbox.get_affected_dims(
            EdgePreservingRoughness3D(0.1f0), info, (:x, :y, :z, :time)
        ) == (:x, :y, :z)
    end

    @testset "scale_regularization rescales the intensity threshold too" begin
        # δ is an absolute intensity, not a weight, so scaling the image by `factor` has to scale δ by the
        # same factor for the term to keep its meaning.
        original = EdgePreservingRoughness2D(0.2; δ = 0.1)
        scaled = MriReconstructionToolbox.scale_regularization(original, 2.5)
        @test scaled.λ ≈ 0.5
        @test scaled.δ ≈ 0.25

        # The invariant `scale_regularization` has to satisfy: on an image scaled by `factor`, the scaled
        # term must equal `factor²` times the original term on the original image -- `factor²` being how the
        # least-squares data term itself scales, so the balance between the two is preserved. The ℓ₁-type
        # terms satisfy the same identity.
        x = randn(6, 6)
        @test MriReconstructionToolbox.calculate(scaled, x .* 2.5) ≈
            2.5^2 * MriReconstructionToolbox.calculate(original, x)
        l1_original, l1_scaled = L1Image(0.2), MriReconstructionToolbox.scale_regularization(L1Image(0.2), 2.5)
        @test MriReconstructionToolbox.calculate(l1_scaled, x .* 2.5) ≈
            2.5^2 * MriReconstructionToolbox.calculate(l1_original, x)
    end
end

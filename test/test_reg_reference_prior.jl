using TestItems

@testitem "ReferencePrior regularization" tags = [:regularization] setup = [RegTestSetup] begin
    using LinearAlgebra

    @testset "materialize" for threaded in [false, true]
        x_ref = randn(5, 5)
        x = randn(5, 5)
        λ = 0.3
        @test MriReconstructionToolbox.calculate(ReferencePrior(λ, x_ref), x; threaded) ≈
            λ * norm(x .- x_ref, 1)
    end

    @testset "complex data" begin
        x_ref = randn(ComplexF64, 4, 4)
        x = randn(ComplexF64, 4, 4)
        λ = 0.2
        @test MriReconstructionToolbox.calculate(ReferencePrior(λ, x_ref), x; threaded = false) ≈
            λ * sum(abs, x .- x_ref)
    end

    @testset "zero at the reference" begin
        x_ref = randn(4, 4)
        @test MriReconstructionToolbox.calculate(ReferencePrior(0.5, x_ref), copy(x_ref); threaded = false) ≈ 0
    end

    @testset "size mismatch is rejected" begin
        x = Variable(randn(4, 4))
        @test_throws ArgumentError MriReconstructionToolbox.materialize(
            ReferencePrior(0.1, randn(3, 3)), x; threaded = false
        )
    end

    @testset "get_operator and get_affected_dims" begin
        x = randn(4, 4)
        @test get_operator(ReferencePrior(0.1, x), x; threaded = false) isa Eye
        ksp = randn(ComplexF32, 4, 4)
        info = AcquisitionInfo(ksp; image_size = (4, 4))
        # the reference has the size of the full image, so problem decomposition must be blocked
        @test MriReconstructionToolbox.get_affected_dims(ReferencePrior(0.1, x), info, 1:2) == (1, 2)
    end

    @testset "scale_regularization scales λ and the reference" begin
        x_ref = randn(4, 4)
        scaled = MriReconstructionToolbox.scale_regularization(ReferencePrior(0.2, x_ref), 2.5)
        @test scaled.λ ≈ 0.5
        @test scaled.reference ≈ x_ref .* 2.5
    end
end

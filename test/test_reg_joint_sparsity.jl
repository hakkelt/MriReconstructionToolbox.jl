using TestItems

@testitem "JointSparsity regularization" tags = [:regularization] setup = [RegTestSetup] begin
    using LinearAlgebra

    l21(x, dim) = sum(sqrt.(sum(abs2, x; dims = dim)))

    @testset "Constructor" begin
        reg = JointSparsity(0.1; dim = 3)
        @test reg.λ == 0.1
        @test reg.dim == 3
        @test JointSparsity(0.1; dim = :echo).dim == :echo
        @test_throws ArgumentError JointSparsity(0.1; dim = 0)
        @test_throws ArgumentError JointSparsity(rand(2, 2); dim = 1)
    end

    @testset "materialize - group dimension is last" for threaded in [false, true]
        x = randn(5, 4, 3)
        λ = 0.3
        @test MriReconstructionToolbox.calculate(JointSparsity(λ; dim = 3), x; threaded) ≈ λ * l21(x, 3)
    end

    @testset "materialize - group dimension in the middle" for threaded in [false, true]
        x = randn(4, 3, 2)
        λ = 0.7
        @test MriReconstructionToolbox.calculate(JointSparsity(λ; dim = 2), x; threaded) ≈ λ * l21(x, 2)
    end

    @testset "materialize - group dimension is first" begin
        x = randn(3, 4, 2)
        λ = 0.2
        @test MriReconstructionToolbox.calculate(JointSparsity(λ; dim = 1), x; threaded = false) ≈ λ * l21(x, 1)
    end

    @testset "complex data" begin
        x = randn(ComplexF64, 4, 3, 2)
        λ = 0.5
        @test MriReconstructionToolbox.calculate(JointSparsity(λ; dim = 2), x; threaded = false) ≈ λ * l21(x, 2)
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y, :echo)}(randn(4, 4, 3))
        reg = JointSparsity(0.25; dim = :echo)
        op = get_operator(reg, x; threaded = false)
        @test op isa MriReconstructionToolbox.NamedDimsOp
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈ 0.25 * l21(unname(x), 3)
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 4)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test MriReconstructionToolbox.get_affected_dims(JointSparsity(0.1f0; dim = 3), info, 1:3) == (3,)
        @test MriReconstructionToolbox.get_affected_dims(
            JointSparsity(0.1f0; dim = :echo), info, (:x, :y, :echo)
        ) == (:echo,)
    end

    @testset "scale_regularization" begin
        reg = MriReconstructionToolbox.scale_regularization(JointSparsity(0.2; dim = 3), 2.5)
        @test reg.λ ≈ 0.5
        @test reg.dim == 3
    end
end

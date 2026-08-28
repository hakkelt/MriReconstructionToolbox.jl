using TestItems

@testitem "NonNegative and BoxConstraint regularizations" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    @testset "NonNegative" for threaded in [false, true]
        @test MriReconstructionToolbox.calculate(NonNegative(), abs.(randn(4, 4)); threaded) == 0
        @test MriReconstructionToolbox.calculate(NonNegative(), [-1.0 1.0; 1.0 1.0]; threaded) == Inf
    end

    @testset "BoxConstraint" begin
        @test MriReconstructionToolbox.calculate(BoxConstraint(0.0, 1.0), rand(4, 4); threaded = false) == 0
        @test MriReconstructionToolbox.calculate(BoxConstraint(0.0, 1.0), [2.0 0.5]; threaded = false) == Inf
        @test MriReconstructionToolbox.calculate(
            BoxConstraint(zeros(1, 2), ones(1, 2)), [0.5 0.5]; threaded = false
        ) == 0
        @test_throws ArgumentError BoxConstraint(1.0, 0.0)
    end

    @testset "complex data is rejected" begin
        x = Variable(randn(ComplexF64, 4, 4))
        @test_throws ArgumentError MriReconstructionToolbox.materialize(NonNegative(), x; threaded = false)
        @test_throws ArgumentError MriReconstructionToolbox.materialize(
            BoxConstraint(0.0, 1.0), x; threaded = false
        )
    end

    @testset "get_operator and get_affected_dims" begin
        x = randn(4, 4)
        @test get_operator(NonNegative(), x; threaded = false) isa Eye
        @test get_operator(BoxConstraint(0.0, 1.0), x; threaded = false) isa Eye
        ksp = randn(ComplexF32, 4, 4)
        info = AcquisitionInfo(ksp; image_size = (4, 4))
        @test MriReconstructionToolbox.get_affected_dims(NonNegative(), info, 1:2) == ()
        @test MriReconstructionToolbox.get_affected_dims(BoxConstraint(0.0, 1.0), info, 1:2) == ()
        # array-valued bounds have the size of the full image, so decomposition must be blocked
        @test MriReconstructionToolbox.get_affected_dims(
            BoxConstraint(zeros(4, 4), ones(4, 4)), info, 1:2
        ) == (1, 2)
    end

    @testset "scale_regularization" begin
        @test MriReconstructionToolbox.scale_regularization(NonNegative(), 2.5) isa NonNegative
        scaled = MriReconstructionToolbox.scale_regularization(BoxConstraint(0.5, 1.0), 2.0)
        @test scaled.lower ≈ 1.0
        @test scaled.upper ≈ 2.0
    end
end


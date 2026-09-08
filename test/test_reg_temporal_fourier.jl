using TestItems

@testitem "L1TemporalFourier regularization" tags = [:regularization] setup = [RegTestSetup] begin

    @testset "Constructor" begin
        reg = L1TemporalFourier(0.1)
        @test reg.λ == 0.1
        @test isnothing(reg.time_dim)

        reg2 = L1TemporalFourier(0.2; time_dim = 3)
        @test reg2.time_dim == 3

        reg3 = L1TemporalFourier(0.3; time_dim = :time)
        @test reg3.time_dim == :time

        @test_throws ArgumentError L1TemporalFourier(0.1; time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float32, 8, 8, 10)
        reg = L1TemporalFourier(0.1f0; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == size(x)
        @test size(op, 2) == size(x)
        result = op * x
        @test size(result) == size(x)
    end

    @testset "materialize" begin
        x = Variable(Float32, 8, 8, 10)
        reg = L1TemporalFourier(0.1f0; time_dim = 3)
        term = MriReconstructionToolbox.materialize(reg, x; threaded = false)
        @test term !== nothing
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 10)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        reg = L1TemporalFourier(0.1f0; time_dim = 3)
        dims = MriReconstructionToolbox.get_affected_dims(reg, info, 1:3)
        @test dims == (3,)

        # Named image dims must yield Symbols so that task splitting can
        # setdiff them against Symbol batch dims.
        reg_named = L1TemporalFourier(0.1f0; time_dim = :time)
        dims_named = MriReconstructionToolbox.get_affected_dims(reg_named, info, (:x, :y, :time))
        @test dims_named == (:time,)
    end
end

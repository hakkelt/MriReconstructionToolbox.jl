using TestItems

@testitem "TemporalFourier regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    @testset "Constructor" begin
        reg = TemporalFourier(0.1)
        @test reg.λ == 0.1
        @test isnothing(reg.time_dim)

        reg2 = TemporalFourier(0.2; time_dim = 3)
        @test reg2.time_dim == 3

        reg3 = TemporalFourier(0.3; time_dim = :time)
        @test reg3.time_dim == :time

        @test_throws ArgumentError TemporalFourier(0.1; time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float32, 8, 8, 10)
        reg = TemporalFourier(0.1f0; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == size(x)
        @test size(op, 2) == size(x)
        result = op * x
        @test size(result) == size(x)
    end

    @testset "materialize" begin
        x = Variable(Float32, 8, 8, 10)
        reg = TemporalFourier(0.1f0; time_dim = 3)
        term = MriReconstructionToolbox.materialize(reg, x; threaded = false)
        @test term !== nothing
    end

    @testset "get_affected_dims" begin
        ksp = randn(ComplexF32, 8, 8, 10)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        reg = TemporalFourier(0.1f0; time_dim = 3)
        dims = MriReconstructionToolbox.get_affected_dims(reg, info, 1:3)
        @test dims == (3,)

        # Named image dims must yield Symbols so that problem decomposition can
        # setdiff them against Symbol batch dims.
        reg_named = TemporalFourier(0.1f0; time_dim = :time)
        dims_named = MriReconstructionToolbox.get_affected_dims(reg_named, info, (:x, :y, :time))
        @test dims_named == (:time,)
    end
end

@testitem "LowRank regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    @testset "Constructor" begin
        reg = LowRank(0.1)
        @test reg.λ == 0.1
        @test isnothing(reg.time_dim)

        reg2 = LowRank(0.2; time_dim = 3)
        @test reg2.time_dim == 3

        @test_throws ArgumentError LowRank(0.1; time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10)
        reg = LowRank(0.1; time_dim = 3)
        op = get_operator(reg, x; threaded)
        # Should reshape to Casorati matrix: (8*8, 10)
        @test size(op, 1) == (64, 10)
        @test size(op, 2) == size(x)
        result = op * x
        @test size(result) == (64, 10)
    end

    @testset "get_operator with batch dims" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10, 3)
        reg = LowRank(0.1; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == (64, 10, 3)
        result = op * x
        @test size(result) == (64, 10, 3)
    end

    @testset "materialize" begin
        x = Variable(Float64, 8, 8, 10)
        reg = LowRank(0.1; time_dim = 3)
        term = MriReconstructionToolbox.materialize(reg, x; threaded = false)
        @test term !== nothing
    end

    @testset "get_affected_dims" begin
        reg = LowRank(0.1; time_dim = 3)
        @test MriReconstructionToolbox.get_affected_dims(reg, nothing, 1:4) == 1:3

        reg_named = LowRank(0.1; time_dim = :time)
        dims_named = MriReconstructionToolbox.get_affected_dims(reg_named, nothing, (:x, :y, :time, :coil))
        @test dims_named == (:x, :y, :time)
    end

    @testset "LowRank prevents decomposition across time" begin
        nx, ny, nt = 8, 8, 5
        ksp = NamedDimsArray{(:kx, :ky, :time)}(rand(ComplexF32, nx, ny, nt))
        acq = AcquisitionInfo(ksp)
        config = Config(; verbose = false)

        plan_noreg = MriReconstructionToolbox.get_problem_decomposition_plan(acq, (), config)
        @test !isnothing(plan_noreg) # :time is a batch dim without regularization

        reg = LowRank(0.1; time_dim = :time)
        plan = MriReconstructionToolbox.get_problem_decomposition_plan(acq, (reg,), config)
        @test isnothing(plan) # LowRank couples the time dimension
    end
end

@testitem "RankLimit regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators

    @testset "Constructor" begin
        reg = RankLimit(5)
        @test reg.max_rank == 5
        @test isnothing(reg.time_dim)

        @test_throws ArgumentError RankLimit(0)
        @test_throws ArgumentError RankLimit(-1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(Float64, 8, 8, 10)
        reg = RankLimit(3; time_dim = 3)
        op = get_operator(reg, x; threaded)
        @test size(op, 1) == (64, 10)
        @test size(op, 2) == size(x)
    end

    @testset "materialize" begin
        x = Variable(Float64, 8, 8, 10)
        reg = RankLimit(3; time_dim = 3)
        term = MriReconstructionToolbox.materialize(reg, x; threaded = false)
        @test term !== nothing
    end
end

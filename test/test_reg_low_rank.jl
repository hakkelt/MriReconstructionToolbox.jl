using TestItems

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

@testitem "LocallyLowRank regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims

    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    function reference_llr(x, λ, block_size, nt)
        value = 0.0
        for i in 1:block_size[1]:size(x, 1), j in 1:block_size[2]:size(x, 2)
            irange = i:min(i + block_size[1] - 1, size(x, 1))
            jrange = j:min(j + block_size[2] - 1, size(x, 2))
            M = reshape(x[irange, jrange, :], length(irange) * length(jrange), nt)
            value += λ * sum(svdvals(M))
        end
        return value
    end

    @testset "Constructor" begin
        reg = LocallyLowRank(0.1; block_size = 4)
        @test reg.λ == 0.1
        @test reg.block_size == 4
        @test isnothing(reg.time_dim)
        @test LocallyLowRank(0.1; block_size = (4, 2), time_dim = 3).block_size == (4, 2)
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 0)
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 4, time_dim = -1)
    end

    @testset "get_operator" for threaded in [false, true]
        x = randn(8, 8, 4)
        op = get_operator(LocallyLowRank(0.1; block_size = 4, time_dim = 3), x; threaded)
        @test op isa Eye
        @test op * x ≈ x
    end

    @testset "materialize matches blockwise nuclear norms" for threaded in [false, true]
        x = randn(ComplexF64, 8, 6, 4)
        λ = 0.1
        reg = LocallyLowRank(λ; block_size = (4, 3), time_dim = 3)
        @test MriReconstructionToolbox.calculate(reg, x; threaded) ≈ reference_llr(x, λ, (4, 3), 4)
    end

    @testset "incomplete boundary blocks" begin
        x = randn(7, 5, 3)
        λ = 0.2
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈ reference_llr(x, λ, (4, 4), 3)
    end

    @testset "batch dimensions are handled independently" begin
        x = randn(4, 4, 3, 2)
        λ = 0.15
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        expected = reference_llr(x[:, :, :, 1], λ, (4, 4), 3) + reference_llr(x[:, :, :, 2], λ, (4, 4), 3)
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈ expected
    end

    @testset "prox is exact blockwise singular value thresholding" for threaded in [false, true]
        x = randn(ComplexF64, 8, 4, 3)
        λ, γ = 0.1, 0.7
        reg = LocallyLowRank(λ; block_size = 4, time_dim = 3)
        var = Variable(x)
        term = MriReconstructionToolbox.materialize(reg, var; threaded)
        f = SO.extract_functions(term)
        y = similar(x)
        fy = PC.prox!(y, f, x, γ)

        expected = similar(x)
        expected_value = 0.0
        for i in 1:4:8, j in 1:4:4
            M = reshape(x[i:(i + 3), j:min(j + 3, 4), :], :, 3)
            F = svd(M)
            σ = max.(0.0, F.S .- λ * γ)
            expected[i:(i + 3), j:min(j + 3, 4), :] = reshape(F.U * Diagonal(σ) * F.Vt, 4, :, 3)
            expected_value += λ * sum(σ)
        end
        @test y ≈ expected
        @test fy ≈ expected_value
    end

    @testset "NamedDimsArray input" begin
        x = NamedDimsArray{(:x, :y, :time)}(randn(4, 4, 3))
        reg = LocallyLowRank(0.1; block_size = 2, time_dim = :time)
        op = get_operator(reg, x; threaded = false)
        @test op isa MriReconstructionToolbox.NamedDimsOp
        @test MriReconstructionToolbox.calculate(reg, x; threaded = false) ≈
            reference_llr(unname(x), 0.1, (2, 2), 3)
    end

    @testset "get_affected_dims couples space and time" begin
        ksp = randn(ComplexF32, 8, 8, 10)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        reg = LocallyLowRank(0.1f0; block_size = 4, time_dim = 3)
        @test MriReconstructionToolbox.get_affected_dims(reg, info, 1:3) == 1:3
        reg_named = LocallyLowRank(0.1f0; block_size = 4, time_dim = :time)
        @test MriReconstructionToolbox.get_affected_dims(reg_named, info, (:x, :y, :time)) == (:x, :y, :time)
    end

    @testset "block size validation" begin
        x = Variable(randn(4, 4, 2))
        reg = LocallyLowRank(0.1; block_size = 8, time_dim = 3)
        @test_throws ArgumentError MriReconstructionToolbox.materialize(reg, x; threaded = false)
        reg_wrong_rank = LocallyLowRank(0.1; block_size = (4, 4, 4), time_dim = 3)
        @test_throws ArgumentError MriReconstructionToolbox.materialize(reg_wrong_rank, x; threaded = false)
    end

    @testset "scale_regularization" begin
        reg = MriReconstructionToolbox.scale_regularization(
            LocallyLowRank(0.2; block_size = 4, time_dim = 3), 2.5
        )
        @test reg.λ ≈ 0.5
        @test reg.block_size == 4
        @test reg.time_dim == 3
    end
end


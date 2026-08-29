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
        @test reg.shift == :none
    end
end

@testitem "LocallyLowRank grid shifts" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using Random
    using MriReconstructionToolbox

    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    materialized(reg, x) = SO.extract_functions(
        MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
    )

    function prox_of(reg, x, γ = 1.0)
        y = similar(x)
        PC.prox!(y, materialized(reg, x), x, γ)
        return y
    end

    @testset "Constructor" begin
        @test LocallyLowRank(0.1; block_size = 4).shift == :none
        @test LocallyLowRank(0.1; block_size = 4, shift = :random).shift == :random
        @test_throws ArgumentError LocallyLowRank(0.1; block_size = 4, shift = :sometimes)
    end

    @testset "a shifted grid is exactly the unshifted grid on a circularly shifted image" begin
        x = randn(MersenneTwister(11), ComplexF64, 8, 8, 5)
        unshifted = LocallyLowRank(0.3; block_size = 4, time_dim = 3)
        shifted = LocallyLowRank(0.3; block_size = 4, time_dim = 3, shift = :fixed, rng = MersenneTwister(7))
        f = materialized(shifted, x)
        offset = f.offset[]
        @test offset != (0, 0)   # the seed above must actually produce a shift for this test to mean anything

        y = similar(x)
        PC.prox!(y, f, x, 1.0)
        # The wrapped tiling is still a permutation of the voxels, so shifting the image by -offset,
        # applying the unshifted prox and shifting back must reproduce it bit for bit.
        rolled = circshift(x, (-offset[1], -offset[2], 0))
        expected = circshift(prox_of(unshifted, rolled), (offset[1], offset[2], 0))
        @test y ≈ expected
    end

    @testset "a shifted grid needs a divisible image" begin
        x = Variable(randn(7, 8, 3))
        reg = LocallyLowRank(0.1; block_size = 4, time_dim = 3, shift = :fixed)
        @test_throws ArgumentError MriReconstructionToolbox.materialize(reg, x; threaded = false)
    end

    @testset "shift=:random redraws the origin on every prox call" begin
        x = randn(MersenneTwister(3), 8, 8, 4)
        reg = LocallyLowRank(0.3; block_size = 4, time_dim = 3, shift = :random, rng = MersenneTwister(5))
        f = materialized(reg, x)
        y = similar(x)
        offsets = map(1:12) do _
            PC.prox!(y, f, x, 1.0)
            f.offset[]
        end
        @test length(unique(offsets)) > 1
    end

    @testset "scale_regularization keeps the shift policy" begin
        reg = MriReconstructionToolbox.scale_regularization(
            LocallyLowRank(0.2; block_size = 4, time_dim = 3, shift = :random), 2.0
        )
        @test reg.λ ≈ 0.4
        @test reg.shift == :random
    end
end

@testitem "MultiScaleLowRank regularization" tags = [:regularization] begin
    using Test
    using LinearAlgebra
    using MriReconstructionToolbox
    using AbstractOperators

    const SO = MriReconstructionToolbox.StructuredOptimization
    const PC = MriReconstructionToolbox.ProximalCore

    function prox_of(reg, x, γ = 1.0)
        term = MriReconstructionToolbox.materialize(reg, Variable(x); threaded = false)
        y = similar(x)
        PC.prox!(y, SO.extract_functions(term), x, γ)
        return y
    end

    @testset "Constructor" begin
        reg = MultiScaleLowRank(0.1; block_sizes = (4, 8))
        @test reg.λ == 0.1
        @test reg.block_sizes == (4, 8)
        @test reg.weights === nothing
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = ())
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4, 0))
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4, 8), weights = [1.0])
        @test_throws ArgumentError MultiScaleLowRank(0.1; block_sizes = (4,), shift = :maybe)
    end

    @testset "get_operator" begin
        x = randn(8, 8, 4)
        op = get_operator(MultiScaleLowRank(0.1; block_sizes = (4,), time_dim = 3), x; threaded = false)
        @test op isa Eye
        @test op * x ≈ x
    end

    @testset "a single scale reduces to LocallyLowRank" for threaded in [false, true]
        x = randn(ComplexF64, 8, 8, 5)
        mslr = MultiScaleLowRank(0.3; block_sizes = (4,), time_dim = 3)
        llr = LocallyLowRank(0.3; block_size = 4, time_dim = 3)
        @test MriReconstructionToolbox.calculate(mslr, x; threaded) ≈
            MriReconstructionToolbox.calculate(llr, x; threaded)
        @test prox_of(mslr, x) ≈ prox_of(llr, x)
    end

    @testset "the prox is the weighted average of the per-scale proxes" begin
        x = randn(ComplexF64, 8, 8, 5)
        scales = (2, 4, 8)
        weights = [0.2, 0.5, 0.3]
        mslr = MultiScaleLowRank(0.3; block_sizes = scales, time_dim = 3, weights)
        expected = sum(
            w .* prox_of(LocallyLowRank(0.3; block_size = b, time_dim = 3), x, 0.7)
                for (w, b) in zip(weights, scales)
        )
        @test prox_of(mslr, x, 0.7) ≈ expected
    end

    @testset "the value is the weighted average of the per-scale penalties" begin
        x = randn(8, 8, 4)
        scales = (4, 8)
        mslr = MultiScaleLowRank(0.2; block_sizes = scales, time_dim = 3)
        expected = sum(
            MriReconstructionToolbox.calculate(LocallyLowRank(0.2; block_size = b, time_dim = 3), x) / 2
                for b in scales
        )
        @test MriReconstructionToolbox.calculate(mslr, x) ≈ expected
    end

    @testset "get_affected_dims couples space and time" begin
        reg = MultiScaleLowRank(0.1; block_sizes = (2, 4), time_dim = 3)
        @test MriReconstructionToolbox.get_affected_dims(reg, nothing, 1:4) == 1:3
    end

    @testset "scale_regularization" begin
        reg = MriReconstructionToolbox.scale_regularization(
            MultiScaleLowRank(0.2; block_sizes = (4, 8), time_dim = 3), 2.5
        )
        @test reg.λ ≈ 0.5
        @test reg.block_sizes == (4, 8)
    end
end

@testitem "ProximalAverage" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox

    const PC = MriReconstructionToolbox.ProximalCore
    const PA = MriReconstructionToolbox.ProximalAverage
    const PO = MriReconstructionToolbox.ProximalOperators

    @testset "weights must be a convex combination" begin
        f = PO.NormL1(1.0)
        @test_throws ArgumentError PA((f, f), [0.5])
        @test_throws ArgumentError PA((f, f), [0.7, 0.7])
        @test_throws ArgumentError PA((f, f), [-0.5, 1.5])
        @test PA((f, f), [0.25, 0.75]) isa PA
    end

    @testset "the prox of an average of identical functions is that function's prox" begin
        x = randn(16)
        f = PO.NormL1(0.3)
        avg = PA((f, f, f), fill(1 / 3, 3))
        y, y_ref = similar(x), similar(x)
        PC.prox!(y, avg, x, 0.8)
        PC.prox!(y_ref, f, x, 0.8)
        @test y ≈ y_ref
        @test avg(x) ≈ f(x)
    end
end


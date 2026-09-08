using TestItems

@testitem "HardThreshold regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    @testset "Constructor" begin
        reg = HardThreshold(0.5)
        @test reg.λ == 0.5
        @test reg.domain == :image
        @test HardThreshold(0.5; domain = :wavelet3d, levels = 3).levels == 3
        @test_throws ArgumentError HardThreshold(-0.5)
        @test_throws ArgumentError HardThreshold(0.5; domain = :dct)
    end

    @testset "get_operator" begin
        x = randn(8, 8)
        @test get_operator(HardThreshold(0.5), x; threaded = false) isa Eye
        op = get_operator(HardThreshold(0.5; domain = :wavelet2d), x; threaded = false)
        @test size(op, 2) == size(x)
        named = NamedDimsArray{(:x, :y)}(randn(8, 8))
        @test get_operator(HardThreshold(0.5), named; threaded = false) isa MriReconstructionToolbox.NamedDimsOp
    end

    @testset "calculate counts non-zeros" begin
        x = zeros(4, 4)
        x[1, 1] = 1.0
        x[2, 3] = -2.0
        @test MriReconstructionToolbox.calculate(HardThreshold(0.5), x) ≈ 1.0
    end

    @testset "the prox keeps large coefficients untouched" for λ in (0.05, 0.5)
        x = randn(ComplexF64, 8, 8, 2)
        γ = 0.8
        y, _ = prox_of(HardThreshold(λ), x, γ)
        threshold = sqrt(2 * γ * λ)
        kept = abs.(x) .> threshold
        # Hard thresholding, unlike soft thresholding, does not shrink what it keeps.
        @test y[kept] == x[kept]
        @test all(iszero, y[.!kept])
    end

    @testset "get_affected_dims" begin
        @test MriReconstructionToolbox.get_affected_dims(HardThreshold(0.5), nothing, 1:4) == ()
        @test MriReconstructionToolbox.get_affected_dims(
            HardThreshold(0.5; domain = :wavelet2d), nothing, (:x, :y, :time)
        ) == (:x, :y)
        @test MriReconstructionToolbox.get_affected_dims(
            HardThreshold(0.5; domain = :wavelet3d), nothing, 1:4
        ) == (1, 2, 3)
    end

    @testset "scale_regularization" begin
        # ‖·‖₀ is homogeneous of degree 0, so λ picks up factor², unlike the ℓ₁ terms.
        reg = MriReconstructionToolbox.scale_regularization(HardThreshold(0.2), 3.0)
        @test reg.λ ≈ 1.8
        @test reg.domain == :image

        # The same invariant every term must satisfy: on an image scaled by `factor`, the scaled term
        # equals `factor²` times the original term, matching how the data term scales.
        x = randn(6, 6)
        @test MriReconstructionToolbox.calculate(reg, x .* 3.0) ≈
            3.0^2 * MriReconstructionToolbox.calculate(HardThreshold(0.2), x)
    end
end

@testitem "SparsityLimit regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    @testset "Constructor" begin
        @test SparsityLimit(10).max_nonzeros == 10
        @test_throws ArgumentError SparsityLimit(0)
        @test_throws ArgumentError SparsityLimit(10; domain = :dct)
    end

    @testset "calculate is the indicator of the sparsity ball" begin
        x = zeros(4, 4)
        x[1:3] .= 1.0
        @test MriReconstructionToolbox.calculate(SparsityLimit(3), x) == 0
        @test MriReconstructionToolbox.calculate(SparsityLimit(2), x) == Inf
    end

    @testset "the prox keeps the k largest coefficients" begin
        x = randn(ComplexF64, 8, 8)
        k = 12
        y, _ = prox_of(SparsityLimit(k), x)
        @test count(!iszero, y) == k
        kept = findall(!iszero, vec(y))
        @test vec(y)[kept] == vec(x)[kept]
        # everything dropped must be no larger than everything kept
        dropped = setdiff(eachindex(vec(x)), kept)
        @test maximum(abs, vec(x)[dropped]) <= minimum(abs, vec(x)[kept])
    end

    @testset "the budget blocks task splitting" begin
        @test MriReconstructionToolbox.get_affected_dims(SparsityLimit(4), nothing, (:x, :y, :slice)) ==
            (:x, :y, :slice)
    end

    @testset "scale_regularization is a no-op" begin
        reg = SparsityLimit(7)
        @test MriReconstructionToolbox.scale_regularization(reg, 3.0) === reg
    end
end

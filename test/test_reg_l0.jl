using TestItems

@testitem "L0Image regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    @testset "Constructor" begin
        reg = L0Image(threshold = 0.5)
        @test reg.threshold == 0.5
        @test reg.count === nothing
        @test L0Image(count = 10).count == 10
        @test_throws ArgumentError L0Image()
        @test_throws ArgumentError L0Image(threshold = 0.5, count = 10)
        @test_throws ArgumentError L0Image(threshold = -0.5)
        @test_throws ArgumentError L0Image(count = 0)
    end

    @testset "get_operator" begin
        x = randn(8, 8)
        @test get_operator(L0Image(threshold = 0.5), x; threaded = false) isa Eye
        named = NamedDimsArray{(:x, :y)}(randn(8, 8))
        @test get_operator(L0Image(threshold = 0.5), named; threaded = false) isa MriReconstructionToolbox.NamedDimsOp
    end

    @testset "calculate counts non-zeros (threshold form)" begin
        x = zeros(4, 4)
        x[1, 1] = 1.0
        x[2, 3] = -2.0
        @test MriReconstructionToolbox.calculate(L0Image(threshold = 0.5), x) ≈ 1.0
    end

    @testset "calculate is the indicator of the sparsity ball (count form)" begin
        x = zeros(4, 4)
        x[1:3] .= 1.0
        @test MriReconstructionToolbox.calculate(L0Image(count = 3), x) == 0
        @test MriReconstructionToolbox.calculate(L0Image(count = 2), x) == Inf
    end

    @testset "the prox keeps large coefficients untouched (threshold form)" for λ in (0.05, 0.5)
        x = randn(ComplexF64, 8, 8, 2)
        γ = 0.8
        y, _ = prox_of(L0Image(threshold = λ), x, γ)
        threshold = sqrt(2 * γ * λ)
        kept = abs.(x) .> threshold
        # Hard thresholding, unlike soft thresholding, does not shrink what it keeps.
        @test y[kept] == x[kept]
        @test all(iszero, y[.!kept])
    end

    @testset "the prox keeps the k largest coefficients (count form)" begin
        x = randn(ComplexF64, 8, 8)
        k = 12
        y, _ = prox_of(L0Image(count = k), x)
        @test count(!iszero, y) == k
        kept = findall(!iszero, vec(y))
        @test vec(y)[kept] == vec(x)[kept]
        # everything dropped must be no larger than everything kept
        dropped = setdiff(eachindex(vec(x)), kept)
        @test maximum(abs, vec(x)[dropped]) <= minimum(abs, vec(x)[kept])
    end

    @testset "get_affected_dims" begin
        @test MriReconstructionToolbox.get_affected_dims(L0Image(threshold = 0.5), nothing, 1:4) == ()
        @test MriReconstructionToolbox.get_affected_dims(L0Image(count = 4), nothing, (:x, :y, :slice)) ==
            (:x, :y, :slice)
    end

    @testset "scale_regularization" begin
        # ‖·‖₀ is homogeneous of degree 0, so λ picks up factor², unlike the ℓ₁ terms.
        reg = MriReconstructionToolbox.scale_regularization(L0Image(threshold = 0.2), 3.0)
        @test reg.threshold ≈ 1.8

        # The same invariant every term must satisfy: on an image scaled by `factor`, the scaled term
        # equals `factor²` times the original term, matching how the data term scales.
        x = randn(6, 6)
        @test MriReconstructionToolbox.calculate(reg, x .* 3.0) ≈
            3.0^2 * MriReconstructionToolbox.calculate(L0Image(threshold = 0.2), x)

        # The count form is scale-invariant: no-op.
        count_reg = L0Image(count = 7)
        @test MriReconstructionToolbox.scale_regularization(count_reg, 3.0) === count_reg
    end
end

@testitem "L0Wavelet2D/L0Wavelet3D regularization" tags = [:regularization] setup = [RegTestSetup, ProxOf] begin
    @testset "Constructor" begin
        @test L0Wavelet2D(threshold = 0.5; levels = 3).levels == 3
        @test_throws ArgumentError L0Wavelet2D()
        @test_throws ArgumentError L0Wavelet2D(threshold = 0.5, count = 10)
        @test_throws ArgumentError L0Wavelet3D()
    end

    @testset "get_operator" begin
        x = randn(8, 8)
        op = get_operator(L0Wavelet2D(threshold = 0.5), x; threaded = false)
        @test size(op, 2) == size(x)
    end

    @testset "get_affected_dims" begin
        @test MriReconstructionToolbox.get_affected_dims(
            L0Wavelet2D(threshold = 0.5), nothing, (:x, :y, :time)
        ) == (:x, :y)
        @test MriReconstructionToolbox.get_affected_dims(
            L0Wavelet3D(threshold = 0.5), nothing, 1:4
        ) == (1, 2, 3)
        # The count form couples the whole array, regardless of domain, so it blocks task splitting
        # over every dimension, not just the ones the wavelet transform touches.
        @test MriReconstructionToolbox.get_affected_dims(
            L0Wavelet2D(count = 4), nothing, (:x, :y, :time)
        ) == (:x, :y, :time)
    end

    @testset "scale_regularization is a no-op for the count form" begin
        reg = L0Wavelet2D(count = 7)
        @test MriReconstructionToolbox.scale_regularization(reg, 3.0) === reg
    end
end

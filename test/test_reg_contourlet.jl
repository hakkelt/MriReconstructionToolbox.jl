using TestItems

@testitem "L1Contourlet Regularization" tags = [:regularization] setup = [RegTestSetup] begin
    @testset "constructor" begin
        reg = L1Contourlet(0.1)
        @test reg.λ == 0.1
        @test reg.params isa ContourletParams
    end

    @testset "get_operator - 2D" for threaded in [false, true]
        x = rand(32, 32)
        reg = L1Contourlet(0.1; params = ContourletParams(J = 2, L_array = parabolic_levels(2)))
        op = get_operator(reg, x; threaded)
        @test op isa MriReconstructionToolbox.StackedNSCTOp

        result = op * x
        @test size(result, 1) == 32
        @test size(result, 2) == 32
        @test ndims(result) == 3

        # Round trip via the declared (perfect-reconstruction) inverse.
        x_rec = op' * result
        @test x_rec ≈ x atol = 1.0e-8
    end

    @testset "get_operator - batched 3D input" for threaded in [false, true]
        x = rand(32, 32, 4)
        reg = L1Contourlet(0.1; params = ContourletParams(J = 2, L_array = parabolic_levels(2)))
        op = get_operator(reg, x; threaded)
        @test op isa BatchOp

        result = op * x
        @test size(result)[1:2] == (32, 32)
        @test size(result, 4) == 4

        x_rec = op' * result
        @test x_rec ≈ x atol = 1.0e-8
    end

    @testset "get_operator - precision bridging" for T in [Float32, Float64, ComplexF32, ComplexF64]
        x = rand(T, 32, 32)
        reg = L1Contourlet(0.1; params = ContourletParams(J = 2, L_array = parabolic_levels(2)))
        op = get_operator(reg, x)
        @test domain_type(op) == T
        @test codomain_type(op) == T

        result = op * x
        x_rec = op' * result
        @test x_rec ≈ x atol = 1.0e-6
    end

    @testset "get_operator - dimension check" begin
        x = rand(10)  # 1D
        reg = L1Contourlet(0.1)
        @test_throws ArgumentError get_operator(reg, x)
    end

    @testset "materialize" for threaded in [false, true]
        x = rand(32, 32)
        λ = 0.2
        reg = L1Contourlet(λ; params = ContourletParams(J = 2, L_array = parabolic_levels(2)))
        result = MriReconstructionToolbox.calculate(reg, x; threaded)
        @test result isa Real
        @test result ≥ 0  # L1 norm is non-negative

        op = get_operator(reg, x; threaded = false)
        coeffs = op * x
        manual_result = λ * sum(abs, coeffs)
        @test result ≈ manual_result
    end
end

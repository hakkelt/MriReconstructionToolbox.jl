using TestItems

@testitem "Tikhonov Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "Tikhonov Regularization" for threaded in [false, true]
        @testset "get_operator" begin
            x = rand(10, 10)
            reg = Tikhonov(0.1)
            op = get_operator(reg, x; threaded)
            @test op isa Eye
            @test size(op) == (size(x), size(x))
            result = op * x
            @test result ≈ x
        end

        @testset "materialize - scalar λ" for threaded in [false, true]
            x = rand(5, 5)
            λ = 0.5
            reg = Tikhonov(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            manual_result = sum(abs2, λ .* x)
            @test result ≈ manual_result
        end

        @testset "materialize - matrix λ" for threaded in [false, true]
            x = ones(3, 3)
            λ_matrix = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
            reg = Tikhonov(λ_matrix)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            manual_result = sum(abs2, λ_matrix .* x)
            @test result ≈ manual_result
        end
    end
end

@testitem "L1Image Regularization" tags = [:regularization] begin
    using Test
    using MriReconstructionToolbox
    using AbstractOperators
    using NamedDims
    using Wavelets

    @testset "L1Image Regularization" begin
        @testset "get_operator" for threaded in [false, true]
            x = rand(8, 8)
            reg = L1Image(0.2)
            op = get_operator(reg, x; threaded)
            @test op isa Eye
            @test size(op) == (size(x), size(x))
            result = op * x
            @test result ≈ x
        end

        @testset "materialize" for threaded in [false, true]
            x = rand(6, 6)
            λ = 0.3
            reg = L1Image(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            manual_result = λ * norm(x, 1)
            @test result ≈ manual_result
        end

        @testset "materialize - complex data" for threaded in [false, true]
            x = randn(ComplexF64, 4, 4)
            λ = 0.5
            reg = L1Image(λ)
            result = MriReconstructionToolbox.calculate(reg, x; threaded)
            manual_result = λ * norm(x, 1)
            @test result ≈ manual_result
        end
    end
end

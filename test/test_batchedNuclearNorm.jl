using LinearAlgebra
using ProximalOperators
using Random
using Test

@testset "BatchedNuclearNorm is NuclearNorm slice by slice" begin
    Random.seed!(0)
    for X in (randn(ComplexF64, 12, 5, 4), randn(6, 9, 3))
        f = BatchedNuclearNorm(0.3)
        @test f(X) ≈ sum(k -> NuclearNorm(0.3)(X[:, :, k]), axes(X, 3))
        Y, fy = prox(f, X, 0.7)
        for k in axes(X, 3)
            Yk, fyk = prox(NuclearNorm(0.3), X[:, :, k], 0.7)
            @test Y[:, :, k] ≈ Yk
        end
        @test fy ≈ f(Y)
        Z = copy(X)
        @test prox!(Z, f, Z, 0.7) ≈ fy
        @test Z ≈ Y
    end
end

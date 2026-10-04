using LinearAlgebra
using ProximalOperators
using Random
using Test

@testset "SqrNormL2 with weights that broadcast" begin
    Random.seed!(0)
    x = randn(ComplexF64, 6, 3, 4)
    w = rand(6, 1, 4) .+ 0.5
    W = repeat(w, 1, 3, 1)
    f, F = SqrNormL2(w), SqrNormL2(W)
    @test f(x) ≈ F(x)
    y, Y = similar(x), similar(x)
    @test gradient!(y, f, x) ≈ gradient!(Y, F, x)
    @test y ≈ Y
    for gamma in (0.7, fill(0.7, size(x)))
        @test prox!(y, f, x, gamma) ≈ prox!(Y, F, x, gamma)
        @test y ≈ Y
    end
end

@testset "SqrNormL2 with broadcast weights, threaded and along the first axis" begin
    Random.seed!(1)
    x = randn(ComplexF32, 64, 17, 8, 40)
    for w in (rand(Float32, 64, 17, 1, 40), rand(Float32, 1, 17, 8), rand(Float32, 64))
        w .+= 0.5f0
        W = w .* ones(Float32, size(x))
        for threaded in (true, false)
            f, F = SqrNormL2(w; threaded), SqrNormL2(W; threaded)
            @test f(x) ≈ F(x)
            y, Y = similar(x), similar(x)
            @test gradient!(y, f, x) ≈ gradient!(Y, F, x)
            @test y ≈ Y
            @test prox!(y, f, x, 0.3f0) ≈ prox!(Y, F, x, 0.3f0)
            @test y ≈ Y
        end
    end
end

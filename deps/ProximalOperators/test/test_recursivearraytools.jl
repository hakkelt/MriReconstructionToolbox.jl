using Test
using Random
using LinearAlgebra
using ProximalOperators
using RecursiveArrayTools

Random.seed!(1234)

@testset "RecursiveArrayToolsExt SeparableSum" begin
    x = randn(10)
    X = randn(10,10) .+ im*randn(10,10)
    
    lambdas = (abs.(randn(size(x))), 0.1)
    prox_col = (NormL1(lambdas[1]), NormL2(lambdas[2]))
    
    f = SeparableSum(prox_col)
    
    # Test on ArrayPartition
    xs = ArrayPartition(x, X)
    
    # evaluation
    @test f(xs) == f((x, X))
    
    # prox
    ys, fys = prox(f, xs, 1.0)
    y, fy = prox(f, (x, X), 1.0)
    @test fys == fy
    @test ys.x[1] == y[1]
    @test ys.x[2] == y[2]
    @test typeof(ys) <: ArrayPartition

    # prox!
    ys_mut = ArrayPartition(similar(x), similar(X))
    prox!(ys_mut, f, xs, 1.0)
    @test ys_mut.x[1] == y[1]
    @test ys_mut.x[2] == y[2]

    # prox! with multiple gammas
    ys_mut2 = ArrayPartition(similar(x), similar(X))
    gammas = (0.5, 1.3)
    prox!(ys_mut2, f, xs, gammas)
    y_g, fy_g = prox(f, (x, X), gammas)
    @test ys_mut2.x[1] == y_g[1]
    @test ys_mut2.x[2] == y_g[2]

    # gradient
    fs = (SqrNormL2(), LeastSquares(randn(5,10), randn(5)))
    f2 = SeparableSum(fs)
    x1, x2 = randn(10), randn(10)
    xs2 = ArrayPartition(x1, x2)
    
    grad_xs2, f_xs2 = gradient(f2, xs2)
    grad_x, f_x = gradient(f2, (x1, x2))
    
    @test f_xs2 == f_x
    @test grad_xs2.x[1] == grad_x[1]
    @test grad_xs2.x[2] == grad_x[2]
    @test typeof(grad_xs2) <: ArrayPartition
    
    # gradient!
    grad_xs2_mut = ArrayPartition(similar(x1), similar(x2))
    gradient!(grad_xs2_mut, f2, xs2)
    @test grad_xs2_mut.x[1] == grad_x[1]
    @test grad_xs2_mut.x[2] == grad_x[2]
end

@testset "RecursiveArrayToolsExt PrecomposedSlicedSeparableSum" begin
    fs = (NormL1(), NormL2(), SqrNormL2())

    A1 = (Diagonal(ones(10)), nothing)
    F = qr(randn(5, 5))
    A2 = (nothing, Matrix(F.Q))
    F = qr(randn(5, 5))
    A3 = (nothing, Matrix(F.Q))
    mu = rand(5)
    A3[2] .*= reshape(mu, 5, 1)
    ops = (A1, A2, A3)

    idxs = ((Colon(), nothing), (nothing, 1:5), (nothing, 6:10))
    μs = (1.0, 1.0, mu)

    f = PrecomposedSlicedSeparableSum(fs, idxs, ops, μs)
    x_tuple = (randn(10), rand(10))
    xs = ArrayPartition(x_tuple...)

    # evaluation
    @test f(xs) == f(x_tuple)

    # prox!
    ys_mut = ArrayPartition(zeros(10), zeros(10))
    ys_tuple_mut = (zeros(10), zeros(10))
    
    fy_xs = prox!(ys_mut, f, xs, 1.0)
    fy_tuple = prox!(ys_tuple_mut, f, x_tuple, 1.0)
    
    @test fy_xs == fy_tuple
    @test ys_mut.x[1] == ys_tuple_mut[1]
    @test ys_mut.x[2] == ys_tuple_mut[2]
end

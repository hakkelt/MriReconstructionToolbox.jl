@testitem "PermuteDims" tags = [:linearoperator, :PermuteDims] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing PermuteDims --- ")

    n, m, l = 2, 3, 4
    perm = (3, 1, 2)
    op = PermuteDims(Float64, (n, m, l), perm)

    test_op(op, randn(n, m, l), randn(l, n, m), verb)

    @test size(op) == ((l, n, m), (n, m, l))
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    @test domain_array_type(op) == Array{Float64}
    @test is_thread_safe(op) == true

    x = reshape(collect(1.0:(n * m * l)), n, m, l)
    @test op * x == permutedims(x, perm)
    @test op' * (op * x) == x

    @testset "the operator is orthogonal" begin
        @test is_orthogonal(op)
        @test is_invertible(op)
        @test is_full_row_rank(op)
        @test is_full_column_rank(op)
        @test diag_AcA(op) == 1.0
        @test diag_AAc(op) == 1.0
        # a permutation preserves the norm, which is what being orthogonal buys the solvers
        y = randn(n, m, l)
        @test norm(op * y) ≈ norm(y)
    end

    @testset "constructors" begin
        @test size(PermuteDims((n, m, l), perm)) == size(op)
        @test size(PermuteDims(zeros(n, m, l), perm)) == size(op)
        @test domain_type(PermuteDims(zeros(ComplexF64, n, m), (2, 1))) == ComplexF64
        # `perm` given as a vector is accepted as well
        @test size(PermuteDims(Float64, (n, m, l), [3, 1, 2])) == size(op)
    end

    @testset "an invalid permutation is rejected at construction" begin
        @test_throws ArgumentError PermuteDims(Float64, (n, m, l), (1, 1, 2))
        @test_throws ArgumentError PermuteDims(Float64, (n, m, l), (1, 2, 4))
    end

    @testset "the identity permutation is the identity map" begin
        identity_op = PermuteDims(Float64, (n, m, l), (1, 2, 3))
        @test identity_op * x == x
    end

    @testset "complex input" begin
        complex_op = PermuteDims(ComplexF64, (n, m, l), perm)
        z = randn(ComplexF64, n, m, l)
        w = randn(ComplexF64, l, n, m)
        @test complex_op * z == permutedims(z, perm)
        @test dot(complex_op * z, w) ≈ dot(z, complex_op' * w)
    end
end

@testitem "is_linear and is_affine" tags = [:calculus, :AffineAdd] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)

    n = 4
    L = DiagOp(randn(n))
    A = AffineAdd(MatrixOp(randn(n, n)), randn(n))
    N = Sin(n)
    trait(op) = (is_linear(op), is_affine(op))

    # Leaves: linear, affine, nonlinear.
    @test L isa AffineOperator
    @test trait(L) == (true, true)
    @test trait(A) == (false, true)
    @test trait(N) == (false, false)
    @test trait(AffineAdd(N, randn(n))) == (false, false)

    # A combination is linear when all of its parts are, and affine when all of its parts are.
    for combine in (
            (X, Y) -> X * Y, (X, Y) -> X + Y, (X, Y) -> [X; Y], (X, Y) -> [X Y],
            (X, Y) -> DCAT(X, Y),
        )
        @test trait(combine(L, L)) == (true, true)
        @test trait(combine(L, A)) == (false, true)
        @test trait(combine(A, L)) == (false, true)
        @test trait(combine(L, N)) == (false, false)
    end
    @test trait(3.0 * A) == (false, true)
    @test trait(reshape(A, 2, 2)) == (false, true)
    @test trait(BatchOp(A, (3,))) == (false, true)
    @test trait(BatchOp(L, (3,))) == (true, true)

    # The adjoint of an affine operator is the adjoint of its linear part, so it is linear.
    r = randn(n)
    @test trait(A') == (true, true)
    @test A' * r ≈ remove_displacement(A)' * r
    @test trait((L * A)') == (true, true)
    @test (L * A)' * r ≈ remove_displacement(A)' * (L' * r)
    @test_throws ErrorException N'

    # Combination rules that need a linear operator no longer take an affine one: scaling an
    # affine chain scales its displacement too.
    x = randn(n)
    S = 2.0 * (L * A)
    @test S * x ≈ 2.0 * (L * (A * x))
end

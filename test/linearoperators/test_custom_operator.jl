@testitem "AbstractOperator fallback properties" tags = [:misc] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    import AbstractOperators: domain_type, codomain_type, fun_name
    Random.seed!(0)

    # The minimal operator `docs/src/custom.md` describes, and nothing else.
    struct DenseTestOp <: LinearOperator
        A::Matrix{Float64}
    end
    Base.size(L::DenseTestOp) = ((size(L.A, 1),), (size(L.A, 2),))
    domain_type(::DenseTestOp) = Float64
    codomain_type(::DenseTestOp) = Float64
    fun_name(::DenseTestOp) = "A"
    LinearAlgebra.mul!(y::AbstractArray, L::DenseTestOp, x::AbstractArray) = mul!(y, L.A, x)
    LinearAlgebra.mul!(y::AbstractArray, L::AdjointOperator{DenseTestOp}, x::AbstractArray) =
        mul!(y, L.A.A', x)

    n, m = 5, 4
    A = randn(n, m)
    op = DenseTestOp(A)

    @test is_linear(op) && is_affine(op)
    # A linear operator's displacement is zero, without it being applied.
    @test displacement(op) === 0.0
    # is_thread_safe falls back to false for custom operators
    @test is_thread_safe(op) == false
    # is_sliced falls back to false
    @test is_sliced(op) == false
    # get_slicing_expr falls back to Colon() for non-null operators
    @test AbstractOperators.get_slicing_expr(op) == Colon()
    # get_slicing_mask throws for operators without specialization
    @test_throws ErrorException AbstractOperators.get_slicing_mask(op)
    # has_optimized_normalop falls back to false
    @test AbstractOperators.has_optimized_normalop(op) == false
    # get_normal_op falls back to L' * L
    normal = AbstractOperators.get_normal_op(op)
    x = randn(m)
    @test normal * x ≈ A' * (A * x)
    # diag throws for non-diagonal operators
    @test_throws ErrorException diag(op)
end

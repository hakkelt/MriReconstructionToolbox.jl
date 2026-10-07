@testitem "Accumulating mul!: one pass, no allocation" tags = [:calculus, :linearoperator, :AccumulatingMul] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    Random.seed!(0)

    n = 1 << 16
    ops = Any[
        Eye(Float64, (n,)),
        Zeros(Float64, (n,), Float64, (n,)),
        MatrixOp(randn(40, 30)),
        LMatrixOp(randn(30), 20),
        GetIndex(Float64, (64, 64), (2:40, :)),
        GetIndex(Float64, (n,), collect(1:3:n)),
        ZeroPad((64, 64), (3, 5)),
        3.0 * FiniteDiff(Float64, (2, 512, 32), 2),
    ]
    for threaded in (false, true)
        append!(
            ops, Any[
                DiagOp(randn(n); threaded),
                DiagOp(Float64, (n,), randn(ComplexF64, n); threaded),
                FiniteDiff(Float64, (2, 512, 32), 2; threaded),
                FiniteDiff(Float64, (n,), 1; threaded),
                HigherOrderDiff(Float64, (2, 512, 32), 2, 2; threaded),
                HigherOrderDiff(Float64, (n,), 1, 3; threaded),
            ]
        )
    end

    function allocations(L, α, β)
        x = randn(domain_type(L), size(L, 2))
        y = randn(codomain_type(L), size(L, 1))
        mul!(y, L, x, α, β)
        return @allocated mul!(y, L, x, α, β)
    end

    for A in ops, L in (A, A'), (α, β) in ((0.5, 2.0), (true, true), (2.0, false))
        @test allocations(L, α, β) == 0
    end

    # Ignoring the buffer, `add_mul!` is the accumulating `mul!`.
    D = FiniteDiff(Float64, (20, 30), 2)
    x, y0 = randn(20, 29), randn(20, 30)
    y = copy(y0)
    AbstractOperators.add_mul!(y, D', x, fill(NaN, 20, 30), 0.5, 2.0)
    @test y ≈ 0.5 * (D' * x) + 2.0 * y0
end

@testitem "Accumulating mul!: GetIndex adjoint sums repeated samples" tags = [:linearoperator, :GetIndex, :AccumulatingMul] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra

    G = GetIndex(Float64, (5,), [2, 4, 2])
    y = ones(5)
    mul!(y, G', [1.0, 10.0, 100.0], 1, 1)
    @test y == [1.0, 102.0, 1.0, 11.0, 1.0]
    @test dot(G * [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 10.0, 100.0]) ≈ dot([1.0, 2.0, 3.0, 4.0, 5.0], y .- 1)
end

@testitem "Accumulating mul!: generic method and the operator pool" tags = [:calculus, :OperatorPool, :AccumulatingMul] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, LinearAlgebra, Random
    Random.seed!(0)

    F = DFT(ComplexF64, (64, 64))
    x = randn(ComplexF64, 64, 64)
    y0 = randn(ComplexF64, 64, 64)
    expected = 0.5 * (F * x) + 2.0 * y0
    allocations(y, F, x, α, β) = (mul!(y, F, x, α, β); @allocated mul!(y, F, x, α, β))

    # With β = 0 the result is computed in `y` itself.
    @test allocations(similar(y0), F, x, 0.5, false) == 0
    @test mul!(similar(y0), F, x, 0.5, false) ≈ 0.5 * (F * x)

    # Outside a pool every call computes into a new array.
    @test mul!(copy(y0), F, x, 0.5, 2.0) ≈ expected

    # Inside a pool the array is taken from it and returned after each call.
    pool = OperatorPool()
    with_operator_pool(pool) do
        @test mul!(copy(y0), F, x, 0.5, 2.0) ≈ expected
        @test length(pool.buffers) == 1
        buf = only(pool.buffers)
        @test allocations(copy(y0), F, x, 0.5, 2.0) == 0
        @test only(pool.buffers) === buf
    end

    # Tasks sharing one pool each get an array of their own.
    xs = [randn(ComplexF64, 64, 64) for _ in 1:16]
    expected_k = [0.5 * (F * xs[k]) + 2.0 * y0 for k in 1:16]
    got = Vector{Matrix{ComplexF64}}(undef, 16)
    Threads.@threads for k in 1:16
        with_operator_pool(pool) do
            yk = copy(y0)
            mul!(yk, F, xs[k], 0.5, 2.0)
            got[k] = yk
        end
    end
    @test all(got .≈ expected_k)

    # `Scale` and `Reshape` hand `add_mul!`'s buffer on to an operator that does not accumulate.
    S = 2.0 * F
    buf = similar(y0)
    y = copy(y0)
    AbstractOperators.add_mul!(y, S, x, buf, 0.5, 2.0)
    @test y ≈ F * x + 2.0 * y0
    R = reshape(F, 64 * 64)
    y = vec(copy(y0))
    AbstractOperators.add_mul!(y, R, x, vec(buf), 1, 1)
    @test y ≈ vec(F * x) + vec(y0)
end

@testitem "Accumulating mul!: combinators add their blocks in place" tags = [:calculus, :HCAT, :VCAT, :Sum, :AccumulatingMul] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, LinearAlgebra, Random
    Random.seed!(0)
    const AO = AbstractOperators

    D1, D2 = FiniteDiff((64, 48), 1), FiniteDiff((64, 48), 2)
    F = DFT(ComplexF64, (32, 32))
    ops = Any[
        HCAT(Eye(64), DiagOp(randn(64)), MatrixOp(randn(64, 20))),
        HCAT(D1', D2'),
        VCAT(D1, D2)',
        VCAT(FiniteDiff((2, 300, 10), 2), HigherOrderDiff(Float64, (2, 300, 10), 2, 2))',
        Sum(Eye(64), DiagOp(randn(64)), MatrixOp(randn(64, 64))),
        Sum(F, Eye(ComplexF64, (32, 32))),
        Compose(DiagOp(randn(63, 48)), D1),
        DiagOp(randn(ComplexF64, 32, 32)) * F,
        2.0 * D1,
    ]
    function allocations(L, f!)
        x = AO.allocate_in_domain(L)
        x .= 1
        y = AO.allocate_in_codomain(L)
        y .= 1
        f!(y, L, x)
        return @allocated f!(y, L, x)
    end
    for A in ops, L in (A, A')
        x = AO.allocate_in_domain(L)
        x .= randn.(eltype(x))
        y0 = AO.allocate_in_codomain(L)
        y0 .= randn.(eltype(y0))
        Lx = L * x
        @test mul!(copy(y0), L, x, 0.7, -1.3) ≈ 0.7 .* Lx .+ -1.3 .* y0
        yn = copy(y0)
        fill!(yn, NaN)
        @test mul!(yn, L, x, 2.0, false) ≈ 2 .* Lx
        # Accumulating adds no work arrays to what the plain product needs; a last `Compose` stage
        # that does not accumulate takes its array from the pool.
        with_operator_pool(OperatorPool()) do
            @test allocations(L, (y, L, x) -> mul!(y, L, x, 0.7, -1.3)) <= allocations(L, mul!)
        end
    end
end

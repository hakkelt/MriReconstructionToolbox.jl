@testitem "OperatorPool: Compose reuses recycled buffers" tags = [:calculus, :Compose, :OperatorPool] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    Random.seed!(0)

    w = randn(21, 10)
    build(d) = DiagOp(d) * FiniteDiff((21, 10), 1) * DiagOp(w)
    d1, d2 = randn(20, 10), randn(20, 10)
    x = randn(21, 10)

    pool = OperatorPool()
    op1 = with_operator_pool(() -> build(d1), pool)
    bufs1 = Base.IdSet{Array}(op1.buf)
    y1 = op1 * x
    recycle!(pool, op1)
    @test length(pool.buffers) == length(bufs1)

    op2 = with_operator_pool(() -> build(d2), pool)
    @test all(b -> b in bufs1, op2.buf)
    @test isempty(pool.buffers)
    # Same results as operators built without a pool, including after the buffers held another
    # operator's intermediates.
    @test op2 * x == build(d2) * x
    @test op2' * (op2 * x) == build(d2)' * (build(d2) * x)
    @test y1 == build(d1) * x

    # A buffer of the wrong size is never handed out.
    recycle!(pool, op2)
    op3 = with_operator_pool(() -> DiagOp(randn(5)) * FiniteDiff((6,), 1), pool)
    @test !any(b -> b in bufs1, op3.buf)

    # Without an active pool, recycling is a no-op and building allocates as before.
    recycle!(op3)
    @test build(d1) * x == y1
end

@testitem "OperatorPool: shared by concurrent tasks" tags = [:calculus, :Compose, :OperatorPool] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    Random.seed!(0)

    n = 64
    ds = [randn(n - 1) for _ in 1:32]
    xs = [randn(n) for _ in 1:32]
    expected = [DiagOp(ds[k]) * FiniteDiff((n,), 1) * DiagOp(xs[k]) * xs[k] for k in 1:32]
    got = Vector{Vector{Float64}}(undef, 32)
    pool = OperatorPool()
    Threads.@threads for k in 1:32
        with_operator_pool(pool) do
            op = DiagOp(ds[k]) * FiniteDiff((n,), 1) * DiagOp(xs[k])
            got[k] = op * xs[k]
            recycle!(op)
        end
    end
    @test got == expected
end

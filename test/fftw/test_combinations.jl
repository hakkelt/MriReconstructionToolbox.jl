@testitem "Transform Combinations" tags = [:fftw, :CombinationRules] begin
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: can_be_combined, combine

    n = 8  # Power of 2 for DCT

    # Test DCT combinations
    dct_op = DCT(n)
    idct_op = IDCT(n)

    @test can_be_combined(dct_op, idct_op)
    @test can_be_combined(idct_op, dct_op)

    combined_dct = combine(dct_op, idct_op)
    @test combined_dct isa Eye

    # Test DFT combinations
    dft_op = DFT(ComplexF64, n)
    idft_op = IDFT(n)

    @test can_be_combined(dft_op, idft_op)
    @test can_be_combined(idft_op, dft_op)

    combined_dft = combine(dft_op, idft_op)
    @test combined_dft isa Eye
end

@testitem "SignAlternation pair cancels around a diagonal" tags = [:fftw, :CombinationRules] begin
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: can_be_combined, combine, get_operators, get_normal_op
    using LinearAlgebra: norm

    sz = (8, 8)
    mask = rand(Bool, sz)
    mask[1, 1] = true  # never empty
    F = DFT(ComplexF64, sz)
    S = SignAlternation(ComplexF64, sz, (1, 2))
    P = GetIndex(ComplexF64, sz, (mask,))

    # The MRT-shaped encoding operator: subsample ∘ sign-alternate ∘ DFT.
    A = P * S * F
    AHA = A' * A

    ops = AHA isa Compose ? get_operators(AHA) : (AHA,)
    # `PᴴP` folds to a single diagonal `NormalGetIndex`, and the `±` pair around it is gone
    @test !any(op -> op isa SignAlternation, ops)
    @test any(op -> op isa AbstractOperators.NormalGetIndex, ops)
    # operator-count regression: (ℱ, ↓ᵃ↓, ℱᴴ) and nothing else
    @test length(ops) == 3

    x = randn(ComplexF64, sz)
    @test norm(AHA * x - A' * (A * x)) <= 1.0e-9 * norm(x)
    @test norm(get_normal_op(A) * x - A' * (A * x)) <= 1.0e-9 * norm(x)

    # Guards. Different `dirs` do not cancel (their product alternates over the
    # symmetric difference), and a non-diagonal middle operator does not commute.
    PhP = P' * P
    S1 = SignAlternation(ComplexF64, sz, (1,))
    @test can_be_combined(S, PhP, S)
    @test !can_be_combined(S1, PhP, S)
    @test !can_be_combined(S, F, S)   # a DFT middle is not diagonal
    @test combine(S, PhP, S) === PhP
end

@testitem "Triple combination keeps operator and buffer counts consistent" tags = [:fftw, :CombinationRules, :Compose] begin
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: get_operators, get_normal_op
    using LinearAlgebra: norm

    # `get_normal_op(::Compose)` mirrors the forward buffers, so the two halves share buffer
    # *objects*; removing a triple then makes two aliased buffers adjacent and the constructor
    # allocates a replacement. That path indexed the already-shortened buffer tuple with the
    # pre-combination index and dropped one buffer too many — and a `Compose{N, M}` with
    # `M != N - 1` is not caught anywhere: `mul!` is generated over `M` and silently skips the
    # operators past `buf[M]`, so `AᴴA` quietly lost its outermost factor.
    sz = (8, 8)
    mask = rand(Bool, sz)
    mask[1, 1] = true
    D = DiagOp(rand(ComplexF64, sz))
    F = DFT(ComplexF64, sz)
    S = SignAlternation(ComplexF64, sz, (1, 2))
    P = GetIndex(ComplexF64, sz, (mask,))

    A = P * S * F * D
    for AHA in (A' * A, get_normal_op(A))
        ops = AHA isa Compose ? get_operators(AHA) : (AHA,)
        @test !any(op -> op isa SignAlternation, ops)
        if AHA isa Compose
            @test length(AHA.buf) == length(ops) - 1
            # buffers may be shared on purpose (the two halves mirror each other), but two
            # *adjacent* ones aliasing would have one stage overwrite its own input
            @test all(AHA.buf[i] !== AHA.buf[i + 1] for i in 1:(length(AHA.buf) - 1))
        end
        x = randn(ComplexF64, sz)
        @test norm(AHA * x - A' * (A * x)) <= 1.0e-9 * norm(A' * (A * x))
    end
end

@testitem "SignAlternation is pushed through a batch operator" tags = [:fftw, :CombinationRules, :batching] begin
    using FFTWOperators
    using AbstractOperators
    using AbstractOperators: is_threaded

    T = ComplexF64

    # The alternation's `dirs` are all non-batch dimensions, so it commutes with the batch
    # loop and lands on the wrapped `DiagOp`, which absorbs it: the composition collapses to
    # a single batch operator and the sign pass disappears.
    D = DiagOp(rand(T, 8, 6, 3, 2))
    B = BatchOp(D, (5,), (:_, :_, :b, :_, :_) => (:_, :_, :b, :_, :_))
    x = rand(T, size(B, 2)...)
    for dirs in ((1, 2), (1, 4), (2, 4, 5))
        S = SignAlternation(T, size(B, 1), dirs; threaded = false)
        C = S * B
        @test !(C isa Compose)
        @test C * x == S * (B * x)          # exact: every factor is ±1
        C2 = B * S
        @test !(C2 isa Compose)
        @test C2 * x == B * (S * x)
        @test C' * x == B' * (S' * x)
    end

    # Guards. An alternation over a batch dimension is not constant along the batch loop, and
    # an inner operator that cannot absorb the signs would only pay them inside the loop.
    @test SignAlternation(T, size(B, 1), (3,); threaded = false) * B isa Compose
    @test SignAlternation(T, size(B, 1), (1, 3); threaded = false) * B isa Compose
    Bv = BatchOp(Variation(8, 8), (3,), (:_, :_, :b) => (:_, :_, :b))
    @test SignAlternation(Float64, size(Bv, 2), (1,); threaded = false) isa SignAlternation
    @test Bv * SignAlternation(Float64, size(Bv, 2), (1,); threaded = false) isa Compose

    # The rewritten operator keeps the batch loop's own threading decision.
    Bt = BatchOp(DiagOp(rand(T, 16, 16, 4)), (8,); threaded = true)
    St = SignAlternation(T, size(Bt, 1), (1, 2))
    Ct = St * Bt
    @test !(Ct isa Compose)
    @test is_threaded(Ct) == is_threaded(Bt)
    xt = rand(T, size(Bt, 2)...)
    @test Ct * xt == St * (Bt * xt)
end

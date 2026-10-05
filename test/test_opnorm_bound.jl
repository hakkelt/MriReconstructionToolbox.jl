@testitem "opnorm_bound: leaves and combinators" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit, has_fast_opnorm

    # A leaf with a known exact norm reports that norm, not `Inf`.
    @test opnorm_bound(Eye(Float64, (8, 8))) == 1.0
    @test opnorm_bound(Zeros(Float64, (8, 8), Float64, (8, 8))) == 0.0
    d = randn(8, 8)
    @test opnorm_bound(DiagOp(d)) ≈ maximum(abs, d)

    # Scale is exact, not merely a bound.
    @test opnorm_bound(3.0 * DiagOp(d)) ≈ 3 * maximum(abs, d)

    # Sum obeys the triangle inequality, VCAT/HCAT the block inequalities, DCAT the maximum.
    A = MatrixOp(randn(8, 8), 8)
    B = DiagOp(randn(8, 8))
    @test opnorm_bound(A + B) ≈ opnorm_bound(A) + opnorm_bound(B)
    @test opnorm_bound(vcat(A, B)) ≈ sqrt(opnorm_bound(A)^2 + opnorm_bound(B)^2)
    @test opnorm_bound(hcat(A, B)) ≈ sqrt(opnorm_bound(A)^2 + opnorm_bound(B)^2)
    @test opnorm_bound(DCAT(A, B)) ≈ max(opnorm_bound(A), opnorm_bound(B))

    # Compose multiplies the factors. `A * B` here stays a `Compose`: a `MatrixOp` with a
    # multi-dimensional domain does not absorb a matrix-valued diagonal.
    @test A * B isa AbstractOperators.Compose
    @test opnorm_bound(A * B) ≈ opnorm_bound(A) * opnorm_bound(B)

    # An unknown leaf makes the whole expression unknown rather than wrong.
    F = FiniteDiff((8, 8))
    @test !has_fast_opnorm(F)
    @test opnorm_bound(F) == Inf
    @test opnorm_bound(F * B) == Inf
    @test opnorm_bound(vcat(F, B)) == Inf

    # An `AffineAdd` with a displacement has no operator norm: `powerit` iterates the affine
    # map, so bounding only the linear part would under-report.
    @test opnorm_bound(AffineAdd(A, randn(8, 8))) == Inf
    @test opnorm_bound(AffineAdd(A, zeros(8, 8))) ≈ opnorm_bound(A)
end

@testitem "opnorm_bound: DiagOp over BroadCast is exact" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit

    # One array replicated into `nc` copies and then weighted entry by entry. The
    # submultiplicative product overshoots by up to `sqrt(nc)` on this shape, so the pair is
    # folded before the product is taken.
    Random.seed!(0x5eed)
    nx, ny, nc = 12, 10, 8
    w = randn(ComplexF64, nx, ny, nc)
    S = DiagOp(w) * BroadCast(Eye(@view(w[:, :, 1])), size(w))

    exact = sqrt(maximum(sum(abs2, w; dims = 3)))
    @test opnorm_bound(S) ≈ exact
    @test opnorm_bound(S) ≈ powerit(S; maxit = 2000, rel_margin = 1.0e-14) rtol = 1.0e-6
    # ... and it is strictly better than what the factors alone give.
    @test opnorm_bound(S) < opnorm(DiagOp(w)) * opnorm(BroadCast(Eye(@view(w[:, :, 1])), size(w)))

    # The fold survives being embedded in a longer chain, where the pair is no longer the whole
    # composition: the bound stays finite and still dominates.
    chain = GetIndex(ComplexF64, (nx, ny, nc), (1:(nx ÷ 2), 1:ny, 1:nc)) * S
    @test isfinite(opnorm_bound(chain))
    @test opnorm_bound(chain) ≥ powerit(chain; maxit = 2000, rel_margin = 1.0e-14)

    # A `DiagOp` may hold a single number rather than an array, in which case there is nothing to
    # sum over the broadcast dimensions and the fold declines. The product rule is exact there
    # anyway, since every copy carries the same weight.
    scalar = DiagOp(ComplexF64, (nx, ny, nc), 2.0 + 0im) *
        BroadCast(Eye(ComplexF64, (nx, ny)), (nx, ny, nc))
    @test opnorm_bound(scalar) ≈ 2 * sqrt(nc)
    @test opnorm_bound(scalar) ≥ powerit(scalar; maxit = 2000, rel_margin = 1.0e-14)

    # A single copy (one coil) replicates nothing: the broadcast is the identity, and the pair
    # is just the diagonal.
    one_coil = DiagOp(w[:, :, 1:1]) * BroadCast(Eye(ComplexF64, (nx, ny)), (nx, ny, 1))
    @test opnorm(BroadCast(Eye(ComplexF64, (nx, ny)), (nx, ny, 1))) == 1
    @test opnorm_bound(one_coil) ≈ maximum(abs, w[:, :, 1])
    @test opnorm_bound(one_coil) ≥ powerit(one_coil; maxit = 2000, rel_margin = 1.0e-14)
end

@testitem "opnorm_bound: batched diagonals over BroadCast" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit

    # One array replicated into `nc` copies, each weighted by its own diagonal and then
    # subsampled: the undersampled SENSE encoding. With the weights normalized so that
    # `sum_c |w[:, :, c]|² == 1`, the norm is at most one, whereas the submultiplicative product
    # gives `maximum(abs, w) * sqrt(nc)`.
    Random.seed!(0x5eed)
    nx, ny, nc = 12, 10, 6
    w = randn(ComplexF64, nx, ny, nc)
    w ./= sqrt.(sum(abs2, w; dims = 3))
    masks = fill(rand(ny) .< 0.5, nc)
    blocks = [GetIndex(ComplexF64, (nx, ny), (:, masks[c])) * DiagOp(w[:, :, c]) for c in 1:nc]
    B = BroadCast(Eye(ComplexF64, (nx, ny)), (nx, ny, nc))
    for threaded in (false, true)
        S = BatchOp(blocks; threaded) * B
        exact = powerit(S; maxit = 3000, rel_margin = 1.0e-14)
        @test opnorm_bound(S) ≈ 1.0
        @test opnorm_bound(S) ≥ exact
        @test opnorm_bound(S) < maximum(abs, w) * sqrt(nc)
    end

    # Unnormalized weights: still a bound, and still at most the product.
    v = randn(ComplexF64, nx, ny, nc)
    S = BatchOp([GetIndex(ComplexF64, (nx, ny), (:, masks[c])) * DiagOp(v[:, :, c]) for c in 1:nc]) * B
    @test opnorm_bound(S) ≈ sqrt(maximum(sum(abs2, v; dims = 3)))
    @test opnorm_bound(S) ≥ powerit(S; maxit = 3000, rel_margin = 1.0e-14)

    # Blocks that are bare diagonals give the stacked-diagonal norm exactly.
    S = BatchOp([DiagOp(v[:, :, c]) for c in 1:nc]) * B
    @test opnorm_bound(S) ≈ powerit(S; maxit = 3000, rel_margin = 1.0e-14) rtol = 1.0e-6

    # A block that does not start with a diagonal leaves the product rule in charge.
    S = BatchOp([DiagOp(v[:, :, c]) * MatrixOp(randn(ComplexF64, nx, nx), ny) for c in 1:nc]) * B
    @test opnorm_bound(S) ≈ opnorm_bound(S.A[2]) * opnorm_bound(S.A[1])
    @test opnorm_bound(S) ≥ powerit(S; maxit = 3000, rel_margin = 1.0e-14)
end

@testitem "opnorm_bound: a broadcast scales the norm it wraps" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit

    # Replicating a vector `c` times multiplies its 2-norm by exactly `sqrt(c)`. Forwarding
    # `opnorm` straight to the inner operator, as this once did, under-reported by that factor.
    Random.seed!(7)
    A = MatrixOp(randn(6, 6))          # ℝ^6 -> ℝ^6, so it can be broadcast over a second axis
    R = BroadCast(A, (6, 4))
    @test opnorm(R) ≈ 2 * opnorm(A)
    @test opnorm_bound(R) ≈ 2 * opnorm_bound(A)
    @test opnorm(R) ≈ powerit(R; maxit = 2000, rel_margin = 1.0e-14) rtol = 1.0e-6
end

@testitem "estimate_opnorm: sides and margin" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: estimate_opnorm, opnorm_bound, powerit, has_fast_opnorm

    Random.seed!(11)
    nx, ny, nc = 12, 10, 6
    w = randn(ComplexF64, nx, ny, nc)
    S = DiagOp(w) * BroadCast(Eye(@view(w[:, :, 1])), size(w))
    # A subsampling in front keeps the norm strictly below the bound, so `:upper` and
    # `:accurate` are actually distinguishable.
    A = GetIndex(ComplexF64, (nx, ny, nc), (1:(nx ÷ 2), 1:ny, 1:nc)) * S

    exact = powerit(A; maxit = 3000, rel_margin = 1.0e-14)
    U = opnorm_bound(A)
    @test isfinite(U)

    # `:upper` never falls below the truth; `:accurate` never rises above it.
    @test estimate_opnorm(A) >= exact
    @test estimate_opnorm(A; side = :accurate) <= exact * (1 + 1.0e-8)
    @test estimate_opnorm(A) == U

    @test_throws ArgumentError estimate_opnorm(A; side = :sideways)

    # A tight certificate costs no iterations: the answer is the same for any `maxit`.
    @test estimate_opnorm(A; maxit = 1) == estimate_opnorm(A; maxit = 100)

    # With no certificate available, the result is the residual heuristic, which still errs
    # upwards relative to the power iteration's own lower bound.
    F = FiniteDiff((16, 16))
    @test opnorm_bound(F) == Inf
    @test estimate_opnorm(F) >= powerit(F; maxit = 500, rel_margin = 1.0e-12) * (1 - 1.0e-8)
    @test estimate_opnorm(F; side = :accurate) <= estimate_opnorm(F)

    # `has_fast_opnorm` still short-circuits to the exact value, whichever side is asked for.
    D = DiagOp(randn(8, 8))
    @test has_fast_opnorm(D)
    @test estimate_opnorm(D) == opnorm(D)
    @test estimate_opnorm(D; side = :accurate) == opnorm(D)

    # Every operator goes through the one generic method, so the keywords reach every shape.
    # Per-type `estimate_opnorm` methods used to shadow it and silently rejected them.
    G = FiniteDiff((16, 16))
    for L in (2.0 * G, G', DCAT(G, G), BatchOp(G, 3; threaded = false))
        @test estimate_opnorm(L; rel_margin = 0.05, side = :accurate, maxit = 5) > 0
    end

    # `:upper` is a promise, so the returned value must dominate the iterate it was checked
    # against even in single precision, where narrowing the `Float64` bound rounds to nearest.
    w32 = randn(ComplexF32, nx, ny, nc)
    S32 = DiagOp(w32) * BroadCast(Eye(@view(w32[:, :, 1])), size(w32))
    @test estimate_opnorm(S32) isa Float32
    @test estimate_opnorm(S32) >= powerit(S32; maxit = 2000, rel_margin = 1.0f-7)
    # Narrowing the `Float64` bound rounds up, never to nearest.
    @test estimate_opnorm(S32) >= opnorm_bound(S32)
end

@testitem "estimate_opnorm: Lanczos and the power method" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: estimate_opnorm, opnorm_bound, powerit, _powerit, _powerit_rng

    # Operators with a known spectrum: `Q diag(σ) Qᴴ` for a fixed random unitary `Q`.
    n = 300
    u(rng, a, b, k) = a .+ (b - a) .* rand(rng, k)
    spectra(rng) = [
        "separated" => [1.0; u(rng, 0, 0.7, n - 1)],
        "slow gap" => [1.0; sqrt(0.99); u(rng, 0, 0.9, n - 2)],
        "flat cluster" => [1.0; u(rng, 0.99, 1, 29); u(rng, 0, 0.9, n - 30)],
        "doubled top" => [1.0; 1.0; u(rng, 0, 0.9, n - 2)],
        "geometric" => 0.98 .^ (0:(n - 1)),
        "uniform" => [1.0; u(rng, 0, 1, n - 1)],
        "rank deficient" => [1.0; u(rng, 0, 1, n ÷ 2 - 1); zeros(n - n ÷ 2)],
    ]
    run(L; kw...) = _powerit(L; maxit = 100, rng = _powerit_rng(), upper = Inf, kw...)
    for T in (Float64, Float32, ComplexF64)
        rng = Xoshiro(3)
        Q = Matrix(qr(randn(rng, T, n, n)).Q)
        for (name, σ) in spectra(rng)
            L = MatrixOp(Q * Diagonal(T.(σ)) * Q')
            # Both lower bounds stay at or below the norm, to rounding.
            for method in (:lanczos, :power)
                @test powerit(L; maxit = 500, rel_margin = 1.0e-10, method) <= 1 + 10 * 500 * eps(real(T))
            end
            # Lanczos never needs more steps than the power method to the same margin, and on a
            # slow-gapped, clustered or flat top it is within 1% in 25 steps.
            λ, θ, r = run(L; maxit = 25, rel_margin = 0.01, method = :lanczos)
            name in ("slow gap", "flat cluster", "doubled top", "uniform", "rank deficient") &&
                @test r / (2θ) <= 0.01
            @test run(L; rel_margin = 1.0e-3, method = :lanczos)[1] >= run(L; rel_margin = 1.0e-3, method = :power)[1] * (1 - 1.0e-6)
            # Forced Lanczos measures the residual it returns where the residual enters the value.
            λ, θ, r = run(L; rel_margin = 1.0e-3, method = :lanczos, residual_in_value = true)
            @test sqrt(θ + r) >= 1 - 10 * 100 * eps(real(T))
        end
    end

    # The residual is the measured one on a run past convergence, where the Lanczos estimate is not.
    σ = [1.0; u(Xoshiro(5), 0, 0.7, n - 1)]
    L = MatrixOp(Matrix(Diagonal(Float32.(σ))))
    measured = run(L; maxit = 200, rel_margin = 0.0, method = :lanczos, residual_in_value = true)[3]
    @test measured >= run(L; maxit = 200, rel_margin = 0.0, method = :lanczos)[3]

    # `:auto` runs Lanczos unless the residual enters the value, where it runs the power method.
    F = FiniteDiff((16, 16))
    @test opnorm_bound(F) == Inf
    @test estimate_opnorm(F) == estimate_opnorm(F; method = :power)
    @test estimate_opnorm(F; side = :accurate) == estimate_opnorm(F; side = :accurate, method = :lanczos)
    @test powerit(F) == powerit(F; method = :lanczos)
    @test estimate_opnorm(F; method = :lanczos) >= powerit(F; maxit = 500, rel_margin = 1.0e-12)
    @test_throws ArgumentError estimate_opnorm(F; method = :arnoldi)
    @test_throws ArgumentError powerit(F; method = :arnoldi)

    # A finite certificate is returned as it is whichever iteration certifies it.
    w = randn(Xoshiro(2), ComplexF64, 12, 10, 6)
    S = DiagOp(w) * BroadCast(Eye(@view(w[:, :, 1])), size(w))
    A = GetIndex(ComplexF64, size(w), (1:6, 1:10, 1:6)) * S
    @test estimate_opnorm(A; method = :lanczos) == estimate_opnorm(A; method = :power) == opnorm_bound(A)

    # Repeated calls agree, and a null operator is zero for either iteration.
    @test powerit(F) == powerit(F)
    Z = Zeros(Float64, (5,), Float64, (5,))
    @test powerit(Z; method = :lanczos) == 0
    @test powerit(Z; method = :power) == 0
end

@testitem "opnorm_bound: randomized certificate" tags = [:calculus, :OpnormBound] begin
    using LinearAlgebra, Random
    using AbstractOperators
    using AbstractOperators: opnorm_bound, powerit

    # The failure mode of a structural bound is a silently dropped factor, which a hand-written
    # case does not catch and a randomized nesting does. Every finite bound must dominate the
    # converged power iteration, which approaches the norm from below.
    # Every draw comes from `rng`, including the entries of the leaves: a `randn()` here would
    # read the global generator, and then the number of cases with a finite bound — and so the
    # number of assertions this item runs — would differ from one session to the next.
    leaves = rng -> [
        DiagOp(randn(rng, 8, 8)),
        Eye(Float64, (8, 8)),
        MatrixOp(randn(rng, 8, 8), 8),
        GetIndex(Float64, (8, 8), (1:4, 1:8)),
        Zeros(Float64, (8, 8), Float64, (8, 8)),
    ]

    function random_op(depth, rng)
        depth <= 0 && return rand(rng, leaves(rng))
        a = random_op(depth - 1, rng)
        k = rand(rng, 1:6)
        k == 1 && return rand(rng) * a
        k == 2 && return a' * a
        k == 3 && return AffineAdd(a, randn(rng, size(a, 1)...))
        b = random_op(depth - 1, rng)
        k == 4 && return size(a) == size(b) ? a + b : a
        k == 5 && return size(a, 2) == size(b, 2) ? vcat(a, b) : a
        return size(a, 1) == size(b, 1) ? hcat(a, b) : a
    end

    rng = Xoshiro(20260921)
    # `Ref`, not a plain `Int`: assignment inside the loop body would bind a new local under
    # soft scope and leave the outer count at zero.
    checked = Ref(0)
    for _ in 1:150
        L = try
            random_op(rand(rng, 1:3), rng)
        catch
            continue
        end
        U = opnorm_bound(L)
        isfinite(U) || continue
        lower = try
            powerit(L; maxit = 1000, rel_margin = 1.0e-13)
        catch
            continue
        end
        checked[] += 1
        @test U >= lower * (1 - 1.0e-8)
    end
    @test checked[] > 50  # the generator actually produced bounded operators
end

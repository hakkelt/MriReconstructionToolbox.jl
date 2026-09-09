@testitem "Hankel" tags = [:linearoperator, :Hankel] setup = [TestUtils] begin
    using Random, AbstractOperators, LinearAlgebra
    Random.seed!(0)

    # ---- doctest-style forward on a 1D signal ----
    H = Hankel(Float64, (5,), (3,))
    @test size(H) == ((3, 3), (5,))
    @test H * collect(1.0:5.0) == [1.0 2.0 3.0; 2.0 3.0 4.0; 3.0 4.0 5.0]
    @test domain_type(H) == Float64
    @test codomain_type(H) == Float64
    @test is_thread_safe(H) == true

    # ---- test_op: forward/adjoint/in-place/adjoint-invariance ----
    # 1D, real
    test_op(Hankel(Float64, (9,), (4,)), randn(9), randn(6, 4), verb)
    # 2D, complex, no channels
    test_op(Hankel(ComplexF64, (8, 7), (3, 2)), randn(ComplexF64, 8, 7), randn(ComplexF64, 36, 6), verb)
    # 2D, complex, multi-channel (SAKE layout)
    Hc = Hankel(ComplexF64, (8, 6), (3, 2); nchannels = 4)
    @test size(Hc) == ((30, 24), (8, 6, 4))
    test_op(Hc, randn(ComplexF64, 8, 6, 4), randn(ComplexF64, 30, 24), verb)
    # 3D, complex
    test_op(Hankel(ComplexF64, (6, 5, 4), (2, 2, 2)), randn(ComplexF64, 6, 5, 4), randn(ComplexF64, 60, 8), verb)

    # ---- explicit dot-product adjoint test, ⟨H u, v⟩ == ⟨u, Hᴴ v⟩ ----
    # `test_op` already asserts adjoint invariance; this states it directly on the
    # multi-channel complex layout, which is the one the SAKE/LORAKS prox relies on.
    u = randn(ComplexF64, size(Hc, 2)...)
    v = randn(ComplexF64, size(Hc, 1)...)
    @test dot(Hc * u, v) ≈ dot(u, Hc' * v)

    # ---- normal operator equals the multiplicity diagonal ----
    Nop = AbstractOperators.get_normal_op(Hc)
    d = AbstractOperators.diag_AcA(Hc)
    @test Hc' * (Hc * u) ≈ d .* u
    @test Nop * u ≈ d .* u
    @test is_AcA_diagonal(Hc) == true
    # every element appears in >= 1 window
    @test all(real.(d) .>= 1)
    # interior elements of the (8,6) grid with window (3,2) appear in 3*2 windows
    @test d[4, 3, 1] == 6

    # `diag_AcA` must be exactly the per-sample multiplicity: the number of Hankel entries
    # each k-space sample is copied into. Counted here from the dense operator matrix, which
    # has a single 1 per (window, sample) incidence, so the column sums are the counts.
    Hs = Hankel(ComplexF64, (5, 4), (2, 3); nchannels = 2)
    nin = prod(size(Hs, 2))
    dense = reduce(hcat, [vec(Hs * reshape(ComplexF64.(1:nin .== i), size(Hs, 2)...)) for i in 1:nin])
    @test vec(real.(AbstractOperators.diag_AcA(Hs))) == vec(sum(abs2, dense; dims = 1))
    # and the multiplicity weighting is what makes the weighted adjoint a left inverse of H
    w = randn(ComplexF64, size(Hs, 2)...)
    @test (Hs' * (Hs * w)) ./ AbstractOperators.diag_AcA(Hs) ≈ w

    # ---- opnorm matches the dense operator ----
    Hd = Hankel(ComplexF64, (7,), (3,))
    M = reduce(hcat, [vec(Hd * ComplexF64.(1:7 .== i)) for i in 1:7])
    @test opnorm(Hd) ≈ opnorm(M)
    @test AbstractOperators.has_fast_opnorm(Hd) == true

    # ---- exact rank of a synthetic low-rank Hankel ----
    # a single complex exponential lifts to a rank-1 Hankel matrix
    n = 32
    z1 = cispi(0.17); z2 = cispi(-0.4)
    s = ComplexF64[z1^k + z2^k for k in 0:(n - 1)]
    Hr = Hankel(ComplexF64, (n,), (8,))
    @test rank(Matrix(Hr * s); atol = 1.0e-8) == 2

    # ---- copy semantics ----
    Hcopy = AbstractOperators.copy_operator(Hc)
    @test Hcopy * u ≈ Hc * u
    @test Hcopy.mult === Hc.mult  # read-only, shared

    # ---- properties / errors ----
    @test is_linear(Hc) == true
    @test is_full_column_rank(Hc) == true
    @test_throws ErrorException Hankel(Float64, (5,), (6,))          # window > grid
    @test_throws ErrorException Hankel(Float64, (5,), (0,))          # non-positive window
    @test_throws ErrorException Hankel(Float64, (5,), (3,); structure = :s)  # unimplemented
    @test_throws ErrorException Hankel(randn(5, 5), (3,))            # ndims mismatch

    io = IOBuffer()
    show(io, H)
    @test occursin("𝓗", String(take!(io)))
end

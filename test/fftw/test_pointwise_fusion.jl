@testitem "DFT normalization joins a pointwise run" tags = [:fftw, :DFT, :pointwise] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, FFTW, LinearAlgebra, Random
    const AO = AbstractOperators
    Random.seed!(0)

    function sequential(C, x)
        y = AO.allocate_in_codomain(C)
        AO._pw_sequential!(y, C.A, C.buf, x)
        return y
    end

    n = (8, 6, 3)
    norms = (FFTWOperators.UNNORMALIZED, FFTWOperators.ORTHO, FFTWOperators.FORWARD, FFTWOperators.BACKWARD)
    for T in (ComplexF32, ComplexF64), normalization in norms
        F = DFT(T, n, (1, 2); normalization)
        D = DiagOp(randn(T, n))
        G = AO.get_normal_op(GetIndex(T, n, (:, rand(6, 3) .> 0.5)))
        E = BroadCast(Eye(T, n[1:2]), n)
        for C in (D * F, G * F, D * F', E' * D' * F' * G * F * D * E)
            @test C isa Compose
            @test AO._pw_run_length(C.A) > 1 || AO._pw_run_length(C.A[2:end]) > 1
            x = randn(T, size(C, 2))
            @test C * x == sequential(C, x)
        end
    end

    # A real-input transform keeps its own conversion in the core.
    F = DFT(Float64, (8, 6))
    D = DiagOp(randn(ComplexF64, 8, 6))
    C = D * F
    x = randn(8, 6)
    @test C * x == sequential(C, x)
    r = randn(ComplexF64, 8, 6)
    @test C' * r ≈ F' * (D' * r)
end

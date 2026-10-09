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

@testitem "SignAlternation joins a pointwise run" tags = [:fftw, :Shift, :pointwise] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, LinearAlgebra, Random
    const AO = AbstractOperators
    Random.seed!(0)

    function sequential(C, x)
        y = AO.allocate_in_codomain(C)
        AO._pw_sequential!(y, C.A, C.buf, x)
        return y
    end

    # A sign alternation next to a `DiagOp` merges into it, so the runs pair it with a mask, a
    # broadcast and a transform.
    for T in (Float64, ComplexF32), n in ((8, 6, 4), (7, 5, 3), (6, 5, 4)), threaded in (false, true)
        Tc = complex(T)
        G = AO.get_normal_op(GetIndex(T, n, (:, rand(n[2], n[3]) .> 0.5)))
        Gc = AO.get_normal_op(GetIndex(Tc, n, (:, rand(n[2], n[3]) .> 0.5)))
        E = BroadCast(Eye(T, n[1:2]), n; threaded)
        F = DFT(Tc, n; threaded)
        for dirs in ((1,), (2,), (3,), (1, 2), (2, 3), (1, 3), (1, 2, 3))
            S = SignAlternation(T, n, dirs; threaded)
            @test AO._pw_kind(S) isa AO.PwMapKind
            Sc = SignAlternation(Tc, n, dirs; threaded)
            for C in (G * S, S * G, S * E, E' * S, Sc * F, Gc * Sc * F)
                @test C isa Compose
                @test AO._pw_run_length(C.A) == length(C.A)
                x = randn(domain_type(C), size(C, 2))
                @test C * x == sequential(C, x)
            end
        end
    end

    # Arrays long enough that the kernel cuts them into several segments.
    for n in ((3000,), (4099, 3), (2, 3, 2050), (5, 1000, 3))
        G = AO.get_normal_op(GetIndex(Float64, n, (rand(n...) .> 0.5,)))
        for m in 1:(2^length(n) - 1)
            dirs = Tuple(d for d in 1:length(n) if isodd(m >> (d - 1)))
            S = SignAlternation(Float64, n, dirs; threaded = false)
            x = randn(n)
            @test (G * S) * x == G * (S * x)
        end
    end
end

@testitem "SignAlternation joins a pointwise run on device arrays" tags = [:gpu, :fftw, :pointwise] setup = [TestUtils, GpuEnvSetup] begin
    using GPUEnv, Random, AbstractOperators, FFTWOperators
    const AO = AbstractOperators

    for backend in gpu_backends()
        Random.seed!(0)
        T = ComplexF32
        dev(a) = to_gpu(backend, a)
        for n in ((8, 6, 4), (7, 5, 3)), dirs in ((1,), (2, 3), (1, 2, 3))
            x = randn(T, n[1:2])
            S, hS = SignAlternation(T, n, dirs; array_type = typeof(dev(zeros(T, n)))), SignAlternation(T, n, dirs)
            @test AO._pw_kind(S) isa AO.PwMapKind
            E, hE = BroadCast(Eye(dev(zeros(T, n[1:2]))), n), BroadCast(Eye(T, n[1:2]), n)
            C, hC = S * E, hS * hE
            @test C isa Compose
            @test AO._pw_run_length(C.A) == 2
            @test Array(C * dev(x)) == hC * x
            @test Array(C' * dev(hC * x)) ≈ hC' * (hC * x)
        end
    end
end

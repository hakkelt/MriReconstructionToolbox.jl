@testitem "WaveletOp" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, WaveletOperators

    ########## WaveletOp ############
    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n = 8
    op = WaveletOp(ComplexF64, wavelet(WT.db4), (n,))
    x1 = randn(ComplexF64, n)
    y1 = test_op(op, x1, randn(ComplexF64, n), verb)
    y2 = dwt(x1, wavelet(WT.db4))

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "WaveletOp (GPU)" tags = [:gpu, :wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n = 8
        op = WaveletOp(Float64, wavelet(WT.db4), (n,); array_type = gpu_wrapper(backend, Float64, n))
        x1 = gpu_randn(backend, n)
        y1 = test_op(op, x1, gpu_randn(backend, n), false)
        @test domain_array_type(op) <: backend.array_type
        @test codomain_array_type(op) <: backend.array_type

        y2 = dwt(collect(x1), wavelet(WT.db4))
        @test norm(collect(y1) .- y2) <= 1.0e-12
    end
end

@testitem "WaveletOp: normal operator is the identity for orthogonal families" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators
    Random.seed!(1)

    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))

    @test AbstractOperators.has_optimized_normalop(op)
    Wop = AbstractOperators.get_normal_op(op)
    @test Wop isa Eye

    x = randn(n)
    @test norm(op' * (op * x) .- x) <= 1.0e-12
    @test norm(Wop * x .- x) <= 1.0e-12

    # A biorthogonal family (lifting-scheme CDF) is not self-adjoint: the identity trait must
    # not fire for it.
    bop = WaveletOp(Float64, wavelet(WT.cdf97, WT.Lifting), (n,))
    @test !AbstractOperators.has_optimized_normalop(bop)
    @test !AbstractOperators.is_AcA_diagonal(bop)
    @test !AbstractOperators.is_AAc_diagonal(bop)
    @test_throws ArgumentError AbstractOperators.get_normal_op(bop)
    @test_throws ArgumentError AbstractOperators.diag_AcA(bop)
end

@testitem "WaveletOp: copy_operator" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, LinearAlgebra, Random, AbstractOperators, WaveletOperators
    Random.seed!(6)

    n = 8
    op = WaveletOp(Float64, wavelet(WT.db4), (n,))

    # No constraint: reproduces the operator exactly.
    op2 = copy_operator(op)
    @test op2 isa WaveletOp
    @test typeof(op2) === typeof(op)
    x = randn(n)
    y1, y2 = zeros(n), zeros(n)
    mul!(y1, op, x)
    mul!(y2, op2, x)
    @test y1 ≈ y2

    # `threaded` is accepted (vacuously, since WaveletOp never has a threaded path).
    op3 = copy_operator(op; threaded = false)
    @test domain_array_type(op3) <: Array{Float64}

    # `storage_type` rebuilds the storage-tracking type parameter.
    op4 = copy_operator(op; storage_type = Array{Float64})
    @test op4 isa WaveletOp
    @test domain_array_type(op4) <: Array{Float64}
    @test codomain_array_type(op4) <: Array{Float64}
end

@testitem "WaveletOp constructor errors" tags = [:wavelet, :WaveletOp] setup = [TestUtils] begin
    using Wavelets, WaveletOperators
    wt = wavelet(WT.db4)

    # 1D: odd dimension
    @test_throws ArgumentError WaveletOp(Float64, wt, 5)
    # 1D: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, 8, 100)

    # ND: odd dimension in tuple
    @test_throws ArgumentError WaveletOp(Float64, wt, (5, 8))
    # ND: too many levels
    @test_throws ArgumentError WaveletOp(Float64, wt, (8, 8), 100)
end

@testmodule HigherOrderDiffReference begin
    export forward_reference, adjoint_reference

    # `order` repeated `diff`s, in BigFloat so the reference carries no rounding of its own.
    function forward_reference(x, d, order)
        r = big.(x)
        for _ in 1:order
            r = diff(r; dims = d)
        end
        return r
    end

    # The adjoint of one forward difference is the difference of its input padded by a zero
    # slab on either side; the adjoint of the order-`order` one repeats it.
    function adjoint_reference(y, d, order)
        r = big.(y)
        for _ in 1:order
            z = zeros(eltype(r), ntuple(i -> i == d ? 1 : size(r, i), ndims(r)))
            r = cat(z, r; dims = d) - cat(r, z; dims = d)
        end
        return r
    end
end

@testitem "HigherOrderDiff: against repeated differences" tags = [:linearoperator, :HigherOrderDiff] setup = [
    TestUtils, HigherOrderDiffReference,
] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    # The single-pass stencil rounds differently from the repeated differences: allow `4K`
    # units of the input's largest entry.
    shapes = [(7,), (2, 40, 6), (3, 17, 5), (6, 5, 4, 3), (9, 1), (3, 4)]
    for T in (Float32, Float64, ComplexF64), dims in shapes, d in eachindex(dims), order in 1:4, threaded in (false, true)
        dims[d] <= order && continue
        op = HigherOrderDiff(T, dims, d, order; threaded)
        x = randn(T, dims)
        y = op * x
        @test size(y) == size(op, 1)
        @test maximum(abs, y - forward_reference(x, d, order)) <= 4order * eps(real(T)) * maximum(abs, x)
        g = randn(T, size(op, 1))
        @test maximum(abs, op' * g - adjoint_reference(g, d, order)) <= 4order * eps(real(T)) * maximum(abs, g)
        @test dot(op * x, g) ≈ dot(x, op' * g)
        if order == 1
            # Order 1 is FiniteDiff's own expression, bit for bit.
            F = FiniteDiff(T, dims, d; threaded)
            @test y == F * x
            @test op' * g == F' * g
        end
    end

    op = HigherOrderDiff(Float64, (10,), 1, 2)
    test_op(op, randn(10), randn(8), verb)
    @test op * collect(1.0:10.0) .^ 2 == fill(2.0, 8)

    # The threaded and serial kernels agree exactly, on a size where the policy threads.
    for dims in [(3, 512, 64), (64, 64, 32)], d in eachindex(dims)
        x = randn(dims)
        serial = HigherOrderDiff(Float64, dims, d, 2; threaded = false)
        threaded = HigherOrderDiff(Float64, dims, d, 2; threaded = true)
        @test threaded * x == serial * x
        g = randn(size(serial, 1))
        @test threaded' * g == serial' * g
    end
end

@testitem "HigherOrderDiff: constructors and properties" tags = [:linearoperator, :HigherOrderDiff] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra

    op = HigherOrderDiff(Float64, (5, 6, 2), 2, 3)
    @test size(op) == ((5, 3, 2), (5, 6, 2))
    @test domain_type(op) == codomain_type(op) == Float64
    @test is_linear(op)
    @test is_full_row_rank(op)
    @test !is_full_column_rank(op)
    @test !is_AcA_diagonal(op) && !is_AAc_diagonal(op)
    @test HigherOrderDiff(zeros(ComplexF32, 4, 5), 1, 2) isa HigherOrderDiff{2, 1, 2, ComplexF32}
    @test HigherOrderDiff((4, 5), 2, 2) == HigherOrderDiff(Float64, (4, 5), Val(2), Val(2))
    @test_throws ArgumentError HigherOrderDiff(Float64, (3,), 1, 3)
    @test_throws ArgumentError HigherOrderDiff(Float64, (5,), 1, 0)
    @test_throws ErrorException HigherOrderDiff(Float64, (5, 5), 3, 2)

    for (o, name) in (
            (HigherOrderDiff((8,), 1, 2), "δx²"), (HigherOrderDiff((8, 8), 2, 3), "δy³"),
            (HigherOrderDiff((4, 4, 13), 3, 12), "δz¹²"), (HigherOrderDiff((3, 3, 3, 3), 4, 2), "δx4²"),
        )
        @test occursin(name, sprint(show, o))
    end

    c = copy_operator(op; threaded = false)
    @test c isa typeof(HigherOrderDiff(Float64, (5, 6, 2), 2, 3; threaded = false))
end

@testitem "HigherOrderDiff: chained differences combine" tags = [:linearoperator, :HigherOrderDiff, :combination] setup = [
    TestUtils,
] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    F1, F2, F3 = FiniteDiff((12, 5)), FiniteDiff((11, 5)), FiniteDiff((10, 5))
    @test AbstractOperators.can_be_combined(F2, F1)
    @test F2 * F1 isa HigherOrderDiff{2, 1, 2}
    @test F3 * F2 * F1 isa HigherOrderDiff{2, 1, 3}
    @test F3 * (F2 * F1) isa HigherOrderDiff{2, 1, 3}
    @test (F3 * F2) * F1 * HigherOrderDiff(Float64, (13, 5), 1, 1) * FiniteDiff((14, 5)) isa HigherOrderDiff{2, 1, 5}
    @test FiniteDiff((11, 5), 2) * FiniteDiff((11, 6), 2) isa HigherOrderDiff{2, 2, 2}

    # Adjoints: `F1' * F2'` is `(F2 * F1)'`.
    @test F1' * F2' isa AdjointOperator{<:HigherOrderDiff{2, 1, 2}}
    @test F1' * F2' * F3' isa AdjointOperator{<:HigherOrderDiff{2, 1, 3}}

    x, y = randn(12, 5), randn(9, 5)
    @test (F3 * F2 * F1) * x ≈ F3 * (F2 * (F1 * x))
    @test (F1' * F2' * F3') * y ≈ F1' * (F2' * (F3' * y))

    # Different directions, element types or a normal operator do not combine.
    @test !AbstractOperators.can_be_combined(FiniteDiff((11, 5), 2), FiniteDiff((12, 5), 1))
    @test !(FiniteDiff((11, 5), 2) * FiniteDiff((12, 5), 1) isa HigherOrderDiff)
    @test !AbstractOperators.can_be_combined(FiniteDiff(Float32, (11,)), FiniteDiff(Float64, (12,)))
    @test !(F1' * F1 isa HigherOrderDiff)

    # The combined operator is threaded when either part was, subject to the policy.
    n = 1 << 16
    serial, threaded = FiniteDiff(Float64, (n,); threaded = false), FiniteDiff(Float64, (n - 1,); threaded = true)
    @test is_threaded(threaded * serial) == is_threaded(threaded)
    @test !is_threaded(FiniteDiff(Float64, (n - 1,); threaded = false) * serial)
end

@testitem "HigherOrderDiff (GPU)" tags = [:gpu, :linearoperator, :HigherOrderDiff] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n, m = 10, 5
        op = HigherOrderDiff(Float64, (n, m), 1, 2; array_type = gpu_wrapper(backend, Float64, n, m))
        test_op(op, gpu_randn(backend, n, m), gpu_randn(backend, n - 2, m), false)

        op2 = HigherOrderDiff(Float64, (n, m), 2, 3; array_type = gpu_wrapper(backend, Float64, n, m))
        test_op(op2, gpu_randn(backend, n, m), gpu_randn(backend, n, m - 3), false)

        F = FiniteDiff(Float64, (n,); array_type = gpu_wrapper(backend, Float64, n))
        F2 = FiniteDiff(Float64, (n - 1,); array_type = gpu_wrapper(backend, Float64, n - 1))
        @test F2 * F isa HigherOrderDiff
    end
end

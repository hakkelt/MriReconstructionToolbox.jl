@testitem "SymmetrizedVariation: basic mul" tags = [:linearoperator, :SymmetrizedVariation] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing SymmetrizedVariation: basic mul --- ")

    for threaded in (false, true)
        n, m = 10, 5
        verb && println("  - threaded = $threaded")
        op = SymmetrizedVariation(Float64, (n, m); threaded)
        test_op(op, randn(n * m, 2), randn(n * m, 3), verb)

        @test size(op) == ((n * m, 3), (n * m, 2))
        @test domain_type(op) == Float64
        @test codomain_type(op) == Float64
        @test domain_array_type(op) == Array{Float64}
        @test codomain_array_type(op) == Array{Float64}
        @test is_thread_safe(op) == true

        # A constant field has zero symmetrized gradient.
        @test op * ones(n * m, 2) ≈ zeros(n * m, 3)

        # The entries are exactly the ones the definition prescribes, expressed through `Variation`'s own
        # directional derivatives, so the two discretizations cannot drift apart unnoticed.
        Ʋ = Variation(Float64, (n, m); threaded = false)
        derivative(v, d) = (Ʋ * reshape(v, n, m))[:, d]
        w1, w2 = randn(n * m), randn(n * m)
        result = op * hcat(w1, w2)
        @test result[:, 1] ≈ derivative(w1, 1)
        @test result[:, 2] ≈ derivative(w2, 2)
        @test result[:, 3] ≈ (sqrt(2) / 2) .* (derivative(w2, 1) .+ derivative(w1, 2))
    end
end

@testitem "SymmetrizedVariation: 3D, constructors and properties" tags = [:linearoperator, :SymmetrizedVariation] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing SymmetrizedVariation: 3D, constructors and properties --- ")

    n, m, l = 4, 3, 3
    M = n * m * l
    for threaded in (false, true)
        op = SymmetrizedVariation(Float64, (n, m, l); threaded)
        # 3 diagonal plus 3 off-diagonal entries of a symmetric 3x3 tensor
        @test size(op) == ((M, 6), (M, 3))
        test_op(op, randn(M, 3), randn(M, 6), verb)
        @test op * ones(M, 3) ≈ zeros(M, 6)
    end

    @testset "constructors agree" begin
        expected = size(SymmetrizedVariation(Float64, (n, m)))
        @test size(SymmetrizedVariation(Float64, n, m)) == expected
        @test size(SymmetrizedVariation((n, m))) == expected
        @test size(SymmetrizedVariation(n, m)) == expected
        @test size(SymmetrizedVariation(zeros(n, m))) == expected
        @test domain_type(SymmetrizedVariation(zeros(ComplexF64, n, m))) == ComplexF64
    end

    @testset "rejected inputs" begin
        # A one-dimensional grid has no symmetrized gradient to speak of, and a singleton dimension has no
        # finite difference -- the same restrictions `Variation` imposes.
        @test_throws ErrorException SymmetrizedVariation(Float64, (4,))
        @test_throws ArgumentError SymmetrizedVariation(Float64, (4, 1))
    end

    @testset "the off-diagonal weight makes the row norm a Frobenius norm" begin
        # For a field whose symmetrized gradient is a known tensor, the Euclidean norm of a stored row must
        # equal the Frobenius norm of the full symmetric matrix, including both copies of each off-diagonal.
        op = SymmetrizedVariation(Float64, (n, m); threaded = false)
        w = randn(n * m, 2)
        result = op * w
        Ʋ = Variation(Float64, (n, m); threaded = false)
        derivative(v, d) = (Ʋ * reshape(v, n, m))[:, d]
        e11 = derivative(w[:, 1], 1)
        e22 = derivative(w[:, 2], 2)
        e12 = (derivative(w[:, 2], 1) .+ derivative(w[:, 1], 2)) ./ 2
        frobenius = sqrt.(e11 .^ 2 .+ e22 .^ 2 .+ 2 .* e12 .^ 2)
        @test vec(sqrt.(sum(abs2, result; dims = 2))) ≈ frobenius
    end

    @testset "a symmetric gradient field is reproduced exactly" begin
        # w = ∇φ for a quadratic φ has ℰw = the (constant) Hessian of φ. Away from the boundaries, where the
        # mirrored differences apply, the operator must return exactly that Hessian.
        nx, ny = 8, 8
        φ = [1.5 * i^2 + 0.5 * j^2 + 2.0 * i * j for i in 1:nx, j in 1:ny]
        Ʋ = Variation(Float64, (nx, ny); threaded = false)
        w = Ʋ * φ
        result = reshape(SymmetrizedVariation(Float64, (nx, ny); threaded = false) * w, nx, ny, 3)
        interior = (3:(nx - 1), 3:(ny - 1))
        @test all(≈(3.0), result[interior..., 1])          # ∂ₓₓφ
        @test all(≈(1.0), result[interior..., 2])          # ∂ᵧᵧφ
        @test all(≈(sqrt(2) * 2.0), result[interior..., 3]) # √2 ⋅ ∂ₓᵧφ
    end
end

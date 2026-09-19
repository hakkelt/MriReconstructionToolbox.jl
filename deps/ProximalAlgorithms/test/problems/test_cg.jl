using Test
using LinearAlgebra
using ProximalAlgorithms
using Random

@testset "CG" begin
    @testset "Real inputs" begin
        n = 100
        A = rand(n,n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        
        # Test basic CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
        
        # Test with preconditioner
        P = Diagonal(diag(A))  # Jacobi preconditioner
        pcg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, P=P)
        x, it = pcg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Complex inputs" begin
        n = 100
        A = rand(ComplexF64, n,n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Custom operator" begin
        # Define simple operator that implements mul!
        struct DiagonalOperator{T}
            diag::Vector{T}
        end
        
        function LinearAlgebra.mul!(y, A::DiagonalOperator, x)
            y .= A.diag .* x
            return y
        end

        function Base.:*(A::DiagonalOperator, x)
            return A.diag .* x
        end
        
        n = 100
        d = rand(n) .+ 1  # Ensure positive diagonal
        A = DiagonalOperator(d)
        b = rand(n)
        x0 = zeros(n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(d .* x - b) < 1e-6
    end
    
    @testset "Ridge regression" begin
        n = 100
        A = rand(n, n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        λ = 0.1  # Regularization parameter

        # Test ridge regression with CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2

        # Test ridge regression with complex inputs
        A = rand(ComplexF64, n, n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)

        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2
    end

    @testset "Preconditioned ridge regression" begin
        # The preconditioned iteration used to drop the λ term from both the residual and the
        # matrix-vector product, so `CG(; P, λ)` silently solved the *unregularized* problem --
        # a wrong answer rather than an error. Both must land on the same solution, `(A + λI) \ b`.
        n = 60
        A = rand(n, n)
        A = A'A + I
        b = rand(n)
        x0 = zeros(n)
        λ = 0.7
        expected = (A + λ * I) \ b

        x_plain, _ = ProximalAlgorithms.CG(x0 = x0, A = A, b = b, λ = λ, maxit = 500)()
        @test norm(x_plain - expected) / norm(expected) < 1.0e-6

        P = Diagonal(diag(A) .+ λ)
        x_pre, _ = ProximalAlgorithms.CG(x0 = x0, A = A, b = b, P = P, λ = λ, maxit = 500)()
        @test norm(x_pre - expected) / norm(expected) < 1.0e-6
    end

    @testset "PCGNR with a normal operator of a different type" begin
        # `PCGNRIteration` stores `A'A` and `A'b`, so its type parameters must name those, not
        # the un-composed `A` and the measurement `b`. Naming them after the arguments made
        # every operator whose normal form has a different type fail to `convert`; an
        # `UpperTriangular` is the smallest example (`A'A` is a dense `Matrix`).
        n = 40
        A = UpperTriangular(rand(n, n) + n * I)
        b = rand(n)
        x0 = zeros(n)
        expected = (A'A) \ (A'b)

        x, _ = ProximalAlgorithms.CGNR(x0 = x0, A = A, b = b, P = Diagonal(ones(n)), maxit = 500)()
        @test norm(x - expected) / norm(expected) < 1.0e-6
    end
end

using LinearAlgebra
using Random

@testset "Factorizations grant themselves BLAS threads past a soft default" begin
    NT = ProximalOperators.NestedThreading
    PO = ProximalOperators
    Random.seed!(0)

    blas = BLAS.get_num_threads()
    counted(X) = PO.with_factorization_threads(BLAS.get_num_threads, X)
    old = PO.FACTORIZATION_THREAD_WORK[]
    try
        PO.FACTORIZATION_THREAD_WORK[] = 8 * 4 * 4
        @test PO.factorization_work(zeros(8, 4)) == 128
        NT.with_thread_default(1; only = (:blas,)) do
            @test counted(zeros(4, 4)) == 1             # below the gate: the default stands
            @test counted(zeros(8, 4)) == blas          # at the gate: BLAS's own count back
            NT.with_thread_budget(1) do                 # a hard limit is never granted past
                @test counted(zeros(8, 4)) == 1
            end
        end
        @test BLAS.get_num_threads() == blas

        # The operators that use the gate keep computing the same thing through it.
        PO.FACTORIZATION_THREAD_WORK[] = 1
        X = randn(12, 7)
        NT.with_thread_default(1; only = (:blas,)) do
            for f in (NuclearNorm(0.3), IndStiefel())
                y, fy = prox(f, X, 0.7)
                y_naive, fy_naive = ProximalOperators.prox_naive(f, X, 0.7)
                @test y ≈ y_naive
                @test fy ≈ fy_naive
            end
            S = Symmetric(X' * X - 3I)
            y, _ = prox(IndPSD(), S, 1.0)
            y_naive, _ = ProximalOperators.prox_naive(IndPSD(), S, 1.0)
            @test Matrix(y) ≈ Matrix(y_naive)
        end
        @test BLAS.get_num_threads() == blas
    finally
        PO.FACTORIZATION_THREAD_WORK[] = old
    end
end

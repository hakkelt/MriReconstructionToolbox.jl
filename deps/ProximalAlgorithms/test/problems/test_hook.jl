using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: NormL1
using ProximalAlgorithms
using LinearAlgebra
using Test

@testset "Iteration hook" begin
    A = [
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = [1.0, 2.0, 3.0, 4.0]
    n = size(A, 2)
    x0 = zeros(n)

    f = ProximalAlgorithms.AutoDifferentiable(x -> norm(A * x - b)^2 / 2, AutoZygote())
    g = NormL1(0.1 * norm(A'b, Inf))
    Lf = opnorm(A)^2

    maxit = 25

    # The hook fires once per iteration, after that iteration's state exists and before the
    # termination test, regardless of `verbose` and `freq`.
    seen = Int[]
    objectives = Float64[]
    hook = (k, alg, iter, state) -> begin
        push!(seen, k)
        push!(objectives, norm(A * state.x - b)^2 / 2 + g(state.x))
    end
    solver = ProximalAlgorithms.ForwardBackward(maxit = maxit, tol = 0, hook = hook)
    x, it = solver(x0 = x0, f = f, g = g, Lf = Lf)

    @test seen == collect(1:it)
    @test length(objectives) == it
    @test objectives[end] <= objectives[1]

    # The default is no hook, and the same run without one gives the same answer.
    x_ref, it_ref = ProximalAlgorithms.ForwardBackward(maxit = maxit, tol = 0)(
        x0 = x0, f = f, g = g, Lf = Lf
    )
    @test it == it_ref
    @test x == x_ref

    # A `nothing` hook is dispatched away rather than branched on, so it costs no allocation.
    @test ProximalAlgorithms._run_hook(nothing, 1, nothing, nothing, nothing) === nothing
    @test (@allocated ProximalAlgorithms._run_hook(nothing, 1, nothing, nothing, nothing)) == 0
end

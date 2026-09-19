using ProximalOperators
using Test

@testset "ProximalAverage" begin

    T = Float64

    @testset "constructor" begin
        f = NormL1(T(1))
        @test ProximalAverage(f, f).weights == [0.5, 0.5]
        @test_throws ArgumentError ProximalAverage((f, f), [0.5])
        @test_throws ArgumentError ProximalAverage((f, f), [-0.5, 1.5])
        @test_throws ArgumentError ProximalAverage((f, f), [0.5, 0.7])
    end

    @testset "prox is the weighted average of the proxes" begin
        f = NormL1(T(0.3))
        g = SqrNormL2(T(2))
        w = [T(0.25), T(0.75)]
        h = ProximalAverage((f, g), w)

        x = T[1.0, -2.0, 0.05, 3.5]
        gamma = T(0.7)

        yf, = prox(f, x, gamma)
        yg, = prox(g, x, gamma)
        y, hy = prox(h, x, gamma)

        @test y ≈ w[1] .* yf .+ w[2] .* yg
        # `prox!` reports the value at the point it wrote, which is the weighted average of the
        # component values there -- not the average of the components at their own prox points.
        @test hy ≈ w[1] * f(y) + w[2] * g(y)

        predicates_test(h)
        @test ProximalCore.is_convex(h) == true
        @test ProximalCore.is_smooth(h) == false

        prox_test(h, x, gamma)
    end

    @testset "a single component reduces to that component" begin
        f = NormL1(T(0.4))
        h = ProximalAverage((f,), [T(1)])
        x = T[0.2, -0.9, 1.3]
        y_h, = prox(h, x, T(0.5))
        y_f, = prox(f, x, T(0.5))
        @test y_h ≈ y_f
    end

    @testset "prox! accepts an aliasing output" begin
        f = NormL1(T(0.3))
        g = SqrNormL2(T(2))
        h = ProximalAverage(f, g)
        x = T[1.0, -2.0, 0.05, 3.5]
        expected, = prox(h, x, T(0.7))
        z = copy(x)
        prox!(z, h, z, T(0.7))
        @test z ≈ expected
    end

end

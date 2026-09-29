using Test
using ProximalOperators

# Parameters given as Float64 literals on Float32 (or ComplexF32) data: the value comes back in the
# data's precision, and `y` holds what the prox computes, rounded to its element type. Sizes cover
# the serial path and, when there are threads, the threaded one.
@testset "Float64 parameters on Float32 data ($n elements)" for n in (64, 1 << 18)
    fs = (NormL1(0.3), NormL1(fill(0.3, n)), SqrNormL2(0.3), NormL2(0.3), ElasticNet(0.3, 0.2))
    for T in (Float32, ComplexF32), f in fs
        W = T <: Complex ? ComplexF64 : Float64
        x = randn(T, n)
        y = similar(x)
        fy = prox!(y, f, x, 0.5)
        @test fy isa Real
        y64 = similar(x, W)
        fy64 = prox!(y64, f, W.(x), 0.5)
        @test y ≈ y64 rtol = 1.0e-5
        @test fy ≈ fy64 rtol = 1.0e-4
        @test f(x) ≈ f(W.(x)) rtol = 1.0e-4
    end
end

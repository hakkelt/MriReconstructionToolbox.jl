using ProximalOperators
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random

const SUITE = BenchmarkGroup()

k = "IndPSD"
SUITE[k] = BenchmarkGroup(["IndPSD"])
for T in [Float64, Complex{Float64}]
    for n in [10, 20, 50]
        SUITE[k][T, n] = @benchmarkable prox!(Y, f, X) setup=begin
            f = IndPSD()
            W = if $T <: Real
                Symmetric
            else
                Hermitian
            end
            Random.seed!(0)
            A = randn($T, $n, $n)
            X = W((A + A')/2)
            Y = similar(X)
        end
    end
end

k = "IndBox"
SUITE[k] = BenchmarkGroup(["IndBox"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        low = -ones($T, 10000)
        upp = +ones($T, 10000)
        f = IndBox(low, upp)
        x = [-2*ones($T, 3000); zeros($T, 4000); ones($T, 3000)]
        y = similar(x)
    end
end

k = "IndNonnegative"
SUITE[k] = BenchmarkGroup(["IndNonnegative"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndNonnegative()
        x = [-2*ones($T, 3000); zeros($T, 4000); ones($T, 3000)]
        y = similar(x)
    end
end

k = "NormL2"
SUITE[k] = BenchmarkGroup(["NormL2"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = NormL2()
        x = ones($T, 10000)
        y = similar(x)
    end
end

k = "LeastSquares"
SUITE[k] = BenchmarkGroup(["LeastSquares"])
for (T, s, sparse, iterative) in Iterators.product(
    [Float64, ComplexF64],
    [(5, 11), (11, 5)],
    [false, true],
    [false, true],
)
    SUITE[k][(T, s, sparse, iterative)] = @benchmarkable prox!(y, f, x) setup=begin
        A = if $sparse sparse(ones($T, $s)) else ones($T, $s) end
        b = ones($T, $(s[1]))
        f = LeastSquares(A, b, iterative=$iterative)
        x = ones($T, $(s[2]))
        y = similar(x)
    end
end

k = "IndExpPrimal"
SUITE[k] = BenchmarkGroup(["IndExpPrimal"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndExpPrimal()
        x = [0.537667139546100, 1.833885014595086, -2.258846861003648]
        y = similar(x)
    end
end

k = "IndSimplex"
SUITE[k] = BenchmarkGroup(["IndSimplex"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndSimplex()
        x = collect($T, -0.5:0.001:2.0)
        y = similar(x)
    end
end

k = "IndBallL1"
SUITE[k] = BenchmarkGroup(["IndBallL1"])
for T in [Float32, Float64]
    SUITE[k][T] = @benchmarkable prox!(y, f, x) setup=begin
        f = IndBallL1()
        x = collect($T, -2.0:0.001:0.5)
        y = similar(x)
    end
end

# ---------------------------------------------------------------------------
# Threading
#
# The groups above run at sizes where every operator is below its threshold and
# therefore serial -- which is the right thing for them to measure, since a
# serial regression is the one this change must not cause. These add the other
# half: sizes above the threshold, where the threaded and serial paths differ,
# with both built explicitly so a comparison is possible without restarting
# Julia at a different thread count.
#
# Run with threads, e.g. `julia --project=benchmark -t 8 benchmark/benchmarks.jl`.
# At `-t 1` the two entries measure the same code and should agree.

k = "threading"
SUITE[k] = BenchmarkGroup(["threading"])
for (name, build) in [
    ("NormL1", th -> NormL1(0.3; threaded = th)),                     # arithmetic
    ("SqrNormL2", th -> SqrNormL2(0.7; threaded = th)),               # memory bound
    ("LogBarrier", th -> LogBarrier(1.0, 0.0, 1.0; threaded = th)),   # transcendental
]
    for T in [Float32, Float64], n in [2^12, 2^16, 2^20], threaded in [false, true]
        SUITE[k][(name, T, n, threaded)] = @benchmarkable prox!(y, f, x, $(T(0.7))) setup = begin
            Random.seed!(0)
            f = $build($threaded)
            # LogBarrier needs a positive argument; the others do not care
            x = abs.(randn($T, $n)) .+ $T(0.5)
            y = similar(x)
        end
    end
end

# ---------------------------------------------------------------------------
# Device storage
#
# `JLArray` executes on the CPU, so these numbers are not a GPU measurement and
# must not be read as one. What they do measure is the *dispatch*: that a device
# array takes the broadcast kernels and the bisection algorithms rather than the
# scalar loops, and that neither path regresses. A real device measurement needs
# a real device.

k = "device"
SUITE[k] = BenchmarkGroup(["device"])
let jl = Base.get_extension(ProximalOperators, :GpuExt) === nothing ? nothing : nothing
    # JLArrays is a benchmark-only dependency; skip the group when it is absent
    if isdefined(Main, :JLArrays) || (Base.find_package("JLArrays") !== nothing)
        @eval using JLArrays
        for (name, build) in [
            ("NormL1", () -> NormL1(0.3)),
            ("IndSimplex", () -> IndSimplex(1.0)),   # bisection rather than Condat
            ("IndBallL1", () -> IndBallL1(1.0)),     # bisection rather than Condat
        ]
            for n in [2^12, 2^16]
                SUITE[k][(name, n)] = @benchmarkable prox!(y, f, x, 0.7) setup = begin
                    Random.seed!(0)
                    f = $build()
                    x = JLArray(randn($n))
                    y = similar(x)
                end
            end
        end
    end
end

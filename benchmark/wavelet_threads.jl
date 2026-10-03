# Threaded against serial 2-D and 3-D wavelet transforms (db2, 3 levels, ComplexF32) over a range
# of sizes, to place `threading_threshold(WaveletOp)`; and, for a two-frame batch, threading across
# the frames against threading inside each frame. Run at several thread counts:
#
#   julia --project=test --threads=N benchmark/wavelet_threads.jl [reps]
#
# Run it through benchmark/slurm/wavelet_threads.sh, which sweeps the thread counts.

using MriReconstructionToolbox
using MriReconstructionToolbox: AbstractOperators, WaveletOperators
const BatchOp = AbstractOperators.BatchOp
const WaveletOp = WaveletOperators.WaveletOp
using Wavelets: wavelet, WT
using LinearAlgebra, Printf, Random

const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const T = ComplexF32
const W = wavelet(WT.db2)
const LEVELS = 3

# `threaded = true` resolves through the size threshold under test, so the threaded variant is
# built with its threading flag set directly.
function variants(dims)
    serial = WaveletOp(T, W, dims, LEVELS; threaded = false)
    threaded = typeof(serial).name.wrapper{typeof(serial).parameters[1:4]..., true}(serial.wavelet, serial.dim_in, serial.levels)
    return serial, threaded
end

# Forward plus adjoint, minimum over `REPS` rounds, the variants interleaved round by round.
function time_pair(ops, x)
    y = similar(x)
    xb = similar(x)
    for op in ops
        mul!(y, op, x)
        mul!(xb, op', y)
    end
    best = fill(Inf, length(ops))
    for _ in 1:REPS, (i, op) in enumerate(ops)
        t = @elapsed (mul!(y, op, x); mul!(xb, op', y))
        best[i] = min(best[i], t)
    end
    return best .* 1.0e3
end

Random.seed!(0)
println("threads = ", Threads.nthreads())
@printf("%-14s %10s %12s %12s %8s\n", "size", "elements", "serial [ms]", "thread [ms]", "speedup")
for dims in ((32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (32, 32, 32), (64, 64, 64), (128, 128, 128))
    x = randn(T, dims)
    ts, tt = time_pair(variants(dims), x)
    @printf("%-14s %10d %12.3f %12.3f %8.2f\n", join(dims, "x"), prod(dims), ts, tt, ts / tt)
end

println()
println("two-frame batch: across the frames (serial transforms) against inside each frame")
@printf("%-14s %14s %14s %8s\n", "size", "across [ms]", "inside [ms]", "inside/across")
for dims in ((128, 128), (256, 256), (512, 512))
    serial, threaded = variants(dims)
    across = BatchOp(serial, 2; threaded = true)
    inside = BatchOp(threaded, 2; threaded = false)
    x = randn(T, dims..., 2)
    ta, ti = time_pair((across, inside), x)
    @printf("%-14s %14.3f %14.3f %8.2f\n", join(dims, "x"), ta, ti, ti / ta)
end

# Kernel-package sweep for ProximalOperators' threading policy.
#
# Two questions are answered here, and both feed source code rather than a report:
#
#   1. Which kernel package wins each cost class?  The answer decides what goes
#      into `Project.toml` -- a package that does not win a class it is a
#      candidate for is not added, and the negative result is recorded at the
#      bottom of this file so it is not re-litigated.
#   2. Where is the crossover between the serial and the threaded kernel?  The
#      answer becomes the `threading_threshold` methods in
#      `src/utilities/execution.jl`, each carrying a PROVENANCE comment naming
#      the machine, thread count and date that produced it.
#
# Run with the thread count the thresholds are meant for, e.g.
#
#     taskset -c 0-15 julia --project=benchmark -t 8 benchmark/threading_sweep.jl
#
# Three kernels stand in for the three cost classes of `execution.jl`:
#
#   soft_threshold  ARITHMETIC       write-only, a few flops per element
#   sqrnorm         MEMORY_BOUND     write plus a fused reduction
#   logistic        TRANSCENDENTAL   exp/log per element
#
# The fused write-and-reduce shape of `sqrnorm` is the one that matters most: it
# is what most of `src/functions/` looks like, and only `Polyester.@batch
# reduction` expresses it in a single pass.  The other packages are measured on
# the honest two-pass equivalent (broadcast, then `sum`), which is what using
# them would actually cost.

using BenchmarkTools
using Printf

using Polyester: @batch

const HAVE_FASTBROADCAST = try; using FastBroadcast; true; catch; false; end
const HAVE_LOOPVEC       = try; using LoopVectorization; true; catch; false; end
const HAVE_STRIDED       = try; using Strided; true; catch; false; end

const SIZES   = [2^k for k in 6:2:22]
const ELTYPES = (Float32, Float64, ComplexF64)

soft(x, gl) = x + (x <= -gl ? gl : (x >= gl ? -gl : -x))
soft(x::Complex, gl) = sign(x) * (abs(x) <= gl ? zero(abs(x)) : abs(x) - gl)

# ---------------------------------------------------------------- arithmetic

function soft_serial!(y, x, gl)
    @inbounds @simd for i in eachindex(x, y)
        y[i] = soft(x[i], gl)
    end
    y
end

function soft_batch!(y, x, gl)
    @batch minbatch = 4096 for i in eachindex(x, y)
        @inbounds y[i] = soft(x[i], gl)
    end
    y
end

soft_fastbc!(y, x, gl) = (@.. thread = true y = soft(x, gl); y)
soft_turbo!(y, x, gl)  = (@turbo @. y = soft(x, gl); y)
soft_strided!(y, x, gl) = (@strided @. y = soft(x, gl); y)

# -------------------------------------------------------------- memory bound
# write y and reduce sum(abs2, y) -- the shape of SqrNormL2 and of most prox!

function sqrnorm_serial!(y, x, g)
    acc = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x, y)
        yi = x[i] * g
        y[i] = yi
        acc += abs2(yi)
    end
    acc
end

function sqrnorm_batch!(y, x, g)
    acc = real(eltype(x))(0)
    @batch reduction = ((+, acc),) minbatch = 4096 for i in eachindex(x, y)
        @inbounds yi = x[i] * g
        @inbounds y[i] = yi
        acc += abs2(yi)
    end
    acc
end

sqrnorm_fastbc!(y, x, g)  = (@.. thread = true y = x * g; sum(abs2, y))
sqrnorm_turbo!(y, x, g)   = (@turbo @. y = x * g; sum(abs2, y))
sqrnorm_strided!(y, x, g) = (@strided @. y = x * g; sum(abs2, y))

# ------------------------------------------------------------ transcendental
# LogisticLoss' inner pass: write the Newton residual, accumulate the value

logi(x, mu) = log(one(real(x)) + exp(-mu * real(x)))

function logistic_serial!(y, x, mu)
    acc = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x, y)
        yi = logi(x[i], mu)
        y[i] = yi
        acc += yi
    end
    acc
end

function logistic_batch!(y, x, mu)
    acc = real(eltype(x))(0)
    @batch reduction = ((+, acc),) minbatch = 1024 for i in eachindex(x, y)
        @inbounds yi = logi(x[i], mu)
        @inbounds y[i] = yi
        acc += yi
    end
    acc
end

logistic_fastbc!(y, x, mu)  = (@.. thread = true y = logi(x, mu); sum(y))
logistic_turbo!(y, x, mu)   = (@turbo @. y = logi(x, mu); sum(y))
logistic_strided!(y, x, mu) = (@strided @. y = logi(x, mu); sum(y))

# ------------------------------------------------------------------- driver

struct Kernel
    name::String
    class::String
    impls::Vector{Pair{String, Any}}
    outtype::Symbol      # :same or :real
end

const KERNELS = let
    arith = ["serial" => soft_serial!, "@batch" => soft_batch!]
    mem   = ["serial" => sqrnorm_serial!, "@batch" => sqrnorm_batch!]
    trans = ["serial" => logistic_serial!, "@batch" => logistic_batch!]
    if HAVE_FASTBROADCAST
        push!(arith, "@.." => soft_fastbc!)
        push!(mem,   "@.." => sqrnorm_fastbc!)
        push!(trans, "@.." => logistic_fastbc!)
    end
    if HAVE_LOOPVEC
        push!(arith, "@turbo" => soft_turbo!)
        push!(mem,   "@turbo" => sqrnorm_turbo!)
        push!(trans, "@turbo" => logistic_turbo!)
    end
    if HAVE_STRIDED
        push!(arith, "@strided" => soft_strided!)
        push!(mem,   "@strided" => sqrnorm_strided!)
        push!(trans, "@strided" => logistic_strided!)
    end
    [
        Kernel("soft_threshold", "ARITHMETIC", arith, :same),
        Kernel("sqrnorm", "MEMORY_BOUND", mem, :same),
        Kernel("logistic", "TRANSCENDENTAL", trans, :real),
    ]
end

function measure(kern, T, n)
    x = T <: Complex ? randn(T, n) : randn(T, n)
    y = kern.outtype === :real ? similar(x, real(T)) : similar(x)
    p = real(T)(0.5)
    times = Dict{String, Float64}()
    for (name, fn) in kern.impls
        try
            fn(y, x, p)   # compile / validate
            b = @benchmark $fn($y, $x, $p) samples = 200 evals = 1 seconds = 1.0
            times[name] = minimum(b).time
        catch err
            times[name] = NaN
        end
    end
    times
end

function main()
    @printf("# threading_sweep  julia %s  nthreads=%d  %s\n",
            VERSION, Threads.nthreads(), Sys.CPU_NAME)
    @printf("# FastBroadcast=%s LoopVectorization=%s Strided=%s\n",
            HAVE_FASTBROADCAST, HAVE_LOOPVEC, HAVE_STRIDED)
    for kern in KERNELS
        names = first.(kern.impls)
        println("\n## ", kern.name, "  (", kern.class, ")")
        @printf("%-12s %9s", "eltype", "n")
        for nm in names
            @printf(" %12s", nm)
        end
        @printf(" %14s\n", "best")
        for T in ELTYPES
            for n in SIZES
                t = measure(kern, T, n)
                @printf("%-12s %9d", string(T), n)
                for nm in names
                    @printf(" %12.1f", t[nm])
                end
                ok = filter(p -> !isnan(p.second), collect(t))
                best = isempty(ok) ? "-" : first(argmin(p -> p.second, ok))
                sp = isempty(ok) ? 0.0 : t["serial"] / minimum(p -> p.second, ok)
                @printf(" %10s %4.2fx\n", best, sp)
            end
        end
    end
end

main()

# ---------------------------------------------------------------------------
# RESULTS  --  2026-09-16, znver2 (96 cores, taskset 0-15), julia 1.13.0, -t 8
# ---------------------------------------------------------------------------
#
# Dependency decisions.  Re-run this file before changing any of them.
#
#   Polyester           IN.  Wins ARITHMETIC and MEMORY_BOUND above threshold,
#                            and `@batch reduction` is the only construct here
#                            that expresses the fused write-and-reduce shape in
#                            a single pass.  The two-pass alternatives
#                            (broadcast, then `sum`) lose that class outright:
#                            at Float64 n = 2^20, `@batch` 252 us vs `@..`
#                            1055 us vs `@strided` 991 us vs serial 376 us.
#
#   LoopVectorization   IN.  Owns TRANSCENDENTAL completely, and by a margin
#                            nothing else approaches, because SLEEFPirates
#                            vectorises exp/log where Base does not:
#                              Float64 n = 2^12  serial 84.6us  @turbo 18.3us  @tturbo  5.4us
#                              Float64 n = 2^22  serial 88.3ms  @batch 19.2ms  @tturbo  4.2ms
#                              Float32 n = 2^22  serial 86.5ms  @batch 19.0ms  @tturbo  1.8ms
#                            i.e. 4.6x (Float64) to 10x (Float32) better than
#                            the best non-LV option, and 3-7x better than
#                            serial at sizes far below any threading threshold.
#                            Real eltypes only: `check_args` rejects Complex
#                            arrays and silently falls back, so the kernels
#                            gate on `isreal` and pass `warn_check_args=false`.
#
#   FastBroadcast       OUT. No clear margin.  On ARITHMETIC it lands within
#                            ~20% of `@batch` either way (Float64 n = 2^20:
#                            61.0us vs 62.0us; ComplexF64 n = 2^20: 5.35ms vs
#                            6.42ms), and it cannot express the fused
#                            reduction at all.  Its apparent 6x win on small
#                            complex inputs in the first pass was not codegen
#                            but an absent size guard -- `@.. thread=false`
#                            measures identical to the plain `@inbounds @simd`
#                            loop (n = 2^10: 48.98us vs 49.52us), so it was
#                            simply threading at a size where `@batch`'s
#                            `minbatch` still held it serial.  Our own
#                            `should_thread` makes that choice explicitly.
#
#   Strided             OUT. No clear margin.  Wins exactly one cell family --
#                            ComplexF64 ARITHMETIC at n >= 2^20, by ~1.25x over
#                            `@..` and ~1.55x over `@batch` -- and loses
#                            everywhere else, often badly (Float32
#                            MEMORY_BOUND n = 2^18: 120us vs 38us serial).
#
# Thresholds.  All three cost-class constants in src/utilities/execution.jl are
# confirmed by this sweep rather than assumed; the crossovers measured were
#
#   ARITHMETIC       2^15   serial 5.40us vs @batch 2.71us at Float64 2^14;
#                           6.80us vs 2.67us at Float32 2^16 (2.55x)
#   MEMORY_BOUND     2^18   serial wins every smaller size; @batch 1.15x at
#                           Float64 2^18, 1.49x at 2^20, 2.11x at 2^22
#   TRANSCENDENTAL   2^10   @turbo wins below (Float32 2^6: 180ns vs 1052ns),
#                           @tturbo above (Float32 2^10: 1462ns vs 2053ns)
#
# Note the shape of the MEMORY_BOUND column: `@batch reduction` costs real
# overhead even under `minbatch` (Float32 n = 2^10: 992ns vs 80ns serial), which
# is why `_map_reduce_prox!` branches to a plain `@inbounds @simd` loop rather
# than relying on `minbatch` to degrade gracefully.
#
# One trap worth recording, because the sweep above cannot catch it and the
# end-to-end numbers can.  `NestedThreading.@budgeted` chooses whether to exclude
# the `:polyester` pool from the budget scope by reading the *name* of the macro
# it wraps, and it reads that name off the macrocall head.  Emitting the head as
# `GlobalRef(Polyester, Symbol("@batch"))` makes it unreadable, so the scope
# restricts the very pool the loop then runs on.  The loops still ran, and
# still produced correct results, but every threaded elementwise operator came
# out *slower than serial*: at n = 2^20, NormL1 measured 1197us threaded against
# 467us serial (0.39x).  With the head emitted as `Polyester.@batch` (see
# `_BATCH_HEAD` in src/utilities/execution.jl) the same case is 1.5x, and
# IndNonnegative 3.7x, on a 4-thread host.  Any change to how the loop macros
# build that expression should be checked against a real `prox!` speedup, not
# only against the test suite.
